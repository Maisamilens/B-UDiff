import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from medpy import metric
from monai.data import DataLoader
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm

from light_training.dataloading.dataset import MedicalDataset, get_train_test_loader_from_test_list
from test_list_brats2025 import test_list


TARGETS = ["WT", "TC", "ET"]


def filte_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    return {str(k)[7:] if str(k).startswith("module") else str(k): v for k, v in sd.items()}


def regions(mask):
    return {
        "WT": mask > 0,
        "TC": (mask == 1) | (mask == 3),
        "ET": mask == 3,
    }


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    return float(metric.binary.dc(pred, gt))


def hd95_score(pred, gt, spacing):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 50.0
    return float(metric.binary.hd95(pred, gt, voxelspacing=spacing))


def metadata_name(properties):
    name = properties.get("name", "case")
    while isinstance(name, (list, tuple)) and name:
        name = name[0]
    return str(name)


def metadata_spacing(properties):
    spacing = properties.get("spacing", [1.0, 1.0, 1.0])
    if isinstance(spacing, torch.Tensor):
        spacing = spacing.detach().cpu().numpy()
    if isinstance(spacing, np.ndarray):
        spacing = spacing.reshape(-1).tolist()
    elif isinstance(spacing, (list, tuple)):
        flat = []
        for item in spacing:
            if isinstance(item, torch.Tensor):
                flat.extend(item.detach().cpu().numpy().reshape(-1).tolist())
            elif isinstance(item, np.ndarray):
                flat.extend(item.reshape(-1).tolist())
            elif isinstance(item, (list, tuple)):
                flat.extend(np.asarray(item).reshape(-1).tolist())
            else:
                flat.append(item)
        spacing = flat
    spacing = [float(v) for v in spacing[:3]]
    return spacing if len(spacing) == 3 else [1.0, 1.0, 1.0]


def evaluate_prediction(case_name, variant, pred, label, spacing):
    pred_r = regions(pred)
    gt_r = regions(label)
    row = {"case": case_name, "variant": variant}
    for target in TARGETS:
        row[f"{target}_dice"] = dice_score(pred_r[target], gt_r[target])
        row[f"{target}_hd95"] = hd95_score(pred_r[target], gt_r[target], spacing)
    row["Avg_dice"] = float(np.mean([row[f"{t}_dice"] for t in TARGETS]))
    row["Avg_hd95"] = float(np.mean([row[f"{t}_hd95"] for t in TARGETS]))
    return row


def mean_std(values, percent=False):
    arr = np.asarray(values, dtype=float)
    if percent:
        arr = arr * 100.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def ci95(values, percent=False):
    arr = np.asarray(values, dtype=float)
    if percent:
        arr = arr * 100.0
    mean = float(np.mean(arr))
    if len(arr) < 2:
        return mean, mean
    delta = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(len(arr))
    return mean - delta, mean + delta


def paired_pvalue(full_values, variant_values):
    try:
        from scipy import stats
    except Exception:
        return "-"
    if len(full_values) != len(variant_values) or len(full_values) < 2:
        return "-"
    p = stats.ttest_rel(full_values, variant_values, nan_policy="omit").pvalue
    return "-" if np.isnan(p) else f"{float(p):.4g}"


def load_model(checkpoint, device):
    from diffunet.diffunet_model import DiffUNet

    model = DiffUNet(4, 4)
    model.load_state_dict(filte_state_dict(torch.load(checkpoint, map_location="cpu")))
    model.to(device)
    model.eval()
    return model


def predict_variant(model, inferer, image, variant, device):
    old_steps = model.denoise_model.timesteps
    old_accumulation = model.denoise_model.accumulation
    old_mode = model.ablation_mode
    try:
        if variant == "Full":
            model.set_ablation_mode("full")
        elif variant == "w/o MBA":
            model.set_ablation_mode("no_mba")
        elif variant == "w/o boundary logits":
            model.set_ablation_mode("no_boundary_logits")
        elif variant == "Denoise only":
            model.set_ablation_mode("denoise_only")
        elif variant == "Boundary only":
            model.set_ablation_mode("boundary_only")
        elif variant == "w/o PURE":
            model.set_ablation_mode("full")
            model.denoise_model.accumulation = False
        elif variant == "1-step DDIM":
            model.set_ablation_mode("full")
            model.denoise_model.timesteps = 1
        else:
            raise ValueError(f"Unknown variant: {variant}")

        with torch.no_grad():
            output = inferer(image, lambda x: model(x, ddim=True))
            if isinstance(output, tuple):
                output = output[0]
            return output.argmax(dim=1).cpu().numpy()[0]
    finally:
        model.denoise_model.timesteps = old_steps
        model.denoise_model.accumulation = old_accumulation
        model.set_ablation_mode(old_mode)


def write_outputs(out_dir, method_rows):
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_path = out_dir / "brats2025_ablation_per_case_metrics.csv"
    fields = ["case", "variant"]
    for target in TARGETS:
        fields += [f"{target}_dice", f"{target}_hd95"]
    fields += ["Avg_dice", "Avg_hd95"]
    with per_case_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(method_rows)

    variants = []
    for row in method_rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])

    full_rows = [r for r in method_rows if r["variant"] == "Full"]
    full_by_case = {r["case"]: r for r in full_rows}

    summary_path = out_dir / "brats2025_ablation_summary_table.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Variant",
            "WT Dice", "WT HD95",
            "TC Dice", "TC HD95",
            "ET Dice", "ET HD95",
            "Avg Dice", "Dice CI", "Dice p-value",
            "Avg HD95", "HD95 CI", "HD95 p-value",
        ])
        for variant in variants:
            rows = [r for r in method_rows if r["variant"] == variant]
            values = [variant]
            for target in TARGETS:
                dm, ds = mean_std([r[f"{target}_dice"] for r in rows], percent=True)
                hm, hs = mean_std([r[f"{target}_hd95"] for r in rows])
                values += [f"{dm:.2f} +/- {ds:.2f}", f"{hm:.2f} +/- {hs:.2f}"]

            avg_dm, avg_ds = mean_std([r["Avg_dice"] for r in rows], percent=True)
            avg_hm, avg_hs = mean_std([r["Avg_hd95"] for r in rows])
            dice_ci = ci95([r["Avg_dice"] for r in rows], percent=True)
            hd_ci = ci95([r["Avg_hd95"] for r in rows])

            aligned = [r for r in rows if r["case"] in full_by_case]
            dice_p = "-" if variant == "Full" else paired_pvalue(
                [full_by_case[r["case"]]["Avg_dice"] for r in aligned],
                [r["Avg_dice"] for r in aligned],
            )
            hd_p = "-" if variant == "Full" else paired_pvalue(
                [full_by_case[r["case"]]["Avg_hd95"] for r in aligned],
                [r["Avg_hd95"] for r in aligned],
            )

            values += [
                f"{avg_dm:.2f} +/- {avg_ds:.2f}",
                f"({dice_ci[0] / 100:.3f}, {dice_ci[1] / 100:.3f})",
                dice_p,
                f"{avg_hm:.2f} +/- {avg_hs:.2f}",
                f"({hd_ci[0]:.2f}, {hd_ci[1]:.2f})",
                hd_p,
            ]
            writer.writerow(values)
    return per_case_path, summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/fullres/train")
    parser.add_argument("--checkpoint", default="./logs/diffunet/model/best_model_0.9027.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="./ablation_results")
    parser.add_argument("--max-cases", type=int, default=25)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--variants", nargs="*", default=[
        "Full", "w/o MBA", "w/o PURE", "1-step DDIM",
        "w/o boundary logits", "Denoise only", "Boundary only",
    ])
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    if args.all_cases:
        dataset = MedicalDataset(sorted(str(p) for p in Path(args.data_dir).glob("*.npz")))
    else:
        _, dataset = get_train_test_loader_from_test_list(args.data_dir, test_list)
    if args.max_cases is not None and not args.all_cases:
        dataset.datalist = dataset.datalist[: args.max_cases]

    model = load_model(args.checkpoint, args.device)
    inferer = SlidingWindowInferer(roi_size=[128, 128, 128], sw_batch_size=1, overlap=0.5, mode="gaussian")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    rows = []
    for batch in tqdm(loader, total=len(loader), desc="Ablation cases"):
        image = batch["data"].float().to(args.device)
        label = batch["seg"][:, 0].long().cpu().numpy()[0]
        properties = batch["properties"]
        case_name = metadata_name(properties)
        spacing = metadata_spacing(properties)
        for variant in args.variants:
            pred = predict_variant(model, inferer, image, variant, args.device)
            rows.append(evaluate_prediction(case_name, variant, pred, label, spacing))

    per_case_path, summary_path = write_outputs(Path(args.out_dir), rows)
    print(f"Saved per-case metrics: {per_case_path}")
    print(f"Saved summary table: {summary_path}")


if __name__ == "__main__":
    main()
