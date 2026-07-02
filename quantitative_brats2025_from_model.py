import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from medpy import metric
from monai.inferers import SlidingWindowInferer
from monai.data import DataLoader
from tqdm import tqdm

from light_training.dataloading.dataset import MedicalDataset, get_train_test_loader_from_test_list
from test_list_brats2025 import test_list


TARGETS = ["WT", "TC", "ET"]


def filte_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    new_sd = {}
    for k, v in sd.items():
        k = str(k)
        new_sd[k[7:] if k.startswith("module") else k] = v
    return new_sd


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


def regions(mask):
    return {
        "WT": mask > 0,
        "TC": (mask == 1) | (mask == 3),
        "ET": mask == 3,
    }


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


def format_cell(mean, std):
    return f"{mean:.2f} +/- {std:.2f}"


def save_csv(path, rows):
    fields = ["case"]
    for target in TARGETS:
        fields += [f"{target}_dice", f"{target}_hd95"]
    fields += ["Avg_dice", "Avg_hd95"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_summary(path, method, rows):
    headers = [
        "Method",
        "WT Dice", "WT HD95",
        "TC Dice", "TC HD95",
        "ET Dice", "ET HD95",
        "Avg Dice", "Dice CI", "Dice p-value",
        "Avg HD95", "HD95 CI", "HD95 p-value",
    ]
    values = [method]
    for target in TARGETS:
        dice_m, dice_s = mean_std([r[f"{target}_dice"] for r in rows], percent=True)
        hd_m, hd_s = mean_std([r[f"{target}_hd95"] for r in rows])
        values += [format_cell(dice_m, dice_s), format_cell(hd_m, hd_s)]

    avg_dice_m, avg_dice_s = mean_std([r["Avg_dice"] for r in rows], percent=True)
    avg_hd_m, avg_hd_s = mean_std([r["Avg_hd95"] for r in rows])
    dice_ci = ci95([r["Avg_dice"] for r in rows], percent=True)
    hd_ci = ci95([r["Avg_hd95"] for r in rows])
    values += [
        format_cell(avg_dice_m, avg_dice_s),
        f"({dice_ci[0] / 100:.3f}, {dice_ci[1] / 100:.3f})",
        "-",
        format_cell(avg_hd_m, avg_hd_s),
        f"({hd_ci[0]:.2f}, {hd_ci[1]:.2f})",
        "-",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(values)
    return headers, values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/fullres/train")
    parser.add_argument("--checkpoint", default="./logs/diffunet/model/final_model_0.8973.pt")
    parser.add_argument("--method", default="DiffUNet")
    parser.add_argument("--out-dir", default="./quantitative_results")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--max-cases", type=int, default=None, help="Use a small number for quick testing.")
    parser.add_argument("--all-cases", action="store_true", help="Evaluate all .npz cases in --data-dir.")
    args = parser.parse_args()

    from diffunet.diffunet_model import DiffUNet

    if args.all_cases:
        val_ds = MedicalDataset(sorted(str(p) for p in Path(args.data_dir).glob("*.npz")))
    else:
        _, val_ds = get_train_test_loader_from_test_list(args.data_dir, test_list)
    if args.max_cases is not None:
        val_ds.datalist = val_ds.datalist[: args.max_cases]

    model = DiffUNet(4, 4)
    model.load_state_dict(filte_state_dict(torch.load(args.checkpoint, map_location="cpu")))
    model.to(args.device)
    model.eval()

    inferer = SlidingWindowInferer(
        roi_size=args.patch_size,
        sw_batch_size=1,
        overlap=0.5,
        mode="gaussian",
    )

    rows = []
    loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    for batch in tqdm(loader, total=len(loader), desc="Evaluating"):
        image = batch["data"].float().to(args.device)
        label = batch["seg"][:, 0].long().cpu().numpy()[0]
        properties = batch["properties"]
        case_name = metadata_name(properties)
        spacing = metadata_spacing(properties)

        with torch.no_grad():
            output = inferer(image, lambda x: model(x, ddim=True))
            if isinstance(output, tuple):
                output = output[0]
            pred = output.argmax(dim=1).cpu().numpy()[0]

        pred_regions = regions(pred)
        gt_regions = regions(label)
        row = {"case": case_name}
        for target in TARGETS:
            row[f"{target}_dice"] = dice_score(pred_regions[target], gt_regions[target])
            row[f"{target}_hd95"] = hd95_score(pred_regions[target], gt_regions[target], spacing)
        row["Avg_dice"] = float(np.mean([row[f"{t}_dice"] for t in TARGETS]))
        row["Avg_hd95"] = float(np.mean([row[f"{t}_hd95"] for t in TARGETS]))
        rows.append(row)

        print(
            f"{case_name}: "
            f"WT Dice={row['WT_dice'] * 100:.2f}, HD95={row['WT_hd95']:.2f}; "
            f"TC Dice={row['TC_dice'] * 100:.2f}, HD95={row['TC_hd95']:.2f}; "
            f"ET Dice={row['ET_dice'] * 100:.2f}, HD95={row['ET_hd95']:.2f}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    suffix = f"{args.method}_{len(rows)}cases"
    per_case_path = out_dir / f"{suffix}_per_case_metrics.csv"
    summary_path = out_dir / f"{suffix}_summary_table.csv"
    save_csv(per_case_path, rows)
    headers, values = save_summary(summary_path, args.method, rows)
    print(",".join(headers))
    print(",".join(values))
    print(f"Saved per-case metrics: {per_case_path}")
    print(f"Saved summary table: {summary_path}")


if __name__ == "__main__":
    main()
