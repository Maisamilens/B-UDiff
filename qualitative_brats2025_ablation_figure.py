import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from medpy import metric
from matplotlib.patches import Rectangle
from monai.inferers import SlidingWindowInferer


LABEL_COLORS = {
    1: np.array([255, 107, 129], dtype=np.float32) / 255.0,
    2: np.array([255, 232, 128], dtype=np.float32) / 255.0,
    3: np.array([31, 177, 112], dtype=np.float32) / 255.0,
}


def filte_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    return {str(k)[7:] if str(k).startswith("module") else str(k): v for k, v in sd.items()}


def normalize_slice(img):
    img = np.asarray(img, dtype=np.float32)
    mask = img != 0
    lo, hi = np.percentile(img[mask] if mask.any() else img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    return (np.clip(img, lo, hi) - lo) / (hi - lo)


def overlay_labels(base, labels, alpha=0.58):
    gray = normalize_slice(base)
    rgb = np.stack([gray, gray, gray], axis=-1)
    for label, color in LABEL_COLORS.items():
        mask = labels == label
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return rgb


def choose_slice(seg):
    areas = np.sum(seg > 0, axis=(1, 2))
    return int(np.argmax(areas)) if np.max(areas) > 0 else seg.shape[0] // 2


def add_tumor_box(ax, seg_slice):
    coords = np.argwhere(seg_slice > 0)
    if coords.size == 0:
        return
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    pad = 6
    x0, y0 = max(int(x0) - pad, 0), max(int(y0) - pad, 0)
    x1, y1 = min(int(x1) + pad, seg_slice.shape[1] - 1), min(int(y1) + pad, seg_slice.shape[0] - 1)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#d7191c", lw=1.4))


def load_case(data_dir, case_name):
    data_dir = Path(data_dir)
    image_path = data_dir / f"{case_name}.npy"
    seg_path = data_dir / f"{case_name}_seg.npy"
    if image_path.exists() and seg_path.exists():
        return np.load(image_path, mmap_mode="r"), np.load(seg_path, mmap_mode="r")[0]
    npz = np.load(data_dir / f"{case_name}.npz")
    return npz["data"], npz["seg"][0]


def representative_cases(metrics_csv, count):
    with open(metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["Avg_dice"]))
    idxs = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[i]["case"] for i in idxs]


def predict_logits(inferer, image, fn, device):
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(image).copy()).unsqueeze(0).float().to(device)
        out = inferer(x, fn)
        if isinstance(out, tuple):
            out = out[0]
        return out.argmax(dim=1).cpu().numpy()[0]


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


def hd95_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 50.0
    return float(metric.binary.hd95(pred, gt, voxelspacing=[1.0, 1.0, 1.0]))


def add_metrics(rows, case_name, variant, pred, seg):
    pred_r = regions(pred)
    gt_r = regions(seg)
    row = {"case": case_name, "variant": variant}
    for target in ["WT", "TC", "ET"]:
        row[f"{target}_dice"] = dice_score(pred_r[target], gt_r[target])
        row[f"{target}_hd95"] = hd95_score(pred_r[target], gt_r[target])
    row["Avg_dice"] = float(np.mean([row[f"{t}_dice"] for t in ["WT", "TC", "ET"]]))
    row["Avg_hd95"] = float(np.mean([row[f"{t}_hd95"] for t in ["WT", "TC", "ET"]]))
    rows.append(row)


def write_ablation_metrics(out_png, rows):
    out_dir = Path(out_png).parent
    per_case = out_dir / "brats2025_ablation_inference_per_case.csv"
    summary = out_dir / "brats2025_ablation_inference_summary.csv"
    fields = ["case", "variant", "WT_dice", "WT_hd95", "TC_dice", "TC_hd95", "ET_dice", "ET_hd95", "Avg_dice", "Avg_hd95"]
    with per_case.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    variants = []
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])
    with summary.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Variant", "WT Dice", "WT HD95", "TC Dice", "TC HD95", "ET Dice", "ET HD95", "Avg Dice", "Avg HD95"])
        for variant in variants:
            vr = [r for r in rows if r["variant"] == variant]
            values = [variant]
            for target in ["WT", "TC", "ET"]:
                values.append(f"{np.mean([r[f'{target}_dice'] for r in vr]) * 100:.2f}")
                values.append(f"{np.mean([r[f'{target}_hd95'] for r in vr]):.2f}")
            values.append(f"{np.mean([r['Avg_dice'] for r in vr]) * 100:.2f}")
            values.append(f"{np.mean([r['Avg_hd95'] for r in vr]):.2f}")
            writer.writerow(values)
    return per_case, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/fullres/train")
    parser.add_argument("--checkpoint", default="./logs/diffunet/model/best_model_0.9027.pt")
    parser.add_argument("--out", default="./visualization_outputs/brats2025_ablation_inference_paperstyle.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--modality", type=int, default=2)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--metrics-csv", default="./quantitative_results/DiffUNet_250cases_per_case_metrics.csv")
    args = parser.parse_args()

    from diffunet.diffunet_model import DiffUNet

    full_model = DiffUNet(4, 4)
    full_model.load_state_dict(filte_state_dict(torch.load(args.checkpoint, map_location="cpu")))
    full_model.to(args.device).eval()

    no_bta_model = DiffUNet(4, 4, bta=False)
    no_bta_model.load_state_dict(filte_state_dict(torch.load(args.checkpoint, map_location="cpu")))
    no_bta_model.to(args.device).eval()

    one_step_model = DiffUNet(4, 4)
    one_step_model.load_state_dict(filte_state_dict(torch.load(args.checkpoint, map_location="cpu")))
    one_step_model.denoise_model.timesteps = 1
    one_step_model.to(args.device).eval()

    inferer = SlidingWindowInferer(roi_size=[128, 128, 128], sw_batch_size=1, overlap=0.5, mode="gaussian")
    cases = args.cases or representative_cases(args.metrics_csv, args.case_count)

    cols = ["Image", "GT", "Full", "1-step", "No BTA", "Denoise only", "Boundary only"]
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(len(cols) * 2.05, len(cases) * 2.25))
    if len(cases) == 1:
        axes = axes[None, :]

    metric_rows = []
    for row_idx, case_name in enumerate(cases):
        image, seg = load_case(args.data_dir, case_name)
        z = choose_slice(seg)
        base = image[args.modality, z]
        gt_slice = seg[z]

        full_pred = predict_logits(inferer, image, lambda x: full_model(x, ddim=True), args.device)
        one_step_pred = predict_logits(inferer, image, lambda x: one_step_model(x, ddim=True), args.device)
        no_bta_pred = predict_logits(inferer, image, lambda x: no_bta_model(x, ddim=True), args.device)
        denoise_pred = predict_logits(
            inferer,
            image,
            lambda x: full_model.denoise_model(x, embeddings=None, ddim=True),
            args.device,
        )
        boundary_pred = predict_logits(
            inferer,
            image,
            lambda x: full_model.edge_model(x)[0],
            args.device,
        )
        for variant, pred in [
            ("Full", full_pred),
            ("1-step", one_step_pred),
            ("No BTA", no_bta_pred),
            ("Denoise only", denoise_pred),
            ("Boundary only", boundary_pred),
        ]:
            add_metrics(metric_rows, case_name, variant, pred, seg)

        panels = [
            normalize_slice(base),
            overlay_labels(base, gt_slice),
            overlay_labels(base, full_pred[z]),
            overlay_labels(base, one_step_pred[z]),
            overlay_labels(base, no_bta_pred[z]),
            overlay_labels(base, denoise_pred[z]),
            overlay_labels(base, boundary_pred[z]),
        ]
        for col_idx, panel in enumerate(panels):
            axes[row_idx, col_idx].imshow(panel, cmap="gray" if col_idx == 0 else None)
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_linewidth(1.1)
                spine.set_color("white")
        add_tumor_box(axes[row_idx, 0], gt_slice)
        axes[row_idx, 0].set_ylabel(case_name, fontsize=8, rotation=90, labelpad=18)

    for col_idx, name in enumerate(cols):
        axes[-1, col_idx].set_xlabel(name, fontsize=12, labelpad=8)
    fig.text(0.012, 0.5, "BraTS2025 Glioma", va="center", rotation=90, fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0.05, 0.04, 1.0, 0.995], w_pad=0.28, h_pad=0.28)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    per_case, summary = write_ablation_metrics(out, metric_rows)
    print(f"Saved ablation figure: {out}")
    print(f"Saved ablation per-case metrics: {per_case}")
    print(f"Saved ablation summary metrics: {summary}")
    print("Cases:", ", ".join(cases))
    print("Note: these are inference-time ablations from one checkpoint, not separately retrained ablation models.")


if __name__ == "__main__":
    main()
