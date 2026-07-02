import argparse
import csv
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


LABEL_COLORS = {
    1: np.array([255, 107, 129], dtype=np.float32) / 255.0,
    2: np.array([255, 232, 128], dtype=np.float32) / 255.0,
    3: np.array([31, 177, 112], dtype=np.float32) / 255.0,
}

TARGETS = {
    "WT": lambda x: x > 0,
    "TC": lambda x: (x == 1) | (x == 3),
    "ET": lambda x: x == 3,
}


def normalize_slice(img):
    img = np.asarray(img, dtype=np.float32)
    mask = img != 0
    values = img[mask] if mask.any() else img.reshape(-1)
    lo, hi = np.percentile(values, [1, 99])
    if hi <= lo:
        return np.zeros_like(img, dtype=np.float32)
    img = np.clip(img, lo, hi)
    return (img - lo) / (hi - lo)


def overlay_labels(base, labels, alpha=0.58):
    gray = normalize_slice(base)
    rgb = np.stack([gray, gray, gray], axis=-1)
    labels = np.asarray(labels)
    for label, color in LABEL_COLORS.items():
        mask = labels == label
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return rgb


def boundary(mask):
    mask = np.asarray(mask).astype(bool)
    eroded = mask.copy()
    eroded[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    return mask & ~eroded


def contour_overlay(base, gt, pred):
    gray = normalize_slice(base)
    rgb = np.stack([gray, gray, gray], axis=-1)
    gt_b = boundary(gt > 0)
    pred_b = boundary(pred > 0)
    rgb[gt_b] = np.array([0.0, 1.0, 0.1])
    rgb[pred_b] = np.array([1.0, 0.05, 0.05])
    both = gt_b & pred_b
    rgb[both] = np.array([1.0, 1.0, 0.0])
    return rgb


def entropy_from_prob(prob):
    prob = np.asarray(prob, dtype=np.float32)
    entropy = -np.sum(prob * np.log(np.clip(prob, 1e-7, 1.0)), axis=0)
    return entropy / np.log(prob.shape[0])


def choose_slices(seg, pred, count=3):
    volume = (np.asarray(seg) > 0) | (np.asarray(pred) > 0)
    areas = np.sum(volume, axis=(1, 2))
    nonzero = np.flatnonzero(areas)
    if nonzero.size == 0:
        mid = seg.shape[0] // 2
        return [mid]
    if count == 1:
        return [int(nonzero[np.argmax(areas[nonzero])])]
    idxs = np.linspace(0, nonzero.size - 1, count, dtype=int)
    return [int(nonzero[i]) for i in idxs]


def crop_to_tumor(arrays, mask, pad=20):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return arrays
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    y0 = max(int(y0) - pad, 0)
    x0 = max(int(x0) - pad, 0)
    y1 = min(int(y1) + pad, mask.shape[0] - 1)
    x1 = min(int(x1) + pad, mask.shape[1] - 1)
    return [arr[y0 : y1 + 1, x0 : x1 + 1] for arr in arrays]


def add_box(ax, mask, color="#d7191c"):
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    pad = 6
    x0 = max(int(x0) - pad, 0)
    y0 = max(int(y0) - pad, 0)
    x1 = min(int(x1) + pad, mask.shape[1] - 1)
    y1 = min(int(y1) + pad, mask.shape[0] - 1)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.2))


def load_case(data_dir, case):
    image = np.load(Path(data_dir) / f"{case}.npy", mmap_mode="r")
    seg = np.load(Path(data_dir) / f"{case}_seg.npy", mmap_mode="r")[0]
    return image, seg


def read_metrics(metrics_csv):
    path = Path(metrics_csv)
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def select_cases(pred_dir, metrics, count):
    cases = sorted(p.stem for p in Path(pred_dir).glob("*.npy") if not p.stem.endswith("_prob"))
    if not metrics:
        return cases[:count]
    scored = [c for c in cases if c in metrics and metrics[c].get("Avg_dice")]
    scored.sort(key=lambda c: float(metrics[c]["Avg_dice"]))
    if len(scored) <= count:
        return scored
    idxs = np.linspace(0, len(scored) - 1, count, dtype=int)
    return [scored[i] for i in idxs]


def save_case_panel(case, image, seg, pred, prob, out_path, modality):
    z = choose_slices(seg, pred, 1)[0]
    base = np.asarray(image[modality, z])
    gt = np.asarray(seg[z])
    pr = np.asarray(pred[z])
    ent = entropy_from_prob(prob)[z] if prob is not None else None

    display_base, display_gt, display_pr = crop_to_tumor([base, gt, pr], (gt > 0) | (pr > 0))
    fig, axes = plt.subplots(1, 5 if ent is not None else 4, figsize=(16, 3.5))
    axes[0].imshow(normalize_slice(display_base), cmap="gray")
    add_box(axes[0], display_gt > 0)
    axes[0].set_title("Image")
    axes[1].imshow(overlay_labels(display_base, display_gt))
    axes[1].set_title("Ground truth")
    axes[2].imshow(overlay_labels(display_base, display_pr))
    axes[2].set_title("DiffUNet prediction")
    axes[3].imshow(contour_overlay(display_base, display_gt, display_pr))
    axes[3].set_title("Contours: GT green, Pred red")
    if ent is not None:
        display_ent = crop_to_tumor([ent], (gt > 0) | (pr > 0))[0]
        axes[4].imshow(normalize_slice(display_base), cmap="gray")
        axes[4].imshow(display_ent, cmap="magma", alpha=0.65, vmin=0, vmax=1)
        axes[4].set_title("Prediction entropy")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(case, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_multislice_panel(case, image, seg, pred, prob, out_path, modality):
    slices = choose_slices(seg, pred, 3)
    fig, axes = plt.subplots(len(slices), 5, figsize=(15, 8.5))
    if len(slices) == 1:
        axes = axes[None, :]
    ent = entropy_from_prob(prob) if prob is not None else None
    for row, z in enumerate(slices):
        base = np.asarray(image[modality, z])
        gt = np.asarray(seg[z])
        pr = np.asarray(pred[z])
        mask = (gt > 0) | (pr > 0)
        base_c, gt_c, pr_c = crop_to_tumor([base, gt, pr], mask)
        axes[row, 0].imshow(normalize_slice(base_c), cmap="gray")
        axes[row, 1].imshow(overlay_labels(base_c, gt_c))
        axes[row, 2].imshow(overlay_labels(base_c, pr_c))
        axes[row, 3].imshow(contour_overlay(base_c, gt_c, pr_c))
        if ent is not None:
            ent_c = crop_to_tumor([ent[z]], mask)[0]
            axes[row, 4].imshow(normalize_slice(base_c), cmap="gray")
            axes[row, 4].imshow(ent_c, cmap="magma", alpha=0.65, vmin=0, vmax=1)
        else:
            axes[row, 4].imshow(normalize_slice(base_c), cmap="gray")
        axes[row, 0].set_ylabel(f"z={z}", fontsize=10)
    for col, title in enumerate(["Image", "GT", "Prediction", "Contours", "Entropy"]):
        axes[0, col].set_title(title, fontsize=11)
    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{case}: multi-slice tumor coverage", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_target_panel(case, image, seg, pred, out_path, modality):
    z = choose_slices(seg, pred, 1)[0]
    base = np.asarray(image[modality, z])
    gt = np.asarray(seg[z])
    pr = np.asarray(pred[z])
    fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.5))
    for row, (target, fn) in enumerate(TARGETS.items()):
        gt_m = fn(gt)
        pr_m = fn(pr)
        base_c, gt_c, pr_c = crop_to_tumor([base, gt_m.astype(np.uint8), pr_m.astype(np.uint8)], gt_m | pr_m)
        axes[row, 0].imshow(normalize_slice(base_c), cmap="gray")
        axes[row, 0].imshow(gt_c, cmap="Greens", alpha=0.45)
        axes[row, 1].imshow(normalize_slice(base_c), cmap="gray")
        axes[row, 1].imshow(pr_c, cmap="Reds", alpha=0.45)
        axes[row, 2].imshow(contour_overlay(base_c, gt_c, pr_c))
        axes[row, 0].set_ylabel(target, fontsize=12)
    for col, title in enumerate(["GT", "Prediction", "Contour agreement"]):
        axes[0, col].set_title(title)
    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(case, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_montage(cases, data_dir, pred_dir, out_path, modality):
    cols = ["Image", "GT", "Prediction", "Contours", "Entropy"]
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(len(cols) * 2.2, len(cases) * 2.1))
    if len(cases) == 1:
        axes = axes[None, :]
    for row, case in enumerate(cases):
        image, seg = load_case(data_dir, case)
        pred = np.load(Path(pred_dir) / f"{case}.npy", mmap_mode="r")
        prob_path = Path(pred_dir) / f"{case}_prob.npz"
        prob = np.load(prob_path)["prob"] if prob_path.exists() else None
        ent = entropy_from_prob(prob) if prob is not None else None
        z = choose_slices(seg, pred, 1)[0]
        base = np.asarray(image[modality, z])
        gt = np.asarray(seg[z])
        pr = np.asarray(pred[z])
        mask = (gt > 0) | (pr > 0)
        base_c, gt_c, pr_c = crop_to_tumor([base, gt, pr], mask)
        axes[row, 0].imshow(normalize_slice(base_c), cmap="gray")
        axes[row, 1].imshow(overlay_labels(base_c, gt_c))
        axes[row, 2].imshow(overlay_labels(base_c, pr_c))
        axes[row, 3].imshow(contour_overlay(base_c, gt_c, pr_c))
        if ent is not None:
            ent_c = crop_to_tumor([ent[z]], mask)[0]
            axes[row, 4].imshow(normalize_slice(base_c), cmap="gray")
            axes[row, 4].imshow(ent_c, cmap="magma", alpha=0.65, vmin=0, vmax=1)
        axes[row, 0].set_ylabel(case, fontsize=7)
    for col, title in enumerate(cols):
        axes[-1, col].set_xlabel(title, fontsize=12)
    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_html(out_dir, generated):
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>BraTS2025 DiffUNet Prediction Visualizations</title>",
        "<style>body{font-family:Arial,sans-serif;margin:28px;background:#fafafa;color:#222}"
        "img{max-width:100%;border:1px solid #ddd;background:white;margin:8px 0 24px}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}"
        "h1,h2{margin-bottom:6px}.card{background:white;padding:14px;border:1px solid #ddd}</style>",
        "</head><body><h1>BraTS2025 Glioma DiffUNet Prediction Visualizations</h1>",
        "<p>Generated from saved prediction masks and probability volumes.</p>",
    ]
    for section, paths in generated.items():
        lines.append(f"<h2>{html.escape(section)}</h2><div class='grid'>")
        for path in paths:
            rel = Path(path).name
            lines.append(f"<div class='card'><strong>{html.escape(rel)}</strong><br><img src='{html.escape(rel)}'></div>")
        lines.append("</div>")
    lines.append("</body></html>")
    (Path(out_dir) / "prediction_visualization_report.html").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/fullres/train")
    parser.add_argument("--pred-dir", default="./prediction_results/DiffUNet_selected")
    parser.add_argument("--metrics-csv", default="./quantitative_results/DiffUNet_250cases_per_case_metrics.csv")
    parser.add_argument("--out-dir", default="./publication_prediction_visuals")
    parser.add_argument("--case-count", type=int, default=24)
    parser.add_argument("--single-case-count", type=int, default=12)
    parser.add_argument("--modality", type=int, default=2, help="0=t1c, 1=t1n, 2=t2f/FLAIR, 3=t2w")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(args.metrics_csv)
    cases = select_cases(args.pred_dir, metrics, args.case_count)
    single_cases = cases[: args.single_case_count]

    generated = {"Summary montages": [], "Case panels": [], "Multi-slice panels": [], "Target panels": []}
    montage_path = out_dir / "brats2025_publication_montage_representative.png"
    save_montage(cases, args.data_dir, args.pred_dir, montage_path, args.modality)
    generated["Summary montages"].append(montage_path)

    for case in single_cases:
        image, seg = load_case(args.data_dir, case)
        pred = np.load(Path(args.pred_dir) / f"{case}.npy", mmap_mode="r")
        prob_path = Path(args.pred_dir) / f"{case}_prob.npz"
        prob = np.load(prob_path)["prob"] if prob_path.exists() else None
        panel = out_dir / f"{case}_overlay_entropy_panel.png"
        multi = out_dir / f"{case}_multislice_panel.png"
        target = out_dir / f"{case}_wt_tc_et_panel.png"
        save_case_panel(case, image, seg, pred, prob, panel, args.modality)
        save_multislice_panel(case, image, seg, pred, prob, multi, args.modality)
        save_target_panel(case, image, seg, pred, target, args.modality)
        generated["Case panels"].append(panel)
        generated["Multi-slice panels"].append(multi)
        generated["Target panels"].append(target)

    write_html(out_dir, generated)
    print(f"Saved publication prediction visualizations to {out_dir}")
    print("Representative cases:", ", ".join(cases))


if __name__ == "__main__":
    main()
