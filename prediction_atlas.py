import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from generate_publication_prediction_visuals import (
    choose_slices,
    crop_to_tumor,
    load_case,
    normalize_slice,
    overlay_labels,
)


def read_metrics(metrics_csv):
    path = Path(metrics_csv)
    if not path.exists():
        return {}
    with path.open(newline="") as f:
        return {row["case"]: row for row in csv.DictReader(f)}


def parse_cases(items):
    if not items:
        return None
    if len(items) == 1 and "," in items[0]:
        return [x.strip() for x in items[0].split(",") if x.strip()]
    return items


def case_list(pred_dir, metrics, explicit_cases=None):
    if explicit_cases:
        return explicit_cases
    cases = sorted(p.stem for p in Path(pred_dir).glob("*.npy"))
    if metrics:
        cases = [c for c in cases if c in metrics]
        cases.sort(key=lambda c: float(metrics[c].get("Avg_dice", 0.0)))
    return cases


def render_tile(data_dir, pred_dir, case, modality):
    image, seg = load_case(data_dir, case)
    pred = np.load(Path(pred_dir) / f"{case}.npy", mmap_mode="r")
    z = choose_slices(seg, pred, 1)[0]
    base = np.asarray(image[modality, z])
    pr = np.asarray(pred[z])
    mask = (np.asarray(seg[z]) > 0) | (pr > 0)
    base_c, pr_c = crop_to_tumor([base, pr], mask)
    return overlay_labels(base_c, pr_c)


def save_page(cases, data_dir, pred_dir, metrics, out_path, modality, cols):
    rows = math.ceil(len(cases) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.15, rows * 2.35))
    axes = np.asarray(axes).reshape(rows, cols)
    for ax in axes.reshape(-1):
        ax.axis("off")
    for idx, case in enumerate(cases):
        ax = axes[idx // cols, idx % cols]
        ax.imshow(render_tile(data_dir, pred_dir, case, modality))
        title = case
        if case in metrics and metrics[case].get("Avg_dice"):
            title += f"\nDice {float(metrics[case]['Avg_dice']) * 100:.1f}"
        ax.set_title(title, fontsize=7, pad=2)
    fig.tight_layout(pad=0.6)
    fig.savefig(out_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_compare_montage(cases, data_dir, pred_dir, metrics, out_path, modality):
    cols = ["Image", "GT", "Prediction"]
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(8.4, len(cases) * 2.0))
    if len(cases) == 1:
        axes = axes[None, :]
    for row, case in enumerate(cases):
        image, seg = load_case(data_dir, case)
        pred = np.load(Path(pred_dir) / f"{case}.npy", mmap_mode="r")
        z = choose_slices(seg, pred, 1)[0]
        base = np.asarray(image[modality, z])
        gt = np.asarray(seg[z])
        pr = np.asarray(pred[z])
        mask = (gt > 0) | (pr > 0)
        base_c, gt_c, pr_c = crop_to_tumor([base, gt, pr], mask)
        axes[row, 0].imshow(normalize_slice(base_c), cmap="gray")
        axes[row, 1].imshow(overlay_labels(base_c, gt_c))
        axes[row, 2].imshow(overlay_labels(base_c, pr_c))
        ylabel = case
        if case in metrics and metrics[case].get("Avg_dice"):
            ylabel += f"\nAvg Dice {float(metrics[case]['Avg_dice']) * 100:.1f}"
        axes[row, 0].set_ylabel(ylabel, fontsize=7)
    for col, title in enumerate(cols):
        axes[-1, col].set_xlabel(title, fontsize=12)
    for ax in axes.reshape(-1):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/fullres/train")
    parser.add_argument("--pred-dir", default="./prediction_results/DiffUNet_selected")
    parser.add_argument("--metrics-csv", default="./quantitative_results/DiffUNet_250cases_per_case_metrics.csv")
    parser.add_argument("--out-dir", default="./publication_prediction_visuals/atlas")
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--modality", type=int, default=2)
    parser.add_argument("--cases", nargs="*", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(args.metrics_csv)
    cases = case_list(args.pred_dir, metrics, parse_cases(args.cases))

    for page_idx in range(math.ceil(len(cases) / args.page_size)):
        page_cases = cases[page_idx * args.page_size : (page_idx + 1) * args.page_size]
        save_page(
            page_cases,
            args.data_dir,
            args.pred_dir,
            metrics,
            out_dir / f"prediction_overlay_atlas_page_{page_idx + 1:02d}.png",
            args.modality,
            args.cols,
        )

    if args.cases:
        save_compare_montage(
            cases,
            args.data_dir,
            args.pred_dir,
            metrics,
            out_dir / "requested_cases_image_gt_prediction_montage.png",
            args.modality,
        )

    print(f"Saved atlas pages for {len(cases)} cases to {out_dir}")


if __name__ == "__main__":
    main()
