import argparse
import csv
import glob
import math
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from medpy import metric


TARGETS = ["WT", "TC", "ET"]


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


def to_regions(mask):
    return {
        "WT": mask > 0,
        "TC": (mask == 1) | (mask == 3),
        "ET": mask == 3,
    }


def read_mask(path):
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img).astype(np.int16)
    spacing = tuple(reversed(img.GetSpacing()))
    return arr, spacing


def find_gt(raw_dir, case_name):
    candidates = [
        Path(raw_dir) / case_name / f"{case_name}-seg.nii.gz",
        Path(raw_dir) / case_name / "seg.nii.gz",
        Path(raw_dir) / case_name / f"{case_name}_seg.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Ground truth segmentation not found for {case_name}")


def find_prediction(pred_dir, case_name):
    pred_dir = Path(pred_dir)
    candidates = [
        pred_dir / f"{case_name}.nii.gz",
        pred_dir / f"{case_name}-seg.nii.gz",
        pred_dir / f"{case_name}_seg.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def find_region_prediction(pred_dir, case_name, target):
    pred_dir = Path(pred_dir)
    candidates = [
        pred_dir / f"{case_name}-{target}.nii.gz",
        pred_dir / f"{case_name}_{target}.nii.gz",
        pred_dir / target / f"{case_name}.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def case_names_from_pred_dir(pred_dir):
    names = set()
    for path in glob.glob(str(Path(pred_dir) / "*.nii.gz")):
        name = Path(path).name.replace(".nii.gz", "")
        for suffix in ["-WT", "-TC", "-ET", "_WT", "_TC", "_ET", "-seg", "_seg"]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        names.add(name)
    return sorted(names)


def evaluate_case(raw_dir, pred_dir, case_name):
    gt_path = find_gt(raw_dir, case_name)
    gt, spacing = read_mask(gt_path)
    gt_regions = to_regions(gt)

    pred_path = find_prediction(pred_dir, case_name)
    if pred_path is not None:
        pred, _ = read_mask(pred_path)
        pred_regions = to_regions(pred)
    else:
        pred_regions = {}
        for target in TARGETS:
            region_path = find_region_prediction(pred_dir, case_name, target)
            if region_path is None:
                raise FileNotFoundError(f"Prediction for {case_name} {target} not found")
            pred_region, _ = read_mask(region_path)
            pred_regions[target] = pred_region > 0

    row = {"case": case_name}
    for target in TARGETS:
        row[f"{target}_dice"] = dice_score(pred_regions[target], gt_regions[target])
        row[f"{target}_hd95"] = hd95_score(pred_regions[target], gt_regions[target], spacing)
    row["Avg_dice"] = float(np.mean([row[f"{t}_dice"] for t in TARGETS]))
    row["Avg_hd95"] = float(np.mean([row[f"{t}_hd95"] for t in TARGETS]))
    return row


def mean_std(values, percent=False):
    arr = np.asarray(values, dtype=float)
    if percent:
        arr = arr * 100.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def ci95(values, percent=False):
    arr = np.asarray(values, dtype=float)
    if percent:
        arr = arr * 100.0
    n = len(arr)
    if n < 2:
        return float(arr[0]), float(arr[0])
    sem = np.std(arr, ddof=1) / math.sqrt(n)
    delta = 1.96 * sem
    mean = float(np.mean(arr))
    return mean - delta, mean + delta


def paired_t_pvalue(ours, other):
    try:
        from scipy import stats
    except Exception:
        return None
    ours = np.asarray(ours, dtype=float)
    other = np.asarray(other, dtype=float)
    if len(ours) != len(other):
        return None
    return float(stats.ttest_rel(ours, other, nan_policy="omit").pvalue)


def read_comparator_csv(path, metric_name):
    if not path:
        return None
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[row["case"]] = float(row[metric_name])
    return rows


def write_csv(path, rows):
    fieldnames = ["case"]
    for target in TARGETS:
        fieldnames += [f"{target}_dice", f"{target}_hd95"]
    fieldnames += ["Avg_dice", "Avg_hd95"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_cell(mean, std):
    return f"{mean:.2f} +/- {std:.2f}"


def build_table_line(method, rows, comparator_csv=None):
    parts = [method]
    for target in TARGETS:
        dice_m, dice_s = mean_std([r[f"{target}_dice"] for r in rows], percent=True)
        hd_m, hd_s = mean_std([r[f"{target}_hd95"] for r in rows])
        parts.append(format_cell(dice_m, dice_s))
        parts.append(format_cell(hd_m, hd_s))

    avg_dice_m, avg_dice_s = mean_std([r["Avg_dice"] for r in rows], percent=True)
    avg_hd_m, avg_hd_s = mean_std([r["Avg_hd95"] for r in rows])
    dice_ci = ci95([r["Avg_dice"] for r in rows], percent=True)
    hd_ci = ci95([r["Avg_hd95"] for r in rows])

    dice_p = "-"
    hd_p = "-"
    if comparator_csv:
        comp_dice = read_comparator_csv(comparator_csv, "Avg_dice")
        comp_hd = read_comparator_csv(comparator_csv, "Avg_hd95")
        aligned = [r for r in rows if r["case"] in comp_dice and r["case"] in comp_hd]
        dice_p_val = paired_t_pvalue([r["Avg_dice"] for r in aligned], [comp_dice[r["case"]] for r in aligned])
        hd_p_val = paired_t_pvalue([r["Avg_hd95"] for r in aligned], [comp_hd[r["case"]] for r in aligned])
        dice_p = "-" if dice_p_val is None else f"{dice_p_val:.4g}"
        hd_p = "-" if hd_p_val is None else f"{hd_p_val:.4g}"

    parts += [
        format_cell(avg_dice_m, avg_dice_s),
        f"({dice_ci[0] / 100:.3f}, {dice_ci[1] / 100:.3f})",
        dice_p,
        format_cell(avg_hd_m, avg_hd_s),
        f"({hd_ci[0]:.2f}, {hd_ci[1]:.2f})",
        hd_p,
    ]
    return parts


def main():
    parser = argparse.ArgumentParser(
        description="Compute BraTS quantitative table with WT/TC/ET Dice and HD95."
    )
    parser.add_argument("--raw-dir", required=True, help="Raw BraTS directory containing case folders and GT seg files.")
    parser.add_argument("--pred-dir", required=True, help="Prediction directory containing .nii.gz outputs.")
    parser.add_argument("--method", default="DiffUNet", help="Method name printed in the summary table.")
    parser.add_argument("--out-dir", default="quantitative_results", help="Output directory.")
    parser.add_argument("--comparator-csv", default=None, help="Optional per-case CSV from another method for paired p-values.")
    args = parser.parse_args()

    case_names = case_names_from_pred_dir(args.pred_dir)
    if not case_names:
        raise RuntimeError(f"No .nii.gz predictions found in {args.pred_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    rows = [evaluate_case(args.raw_dir, args.pred_dir, case) for case in case_names]
    per_case_csv = out_dir / f"{args.method}_per_case_metrics.csv"
    write_csv(per_case_csv, rows)

    headers = [
        "Method",
        "WT Dice", "WT HD95",
        "TC Dice", "TC HD95",
        "ET Dice", "ET HD95",
        "Avg Dice", "Dice CI", "Dice p-value",
        "Avg HD95", "HD95 CI", "HD95 p-value",
    ]
    table_line = build_table_line(args.method, rows, args.comparator_csv)
    table_csv = out_dir / f"{args.method}_summary_table.csv"
    with open(table_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerow(table_line)

    print(",".join(headers))
    print(",".join(table_line))
    print(f"Saved per-case metrics: {per_case_csv}")
    print(f"Saved summary table: {table_csv}")


if __name__ == "__main__":
    main()
