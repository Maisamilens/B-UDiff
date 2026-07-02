import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TARGETS = ["WT", "TC", "ET", "Avg"]


def load_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in list(row.items()):
            if key != "case":
                row[key] = float(value)
    return rows


def describe(values):
    arr = np.asarray(values, dtype=float)
    q1, median, q3 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)),
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "ci95_low": float(np.mean(arr) - 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))),
        "ci95_high": float(np.mean(arr) + 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))),
    }


def try_stats(rows):
    out = {}
    try:
        from scipy import stats
    except Exception:
        return out
    avg_dice = np.asarray([r["Avg_dice"] for r in rows])
    avg_hd95 = np.asarray([r["Avg_hd95"] for r in rows])
    out["pearson_avg_dice_vs_hd95"] = {
        "r": float(stats.pearsonr(avg_dice, avg_hd95).statistic),
        "p": float(stats.pearsonr(avg_dice, avg_hd95).pvalue),
    }
    out["spearman_avg_dice_vs_hd95"] = {
        "rho": float(stats.spearmanr(avg_dice, avg_hd95).statistic),
        "p": float(stats.spearmanr(avg_dice, avg_hd95).pvalue),
    }
    for target in ["WT", "TC", "ET"]:
        values = np.asarray([r[f"{target}_dice"] for r in rows])
        if len(values) <= 5000:
            shapiro = stats.shapiro(values)
            out[f"{target}_dice_shapiro_normality"] = {
                "W": float(shapiro.statistic),
                "p": float(shapiro.pvalue),
            }
    return out


def write_descriptive_csv(out_path, rows):
    fields = ["metric", "n", "mean", "std", "median", "q1", "q3", "min", "max", "ci95_low", "ci95_high"]
    records = []
    for target in ["WT", "TC", "ET"]:
        for metric in ["dice", "hd95"]:
            values = [r[f"{target}_{metric}"] for r in rows]
            rec = {"metric": f"{target}_{metric}"}
            rec.update(describe(values))
            records.append(rec)
    for metric in ["dice", "hd95"]:
        values = [r[f"Avg_{metric}"] for r in rows]
        rec = {"metric": f"Avg_{metric}"}
        rec.update(describe(values))
        records.append(rec)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return records


def plot_distributions(out_dir, rows):
    dice_data = [[r[f"{t}_dice"] * 100 for r in rows] for t in TARGETS]
    hd_data = [[r[f"{t}_hd95"] for r in rows] for t in TARGETS]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot(dice_data, labels=TARGETS, patch_artist=True)
    axes[0].set_title("Dice Distribution by BraTS Target")
    axes[0].set_ylabel("Dice (%)")
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].boxplot(hd_data, labels=TARGETS, patch_artist=True)
    axes[1].set_title("HD95 Distribution by BraTS Target")
    axes[1].set_ylabel("HD95")
    axes[1].grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(out_dir / "brats2025_metric_boxplots.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist([r["Avg_dice"] * 100 for r in rows], bins=25, color="#4daf4a", edgecolor="white")
    axes[0].set_title("Average Dice Histogram")
    axes[0].set_xlabel("Avg Dice (%)")
    axes[0].set_ylabel("Cases")
    axes[1].hist([r["Avg_hd95"] for r in rows], bins=25, color="#377eb8", edgecolor="white")
    axes[1].set_title("Average HD95 Histogram")
    axes[1].set_xlabel("Avg HD95")
    plt.tight_layout()
    fig.savefig(out_dir / "brats2025_metric_histograms.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(out_dir, rows):
    keys = ["WT_dice", "TC_dice", "ET_dice", "Avg_dice", "WT_hd95", "TC_hd95", "ET_hd95", "Avg_hd95"]
    data = np.asarray([[r[k] for k in keys] for r in rows], dtype=float)
    corr = np.corrcoef(data, rowvar=False)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right")
    ax.set_yticks(range(len(keys)), keys)
    for i in range(len(keys)):
        for j in range(len(keys)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Metric Correlation Heatmap")
    plt.tight_layout()
    fig.savefig(out_dir / "brats2025_metric_correlation_heatmap.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_rankings(out_dir, rows):
    sorted_rows = sorted(rows, key=lambda r: r["Avg_dice"])
    fields = ["case", "WT_dice", "WT_hd95", "TC_dice", "TC_hd95", "ET_dice", "ET_hd95", "Avg_dice", "Avg_hd95"]
    for name, subset in [("worst_20_cases.csv", sorted_rows[:20]), ("best_20_cases.csv", sorted_rows[-20:][::-1])]:
        with (out_dir / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(subset)


def main():
    metrics_csv = Path("quantitative_results/DiffUNet_250cases_per_case_metrics.csv")
    out_dir = Path("statistical_analysis_outputs")
    out_dir.mkdir(exist_ok=True)
    rows = load_rows(metrics_csv)

    descriptive = write_descriptive_csv(out_dir / "descriptive_statistics.csv", rows)
    extra = try_stats(rows)
    with (out_dir / "statistical_tests.json").open("w") as f:
        json.dump({"descriptive": descriptive, "tests": extra}, f, indent=2)

    plot_distributions(out_dir, rows)
    plot_correlation_heatmap(out_dir, rows)
    write_rankings(out_dir, rows)
    print(f"Saved statistical analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()
