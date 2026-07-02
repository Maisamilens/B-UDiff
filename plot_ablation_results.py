import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_mean(cell):
    return float(cell.split("+/-")[0].strip())


def main():
    summary_csv = Path("ablation_results/brats2025_ablation_summary_table.csv")
    out_dir = Path("visualization_outputs")
    out_dir.mkdir(exist_ok=True)

    with summary_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    variants = [r["Variant"] for r in rows]
    dice = [parse_mean(r["Avg Dice"]) for r in rows]
    hd95 = [parse_mean(r["Avg HD95"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#4daf4a", "#377eb8", "#984ea3", "#ff7f00", "#e41a1c", "#a65628", "#999999"]

    axes[0].bar(variants, dice, color=colors[: len(variants)])
    axes[0].set_title("BraTS2025 Ablation: Avg Dice")
    axes[0].set_ylabel("Dice (%)")
    axes[0].set_ylim(0, 100)
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)
    for i, v in enumerate(dice):
        axes[0].text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=8)

    axes[1].bar(variants, hd95, color=colors[: len(variants)])
    axes[1].set_title("BraTS2025 Ablation: Avg HD95")
    axes[1].set_ylabel("HD95")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    for i, v in enumerate(hd95):
        axes[1].text(i, v + 1.2, f"{v:.1f}", ha="center", fontsize=8)

    plt.tight_layout()
    out = out_dir / "brats2025_ablation_statistics.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation statistics plot: {out}")


if __name__ == "__main__":
    main()
