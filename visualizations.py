import ast
import csv
import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "visualization_outputs"
TRAIN_LOG = ROOT / "training details progress from epoch 0 to 499.txt"


def ensure_out_dir():
    OUT_DIR.mkdir(exist_ok=True)


def parse_training_log():
    text = TRAIN_LOG.read_text(errors="ignore")

    metric_re = re.compile(
        r"Epoch\s+(\d+):\s+WT=([0-9.]+),\s+TC=([0-9.]+),\s+ET=([0-9.]+),\s+Mean=([0-9.]+)"
    )
    train_re = re.compile(
        r"Epoch\s+(\d+):.*?loss=([0-9.eE+-]+),\s+lr=([0-9.eE+-]+)"
    )

    metrics = []
    seen_metrics = set()
    for match in metric_re.finditer(text):
        epoch = int(match.group(1))
        if epoch in seen_metrics:
            continue
        seen_metrics.add(epoch)
        metrics.append(
            {
                "epoch": epoch,
                "WT": float(match.group(2)),
                "TC": float(match.group(3)),
                "ET": float(match.group(4)),
                "Mean": float(match.group(5)),
            }
        )

    train = []
    seen_train = set()
    for match in train_re.finditer(text):
        epoch = int(match.group(1))
        if epoch in seen_train:
            continue
        seen_train.add(epoch)
        train.append(
            {
                "epoch": epoch,
                "loss": float(match.group(2)),
                "lr": float(match.group(3)),
            }
        )

    return metrics, train


def save_csv(name, rows, columns):
    path = OUT_DIR / name
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_training_metrics(metrics):
    epochs = [m["epoch"] for m in metrics]
    plt.figure(figsize=(10, 6))
    for key, color in [
        ("WT", "#2b8cbe"),
        ("TC", "#31a354"),
        ("ET", "#de2d26"),
        ("Mean", "#6a51a3"),
    ]:
        plt.plot(epochs, [m[key] for m in metrics], marker="o", linewidth=2, label=key, color=color)
    plt.title("DiffUNet BraTS2025 Validation Dice Across Training")
    plt.xlabel("Epoch")
    plt.ylabel("Dice score")
    plt.ylim(0.55, 0.95)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "training_dice_curves.png", dpi=220)
    plt.close()


def plot_loss_lr(train):
    epochs = [m["epoch"] for m in train]
    losses = [m["loss"] for m in train]
    lrs = [m["lr"] for m in train]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(epochs, losses, color="#c51b7d", linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss", color="#c51b7d")
    ax1.tick_params(axis="y", labelcolor="#c51b7d")
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(epochs, lrs, color="#2166ac", linewidth=1.8)
    ax2.set_ylabel("Learning rate", color="#2166ac")
    ax2.tick_params(axis="y", labelcolor="#2166ac")

    plt.title("Training Loss and Polynomial Learning Rate Schedule")
    fig.tight_layout()
    plt.savefig(OUT_DIR / "training_loss_lr.png", dpi=220)
    plt.close()


def plot_final_metrics(metrics):
    final = metrics[-1]
    keys = ["WT", "TC", "ET", "Mean"]
    vals = [final[k] for k in keys]
    colors = ["#2b8cbe", "#31a354", "#de2d26", "#6a51a3"]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(keys, vals, color=colors)
    plt.title(f"Final Validation Dice at Epoch {final['epoch']}")
    plt.ylabel("Dice score")
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.2)
    for bar, val in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.015, f"{val:.4f}", ha="center")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "final_validation_metrics.png", dpi=220)
    plt.close()


def read_test_list_count():
    test_file = ROOT / "test_list_brats2025.py"
    tree = ast.parse(test_file.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "test_list":
                    return len(ast.literal_eval(node.value))
    return None


def plot_dataset_summary():
    train_npz = glob.glob(str(ROOT / "data" / "fullres" / "train" / "*.npz"))
    val_npz = glob.glob(str(ROOT / "data" / "fullres" / "val" / "*.npz"))
    test_count = read_test_list_count() or 0
    train_split = max(len(train_npz) - test_count, 0)

    rows = [
        {"split": "train_fullres_npz_total", "count": len(train_npz)},
        {"split": "internal_train_after_test_list", "count": train_split},
        {"split": "internal_validation_test_list", "count": test_count},
        {"split": "external_val_fullres_npz", "count": len(val_npz)},
    ]
    save_csv("dataset_summary.csv", rows, ["split", "count"])

    plt.figure(figsize=(8, 5))
    labels = ["Internal train", "Internal validation", "External val folder"]
    values = [train_split, test_count, len(val_npz)]
    colors = ["#4daf4a", "#377eb8", "#ff7f00"]
    plt.bar(labels, values, color=colors)
    plt.title("BraTS2025 Preprocessed Dataset Counts")
    plt.ylabel("Cases")
    plt.xticks(rotation=12, ha="right")
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.01, str(v), ha="center")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "dataset_counts.png", dpi=220)
    plt.close()

    return rows


def plot_sample_case():
    npz_files = sorted(glob.glob(str(ROOT / "data" / "fullres" / "train" / "*.npz")))
    if not npz_files:
        return None

    npz_path = Path(npz_files[0])
    image_path = npz_path.with_suffix(".npy")
    seg_path = npz_path.with_name(npz_path.stem + "_seg.npy")
    if not image_path.exists() or not seg_path.exists():
        return None

    image = np.load(image_path, mmap_mode="r")
    seg = np.load(seg_path, mmap_mode="r")
    z = image.shape[1] // 2

    fig, axes = plt.subplots(1, 5, figsize=(14, 4))
    modality_names = ["T1c", "T1n", "T2f", "T2w"]
    for i in range(4):
        sl = np.array(image[i, z])
        lo, hi = np.percentile(sl, [1, 99])
        axes[i].imshow(np.clip(sl, lo, hi), cmap="gray")
        axes[i].set_title(modality_names[i])
        axes[i].axis("off")

    axes[4].imshow(np.array(image[0, z]), cmap="gray")
    axes[4].imshow(np.array(seg[0, z]), cmap="Set1", alpha=0.45, vmin=0, vmax=3)
    axes[4].set_title("Seg overlay")
    axes[4].axis("off")
    fig.suptitle(f"Sample Preprocessed Case: {npz_path.stem}", y=1.02)
    plt.tight_layout()
    out = OUT_DIR / "sample_case_modalities_segmentation.png"
    plt.savefig(out, dpi=220, bbox_inches="tight")
    plt.close()
    return out


def code_inventory():
    rows = []
    for path in sorted(ROOT.rglob("*.py")):
        if "visualization_outputs" in path.parts:
            continue
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        rows.append(
            {
                "file": str(path.relative_to(ROOT)),
                "lines": len(lines),
                "imports": sum(1 for line in lines if line.strip().startswith(("import ", "from "))),
            }
        )
    save_csv("python_file_inventory.csv", rows, ["file", "lines", "imports"])

    top = sorted(rows, key=lambda r: r["lines"], reverse=True)[:12]
    plt.figure(figsize=(10, 6))
    labels = [r["file"] for r in top]
    values = [r["lines"] for r in top]
    plt.barh(labels[::-1], values[::-1], color="#756bb1")
    plt.title("Largest Python Files in the DiffUNet Project")
    plt.xlabel("Lines of code")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "python_file_inventory.png", dpi=220)
    plt.close()
    return rows


def plot_workflow():
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axis("off")
    boxes = [
        ("Raw BraTS2025\nNIfTI files", 0.04),
        ("Preprocessing\n1_preprocessing_brats2025.py", 0.26),
        ("Preprocessed arrays\ndata/fullres/train", 0.49),
        ("Training\n3_train_diffunet_brats2025.py", 0.70),
        ("Checkpoints + metrics\nlogs/diffunet/model", 0.89),
    ]
    for text, x in boxes:
        ax.text(
            x,
            0.55,
            text,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="#f7f7f7", ec="#525252", lw=1.2),
            fontsize=10,
        )
    for x1, x2 in zip([0.12, 0.35, 0.58, 0.79], [0.19, 0.42, 0.63, 0.82]):
        ax.annotate("", xy=(x2, 0.55), xytext=(x1, 0.55), arrowprops=dict(arrowstyle="->", lw=1.7))
    ax.text(0.70, 0.18, "Inference: 4_predict_brats2025.py", ha="center", fontsize=10)
    ax.annotate("", xy=(0.82, 0.42), xytext=(0.72, 0.23), arrowprops=dict(arrowstyle="->", lw=1.4))
    plt.tight_layout()
    plt.savefig(OUT_DIR / "project_workflow.png", dpi=220)
    plt.close()


def write_report(metrics, dataset_rows):
    best = max(metrics, key=lambda m: m["Mean"])
    final = metrics[-1]
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>DiffUNet BraTS2025 Visualization Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    img {{ max-width: 100%; border: 1px solid #ddd; margin: 12px 0 28px; }}
    table {{ border-collapse: collapse; margin: 10px 0 24px; }}
    td, th {{ border: 1px solid #ddd; padding: 8px 10px; }}
  </style>
</head>
<body>
  <h1>DiffUNet BraTS2025 Visualization Report</h1>
  <p><b>Final epoch:</b> {final['epoch']} &nbsp; <b>Final mean Dice:</b> {final['Mean']:.4f}</p>
  <p><b>Best logged epoch:</b> {best['epoch']} &nbsp; <b>Best mean Dice:</b> {best['Mean']:.4f}</p>
  <h2>Training Curves</h2>
  <img src="training_dice_curves.png" alt="Training Dice curves">
  <img src="training_loss_lr.png" alt="Training loss and learning rate">
  <h2>Final Metrics</h2>
  <img src="final_validation_metrics.png" alt="Final validation metrics">
  <h2>Dataset Summary</h2>
  <table>
    <tr><th>Split</th><th>Count</th></tr>
    {''.join(f"<tr><td>{r['split']}</td><td>{r['count']}</td></tr>" for r in dataset_rows)}
  </table>
  <img src="dataset_counts.png" alt="Dataset counts">
  <h2>Sample Case and Project Structure</h2>
  <img src="sample_case_modalities_segmentation.png" alt="Sample case modalities and segmentation">
  <img src="project_workflow.png" alt="Project workflow">
  <img src="python_file_inventory.png" alt="Python file inventory">
</body>
</html>
"""
    (OUT_DIR / "report.html").write_text(html)


def main():
    ensure_out_dir()
    metrics, train = parse_training_log()
    if not metrics:
        raise RuntimeError("No validation metrics found in training log.")

    save_csv("training_validation_metrics.csv", metrics, ["epoch", "WT", "TC", "ET", "Mean"])
    save_csv("training_loss_lr.csv", train, ["epoch", "loss", "lr"])
    plot_training_metrics(metrics)
    if train:
        plot_loss_lr(train)
    plot_final_metrics(metrics)
    dataset_rows = plot_dataset_summary()
    plot_sample_case()
    code_inventory()
    plot_workflow()
    write_report(metrics, dataset_rows)
    print(f"Visualization pack generated: {OUT_DIR}")


if __name__ == "__main__":
    main()
