import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "B-UDiff"
FIG = OUT / "figures"
TAB = OUT / "tables"
DATA = OUT / "data_summaries"
DOC = OUT / "documents"

MODEL_NAME = "B-UDiff"
PAPER_TITLE = "Boundary-Aware Uncertainty-Guided Diffusion Network with Progressive Refinement for 3D Glioma Segmentation"
MODEL_PATH = ROOT / "logs" / "diffunet" / "model" / "final_model_0.8973.pt"
BEST_MODEL_PATH = ROOT / "logs" / "diffunet" / "model" / "best_model_0.9027.pt"
PRED_DIR = ROOT / "prediction_results" / "DiffUNet_selected"
TRAIN_DIR = ROOT / "data" / "fullres" / "train"
LOG_FILE = ROOT / "training details progress from epoch 0 to 499.txt"
METHOD_FILE = Path(r"C:\Users\1411b\.codex\attachments\35a524bf-7eb2-4e41-aa34-68db272198f4\pasted-text.txt")


def ensure_dirs():
    for d in [OUT, FIG, TAB, DATA, DOC]:
        d.mkdir(parents=True, exist_ok=True)


def savefig(path, dpi=400):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()


def parse_training_log():
    text = LOG_FILE.read_text(encoding="utf-8", errors="replace")
    metric_rows = []
    for m in re.finditer(r"Epoch\s+(\d+):\s+WT=([0-9.]+),\s+TC=([0-9.]+),\s+ET=([0-9.]+),\s+Mean=([0-9.]+)", text):
        metric_rows.append({
            "epoch": int(m.group(1)),
            "WT_dice": float(m.group(2)),
            "TC_dice": float(m.group(3)),
            "ET_dice": float(m.group(4)),
            "Mean_dice": float(m.group(5)),
        })
    loss_rows = []
    for m in re.finditer(r"Epoch\s+(\d+):.*?loss=([0-9.]+),\s+lr=([0-9.]+)", text):
        loss_rows.append({
            "epoch": int(m.group(1)),
            "training_loss_last_batch": float(m.group(2)),
            "learning_rate": float(m.group(3)),
        })
    metrics = pd.DataFrame(metric_rows).drop_duplicates("epoch")
    loss = pd.DataFrame(loss_rows).drop_duplicates("epoch")
    return metrics, loss


def summarize_predictions():
    rows = []
    prob_rows = []
    npy_files = sorted(PRED_DIR.glob("*.npy"))
    for pred_path in npy_files:
        case_id = pred_path.stem
        seg = np.load(pred_path, mmap_mode="r")
        total = int(seg.size)
        counts = {int(k): int(v) for k, v in zip(*np.unique(seg, return_counts=True))}
        wt = int(np.count_nonzero(seg > 0))
        tc = int(np.count_nonzero((seg == 1) | (seg == 3)))
        et = int(np.count_nonzero(seg == 3))
        rows.append({
            "case_id": case_id,
            "shape": "x".join(map(str, seg.shape)),
            "voxels_total": total,
            "background_voxels": counts.get(0, 0),
            "class1_NCR_NET_voxels": counts.get(1, 0),
            "class2_ED_voxels": counts.get(2, 0),
            "class3_ET_voxels": counts.get(3, 0),
            "WT_voxels": wt,
            "TC_voxels": tc,
            "ET_voxels": et,
            "WT_percent": 100.0 * wt / total,
            "TC_percent": 100.0 * tc / total,
            "ET_percent": 100.0 * et / total,
        })

        prob_path = pred_path.with_name(f"{case_id}_prob.npz")
        if prob_path.exists():
            prob = np.load(prob_path)["prob"].astype(np.float32)
            eps = 1e-7
            entropy = -np.sum(prob * np.log(np.clip(prob, eps, 1.0)), axis=0) / np.log(prob.shape[0])
            confidence = np.max(prob, axis=0)
            tumor_mask = np.asarray(seg) > 0
            prob_rows.append({
                "case_id": case_id,
                "probability_shape": "x".join(map(str, prob.shape)),
                "mean_entropy": float(entropy.mean()),
                "median_entropy": float(np.median(entropy)),
                "p95_entropy": float(np.percentile(entropy, 95)),
                "mean_confidence": float(confidence.mean()),
                "tumor_mean_entropy": float(entropy[tumor_mask].mean()) if tumor_mask.any() else np.nan,
                "tumor_mean_confidence": float(confidence[tumor_mask].mean()) if tumor_mask.any() else np.nan,
            })
    pred_df = pd.DataFrame(rows)
    prob_df = pd.DataFrame(prob_rows)
    pred_df.to_csv(DATA / "B-UDiff_prediction_mask_summary_250_cases.csv", index=False)
    prob_df.to_csv(DATA / "B-UDiff_probability_uncertainty_summary_250_cases.csv", index=False)
    return pred_df, prob_df


def load_existing_tables():
    tables = {}
    sources = {
        "B-UDiff_250cases_validation_metrics.csv": ROOT / "quantitative_results" / "DiffUNet_250cases_per_case_metrics.csv",
        "B-UDiff_250cases_validation_summary.csv": ROOT / "quantitative_results" / "DiffUNet_250cases_summary_table.csv",
        "B-UDiff_ablation_per_case_metrics.csv": ROOT / "ablation_results" / "brats2025_ablation_per_case_metrics.csv",
        "B-UDiff_ablation_summary.csv": ROOT / "ablation_results" / "brats2025_ablation_summary_table.csv",
        "B-UDiff_statistical_tests.json": ROOT / "statistical_analysis_outputs" / "statistical_tests.json",
    }
    for name, src in sources.items():
        if src.exists():
            dst = TAB / name
            shutil.copy2(src, dst)
            if dst.suffix.lower() == ".csv":
                tables[name] = pd.read_csv(dst)
            elif dst.suffix.lower() == ".json":
                tables[name] = json.loads(dst.read_text(encoding="utf-8"))
    return tables


def plot_training(metrics, loss):
    if not metrics.empty:
        plt.figure(figsize=(11, 6))
        for col, color in [("WT_dice", "#0072B2"), ("TC_dice", "#D55E00"), ("ET_dice", "#009E73"), ("Mean_dice", "#111111")]:
            plt.plot(metrics["epoch"], metrics[col], label=col.replace("_", " "), linewidth=2.2, color=color)
        best = metrics.loc[metrics["Mean_dice"].idxmax()]
        plt.scatter([best["epoch"]], [best["Mean_dice"]], s=80, color="#CC79A7", zorder=5)
        plt.title(f"{MODEL_NAME} Validation Dice Across Training")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.ylim(0.55, 1.0)
        plt.grid(alpha=0.25)
        plt.legend(frameon=False, ncol=4, loc="lower right")
        plt.text(best["epoch"], best["Mean_dice"] + 0.015, f"Best logged mean={best['Mean_dice']:.4f}", ha="center", fontsize=9)
        savefig(FIG / "Figure_01_B-UDiff_validation_dice_training_curve.png")

    if not loss.empty:
        fig, ax1 = plt.subplots(figsize=(11, 6))
        ax1.plot(loss["epoch"], loss["training_loss_last_batch"], color="#0072B2", linewidth=1.7, label="Last-batch training loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training loss", color="#0072B2")
        ax1.tick_params(axis="y", labelcolor="#0072B2")
        ax1.grid(alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(loss["epoch"], loss["learning_rate"], color="#D55E00", linewidth=1.7, label="Learning rate")
        ax2.set_ylabel("Learning rate", color="#D55E00")
        ax2.tick_params(axis="y", labelcolor="#D55E00")
        plt.title(f"{MODEL_NAME} Optimization Trace")
        fig.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.93))
        savefig(FIG / "Figure_02_B-UDiff_training_loss_learning_rate.png")


def plot_prediction_statistics(pred_df, prob_df):
    plt.figure(figsize=(12, 7))
    volume_cols = ["WT_percent", "TC_percent", "ET_percent"]
    colors = ["#56B4E9", "#E69F00", "#009E73"]
    parts = plt.violinplot([pred_df[c].values for c in volume_cols], showmeans=True, showmedians=True)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_alpha(0.65)
    plt.xticks([1, 2, 3], ["Whole tumor", "Tumor core", "Enhancing tumor"])
    plt.ylabel("Predicted volume (% of scan)")
    plt.title(f"{MODEL_NAME} Predicted Tumor Burden Distribution (250 Cases)")
    plt.grid(axis="y", alpha=0.25)
    savefig(FIG / "Figure_03_B-UDiff_prediction_volume_distribution.png")

    if not prob_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].hist(prob_df["mean_entropy"], bins=30, color="#0072B2", alpha=0.82)
        axes[0].set_title("Mean predictive entropy")
        axes[0].set_xlabel("Normalized entropy")
        axes[0].set_ylabel("Cases")
        axes[0].grid(alpha=0.25)
        axes[1].hist(prob_df["mean_confidence"], bins=30, color="#009E73", alpha=0.82)
        axes[1].set_title("Mean max-class confidence")
        axes[1].set_xlabel("Confidence")
        axes[1].set_ylabel("Cases")
        axes[1].grid(alpha=0.25)
        fig.suptitle(f"{MODEL_NAME} Probability and Uncertainty Summary", y=1.02)
        savefig(FIG / "Figure_04_B-UDiff_uncertainty_confidence_distribution.png")


def plot_validation_and_ablation(tables):
    summary = tables.get("B-UDiff_250cases_validation_summary.csv")
    if summary is not None and not summary.empty:
        row = summary.iloc[0].to_dict()
        dice = [row.get("WT Dice ↑", row.get("WT Dice", np.nan)),
                row.get("TC Dice ↑", row.get("TC Dice", np.nan)),
                row.get("ET Dice ↑", row.get("ET Dice", np.nan))]
        hd95 = [row.get("WT HD95 ↓", row.get("WT HD95", np.nan)),
                row.get("TC HD95 ↓", row.get("TC HD95", np.nan)),
                row.get("ET HD95 ↓", row.get("ET HD95", np.nan))]
        dice = [float(str(x).split()[0]) for x in dice]
        hd95 = [float(str(x).split()[0]) for x in hd95]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        axes[0].bar(["WT", "TC", "ET"], dice, color=["#56B4E9", "#E69F00", "#009E73"])
        axes[0].set_ylim(0, 100)
        axes[0].set_ylabel("Dice (%)")
        axes[0].set_title("Labeled validation Dice")
        axes[1].bar(["WT", "TC", "ET"], hd95, color=["#56B4E9", "#E69F00", "#009E73"])
        axes[1].set_ylabel("HD95")
        axes[1].set_title("Labeled validation HD95")
        for ax in axes:
            ax.grid(axis="y", alpha=0.25)
        fig.suptitle(f"{MODEL_NAME} 250-Case Internal Validation Performance", y=1.02)
        savefig(FIG / "Figure_05_B-UDiff_validation_metric_summary.png")

    ablation = tables.get("B-UDiff_ablation_summary.csv")
    if ablation is not None and not ablation.empty:
        variant_col = "Variant" if "Variant" in ablation.columns else ablation.columns[0]
        avg_dice_col = next((c for c in ablation.columns if "Avg Dice" in c), None)
        avg_hd_col = next((c for c in ablation.columns if "Avg HD95" in c), None)
        if avg_dice_col:
            vals = [float(str(x).split()[0]) for x in ablation[avg_dice_col]]
            labels = ablation[variant_col].astype(str).tolist()
            plt.figure(figsize=(12, 6))
            bars = plt.barh(labels[::-1], vals[::-1], color="#0072B2")
            plt.xlabel("Average Dice (%)")
            plt.title(f"{MODEL_NAME} Ablation Study: Average Dice")
            plt.grid(axis="x", alpha=0.25)
            for bar, val in zip(bars, vals[::-1]):
                plt.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
            savefig(FIG / "Figure_06_B-UDiff_ablation_average_dice.png")
        if avg_hd_col:
            vals = [float(str(x).split()[0]) for x in ablation[avg_hd_col]]
            labels = ablation[variant_col].astype(str).tolist()
            plt.figure(figsize=(12, 6))
            bars = plt.barh(labels[::-1], vals[::-1], color="#D55E00")
            plt.xlabel("Average HD95")
            plt.title(f"{MODEL_NAME} Ablation Study: Average HD95")
            plt.grid(axis="x", alpha=0.25)
            for bar, val in zip(bars, vals[::-1]):
                plt.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
            savefig(FIG / "Figure_07_B-UDiff_ablation_average_hd95.png")


def normalize_slice(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def overlay_seg(image, seg, alpha=0.45):
    base = np.dstack([image, image, image])
    colors = {
        1: np.array([0.90, 0.10, 0.10]),
        2: np.array([0.95, 0.70, 0.05]),
        3: np.array([0.00, 0.60, 0.30]),
    }
    out = base.copy()
    for cls, color in colors.items():
        mask = seg == cls
        out[mask] = (1 - alpha) * out[mask] + alpha * color
    return out


def create_case_visuals(pred_df, prob_df):
    case_pool = pred_df.sort_values("WT_voxels", ascending=False)["case_id"].head(12).tolist()
    requested = ["BraTS-GLI-00014-000", "BraTS-GLI-00018-000", "BraTS-GLI-00006-000"]
    cases = []
    for c in requested + case_pool:
        if c not in cases and (PRED_DIR / f"{c}.npy").exists():
            cases.append(c)
    cases = cases[:8]

    fig, axes = plt.subplots(len(cases), 4, figsize=(13, 2.7 * len(cases)))
    if len(cases) == 1:
        axes = np.expand_dims(axes, 0)
    for r, case_id in enumerate(cases):
        seg = np.load(PRED_DIR / f"{case_id}.npy")
        z = int(np.argmax((seg > 0).sum(axis=(0, 1))))
        if (TRAIN_DIR / f"{case_id}.npy").exists():
            img = np.load(TRAIN_DIR / f"{case_id}.npy", mmap_mode="r")
            flair = normalize_slice(img[0, :, :, z])
        else:
            flair = np.zeros(seg[:, :, z].shape, dtype=np.float32)
        seg_sl = seg[:, :, z]
        prob_path = PRED_DIR / f"{case_id}_prob.npz"
        if prob_path.exists():
            prob = np.load(prob_path)["prob"].astype(np.float32)
            eps = 1e-7
            entropy = -np.sum(prob * np.log(np.clip(prob, eps, 1.0)), axis=0) / np.log(prob.shape[0])
            conf = np.max(prob, axis=0)
            entropy_sl = entropy[:, :, z]
            conf_sl = conf[:, :, z]
        else:
            entropy_sl = np.zeros_like(flair)
            conf_sl = np.zeros_like(flair)
        panels = [
            (flair, "MRI", "gray", None),
            (overlay_seg(flair, seg_sl), f"{MODEL_NAME} mask", None, None),
            (entropy_sl, "Uncertainty", "magma", (0, 1)),
            (conf_sl, "Confidence", "viridis", (0, 1)),
        ]
        for c, (arr, title, cmap, clim) in enumerate(panels):
            ax = axes[r, c]
            if cmap:
                im = ax.imshow(np.rot90(arr), cmap=cmap)
                if clim:
                    im.set_clim(*clim)
            else:
                ax.imshow(np.rot90(arr))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=11)
            if c == 0:
                ax.set_ylabel(f"{case_id}\nz={z}", fontsize=9)
    fig.suptitle(f"{MODEL_NAME} High-Resolution Prediction, Uncertainty, and Confidence Panels", y=0.995)
    savefig(FIG / "Figure_08_B-UDiff_case_montage_prediction_uncertainty.png", dpi=450)

    for case_id in cases[:3]:
        seg = np.load(PRED_DIR / f"{case_id}.npy")
        slices = np.argsort((seg > 0).sum(axis=(0, 1)))[-6:]
        slices = sorted(int(s) for s in slices)
        fig, axes = plt.subplots(2, 6, figsize=(15, 5))
        for i, z in enumerate(slices):
            if (TRAIN_DIR / f"{case_id}.npy").exists():
                img = np.load(TRAIN_DIR / f"{case_id}.npy", mmap_mode="r")
                flair = normalize_slice(img[0, :, :, z])
            else:
                flair = np.zeros(seg[:, :, z].shape, dtype=np.float32)
            axes[0, i].imshow(np.rot90(flair), cmap="gray")
            axes[0, i].set_title(f"z={z}", fontsize=9)
            axes[1, i].imshow(np.rot90(overlay_seg(flair, seg[:, :, z])))
            for ax in [axes[0, i], axes[1, i]]:
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].set_ylabel("MRI")
        axes[1, 0].set_ylabel(f"{MODEL_NAME}")
        fig.suptitle(f"{MODEL_NAME} Multi-Slice Prediction Overlay: {case_id}", y=1.03)
        savefig(FIG / f"{case_id}_B-UDiff_multislice_overlay.png", dpi=450)


def create_method_figure():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.axis("off")
    boxes = [
        (0.03, 0.62, 0.18, 0.18, "Multi-modal MRI\nT1/T1ce/T2/FLAIR", "#D9EAF7"),
        (0.29, 0.72, 0.20, 0.15, "Boundary branch\nE_b / D_b", "#D9F0D3"),
        (0.29, 0.42, 0.20, 0.15, "Diffusion branch\nE_d / D_d", "#DDE7F7"),
        (0.56, 0.57, 0.18, 0.16, "MBA\nBoundary feature fusion", "#FCE4D6"),
        (0.80, 0.68, 0.16, 0.15, "MC-Diff\nUncertainty map", "#F4D6E8"),
        (0.80, 0.38, 0.16, 0.15, "PURE\nProgressive refinement", "#E8E2F4"),
        (0.56, 0.20, 0.18, 0.13, "Final 3D glioma\nsegmentation", "#FFF2CC"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#333333", linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, weight="bold")
    arrows = [
        ((0.21, 0.71), (0.29, 0.80)),
        ((0.21, 0.71), (0.29, 0.50)),
        ((0.49, 0.80), (0.56, 0.67)),
        ((0.49, 0.50), (0.56, 0.62)),
        ((0.74, 0.65), (0.80, 0.75)),
        ((0.74, 0.62), (0.80, 0.45)),
        ((0.88, 0.38), (0.65, 0.33)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.7, color="#333333"))
    ax.text(0.03, 0.93, PAPER_TITLE, fontsize=15, weight="bold")
    ax.text(0.03, 0.07, "Training: uncertainty-guided segmentation loss + boundary loss. Inference: iterative denoising with uncertainty-weighted fusion.", fontsize=11)
    savefig(FIG / "Figure_09_B-UDiff_methodology_overview.png", dpi=450)


def df_to_markdown(df):
    if df is None or df.empty:
        return "No table available."
    safe = df.copy()
    safe = safe.astype(str)
    headers = list(safe.columns)
    rows = safe.values.tolist()
    widths = []
    for i, header in enumerate(headers):
        max_cell = max([len(str(r[i])) for r in rows], default=0)
        widths.append(max(len(str(header)), max_cell))
    header_line = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    body = ["| " + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + body)


def write_reports(metrics, loss, pred_df, prob_df, tables):
    software = {}
    try:
        import torch
        software["torch"] = torch.__version__
        software["cuda_available"] = bool(torch.cuda.is_available())
        software["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU/runtime unavailable"
    except Exception as exc:
        software["torch_error"] = str(exc)
    software["numpy"] = np.__version__
    software["pandas"] = pd.__version__

    model_info = {
        "paper_title": PAPER_TITLE,
        "model_name": MODEL_NAME,
        "final_checkpoint": str(MODEL_PATH),
        "best_checkpoint": str(BEST_MODEL_PATH),
        "prediction_directory": str(PRED_DIR),
        "selected_prediction_items": int(len(list(PRED_DIR.glob("*")))),
        "segmentation_npy_files": int(len(list(PRED_DIR.glob("*.npy")))),
        "probability_npz_files": int(len(list(PRED_DIR.glob("*.npz")))),
        "software": software,
    }
    try:
        import torch
        sd = torch.load(MODEL_PATH, map_location="cpu")
        model_info["checkpoint_size_bytes"] = MODEL_PATH.stat().st_size
        model_info["state_dict_tensors"] = len(sd) if hasattr(sd, "__len__") else None
        model_info["state_dict_parameters"] = int(sum(v.numel() for v in sd.values() if hasattr(v, "numel"))) if isinstance(sd, dict) else None
    except Exception as exc:
        model_info["checkpoint_inspection_error"] = str(exc)
    (DATA / "B-UDiff_model_and_path_manifest.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")

    best_logged = {}
    if not metrics.empty:
        best = metrics.loc[metrics["Mean_dice"].idxmax()]
        best_logged = {k: (float(v) if k != "epoch" else int(v)) for k, v in best.to_dict().items()}

    pred_desc = pred_df[["WT_percent", "TC_percent", "ET_percent", "WT_voxels", "TC_voxels", "ET_voxels"]].describe().T
    pred_desc.to_csv(DATA / "B-UDiff_prediction_descriptive_statistics.csv")
    if not prob_df.empty:
        prob_df[["mean_entropy", "p95_entropy", "mean_confidence", "tumor_mean_entropy", "tumor_mean_confidence"]].describe().T.to_csv(DATA / "B-UDiff_uncertainty_descriptive_statistics.csv")

    method_text = METHOD_FILE.read_text(encoding="utf-8", errors="replace") if METHOD_FILE.exists() else ""
    method_excerpt = method_text[:1800].replace("\n", " ")
    validation_summary = ""
    if "B-UDiff_250cases_validation_summary.csv" in tables:
        validation_summary = df_to_markdown(tables["B-UDiff_250cases_validation_summary.csv"])
    ablation_summary = ""
    if "B-UDiff_ablation_summary.csv" in tables:
        ablation_summary = df_to_markdown(tables["B-UDiff_ablation_summary.csv"])

    report = f"""# {PAPER_TITLE}

## Model identity

Model name used throughout this bundle: **{MODEL_NAME}**.

Final checkpoint: `{MODEL_PATH}`

Best checkpoint found in the experiment folder: `{BEST_MODEL_PATH}`

Prediction-result folder: `{PRED_DIR}`

The selected prediction folder contains {model_info['selected_prediction_items']} files: {model_info['segmentation_npy_files']} segmentation masks (`.npy`) and {model_info['probability_npz_files']} probability volumes (`.npz`).

## Main methodology

B-UDiff is a boundary-aware, uncertainty-guided conditional diffusion network for 3D glioma segmentation. The implementation and manuscript text organize the method around four major mechanisms:

1. A boundary-prediction branch that learns tumor-margin-sensitive cues.
2. A diffusion-denoising branch that recovers the segmentation mask from a noise-corrupted label estimate conditioned on multi-modal MRI.
3. Multi-granularity boundary aggregation (MBA), which injects boundary features into the denoising path.
4. Monte Carlo diffusion (MC-Diff) and Progressive Uncertainty-driven REfinement (PURE), which estimate predictive uncertainty and fuse reverse-diffusion predictions.

Method text source: `{METHOD_FILE}`

Short source excerpt: {method_excerpt}

## Experimental design

The experiment follows an internal labeled validation design for BraTS 2025 glioma segmentation. Training used `data/fullres/train`, 5-fold loader logic with fold 0, 1001 training cases and 250 validation cases as recorded in the training log. The validated quantitative result uses the labeled internal split, while the selected prediction folder is summarized as prediction-only output because those masks/probability maps do not by themselves contain ground-truth labels.

Reported tumor regions follow BraTS conventions:

- WT: all non-background tumor voxels.
- TC: class 1 plus class 3 voxels.
- ET: class 3 voxels.

## Materials

Input imaging: multi-parametric 3D MRI volumes with four channels, treated as T1/T1ce/T2/FLAIR-style BraTS inputs by the project scripts. Segmentation labels use four classes: background, NCR/NET, edema, and enhancing tumor.

Training data path: `{TRAIN_DIR}`

Prediction masks and probabilities: `{PRED_DIR}`

## Software and hardware

Runtime used to generate this bundle:

```json
{json.dumps(software, indent=2)}
```

The training script and log reference an RTX 3050-oriented setup, batch size 1, patch size 128 x 128 x 128, SGD optimizer, polynomial learning-rate scheduling, mixed precision infrastructure, and the `segment` Conda environment.

## Training configuration recovered from code and logs

- Script: `3_train_diffunet_brats2025.py`
- Architecture call: `DiffUNet(4, 4)`
- Max epochs in active training log: 500 completed validation reporting epochs, ending at epoch 499.
- Optimizer in code: SGD, learning rate 0.01, weight decay 3e-5, momentum 0.99, Nesterov enabled.
- Patch size: 128 x 128 x 128.
- Batch size: 1.
- Augmentation: enabled.
- Final logged epoch: WT=0.9171, TC=0.9080, ET=0.8667, Mean=0.8973.
- Best logged validation snapshot in parsed log: `{best_logged}`.

## Labeled validation results

{validation_summary}

## Statistical analysis

The statistical outputs copied into this bundle include descriptive statistics, 95% confidence intervals, Dice/HD95 correlation tests, and Shapiro-Wilk normality checks from the existing validated 250-case result.

Key interpretation: the internal validation mean Dice is high for WT, TC, and ET, while Dice and HD95 are strongly negatively correlated, as expected for segmentation quality metrics.

## Ablation study

{ablation_summary}

Important note: the existing ablation report states that the ablation table was computed on 25 internal validation cases and uses controlled inference-time variants from the available checkpoint, not separately retrained models for every row.

## Prediction-only analysis of selected outputs

The selected prediction directory was summarized across all 250 `.npy` masks and paired `.npz` probability volumes. This bundle includes per-case tumor burden, class-voxel counts, confidence, entropy, and tumor-region uncertainty summaries.

Summary of predicted tumor burden:

{df_to_markdown(pred_desc.reset_index().rename(columns={"index": "metric"}))}

## Generated high-resolution figures

- `figures/Figure_01_B-UDiff_validation_dice_training_curve.png`
- `figures/Figure_02_B-UDiff_training_loss_learning_rate.png`
- `figures/Figure_03_B-UDiff_prediction_volume_distribution.png`
- `figures/Figure_04_B-UDiff_uncertainty_confidence_distribution.png`
- `figures/Figure_05_B-UDiff_validation_metric_summary.png`
- `figures/Figure_06_B-UDiff_ablation_average_dice.png`
- `figures/Figure_07_B-UDiff_ablation_average_hd95.png`
- `figures/Figure_08_B-UDiff_case_montage_prediction_uncertainty.png`
- `figures/Figure_09_B-UDiff_methodology_overview.png`
- Case-level multi-slice overlays for the requested cases.

## Limitations and proper reporting language

Use the 250-case labeled validation table as the primary quantitative result. Use selected-folder statistics as prediction-output characterization, not Dice/HD95 performance, unless matching ground-truth masks are explicitly evaluated. For external state-of-the-art comparison tables, add externally trained baseline checkpoints or peer-reviewed numbers with citations; this local repository does not contain honest runnable checkpoints for those baselines.
"""
    (DOC / "B-UDiff_experimental_report.md").write_text(report, encoding="utf-8")

    latex = report.replace("# ", "\\section*{").replace("\n## ", "}\n\\subsection*{")
    (DOC / "B-UDiff_experimental_report_source.txt").write_text(report, encoding="utf-8")


def main():
    ensure_dirs()
    metrics, loss = parse_training_log()
    metrics.to_csv(TAB / "B-UDiff_training_validation_log.csv", index=False)
    loss.to_csv(TAB / "B-UDiff_training_loss_lr_log.csv", index=False)
    pred_df, prob_df = summarize_predictions()
    tables = load_existing_tables()
    plot_training(metrics, loss)
    plot_prediction_statistics(pred_df, prob_df)
    plot_validation_and_ablation(tables)
    create_case_visuals(pred_df, prob_df)
    create_method_figure()
    write_reports(metrics, loss, pred_df, prob_df, tables)
    print(f"Created B-UDiff bundle at: {OUT}")


if __name__ == "__main__":
    main()
