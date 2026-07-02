import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    new_sd = {}
    for k, v in sd.items():
        k = str(k)
        new_sd[k[7:] if k.startswith("module") else k] = v
    return new_sd


def normalize_slice(img):
    img = np.asarray(img, dtype=np.float32)
    mask = img != 0
    if mask.any():
        lo, hi = np.percentile(img[mask], [1, 99])
    else:
        lo, hi = np.percentile(img, [1, 99])
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


def choose_slice(seg):
    areas = np.sum(seg > 0, axis=(1, 2))
    if np.max(areas) == 0:
        return seg.shape[0] // 2
    return int(np.argmax(areas))


def add_tumor_box(ax, seg_slice, color="#d7191c"):
    coords = np.argwhere(seg_slice > 0)
    if coords.size == 0:
        return
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    pad = 6
    x0 = max(int(x0) - pad, 0)
    y0 = max(int(y0) - pad, 0)
    x1 = min(int(x1) + pad, seg_slice.shape[1] - 1)
    y1 = min(int(y1) + pad, seg_slice.shape[0] - 1)
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.4))


def load_case(data_dir, case_name):
    data_dir = Path(data_dir)
    image_path = data_dir / f"{case_name}.npy"
    seg_path = data_dir / f"{case_name}_seg.npy"
    if image_path.exists() and seg_path.exists():
        image = np.load(image_path, mmap_mode="r")
        seg = np.load(seg_path, mmap_mode="r")[0]
    else:
        npz = np.load(data_dir / f"{case_name}.npz")
        image = npz["data"]
        seg = npz["seg"][0]
    return image, seg


def predict_case(model, inferer, image, device):
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(image)).unsqueeze(0).float().to(device)
        output = inferer(x, lambda y: model(y, ddim=True))
        if isinstance(output, tuple):
            output = output[0]
        return output.argmax(dim=1).cpu().numpy()[0]


def read_metrics_cases(metrics_csv, count):
    path = Path(metrics_csv)
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r.get("Avg_dice")]
    rows.sort(key=lambda r: float(r["Avg_dice"]))
    if len(rows) <= count:
        return [r["case"] for r in rows]
    idxs = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[i]["case"] for i in idxs]


def parse_cases(cases):
    if not cases:
        return None
    if len(cases) == 1 and "," in cases[0]:
        return [c.strip() for c in cases[0].split(",") if c.strip()]
    return cases


def parse_method_preds(items):
    parsed = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError("--method-pred entries must look like Name=folder")
        name, folder = item.split("=", 1)
        parsed[name.strip()] = Path(folder.strip())
    return parsed


def load_method_prediction(folder, case_name):
    candidates = [
        folder / f"{case_name}.npy",
        folder / f"{case_name}.nii.gz",
        folder / f"{case_name}-seg.nii.gz",
        folder / f"{case_name}_seg.nii.gz",
    ]
    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".npy":
            arr = np.load(path)
            return arr[0] if arr.ndim == 4 else arr
        import SimpleITK as sitk

        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    raise FileNotFoundError(f"No prediction found for {case_name} in {folder}")


def main():
    parser = argparse.ArgumentParser(description="Create BraTS2025 qualitative visualization grid.")
    parser.add_argument("--data-dir", default="./data/fullres/train")
    parser.add_argument("--checkpoint", default="./logs/diffunet/model/best_model_0.9027.pt")
    parser.add_argument("--out", default="./visualization_outputs/brats2025_qualitative_diffunet.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-size", nargs=3, type=int, default=[128, 128, 128])
    parser.add_argument("--modality", type=int, default=2, help="0=t1c, 1=t1n, 2=t2f, 3=t2w")
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--case-count", type=int, default=6)
    parser.add_argument("--title", default="", help="Optional figure title. Empty string gives a paper-style panel.")
    parser.add_argument(
        "--method-pred",
        action="append",
        default=[],
        help="Optional aligned prediction folder as Name=folder. Can be passed multiple times.",
    )
    parser.add_argument(
        "--expected-methods",
        nargs="*",
        default=[],
        help="Paper-style columns to include after Ours. Missing prediction folders are rendered as N/A.",
    )
    parser.add_argument(
        "--metrics-csv",
        default="./quantitative_results/DiffUNet_250cases_per_case_metrics.csv",
        help="Used to pick representative cases when --cases is not provided.",
    )
    args = parser.parse_args()

    cases = parse_cases(args.cases)
    if cases is None:
        cases = read_metrics_cases(args.metrics_csv, args.case_count)
    if cases is None:
        cases = [p.stem for p in sorted(Path(args.data_dir).glob("*.npz"))[: args.case_count]]
    cases = cases[: args.case_count]
    method_pred_dirs = parse_method_preds(args.method_pred)
    if args.expected_methods:
        extra_methods = [(name, method_pred_dirs.get(name)) for name in args.expected_methods]
    else:
        extra_methods = list(method_pred_dirs.items())

    from diffunet.diffunet_model import DiffUNet

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

    cols = ["Image", "GT", "Ours"] + [name for name, _ in extra_methods]
    fig, axes = plt.subplots(len(cases), len(cols), figsize=(len(cols) * 2.4, len(cases) * 2.35))
    if len(cases) == 1:
        axes = axes[None, :]

    for row_idx, case_name in enumerate(cases):
        image, seg = load_case(args.data_dir, case_name)
        pred = predict_case(model, inferer, image, args.device)
        z = choose_slice(seg)
        base = np.asarray(image[args.modality, z])
        gt_slice = np.asarray(seg[z])
        pred_slice = np.asarray(pred[z])

        axes[row_idx, 0].imshow(normalize_slice(base), cmap="gray")
        add_tumor_box(axes[row_idx, 0], gt_slice)
        axes[row_idx, 1].imshow(overlay_labels(base, gt_slice))
        axes[row_idx, 2].imshow(overlay_labels(base, pred_slice))
        for extra_idx, (method_name, folder) in enumerate(extra_methods, start=3):
            if folder is None:
                axes[row_idx, extra_idx].imshow(normalize_slice(base), cmap="gray")
                axes[row_idx, extra_idx].text(
                    0.5,
                    0.5,
                    "N/A",
                    transform=axes[row_idx, extra_idx].transAxes,
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.65, edgecolor="none", pad=4),
                )
            else:
                method_pred = load_method_prediction(folder, case_name)
                if method_pred.shape != seg.shape:
                    raise ValueError(
                        f"{method_name} prediction for {case_name} has shape {method_pred.shape}, "
                        f"expected preprocessed shape {seg.shape}."
                    )
                axes[row_idx, extra_idx].imshow(overlay_labels(base, method_pred[z]))

        for col_idx in range(len(cols)):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            for spine in axes[row_idx, col_idx].spines.values():
                spine.set_linewidth(1.2)
                spine.set_color("white")
        axes[row_idx, 0].set_ylabel(case_name, fontsize=8, rotation=90, labelpad=18)

    for col_idx, name in enumerate(cols):
        axes[-1, col_idx].set_xlabel(name, fontsize=14, labelpad=8)

    fig.text(0.015, 0.5, "BraTS2025 Glioma", va="center", rotation=90, fontsize=15, fontweight="bold")
    if args.title:
        fig.suptitle(args.title, fontsize=16, y=0.995)
        rect = [0.06, 0.04, 1.0, 0.965]
    else:
        rect = [0.06, 0.04, 1.0, 0.995]
    plt.tight_layout(rect=rect, w_pad=0.35, h_pad=0.35)

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved qualitative figure: {out}")
    print("Cases:", ", ".join(cases))


if __name__ == "__main__":
    main()
