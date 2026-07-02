import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def filte_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    return {str(k)[7:] if str(k).startswith("module") else str(k): v for k, v in sd.items()}


def normalize_slice(img):
    img = np.asarray(img, dtype=np.float32)
    mask = img != 0
    lo, hi = np.percentile(img[mask] if mask.any() else img, [1, 99])
    if hi <= lo:
        return np.zeros_like(img)
    return (np.clip(img, lo, hi) - lo) / (hi - lo)


def select_cases(metrics_csv, count):
    with open(metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda r: float(r["Avg_dice"]))
    idxs = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[i]["case"] for i in idxs]


def load_case(data_dir, case):
    data_dir = Path(data_dir)
    image = np.load(data_dir / f"{case}.npy")
    seg = np.load(data_dir / f"{case}_seg.npy")[0]
    return image, seg


def crop_slices(seg, patch_size=128):
    coords = np.argwhere(seg > 0)
    if coords.size == 0:
        center = np.array(seg.shape) // 2
    else:
        center = coords.mean(axis=0).astype(int)
    starts = []
    ends = []
    for c, dim in zip(center, seg.shape):
        start = int(max(0, min(c - patch_size // 2, dim - patch_size)))
        end = int(min(dim, start + patch_size))
        starts.append(start)
        ends.append(end)
    return tuple(slice(s, e) for s, e in zip(starts, ends))


def pad_to_patch(arr, patch_size=128):
    pads = []
    for dim in arr.shape[-3:]:
        pads.append((0, max(0, patch_size - dim)))
    if arr.ndim == 4:
        return np.pad(arr, [(0, 0)] + pads, mode="constant")
    return np.pad(arr, pads, mode="constant")


class GradCamCapture:
    def __init__(self, module):
        self.activations = None
        self.gradients = None
        self.fwd = module.register_forward_hook(self.forward_hook)
        self.bwd = module.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inputs, output):
        self.activations = output

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def close(self):
        self.fwd.remove()
        self.bwd.remove()


def gradcam_for_case(model, image_patch, seg_patch, device, target_class):
    model.zero_grad(set_to_none=True)
    hook = GradCamCapture(model.edge_model.decoder.stages[-1])
    x = torch.from_numpy(image_patch.copy()).unsqueeze(0).float().to(device)
    logits, _ = model.edge_model(x)
    target_mask = torch.from_numpy((seg_patch == target_class).astype(np.float32)).to(device)
    if target_mask.sum() > 0:
        score = (logits[0, target_class] * target_mask).sum() / target_mask.sum()
    else:
        score = logits[0, target_class].mean()
    score.backward()
    acts = hook.activations.detach()
    grads = hook.gradients.detach()
    weights = grads.mean(dim=(2, 3, 4), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1))[0]
    cam = torch.nn.functional.interpolate(
        cam[None, None],
        size=seg_patch.shape,
        mode="trilinear",
        align_corners=False,
    )[0, 0]
    cam = cam.cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    pred = logits.argmax(dim=1).detach().cpu().numpy()[0]
    hook.close()
    return cam, pred


def full_prediction_prob(case, pred_dir):
    prob_path = Path(pred_dir) / f"{case}_prob.npz"
    if not prob_path.exists():
        return None
    prob = np.load(prob_path)["prob"].astype(np.float32)
    entropy = -np.sum(prob * np.log(prob + 1e-8), axis=0)
    return entropy / (entropy.max() + 1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data/fullres/train")
    parser.add_argument("--checkpoint", default="./logs/diffunet/model/best_model_0.9027.pt")
    parser.add_argument("--pred-dir", default="./prediction_results/DiffUNet_selected")
    parser.add_argument("--out-dir", default="./interpretability_outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-count", type=int, default=6)
    parser.add_argument("--cases", nargs="*", default=None)
    parser.add_argument("--metrics-csv", default="./quantitative_results/DiffUNet_250cases_per_case_metrics.csv")
    args = parser.parse_args()

    from diffunet.diffunet_model import DiffUNet

    cases = args.cases or select_cases(args.metrics_csv, args.case_count)
    model = DiffUNet(4, 4)
    model.load_state_dict(filte_state_dict(torch.load(args.checkpoint, map_location="cpu")))
    model.to(args.device)
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        image, seg = load_case(args.data_dir, case)
        slices = crop_slices(seg)
        image_patch = pad_to_patch(image[(slice(None),) + slices])
        seg_patch = pad_to_patch(seg[slices])
        target_class = 3 if np.any(seg_patch == 3) else (2 if np.any(seg_patch == 2) else 1)
        cam, edge_pred = gradcam_for_case(model, image_patch, seg_patch, args.device, target_class)
        z = int(np.argmax(np.sum(seg_patch > 0, axis=(1, 2))))
        base = normalize_slice(image_patch[2, z])
        entropy = full_prediction_prob(case, args.pred_dir)
        entropy_slice = None
        if entropy is not None:
            entropy_patch = pad_to_patch(entropy[slices])
            entropy_slice = entropy_patch[z]

        fig, axes = plt.subplots(1, 4 if entropy_slice is not None else 3, figsize=(14, 4))
        axes[0].imshow(base, cmap="gray")
        axes[0].set_title("T2-FLAIR")
        axes[1].imshow(base, cmap="gray")
        axes[1].imshow(seg_patch[z] > 0, cmap="autumn", alpha=0.45)
        axes[1].set_title("GT overlay")
        axes[2].imshow(base, cmap="gray")
        axes[2].imshow(cam[z], cmap="jet", alpha=0.55)
        axes[2].set_title(f"Boundary Grad-CAM class {target_class}")
        if entropy_slice is not None:
            axes[3].imshow(base, cmap="gray")
            axes[3].imshow(entropy_slice, cmap="magma", alpha=0.55)
            axes[3].set_title("Prediction entropy")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(case)
        fig.tight_layout()
        fig.savefig(out_dir / f"{case}_gradcam_entropy.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved Grad-CAM-like heatmaps to {out_dir}")


if __name__ == "__main__":
    main()
