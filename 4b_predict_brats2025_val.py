# 4b_predict_brats2025_val.py
import numpy as np
import torch
import os
import glob
import pickle
import SimpleITK as sitk
from monai.inferers import SlidingWindowInferer
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
VAL_DATA_DIR = "./data/fullres/val"
RAW_VAL_DIR = r"D:\Brats 2025\Brats 2025 Glioma Pre Challange\glio_25\glio_val"
SAVE_DIR = "./prediction_results/diffunet/val"
DEVICE = "cuda:0"
PATCH_SIZE = [128, 128, 128]

# >>>> SET THIS TO YOUR BEST MODEL PATH <<<<
MODEL_PATH = "./logs/diffunet/model/final_model_0.8973.pt"
# ============================================================

os.makedirs(SAVE_DIR, exist_ok=True)


def filte_state_dict(sd):
    if "module" in sd:
        sd = sd["module"]
    new_sd = {}
    for k, v in sd.items():
        k = str(k)
        new_k = k[7:] if k.startswith("module") else k
        new_sd[new_k] = v
    del sd
    return new_sd


def uncrop_prediction(pred, crop_bbox, resampled_shape):
    """Pad prediction back to resampled shape using crop bbox."""
    if crop_bbox is None:
        return pred
    result = np.zeros((3,) + tuple(resampled_shape), dtype=np.float32)
    z_start, z_end = crop_bbox[0]
    y_start, y_end = crop_bbox[1]
    x_start, x_end = crop_bbox[2]
    result[:, z_start:z_end, y_start:y_end, x_start:x_end] = pred
    return result


def convert_labels_to_brats(output):
    """Convert 4-class prediction to WT, TC, ET channels."""
    wt = (output > 0).astype(np.float32)
    tc = ((output == 1) | (output == 3)).astype(np.float32)
    et = (output == 3).astype(np.float32)
    return np.stack([wt, tc, et], axis=0)


def resample_back(prediction, original_spacing, target_spacing, original_shape):
    """Resample prediction from target spacing back to original spacing."""
    result = np.zeros((3,) + tuple(original_shape), dtype=np.float32)
    for c in range(3):
        sitk_pred = sitk.GetImageFromArray(prediction[c])
        sitk_pred.SetSpacing(tuple(target_spacing[::-1]))

        original_size = np.array(original_shape)[::-1]
        original_spacing_sitk = np.array(original_spacing[::-1])
        target_spacing_sitk = np.array(target_spacing[::-1])

        new_size = (np.array(sitk_pred.GetSize()) * target_spacing_sitk / original_spacing_sitk).astype(int).tolist()
        new_size = [max(1, s) for s in new_size]

        resampled = sitk.Resample(
            sitk_pred, new_size, sitk.Transform(), sitk.sitkLinear,
            sitk_pred.GetOrigin(), original_spacing_sitk.tolist(),
            sitk_pred.GetDirection(), 0, sitk_pred.GetPixelID()
        )
        res = sitk.GetArrayFromImage(resampled)

        # Crop or pad to match original shape
        result_c = np.zeros(original_shape, dtype=np.float32)
        s = tuple(slice(0, min(r, o)) for r, o in zip(res.shape, original_shape))
        result_c[s] = res[s]
        result[c] = result_c

    return result


if __name__ == "__main__":
    # Load model
    from diffunet.diffunet_model import DiffUNet
    model = DiffUNet(4, 4)
    new_sd = filte_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.load_state_dict(new_sd)
    model.to(DEVICE)
    model.eval()

    window_infer = SlidingWindowInferer(
        roi_size=PATCH_SIZE, sw_batch_size=1, overlap=0.5, mode="gaussian"
    )

    # Get all preprocessed validation cases
    npz_files = sorted(glob.glob(os.path.join(VAL_DATA_DIR, "*.npz")))
    print(f"Found {len(npz_files)} validation cases")

    for npz_file in tqdm(npz_files, desc="Predicting validation cases"):
        case_name = os.path.splitext(os.path.basename(npz_file))[0]

        # Load preprocessed data
        data_npy = np.load(npz_file.replace(".npz", ".npy"), "r") if os.path.exists(npz_file.replace(".npz", ".npy")) else np.load(npz_file)["data"]

        # Load properties
        pkl_path = npz_file.replace(".npz", ".pkl")
        with open(pkl_path, "rb") as f:
            properties = pickle.load(f)

        # Run inference
        image = torch.from_numpy(np.array(data_npy)).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            output = window_infer(image, lambda x: model(x, ddim=True))
            if isinstance(output, tuple):
                output = output[0]
            output = output.argmax(dim=1).cpu().numpy()[0]  # (D, H, W)

        # Convert to WT, TC, ET
        brats_pred = convert_labels_to_brats(output)

        # Uncrop
        if properties.get('crop_bbox') is not None and properties.get('resampled_shape') is not None:
            brats_pred = uncrop_prediction(brats_pred, properties['crop_bbox'], properties['resampled_shape'])

        # Resample back to original spacing
        if properties.get('spacing') is not None and properties.get('original_shape') is not None:
            brats_pred = resample_back(
                brats_pred,
                properties['spacing'],
                [1.0, 1.0, 1.0],
                properties['original_shape']
            )

        # Save as NIfTI (use raw data for reference geometry)
        raw_case_dir = os.path.join(RAW_VAL_DIR, case_name)
        ref_path = os.path.join(raw_case_dir, f"{case_name}-t1c.nii.gz")
        ref_img = sitk.ReadImage(ref_path)

        for i, label_name in enumerate(["WT", "TC", "ET"]):
            pred_sitk = sitk.GetImageFromArray(brats_pred[i])
            pred_sitk.CopyInformation(ref_img)
            save_path = os.path.join(SAVE_DIR, f"{case_name}-{label_name}.nii.gz")
            sitk.WriteImage(pred_sitk, save_path)

    print(f"Predictions saved to {SAVE_DIR}")