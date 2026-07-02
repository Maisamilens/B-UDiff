# 1b_preprocessing_brats2025.py
import os
import numpy as np
import SimpleITK as sitk
import pickle
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
RAW_VAL_DIR = r"D:\Brats 2025\Brats 2025 Glioma Pre Challange\glio_25\glio_val"
OUTPUT_DIR = "./data/fullres/val"
TARGET_SPACING = [1.0, 1.0, 1.0]
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def resample_image(image_array, original_spacing, target_spacing, is_label=False):
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetSpacing(tuple(original_spacing[::-1]))
    original_size = np.array(sitk_image.GetSize())
    original_spacing_sitk = np.array(sitk_image.GetSpacing())
    target_spacing_sitk = np.array(target_spacing[::-1])
    new_size = (original_size * original_spacing_sitk / target_spacing_sitk).astype(int).tolist()
    new_size = [max(1, s) for s in new_size]
    interpolator = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    resampled = sitk.Resample(
        sitk_image, new_size, sitk.Transform(), interpolator,
        sitk_image.GetOrigin(), target_spacing_sitk.tolist(),
        sitk_image.GetDirection(), 0, sitk_image.GetPixelID()
    )
    return sitk.GetArrayFromImage(resampled)

def normalize_modality(data):
    mask = data > 0
    if mask.any():
        mean = data[mask].mean()
        std = data[mask].std()
        if std > 0:
            data[mask] = (data[mask] - mean) / std
    return data

def crop_to_nonzero(data):
    nonzero_mask = np.any(data > 0, axis=0)
    coords = np.argwhere(nonzero_mask)
    if len(coords) == 0:
        return data, None, data.shape[1:]
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1
    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    cropped_data = data[(slice(None),) + slices]
    bbox = [(int(min_c), int(max_c)) for min_c, max_c in zip(min_coords, max_coords)]
    return cropped_data, bbox, data.shape[1:]

if __name__ == "__main__":
    case_dirs = sorted([
        d for d in os.listdir(RAW_VAL_DIR)
        if os.path.isdir(os.path.join(RAW_VAL_DIR, d)) and d.startswith("BraTS-GLI")
    ])
    print(f"Found {len(case_dirs)} validation cases")

    for case_name in tqdm(case_dirs, desc="Preprocessing validation data"):
        case_dir = os.path.join(RAW_VAL_DIR, case_name)
        modality_suffixes = ['t1c', 't1n', 't2f', 't2w']
        modalities = []
        original_spacing = None

        for suffix in modality_suffixes:
            filepath = os.path.join(case_dir, f"{case_name}-{suffix}.nii.gz")
            sitk_img = sitk.ReadImage(filepath)
            if original_spacing is None:
                original_spacing = list(sitk_img.GetSpacing()[::-1])
            arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
            modalities.append(arr)

        original_shape = list(modalities[0].shape)

        # Resample
        resampled = []
        for mod in modalities:
            resampled.append(resample_image(mod, original_spacing, TARGET_SPACING, is_label=False))
        data = np.stack(resampled, axis=0).astype(np.float32)

        # Normalize
        for i in range(4):
            data[i] = normalize_modality(data[i])

        # Crop
        data, bbox, resampled_shape = crop_to_nonzero(data)

        # Dummy seg (zeros) — needed by the dataloader's unpack function
        seg_out = np.zeros((1,) + data.shape[1:], dtype=np.int32)

        # Save
        save_path_npz = os.path.join(OUTPUT_DIR, f"{case_name}.npz")
        np.savez_compressed(save_path_npz, data=data, seg=seg_out)

        properties = {
            'name': [case_name],
            'spacing': original_spacing,
            'original_shape': original_shape,
            'resampled_shape': list(resampled_shape),
            'crop_bbox': bbox,
            'class_locations': {},
        }
        save_path_pkl = os.path.join(OUTPUT_DIR, f"{case_name}.pkl")
        with open(save_path_pkl, 'wb') as f:
            pickle.dump(properties, f)

    print(f"Validation preprocessing complete! {len(case_dirs)} cases saved to {OUTPUT_DIR}")