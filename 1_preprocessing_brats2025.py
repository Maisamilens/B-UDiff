# 1_preprocessing_brats2025.py
import os
import numpy as np
import SimpleITK as sitk
import pickle
import random
from tqdm import tqdm
from scipy.ndimage import find_objects

# ============================================================
# CONFIGURATION - CHANGE THESE PATHS
# ============================================================
RAW_TRAIN_DIR = r"D:\Brats 2025\Brats 2025 Glioma Pre Challange\glio_25\glio_train"
OUTPUT_DIR = "./data/fullres/train"
TARGET_SPACING = [1.0, 1.0, 1.0]  # resample to 1mm isotropic
VAL_SPLIT = 0.2  # 20% of training data for validation
RANDOM_SEED = 42
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

def resample_image(image_array, original_spacing, target_spacing, is_label=False):
    """Resample a 3D numpy array to target spacing using SimpleITK."""
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetSpacing(tuple(original_spacing[::-1]))  # SimpleITK uses (x,y,z)

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
    """Z-score normalization on non-zero voxels."""
    mask = data > 0
    if mask.any():
        mean = data[mask].mean()
        std = data[mask].std()
        if std > 0:
            data[mask] = (data[mask] - mean) / std
    return data


def crop_to_nonzero(data, seg=None):
    """Crop data (and optionally seg) to the non-zero region of data.
    Returns cropped data, cropped seg, and the bounding box slices."""
    nonzero_mask = np.any(data > 0, axis=0)
    coords = np.argwhere(nonzero_mask)

    if len(coords) == 0:
        return data, seg, None, data.shape[1:]

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1
    slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))

    cropped_data = data[(slice(None),) + slices]
    cropped_seg = seg[slices] if seg is not None else None
    bbox = [(int(min_c), int(max_c)) for min_c, max_c in zip(min_coords, max_coords)]

    return cropped_data, cropped_seg, bbox, data.shape[1:]


def compute_class_locations(seg):
    """Compute voxel locations for each foreground class for oversampling."""
    class_locations = {}
    for c in range(1, 4):  # labels 1, 2, 3
        class_mask = seg == c
        coords = np.argwhere(class_mask)  # (N, 3) z,y,x
        if len(coords) > 0:
            # Format: (N, 4) with [class, z, y, x] — matches base_data_loader.py
            class_locations[c] = np.concatenate(
                [np.full((len(coords), 1), c, dtype=np.int32), coords], axis=1
            )
        else:
            class_locations[c] = np.zeros((0, 4), dtype=np.int32)
    return class_locations


def process_case(case_dir, case_name):
    """Process a single case: resample, normalize, crop, save."""
    # Read all modalities
    modality_suffixes = ['t1c', 't1n', 't2f', 't2w']
    modalities = []
    original_spacing = None

    for suffix in modality_suffixes:
        filepath = os.path.join(case_dir, f"{case_name}-{suffix}.nii.gz")
        sitk_img = sitk.ReadImage(filepath)

        if original_spacing is None:
            original_spacing = list(sitk_img.GetSpacing()[::-1])  # (z,y,x)

        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        modalities.append(arr)

    # Read segmentation
    seg_path = os.path.join(case_dir, f"{case_name}-seg.nii.gz")
    has_seg = os.path.exists(seg_path)
    if has_seg:
        seg_sitk = sitk.ReadImage(seg_path)
        seg = sitk.GetArrayFromImage(seg_sitk).astype(np.int32)
    else:
        seg = np.zeros_like(modalities[0], dtype=np.int32)

    original_shape = list(modalities[0].shape)

    # Resample to target spacing
    resampled_modalities = []
    for mod in modalities:
        resampled = resample_image(mod, original_spacing, TARGET_SPACING, is_label=False)
        resampled_modalities.append(resampled)

    if has_seg:
        seg = resample_image(seg, original_spacing, TARGET_SPACING, is_label=True)

    # Stack: (4, D, H, W)
    data = np.stack(resampled_modalities, axis=0).astype(np.float32)

    # Normalize each modality
    for i in range(4):
        data[i] = normalize_modality(data[i])

    # Crop to non-zero region
    data, seg, bbox, resampled_shape = crop_to_nonzero(data, seg)

    # Compute class locations for foreground oversampling
    class_locations = compute_class_locations(seg) if has_seg else {}

    # Prepare seg output: (1, D, H, W)
    seg_out = seg[None].astype(np.int32)

    # Save properties
    properties = {
        'name': [case_name],
        'spacing': original_spacing,
        'original_shape': original_shape,
        'resampled_shape': list(resampled_shape),
        'crop_bbox': bbox,
        'class_locations': class_locations,
    }

    # Save .npz
    save_path_npz = os.path.join(OUTPUT_DIR, f"{case_name}.npz")
    np.savez_compressed(save_path_npz, data=data, seg=seg_out)

    # Save .pkl
    save_path_pkl = os.path.join(OUTPUT_DIR, f"{case_name}.pkl")
    with open(save_path_pkl, 'wb') as f:
        pickle.dump(properties, f)

    return case_name


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    # Get all case directories
    case_dirs = sorted([
        d for d in os.listdir(RAW_TRAIN_DIR)
        if os.path.isdir(os.path.join(RAW_TRAIN_DIR, d)) and d.startswith("BraTS-GLI")
    ])

    print(f"Found {len(case_dirs)} training cases")

    # Process all cases
    processed_names = []
    for case_name in tqdm(case_dirs, desc="Preprocessing training data"):
        case_dir = os.path.join(RAW_TRAIN_DIR, case_name)
        try:
            name = process_case(case_dir, case_name)
            processed_names.append(name)
        except Exception as e:
            print(f"Error processing {case_name}: {e}")

    print(f"Successfully preprocessed {len(processed_names)} cases")

    # Create train/val split
    random.seed(RANDOM_SEED)
    random.shuffle(processed_names)
    val_count = int(len(processed_names) * VAL_SPLIT)
    val_names = processed_names[:val_count]

    # Generate test_list_brats2025.py
    test_list_content = f"""# Auto-generated test list for BraTS 2025
# {len(val_names)} validation cases (20% of {len(processed_names)} training cases)
# Random seed: {RANDOM_SEED}

test_list = {val_names}
"""
    with open("test_list_brats2025.py", "w") as f:
        f.write(test_list_content)

    print(f"Created test_list_brats2025.py with {len(val_names)} validation cases")
    print(f"Preprocessing complete! Data saved to {OUTPUT_DIR}")