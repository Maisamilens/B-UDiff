# check_labels.py - run this once
import SimpleITK as sitk
import numpy as np
import os

raw_dir = r"D:\Brats 2025\Brats 2025 Glioma Pre Challange\glio_25\glio_train"
cases = sorted(os.listdir(raw_dir))[:5]  # check first 5

for case in cases:
    seg_path = os.path.join(raw_dir, case, f"{case}-seg.nii.gz")
    if os.path.exists(seg_path):
        seg = sitk.GetArrayFromImage(sitk.ReadImage(seg_path))
        print(f"{case}: unique labels = {np.unique(seg)}, shape = {seg.shape}, spacing = {sitk.ReadImage(seg_path).GetSpacing()}")
    else:
        print(f"{case}: NO SEG FILE FOUND")