# 5_compute_metrics_brats2025.py
import argparse
import numpy as np
import os
import glob
import SimpleITK as sitk
from light_training.evaluation.metric import dice as dice_metric

# ============================================================
# CONFIGURATION
# ============================================================
RAW_TRAIN_DIR = r"D:\Brats 2025\Brats 2025 Glioma Pre Challange\glio_25\glio_train"
PRED_DIR = "./prediction_results/diffunet"
# ============================================================

def compute_metrics(pred_dir, raw_dir):
    from test_list_brats2025 import test_list

    wt_dices, tc_dices, et_dices = [], [], []

    for case_name in test_list:
        # Load ground truth
        gt_path = os.path.join(raw_dir, case_name, f"{case_name}-seg.nii.gz")
        if not os.path.exists(gt_path):
            continue
        gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))

        # Compute WT, TC, ET from ground truth
        gt_wt = (gt > 0).astype(np.uint8)
        gt_tc = ((gt == 1) | (gt == 3)).astype(np.uint8)
        gt_et = (gt == 3).astype(np.uint8)

        # Load predictions
        pred_wt_path = os.path.join(pred_dir, f"{case_name}-WT.nii.gz")
        pred_tc_path = os.path.join(pred_dir, f"{case_name}-TC.nii.gz")
        pred_et_path = os.path.join(pred_dir, f"{case_name}-ET.nii.gz")

        if not all(os.path.exists(p) for p in [pred_wt_path, pred_tc_path, pred_et_path]):
            print(f"Missing predictions for {case_name}")
            continue

        pred_wt = sitk.GetArrayFromImage(sitk.ReadImage(pred_wt_path))
        pred_tc = sitk.GetArrayFromImage(sitk.ReadImage(pred_tc_path))
        pred_et = sitk.GetArrayFromImage(sitk.ReadImage(pred_et_path))

        # Binarize predictions
        pred_wt = (pred_wt > 0.5).astype(np.uint8)
        pred_tc = (pred_tc > 0.5).astype(np.uint8)
        pred_et = (pred_et > 0.5).astype(np.uint8)

        # Compute dice
        for name, pred, gt_m in [("WT", pred_wt, gt_wt), ("TC", pred_tc, gt_tc), ("ET", pred_et, gt_et)]:
            if pred.sum() > 0 and gt_m.sum() > 0:
                d = dice_metric(pred, gt_m)
            elif gt_m.sum() == 0 and pred.sum() == 0:
                d = 1.0
            else:
                d = 0.0

            if name == "WT": wt_dices.append(d)
            elif name == "TC": tc_dices.append(d)
            else: et_dices.append(d)

    print(f"\n=== Metrics ===")
    print(f"WT Dice: {np.mean(wt_dices):.4f} ± {np.std(wt_dices):.4f}")
    print(f"TC Dice: {np.mean(tc_dices):.4f} ± {np.std(tc_dices):.4f}")
    print(f"ET Dice: {np.mean(et_dices):.4f} ± {np.std(et_dices):.4f}")
    mean_all = (np.mean(wt_dices) + np.mean(tc_dices) + np.mean(et_dices)) / 3
    print(f"Mean Dice: {mean_all:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, default="diffunet")
    args = parser.parse_args()

    pred_dir = os.path.join("./prediction_results", args.pred)
    compute_metrics(pred_dir, RAW_TRAIN_DIR)