# B-UDiff: Boundary-Aware Uncertainty-Guided Diffusion Network with Progressive Refinement for 3D Glioma Segmentation

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-13.1-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Dataset](https://img.shields.io/badge/Dataset-BraTS%202025%20Glioma-orange.svg)](https://www.synapse.org/)

Official implementation of **B-UDiff**, a boundary-aware, uncertainty-guided conditional diffusion framework for 3D glioma sub-region segmentation from multi-parametric MRI. B-UDiff is built on top of the [Diff-UNet](https://github.com/ge-xing/DiffUNet) diffusion-embedded segmentation backbone, extended with a dedicated boundary-prediction branch, a Multi-granularity Boundary Aggregation (MBA) module, a Monte Carlo Diffusion (MC-Diff) uncertainty module, and a Progressive Uncertainty-driven REfinement (PURE) inference strategy.

> **B-UDiff achieves 92.85% / 88.04% / 83.26% Dice** on WT / TC / ET, an average Dice of **88.05%**, and an average HD95 of **4.60 mm** on a 250-case internal validation split of the **BraTS 2025 Glioma** dataset.

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Dataset Preparation (BraTS 2025 Glioma)](#dataset-preparation-brats-2025-glioma)
- [Usage / Full Pipeline](#usage--full-pipeline)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Analysis & Visualization Scripts](#analysis--visualization-scripts)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)
- [Publishing This Repo to GitHub](#publishing-this-repo-to-github)

---

## Highlights

- **Two-branch architecture**: a boundary-prediction branch ($E_b$/$D_b$) and a diffusion-denoising branch ($E_d$/$D_d$), trained and queried jointly.
- **Multi-granularity Boundary Aggregation (MBA)**: fuses boundary features into the denoising pathway at both low-level (concatenation) and high-level (affinity-based, STCN-inspired) feature scales.
- **Monte Carlo Diffusion (MC-Diff)**: derives voxel-wise predictive uncertainty organically from the stochasticity of the diffusion forward process — no dropout or ensembling required.
- **Uncertainty-guided training loss**: reweights the segmentation loss toward hard, boundary-adjacent, and infiltrative voxels via a time-scheduled balance weight $\eta$.
- **Progressive Uncertainty-driven REfinement (PURE)**: fuses predictions from *every* reverse-diffusion step at inference, weighted by stage-specific uncertainty, instead of using only the last step.
- Evaluated on **1,251 training / 219 validation** BraTS 2025 Glioma cases (fold-0, 5-fold split: 1,001 train / 250 held-out validation).

## Architecture

```
                       ┌──────────────────────────┐
   MRI (I) ──────────► │  Boundary Branch (E_b/D_b)│──► Boundary map B̂₀
                       └────────────┬─────────────┘
                                    │  MBA (low-level concat + high-level affinity)
                                    ▼
   Noisy label (Y_t) ─► ┌──────────────────────────┐
                       │ Denoising Branch (E_d/D_d)│──► Segmentation Ŷ_t
                       └──────────────────────────┘
                                    │
                          MC-Diff (entropy over N_s stochastic passes)
                                    │
                        Uncertainty-guided loss (training) /
                        PURE fusion across all steps (inference)
```

See `Fig. 1` of the paper (`arch.png`) for the full training/inference diagram.

## Repository Structure

```
B-UDiff/
├── budiff/                              # Core model package (network, diffusion, MBA, MC-Diff, PURE)
├── B-UDiff/                             # (Model configs / checkpoints subfolder)
├── light_training/                      # Lightweight training engine / trainer utilities
├── data/                                # Raw + preprocessed BraTS 2025 Glioma data (see below)
├── logs/                                # TensorBoard logs + saved checkpoints (best_model_*.pt)
├── prediction_results/                  # Raw inference outputs (.npy masks / .npz probability volumes)
├── publication_prediction_visuals/      # Publication-ready figures (montage, multi-slice overlays)
├── quantitative_results/                # SOTA comparison + metrics tables (CSV/JSON)
├── statistical_analysis_outputs/        # Wilcoxon tests, bootstrap CIs, Shapiro-Wilk outputs
├── visualization_outputs/               # Training curves, loss/LR plots, volume & uncertainty distributions
├── ablation_results/                    # Full ablation study outputs (25-case set)
├── ablation_results_smoke/              # Quick smoke-test ablation run (small subset)
├── interpretability_outputs/            # Grad-CAM / interpretability maps
├── imgs/                                # Static images used in README / paper
│
├── 1_preprocessing_brats2025.py         # Step 1: normalize + resample BraTS 2025 raw data
├── 1b_preprocessing_brats2025.py        # Step 1b: alternate/extended preprocessing (e.g. boundary maps)
├── 1_rename_mri_data_brats2023.py       # (legacy) BraTS 2023 renaming utility
├── 2_preprocessing_brats2023.py         # (legacy) BraTS 2023 preprocessing
├── 2_train_budiff_brats2025.py          # Step 2: train B-UDiff on BraTS 2025 Glioma
├── 3_train_budiff_brats2025.py          # Step 3: continued/fine-tune training run on BraTS 2025
├── 3_train_budiff_brats2023.py          # (legacy) B-UDiff training on BraTS 2023
├── 4_predict_brats2025.py               # Step 4: run inference (train/val split) on BraTS 2025
├── 4b_predict_brats2025_val.py          # Step 4b: inference restricted to the internal validation split
├── 4_predict_brats2023.py               # (legacy) BraTS 2023 inference
├── 5_compute_metrics_brats2025.py       # Step 5: compute Dice / HD95 metrics on BraTS 2025 predictions
├── 5_compute_metrics_brats2023.py       # (legacy) BraTS 2023 metrics
├── test_list_brats2025.py               # Generates the held-out validation case list (BraTS 2025)
├── test_list_brats2023.py               # (legacy) BraTS 2023 case list
├── check_labels.py                      # Sanity-check label maps (class ids, shapes)
├── debug_model_return.py                # Debug utility: inspect raw model outputs
│
├── save_budiff_predictions_brats2025.py # Saves .npy masks + .npz probability volumes for analysis
├── quantitative_brats2025_from_model.py # Recomputes Table (SOTA comparison) directly from checkpoint
├── quantitative_ablation_brats2025.py   # Runs the 7-variant ablation (MBA/PURE/boundary-only/etc.)
├── quantitative_brats_table.py          # Formats final LaTeX/Markdown result tables
├── statistical_analysis_brats2025.py    # Wilcoxon signed-rank tests, bootstrap CIs, Pearson correlation
├── plot_ablation_results.py             # Generates ablation Dice/HD95 bar charts (Fig. 6, Fig. 7)
├── prediction_atlas.py                  # Builds the tumour-burden / volume-distribution atlas (Fig. 3)
├── prediction_visuals.py                # General prediction visualization utilities
├── qualitative_brats2025_figure.py      # Builds the 8-case montage (Fig. 8)
├── qualitative_brats2025_ablation_figure.py # Qualitative comparison across ablation variants
├── generate_gradcam_brats2025.py        # Grad-CAM interpretability maps
├── generate_b_udiff_bundle.py           # Packages model + configs into a distributable bundle
├── visualizations.py                    # Training curve / loss-LR plots
└── README.md                            # (this file)
```

> **Note:** filenames prefixed `1/2/3/4/5` follow the same numbered-pipeline convention as the original [Diff-UNet](https://github.com/ge-xing/DiffUNet) repo; files suffixed `brats2023` are kept only for backward compatibility/reference and are not required for BraTS 2025 runs.

## Environment Setup

Paper-reported training configuration (Table: *Implementation Details*):

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 5080 (single GPU) |
| Framework | PyTorch 2.4.1+cu131 |
| CUDA | 13.1 |
| Environment manager | Conda |

```bash
# 1. Create and activate the environment
conda create -n budiff python=3.10 -y
conda activate budiff

# 2. Install PyTorch (CUDA 13.1 build)
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu131

# 3. Install remaining dependencies
pip install monai dynamic_network_architectures batchgenerators \
            nibabel SimpleITK numpy scipy scikit-image scikit-learn \
            tensorboard matplotlib seaborn pandas tqdm einops

# 4. Clone the repo
git clone https://github.com/<your-username>/B-UDiff.git
cd B-UDiff
```

## Dataset Preparation (BraTS 2025 Glioma)

1. Download the **BraTS 2025 Glioma** training + validation data (four co-registered, skull-stripped MRI modalities: T1, T1ce, T2, FLAIR, resampled to 1 mm³ isotropic resolution).
2. Place the raw data under:
   ```
   ./data/raw_data/BraTS2025-GLI-Challenge-TrainingData/
   ```
3. Run preprocessing:
   ```bash
   python 1_preprocessing_brats2025.py
   python 1b_preprocessing_brats2025.py   # extended preprocessing / boundary-map generation
   ```
   Preprocessed (normalized + resampled) volumes are written to:
   ```
   ./data/fullres/train/
   ```
4. Generate the fold-0 validation case list (1,001 train / 250 held-out):
   ```bash
   python test_list_brats2025.py
   ```
5. (Optional) Verify label integrity:
   ```bash
   python check_labels.py
   ```

Label convention: `0` = background, `1` = NCR/NET, `2` = peritumoral oedema, `3` = enhancing tumour (ET). Evaluation regions are derived as WT = {1,2,3}, TC = {1,3}, ET = {3}.

## Usage / Full Pipeline

```bash
# Step 1 — Preprocess raw BraTS 2025 volumes
python 1_preprocessing_brats2025.py
python 1b_preprocessing_brats2025.py

# Step 2 — Train B-UDiff (main training run)
python 2_train_budiff_brats2025.py

# Step 3 — (optional) continued training / additional epochs
python 3_train_budiff_brats2025.py

# Monitor training
tensorboard --logdir ./logs/

# Step 4 — Run inference with the trained checkpoint
python 4_predict_brats2025.py            # full split
python 4b_predict_brats2025_val.py       # internal validation split only (250 cases)

# Step 5 — Compute Dice / HD95 metrics
python 5_compute_metrics_brats2025.py --pred=budiff

# Save raw prediction masks + probability volumes (for uncertainty/statistical analysis)
python save_budiff_predictions_brats2025.py
```

### Reproducing paper tables & figures

```bash
# SOTA comparison table (Table: Quantitative Comparison)
python quantitative_brats2025_from_model.py
python quantitative_brats_table.py

# Ablation study (7 variants: Full, w/o MBA, w/o PURE, 1-step DDIM,
# w/o boundary logits, Denoise only, Boundary only)
python quantitative_ablation_brats2025.py
python plot_ablation_results.py

# Statistical significance (Wilcoxon signed-rank, bootstrap CI, Shapiro-Wilk)
python statistical_analysis_brats2025.py

# Training dynamics plots (Fig. 1 validation Dice curve, Fig. 2 loss/LR)
python visualizations.py

# Tumour-volume distribution & uncertainty/confidence histograms (Fig. 3, Fig. 4)
python prediction_atlas.py

# Qualitative montage & multi-slice overlays (Fig. 8, case BraTS-GLI-00018-000)
python qualitative_brats2025_figure.py
python qualitative_brats2025_ablation_figure.py

# Grad-CAM interpretability maps
python generate_gradcam_brats2025.py

# Package a distributable model bundle
python generate_b_udiff_bundle.py
```

Key training hyperparameters (defaults in `2_train_budiff_brats2025.py`):

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD (Nesterov momentum 0.99) |
| Initial LR | 1e-2 (polynomial decay) |
| Weight decay | 3e-5 |
| Patch size | 128 × 128 × 128 |
| Batch size | 1 |
| Epochs | 500 |
| Diffusion forward steps (T) | 1,000 |
| MC-Diff stochastic passes (N_s) | 5 |
| Loss weights (λ₁, λ₂) | 1.0, 1.0 |
| Mixed precision | Enabled |

## Results

BraTS 2025 Glioma — 250-case internal validation split (fold 0):

| Region | Dice (%) ↑ | HD95 (mm) ↓ |
|---|---|---|
| Whole Tumour (WT) | 92.85 ± 8.50 | 3.53 ± 4.55 |
| Tumour Core (TC) | 88.04 ± 18.09 | 4.97 ± 7.82 |
| Enhancing Tumour (ET) | 83.26 ± 20.96 | 5.30 ± 10.09 |
| **Average** | **88.05 ± 13.22** | **4.60 ± 6.24** |

Best logged mean validation Dice: **0.9027** at epoch 349 (`best_model_0.9027.pt`, saved under `./logs/`).

Prediction statistics (250 cases): mean max-class confidence 0.9951 ± 0.0029; mean normalized entropy 0.0101 ± 0.0055; tumour-voxel entropy 0.2086 ± 0.0975 — confirming uncertainty concentrates at boundary-adjacent/infiltrative regions.

## Ablation Study

25-case internal validation subset (see `ablation_results/`):

| Variant | Avg Dice (%) | Avg HD95 (mm) | p-value (vs. Full) |
|---|---|---|---|
| **Full (B-UDiff)** | **87.79** | 4.25 | — |
| w/o MBA | 87.53 | 4.41 | 0.039 |
| w/o PURE | 87.68 | 4.62 | 0.071 |
| 1-step DDIM | 87.74 | 4.14 | 0.456 |
| Boundary only | 87.05 | 4.05 | 0.0045 |
| w/o boundary logits | 8.91 | 101.66 | 1.03×10⁻²¹ |
| Denoise only | 5.43 | 102.46 | 4.08×10⁻²¹ |

The boundary-prediction branch is architecturally essential — removing it collapses performance entirely.

## Analysis & Visualization Scripts

| Script | Output |
|---|---|
| `visualizations.py` | Training/validation Dice curve, loss & LR schedule |
| `prediction_atlas.py` | Tumour volume violin plots, entropy/confidence histograms |
| `qualitative_brats2025_figure.py` | 8-case prediction/uncertainty/confidence montage |
| `qualitative_brats2025_ablation_figure.py` | Ablation qualitative comparisons |
| `plot_ablation_results.py` | Ablation Dice & HD95 bar charts |
| `statistical_analysis_brats2025.py` | Wilcoxon tests, bootstrap CIs, correlation analysis |
| `generate_gradcam_brats2025.py` | Grad-CAM interpretability maps |
| `prediction_visuals.py` | Miscellaneous prediction overlays |

Outputs are written to `visualization_outputs/`, `publication_prediction_visuals/`, `statistical_analysis_outputs/`, `interpretability_outputs/`, and `quantitative_results/` respectively.

## Citation

If you use this code or the B-UDiff model in your research, please cite:

```bibtex
@article{budiff2026,
  title   = {Boundary-Aware Uncertainty-Guided Diffusion Network with Progressive Refinement for 3D Glioma Segmentation},
  author  = {<Maisam Abbas, Ran-Zan Wang>},
  journal = {<YTBD>},
  year    = {2026}
}
```

## Acknowledgements

This work builds on the diffusion-embedded segmentation backbone from **Diff-UNet**:

```bibtex
@misc{diffunet,
  title  = {Diff-UNet: A Diffusion Embedded Network for Robust 3D Medical Image Segmentation},
  author = {Xing, Zhaohu and Ge-Xing},
  howpublished = {\url{https://github.com/ge-xing/DiffUNet}},
  year   = {2023}
}
```

We also thank the maintainers of [MONAI](https://github.com/Project-MONAI/MONAI) and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), whose components were used in preprocessing and training infrastructure.

## License

This project is released under the [Apache 2.0 License](LICENSE), consistent with the base Diff-UNet repository.

---
