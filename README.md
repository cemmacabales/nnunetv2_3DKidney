# nnU-Net V2 for Multi-Class 3D Renal Abnormality Segmentation

This repository contains the complete training pipeline, evaluation notebooks, and empirical results for a thesis investigating **single-class versus multi-class 3D semantic segmentation paradigms** for renal abnormalities using **nnU-Net V2**.

The study addresses four identified research gaps in existing renal imaging AI: (1) reliance on coarse bounding-box detection rather than voxel-level segmentation, (2) morphology-dependent performance degradation in existing segmentation systems, (3) single-institution and single-class modeling limitations that ignore co-morbid presentations, and (4) absence of any controlled comparative evidence between single-class and multi-class paradigms under identical conditions.

Three research questions guide the work:
- **RQ1:** Do single-class nnU-Net V2 models (trained in isolation for stones, cysts, or tumors) produce per-class performance consistent with published benchmarks?
- **RQ2:** Does a unified multi-class nnU-Net V2 model achieve superior, equivalent, or inferior per-class segmentation compared to the validated single-class models?
- **RQ3:** What are the computational efficiency trade-offs (inference latency, GPU VRAM) between three sequential single-class passes versus one unified multi-class pass, and how do these affect clinical deployment feasibility?

Segmentation targets four classes from abdominal CT volumes:
- **Kidney** — anatomical anchor and spatial reference
- **Cyst** — including Bosniak I through IV classifications
- **Stone** — calcium oxalate, uric acid, and mixed composition calculi
- **Tumor** — renal cell carcinoma and benign neoplasms

---

## Repository Structure

```
.
├── nnUNetPlans.json                # Preprocessing plans (3d_fullres, batch_size=12)
├── PROJECT_LOG.md                  # Detailed project log and troubleshooting notes
├── RESULTS_AND_DISCUSSION_FINAL.md # Full thesis results chapter with metrics & analysis
├── RESULTS_AND_DISCUSSION_v2.md    # Earlier results draft
├── RESULTS_AND_DISCUSSION.md       # Original results draft
├── Notebooks/                      # Training notebooks (single-class & multi-class)
├── figures/                        # Result figures and visualizations
├── results/                        # Evaluation outputs
│   ├── hd95/                       # HD95 / diameter / density evaluations
│   ├── summaries/                  # Aggregated result summaries
│   └── summaryjsons/               # JSON evaluation outputs per fold
└── thesis/                         # Thesis documents (all versions)
```

---

## Dataset

- **Name:** `Dataset500_KidneyAbnormalities`
- **Size:** 290 CT cases (74 pure nephrolithiasis, 50 pure cyst, 45 pure solid neoplasm, 121 concurrent multi-pathology)
- **Modality:** Contrast and non-contrast abdominal CT
- **Classes:**
  - `0` — Background
  - `1` — Kidney
  - `2` — Cyst
  - `3` — Stone
  - `4` — Tumor
- **Source:** Multi-institutional public repositories:
  - **MSWAL** (Wu et al.) — primary source of multi-label annotated cases covering kidney stones, tumors, and cysts across 694 patients
  - **KiTS23** — kidney tumor segmentation challenge data
  - **Supplementary cases** — additional pure-pathology stone and cyst cases to balance class representation
- **Preprocessing:** HU clipping −135 to 215, isotropic resampling to 1.0 mm, Z-score normalization (global foreground mean/std)
- **Validation:** 5-fold out-of-fold cross-validation (80/20 split per fold); every case serves as validation exactly once

---

## Training

### Environment
- **Platform:** Google Colab Pro+ (NVIDIA A100 40 GB)
- **Framework:** nnU-Net V2 (default 3D U-Net, 6 encoding stages: 32 → 64 → 128 → 256 → 320 → 320)
- **Configuration:** `3d_fullres` (3D full resolution)
- **Patch Size:** 128 × 128 × 128
- **Batch Size:** 2 (tuned for A100 40 GB; default was 12)
- **Loss:** Soft Dice + Cross-Entropy
- **Optimizer:** SGD with Nesterov Momentum
- **Schedule:** 1,000 epochs (250 iterations/epoch)

### Configurations Trained

All four configurations were trained on the same 290-case corpus with identical preprocessing and evaluation protocols:

| Config | Description | Classes | Training Outcome |
|--------|-------------|---------|------------------|
| **A** | Single-class — Stones only | 1 | Catastrophic forgetting (Dice 0.000 across all 5 folds) |
| **B** | Single-class — Cysts only | 1 | Moderate recall, critically low precision (Dice 0.222) |
| **C** | Single-class — Tumors only | 1 | Best single-class performance (Dice 0.613) |
| **D** | **Multi-class** — All four classes | 4 | **Sustained detection across all classes; superior per-class performance** |

### Quick Start (Colab)

1. Upload your `Dataset500_KidneyAbnormalities` folder to Google Drive.
2. Open the desired notebook from the `Notebooks/` folder in Google Colab (e.g., `MULTICLASS.ipynb`).
3. Update `DATASET_DRIVE_PATH` in Cell 2.
4. Run all cells sequentially.
5. If disconnected during training, use the **resume cell** (`--c` flag).
6. After all 5 folds complete, run the evaluation and export cells.

> **Tip:** Preprocessing is the most fragile step in Colab due to memory fragmentation from multiprocessing. Use `-np 2` workers for stability. The notebook includes a resilience cell that backs up `nnUNetPlans.json` and preprocessed metadata to Drive so you can resume without re-running everything.

---

## Evaluation

### Overlap & Classification Metrics
Computed from aggregated confusion matrices across all validation cases per fold:
- **Dice Similarity Coefficient (DSC)** — spatial overlap; gold-standard for medical segmentation
- **Intersection over Union (IoU)** — stricter overlap metric
- **Precision, Recall, F1-Score** — per-class detection reliability
- **Specificity** — background misclassification rate

### Boundary & Morphology Metrics
- **95th Percentile Hausdorff Distance (HD95)** — worst-case surface deviation; critical for surgical planning and tumor margin assessment
- **Equivalent Spherical Diameter (ESD)** — lesion size quantification
- **Mean Hounsfield Unit (HU) Density** — tissue characterization

Computed via the HD95 evaluation pipeline in `results/hd95/`, which is self-contained per configuration.

### Summarizing Results

After running HD95 evaluation, use the helper script to compile statistics:

```bash
cd results/hd95
python generate_summary.py
```

This produces `compiled_summary.md` with per-label, per-fold statistics.

---

## Key Results (Configuration D — Multi-Class, 5-Fold CV Mean ± SD)

| Class | Dice | Precision | Recall | F1 |
|-------|------|-----------|--------|----|
| **Kidney** | 0.955 ± 0.005 | 0.956 ± 0.007 | 0.959 ± 0.005 | 0.957 ± 0.003 |
| **Cyst** | 0.501 ± 0.054 | 0.713 ± 0.070 | 0.703 ± 0.102 | 0.703 ± 0.071 |
| **Stones** | 0.512 ± 0.129 | 0.839 ± 0.089 | 0.628 ± 0.245 | 0.678 ± 0.189 |
| **Tumor** | 0.763 ± 0.050 | 0.858 ± 0.073 | 0.886 ± 0.018 | 0.871 ± 0.045 |

### Head-to-Head vs. Single-Class (RQ2)

| Class | Single-Class Dice | Multi-Class Dice | Relative Improvement |
|-------|-------------------|------------------|----------------------|
| **Cyst** | 0.222 (Config B) | **0.501** | **+125.7%** |
| **Stones** | 0.000 (Config A) | **0.512** | **From catastrophic failure to sustained detection** |
| **Tumor** | 0.613 (Config C) | **0.763** | **+24.5%** |

### Benchmark vs. MSWAL (Wu et al., nnU-Net V2 on 694 patients)

| Class | MSWAL Baseline | This Study (Multi-Class) | Relative Change |
|-------|----------------|--------------------------|-----------------|
| **Cyst** | 0.409 | **0.501** | **+22.5%** |
| **Stones** | 0.231 | **0.512** | **+121.6%** |
| **Tumor** | 0.405 | **0.763** | **+88.4%** |

### Key Findings
- **Anatomical Anchor Hypothesis:** The kidney class (0.59% voxel prevalence, present in 100% of cases) acts as a stable spatial regularizer. By constraining pathological predictions to renal anatomy, the multi-class model reduced cyst false-positive rate by **83.7%** and tumor false-positive rate by **62.2%** relative to single-class counterparts.
- **Catastrophic forgetting in single-class stone training:** Configuration A transiently learned stone features (pseudo-Dice up to 0.542 at epoch 26) but collapsed irreversibly to all-background predictions by epoch 40 due to extreme class imbalance (stones occupy ~1 in 200,000 voxels). The multi-class model avoided this via shared encoder stabilization from the kidney anchor.
- **Clinical deployment feasibility (RQ3):** One multi-class pass (~4.8 s/vol, ~8.9 GB VRAM) replaces three sequential single-class passes (~12.6 s/vol total, ~8.1 GB × 3) — a **61.9% latency reduction** for comprehensive per-volume characterization with comparable memory requirements.
- **Tumor recall stability:** Multi-class training reduced tumor recall variance by **61%** (SD 0.046 → 0.018), indicating the kidney anchor regularizes predictions across diverse fold compositions.

For full statistical analysis, training dynamics, per-fold breakdowns, and clinical deployment profiles, see `RESULTS_AND_DISCUSSION_FINAL.md`.

---

## Files Reference

| File / Directory | Purpose |
|------------------|---------|
| `Notebooks/MULTICLASS.ipynb` | End-to-end Colab notebook: environment setup → preprocessing (with resilience cell) → training (Folds 0–4) → 5-fold CV evaluation → inference profiling (CUDA timers + VRAM) → model export |
| `Notebooks/CYST.ipynb` | Single-class cyst training notebook |
| `Notebooks/STONES.ipynb` | Single-class stone training notebook |
| `Notebooks/TUMOR.ipynb` | Single-class tumor training notebook |
| `nnUNetPlans.json` | Exported plans file with tuned `batch_size=12` for `3d_fullres` on A100 80 GB |
| `PROJECT_LOG.md` | Operational log: preprocessing troubleshooting (`-np 2` recommendation), batch size tuning (2 → 12), ResEnc cost-benefit analysis, Colab resume protocol |
| `RESULTS_AND_DISCUSSION_FINAL.md` | Complete thesis results chapter: RQ1–RQ3 answers, Phase 1 & 2 metrics, training dynamics (catastrophic forgetting analysis), voxel prevalence analysis, head-to-head comparison, clinical deployment profiles |
| `RESULTS_AND_DISCUSSION_v2.md` | Earlier results draft |
| `RESULTS_AND_DISCUSSION.md` | Original results draft |
| `results/hd95/generate_summary.py` | Post-processing script to aggregate per-case CSVs into `compiled_summary.md` |
| `figures/` | Result figures, training curves, and case visualizations |
| `thesis/` | All thesis document versions (PDF & DOCX) |

---

## Citation / Reference

This repository supports a thesis that provides the first controlled comparison of single-class versus multi-class nnU-Net V2 training for renal abnormality segmentation. The primary contribution is empirical evidence that multi-class training with an anatomical anchor (kidney) resolves catastrophic forgetting in ultra-minority classes (stones) and substantially improves per-class performance across cysts and tumors, while reducing inference latency by 61.9% relative to sequential single-class deployment.

If you use this code, dataset configuration, or findings, please cite the nnU-Net V2 framework:

> Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods, 18(2), 203–211.

And the MSWAL benchmark dataset:

> Wu, J., et al. (2023). *MSWAL: A large-scale multi-class segmentation dataset for common abdominal lesions*. (Reference details as published).

---

## License & Notes

- This repository is provided for **academic and research purposes**.
- The dataset contains publicly available CT annotations; please refer to the original sources (KiTS23, MSWAL) for their respective licenses.
- The `nnUNetPlans.json` included here is specific to the `Dataset500_KidneyAbnormalities` corpus and should not be assumed to generalize to other datasets without re-running nnU-Net's experiment planner.
