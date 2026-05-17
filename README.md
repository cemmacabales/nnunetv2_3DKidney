# nnU-Net V2 for 3D Kidney Abnormality Segmentation

This repository contains the training pipeline, evaluation notebooks, and experimental results for multi-class 3D semantic segmentation of renal abnormalities using **nnU-Net v2**.

The project compares **single-class** vs. **multi-class** 3D U-Net training configurations to segment four classes from abdominal CT volumes:
- **Kidney** (anatomical anchor)
- **Cyst**
- **Stone**
- **Tumor**

---

## Repository Structure

```
.
├── FINAL_MULTICLASS.ipynb          # Main Colab training notebook (multi-class)
├── HD95_Evaluation.ipynb           # HD95 / diameter / density evaluation notebook
├── nnUNetPlans.json                # Preprocessing plans (3d_fullres, batch_size=12)
├── PROJECT_LOG.md                  # Detailed project log and troubleshooting notes
├── RESULTS_AND_DISCUSSION_FINAL.md # Full thesis results chapter with metrics & analysis
├── RESULTS_AND_DISCUSSION_v2.md    # Earlier results draft
├── RESULTS_AND_DISCUSSION.md       # Original results draft
├── hd95_results/
│   └── generate_summary.py         # Summarizes CSV evaluation results into a Markdown report
├── Summary/                        # Aggregated result summaries
├── summaryjsons/                   # JSON evaluation outputs
└── THESIS_*.pdf / .docx            # Thesis documents
```

---

## Dataset

- **Name:** `Dataset500_KidneyAbnormalities`
- **Size:** 290 CT cases
- **Modality:** Contrast & non-contrast abdominal CT
- **Classes:**
  - `0` — Background
  - `1` — Kidney
  - `2` — Cyst
  - `3` — Stone
  - `4` — Tumor
- **Source:** Multi-institutional public datasets (KiTS23, MSWAL, and supplementary cases)

---

## Training

### Environment
- **Platform:** Google Colab Pro+ (NVIDIA A100 80 GB)
- **Framework:** nnU-Net v2
- **Configuration:** `3d_fullres` (3D full resolution)
- **Patch Size:** 128 × 128 × 128
- **Batch Size:** 12 (tuned for A100 80 GB)

### Configurations Trained

| Config | Description | Classes |
|--------|-------------|---------|
| **A** | Single-class — Stones only | 1 |
| **B** | Single-class — Cysts only | 1 |
| **C** | Single-class — Tumors only | 1 |
| **D** | **Multi-class** — All four classes | 4 |

### Quick Start (Colab)

1. Upload your `Dataset500_KidneyAbnormalities` folder to Google Drive.
2. Open `FINAL_MULTICLASS.ipynb` in Google Colab.
3. Update `DATASET_DRIVE_PATH` in Cell 4.
4. Run all cells sequentially.
5. If disconnected during training, use the **resume cell** (`--c` flag).
6. After all 5 folds complete, run the export cell to package the model.

> **Tip:** Preprocessing is the most fragile step in Colab. The notebook includes a resilience cell that backs up `nnUNetPlans.json` and preprocessed metadata to Drive so you can resume without re-running everything.

---

## Evaluation

### Overlap & Classification Metrics
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Precision, Recall, F1-Score
- Specificity

### Boundary Metrics
- **95th Percentile Hausdorff Distance (HD95)** — computed via `HD95_Evaluation.ipynb`
- **Equivalent Spherical Diameter (ESD)**
- **Mean Hounsfield Unit (HU) Density**

The `HD95_Evaluation.ipynb` notebook is self-contained per dataset and can be run independently for each configuration.

### Summarizing Results

After running HD95 evaluation, use the helper script to compile statistics:

```bash
cd hd95_results
python generate_summary.py
```

This produces `compiled_summary.md` with per-label, per-fold statistics for HD95, diameter, and density.

---

## Key Results (Configuration D — Multi-Class)

| Class | Dice | Precision | Recall | F1 |
|-------|------|-----------|--------|----|
| **Kidney** | 0.955 | 0.956 | 0.959 | 0.957 |
| **Cyst** | 0.501 | 0.713 | 0.703 | 0.703 |
| **Stones** | 0.512 | 0.839 | 0.628 | 0.678 |
| **Tumor** | 0.763 | 0.858 | 0.886 | 0.871 |

### Highlights
- **Multi-class training outperformed single-class** across every abnormality class.
- **Kidney segmentation** reached near-expert performance (Dice ~0.955), acting as a stable anatomical anchor.
- **Stone detection** improved from **catastrophic failure** (Dice 0.000 in single-class) to **functional screening** (Dice 0.512 in multi-class).
- **Cyst precision** improved by **+166%** relative to the single-class model.
- **Inference latency:** One multi-class pass (~4.8 s/vol) vs. three sequential single-class passes (~12.6 s/vol) — **61.9% faster** for comprehensive characterization.

For full statistical analysis, training dynamics, and benchmark comparisons against MSWAL (Wu et al.), see `RESULTS_AND_DISCUSSION_FINAL.md`.

---

## Files Reference

| File | Purpose |
|------|---------|
| `FINAL_MULTICLASS.ipynb` | End-to-end Colab notebook: setup → preprocessing → training → 5-fold CV evaluation → inference profiling → model export |
| `HD95_Evaluation.ipynb` | Computes HD95, ESD, and HU density from nnU-Net predictions. CPU-only, self-contained per dataset |
| `nnUNetPlans.json` | Exported plans file with tuned `batch_size=12` for 3d_fullres on A100 |
| `PROJECT_LOG.md` | Operational log: preprocessing troubleshooting, batch size tuning, ResEnc discussion, training resumption notes |
| `RESULTS_AND_DISCUSSION_FINAL.md` | Complete thesis results chapter (RQ1–RQ3, Phase 1 & 2, head-to-head comparison, clinical deployment analysis) |
| `hd95_results/generate_summary.py` | Post-processing script to aggregate per-case CSVs into summary tables |

---

## Citation / Reference

This work is part of a thesis investigating single-class vs. multi-class 3D segmentation paradigms for renal abnormalities using nnU-Net v2.

If you use this code or dataset configuration, please cite the relevant nnU-Net v2 publication:

> Isensee, F., et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods, 18(2), 203–211.

---

## License & Notes

- This repository is provided for **academic and research purposes**.
- The dataset contains publicly available CT annotations; please refer to the original sources (KiTS23, MSWAL) for their respective licenses.
- The `nnUNetPlans.json` included here is specific to the `Dataset500_KidneyAbnormalities` corpus and should not be assumed to generalize to other datasets without re-running nnU-Net's experiment planner.
