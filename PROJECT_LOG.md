# nnU-Net V2 Kidney Abnormality Segmentation — Project Log

**Date:** 2026-04-26  
**Dataset:** Dataset500_KidneyAbnormalities (290 cases)  
**Config:** 3D Full Resolution (multi-class: Kidney, Cyst, Stone, Tumor)  
**Hardware:** Google Colab Pro+ (A100 80GB, High-RAM)

---

## 1. Original Notebook Review

### Initial Requirements Check

| Requirement | Status | Notes |
|-------------|--------|-------|
| 4 training configs (A-D) | ❌ Ignored | User decided to focus on multi-class (Config D) only |
| `checkpoint_final.pth` + `checkpoint_best.pth` (5 folds) | ✅ | Native nnU-Net behavior |
| `progress.png` per fold | ✅ | Native nnU-Net behavior |
| `summary.json` from 5-fold CV | ⚠️ Added | Added Section 7 with custom evaluation cell |
| Held-out test inference (60 volumes, 5-fold ensemble) | ❌ Skipped | User has no held-out test set with labels |
| CCA post-processing | ❌ Skipped | User explicitly deferred |
| Inference latency (CUDA events) | ✅ Added | Section 8 with `torch.cuda.Event` timers |
| Peak GPU VRAM (`max_memory_allocated`) | ✅ Added | Section 8 profiling cell |
| Per-class metrics (DSC, IoU, HD95, P, R, F1) | ✅ Added | Section 7.2 computes all 6 metrics |
| Env vars pointing to Drive | ✅ Verified | `DATASET_DRIVE_PATH` and `RESULTS_DRIVE_PATH` set correctly |
| Dataset folder structure validity | ✅ Valid | `dataset.json`, `imagesTr/`, `labelsTr/` verified |

---

## 2. Notebook Revisions Applied

### New Sections Added

#### Section 7: 5-Fold Cross-Validation Evaluation
- **Cell 7.1:** Generates validation predictions fold-by-fold using `nnUNetv2_predict`, merges outputs into `/content/cv_predictions/merged/`
- **Cell 7.2a:** Installs `medpy` and `scikit-learn`
- **Cell 7.2b:** Computes per-case and aggregate metrics (DSC, IoU, HD95, Precision, Recall, F1) per class (Kidney, Cyst, Stone, Tumor), saves `cv_summary.json`

#### Section 8: Custom Inference with Profiling
- **Cell 8.1:** Uses `nnUNetPredictor` Python API for 5-fold ensemble inference
- Records per-volume **latency** (seconds) via `torch.cuda.Event`
- Records **peak VRAM** (GB) via `torch.cuda.max_memory_allocated()`
- Saves predictions + `inference_profile.json`

#### Section 9 & 10: Renumbered
- Old Section 7 (Save to Drive) → **Section 9**
- Old Section 8 (Streamlit Export) → **Section 10**

### Preprocessing Resilience (Cell 4.1)

**Problem:** Colab runtime resets wipe `/content/nnUNet_preprocessed/`, breaking the inspect-plans cell and requiring full re-preprocessing.

**Solution added:**
1. **Validate Drive path** (writable test file)
2. **Fast backup** of `nnUNetPlans.json` specifically (~50 KB, seconds)
3. **Full folder backup** of `nnUNet_preprocessed/` (slow, optional)
4. **Display plans locally** after backup

**Inspect-plans cell updated:**
- Checks if local plans exist
- If missing, **restores from Drive** automatically
- Then displays `3d_fullres` configuration

---

## 3. Preprocessing Troubleshooting

### Problem: `RuntimeError: Some background worker is 6 feet under`

**Initial hypothesis:** RAM exhaustion (even with 50 GB High-RAM).

**Tests run:**
- `-np 2`: Reached 271/290 cases successfully (interrupted, not crashed)
- Default workers (~8-12): Crashed consistently at ~49-60 cases
- `-np 6`: Crashed at ~59 cases

**Diagnostics:**
```bash
!df -h              # 141 GB free disk → not disk space
!ps aux | grep nnunet  # No zombie processes
!du -sh /content/nnUNet_preprocessed/*  # 3.8 GB folder
```

**Root cause:** Memory fragmentation / copy-on-write overhead from Python multiprocessing with too many workers loading large 3D volumes simultaneously. Not a corrupt file (would crash at same case with any worker count).

**Resolution:**
- Use **`-np 2`** to finish preprocessing safely.
- nnU-Net skips already-processed files on re-run.
- **Recommendation for notebook:** Default to `-np 4` as a safe middle ground.

### Full Preprocessed Folder Restore from Drive

**Problem:** Restoring 3.8 GB from Drive takes 1-2+ hours, burning GPU units for file copying.

**Root cause:** Google Drive mounted in Colab has ~5-15 MB/s throughput. `shutil.copytree` is single-threaded.

**Better approach:**
- Only backup/restore **metadata files** (`nnUNetPlans.json`, `splits_final.json`, `dataset_fingerprint.json`) — ~100 KB total, restores in seconds.
- Preprocessed case data can be regenerated if needed, or just re-run preprocessing with `-np 2` for remaining cases.

---

## 4. Batch Size Optimization (GPU Utilization)

**Observation:** Training with default plans (`batch_size=2`) only used **8 GB / 80 GB** VRAM on A100.

**Goal:** Maximize GPU utilization without OOM.

### Batch Size History

| `batch_size` | Training VRAM | Validation Spike | Status |
|--------------|---------------|------------------|--------|
| 2 (default) | ~8 GB | ~12 GB | ✅ Too low |
| 4 | ~16 GB | ~24 GB | ✅ Conservative |
| 8 | ~28 GB | ~40 GB | ✅ Safe |
| **12 (current)** | **~40-45 GB** | **~55-65 GB** | **✅ Recommended** |
| 16 | ~55 GB | ~70-75 GB | ⚠️ Tight margin |
| 20 | ~68 GB | ~80+ GB | ❌ Likely OOM |

**Current setting:** `batch_size = 12` in `nnUNetPlans.json`

**How to change without re-uploading:**
1. Open `/content/nnUNet_preprocessed/Dataset500_KidneyAbnormalities/nnUNetPlans.json` in Colab file viewer
2. Find `"batch_size": 12` in the `3d_fullres` block
3. Change to desired value (e.g., 16)
4. Save (`Ctrl+S`)
5. Copy to results folder:
```python
import shutil
from pathlib import Path
src = Path("/content/nnUNet_preprocessed/Dataset500_KidneyAbnormalities/nnUNetPlans.json")
dst = Path("/content/nnUNet_results/Dataset500_KidneyAbnormalities/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json")
if dst.exists():
    shutil.copy2(src, dst)
```
6. Delete old fold checkpoint to force fresh training:
```python
import shutil
fold0 = Path("/content/nnUNet_results/Dataset500_KidneyAbnormalities/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0")
if fold0.exists():
    shutil.rmtree(fold0)
```

**Does changing batch_size require re-preprocessing?**
❌ **No.** Preprocessing resamples images to disk. Batch size only affects how many patches are grouped during training. It is a runtime parameter.

---

## 5. ResEnc (Residual Encoder) Discussion

**User question:** Would switching to ResEnc be faster / save compute units?

**Answer: No.**

### ResEnc vs Default U-Net

| Preset | VRAM | Training Time (per fold, A100) | Purpose |
|--------|------|-------------------------------|---------|
| Default U-Net | ~8 GB (varies by batch size) | ~15-20 hrs | Fast, standard |
| ResEnc M | ~9 GB | ~12 hrs | Small GPU |
| **ResEnc L** | **~23 GB** | **~35 hrs** | **Accuracy-focused** |
| ResEnc XL | ~37 GB | ~66 hrs | Maximum accuracy |

### Accuracy Gains (from official benchmarks)

| Dataset | Default | ResEnc L | Gain |
|---------|---------|----------|------|
| BTCV | 83.08 | 83.35 | +0.27% |
| ACDC | 91.54 | 91.69 | +0.15% |
| LiTS | 80.09 | 81.60 | +1.51% |
| BraTS | 91.24 | 91.13 | -0.11% (worse) |
| KiTS | 86.04 | 88.17 | +2.13% |
| AMOS | 88.64 | 89.41 | +0.77% |

**Average gain:** ~0.5% to 1.5% Dice. Best case ~2.5%.

### Cost-Benefit Analysis

| Factor | Impact |
|--------|--------|
| Training time | ~2× longer per fold |
| Compute units | ~+75% to +130% (nearly double) |
| Accuracy gain | ~+1% average |
| Lost progress | Folds 0 & 1 already trained = wasted if switching |

**Decision:** ❌ **Do not switch to ResEnc.** Finish default U-Net first. If time/budget allows after fold 4, a single ResEnc run can be done as an ablation comparison for the thesis.

---

## 6. How to Upload Revised `nnUNetPlans.json` in Live Session

**Scenario:** You want to change batch_size mid-training without losing the Colab runtime.

**Steps:**

1. **Stop** the currently running training cell (⏹). Keep the VM alive.
2. **Upload** revised `nnUNetPlans.json` to:
   ```
   /content/nnUNet_preprocessed/Dataset500_KidneyAbnormalities/nnUNetPlans.json
   ```
   (Drag-and-drop in Colab file viewer, overwrite existing.)
3. **Copy to results folder** (so trainer reads new plans):
   ```python
   import shutil
   from pathlib import Path
   src = Path("/content/nnUNet_preprocessed/Dataset500_KidneyAbnormalities/nnUNetPlans.json")
   dst = Path("/content/nnUNet_results/Dataset500_KidneyAbnormalities/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json")
   if dst.exists():
       shutil.copy2(src, dst)
   ```
4. **Delete old fold checkpoint** (critical — otherwise nnU-Net resumes old settings):
   ```python
   import shutil
   fold0 = Path("/content/nnUNet_results/Dataset500_KidneyAbnormalities/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0")
   if fold0.exists():
       shutil.rmtree(fold0)
   ```
5. **Verify** new plans loaded:
   ```python
   import json
   with open("/content/nnUNet_preprocessed/Dataset500_KidneyAbnormalities/nnUNetPlans.json") as f:
       print("batch_size:", json.load(f)["configurations"]["3d_fullres"]["batch_size"])
   ```
6. **Retrain:**
   ```bash
   !nnUNetv2_train 500 3d_fullres 0 --npz
   ```

---

## 7. Current Notebook Structure (Revised)

| Section | Content |
|---------|---------|
| **1** | Runtime & Environment Setup (GPU check, install, mount Drive) |
| **2** | Dataset Configuration (paths, env vars, copy to local) |
| **3** | Dataset Integrity Check (`dataset.json`, label values, image/label count) |
| **4** | Planning & Preprocessing (`-np 2` recommended for Colab stability) |
| **4.1** | Backup Preprocessed Data & Plans to Drive (fast plans backup + full folder) |
| **(Optional)** | Inspect Generated Plans (with auto-restore from Drive) |
| **5** | Model Training (Fold 0–4 cells + resume cell with `--c`) |
| **6** | Find Best Configuration (`nnUNetv2_find_best_configuration`) |
| **7** | 5-Fold CV Evaluation (predict + compute DSC/IoU/HD95/P/R/F1 → `cv_summary.json`) |
| **8** | Custom Inference with Profiling (5-fold ensemble, CUDA timers, VRAM logging) |
| **9** | Save Results to Google Drive (copy `nnUNet_results/` to Drive) |
| **10** | Export Model for Streamlit Deployment (clean zip with weights + metadata) |

---

## 8. Open Decisions / Future Work

| Topic | Decision | Notes |
|-------|----------|-------|
| Single-class configs (A, B, C) | ⏸️ Deferred | User focusing on multi-class (D) first |
| CCA post-processing | ⏸️ Deferred | User explicitly said "don't do this for now" |
| Held-out test set (60 volumes) | ❌ Not available | User has no labeled test set |
| ResEnc comparison | ⏸️ Optional future work | Only if time/budget after default U-Net completes |
| Batch size for remaining folds | 12 (current) | Can bump to 16 for next fold if VRAM stays <50 GB |

---

## 9. Key Files in This Project

| File | Purpose |
|------|---------|
| `kidney_seg_colab.ipynb` | Main training notebook (revised with Sections 7-10) |
| `nnUNetPlans.json` | Plans file (currently `batch_size=12` for 3d_fullres) |
| `Dataset500_KidneyAbnormalities/` | Raw dataset (290 cases, 4-class labels) |
| `cv_summary.json` | Output from Section 7.2 (per-class metrics) |
| `inference_profile.json` | Output from Section 8.1 (latency + VRAM per volume) |

---

*Compiled from discussion on 2026-04-26. For questions, refer to the specific sections above.*
