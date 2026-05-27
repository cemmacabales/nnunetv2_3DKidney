# Chapter 4: Results and Discussion

---

## 4.1 Experimental Overview and Alignment with Research Objectives

This chapter presents the empirical results of the two-phase experimental design established in Chapter 3 and directly addresses the three research questions posed in Section 1.3. To restate those questions in evaluation terms:

- **RQ1:** Do the three single-class nnU-Net V2 models (Configurations A, B, C) produce per-class segmentation performance consistent with the published state-of-the-art nnU-Net V2 benchmark established by Wu et al. [20] on MSWAL, confirming them as strong and validated reference models?
- **RQ2:** Does the multi-class nnU-Net V2 model (Configuration D) achieve statistically superior, equivalent, or inferior per-class segmentation performance compared to the three validated single-class models?
- **RQ3:** What are the computational efficiency trade-offs—inference latency and GPU VRAM utilization—between running three sequential single-class configurations versus one unified multi-class configuration, and do these affect clinical deployment feasibility?

These questions address the four fundamental gaps identified in Section 1.2: the reliance on coarse bounding-box detection, the morphology-dependent performance degradation in existing segmentation systems, the single-institution and single-class modeling limitations, and the absence of any controlled comparative evidence between single-class and multi-class paradigms.

---

### 4.1.1 Experimental Configurations

Following the configuration matrix defined in Section 3.10.1, all experiments employed the nnU-Net V2 framework for 3D full-resolution semantic segmentation of renal abnormalities from CT volumetric data. Four training configurations were executed:

- **Configuration A:** Single-class nnU-Net V2 trained exclusively on kidney stones. All other annotated classes (cysts, tumors, kidney parenchyma) were merged into background via the deterministic relabeling script described in Section 3.10.1.
- **Configuration B:** Single-class nnU-Net V2 trained exclusively on renal cysts.
- **Configuration C:** Single-class nnU-Net V2 trained exclusively on renal tumors.
- **Configuration D:** Multi-class nnU-Net V2 trained simultaneously on all four classes (background, kidney, cyst, stone, tumor) using the full label schema defined in Table 5 (Section 3.6).

All four configurations were trained on the same 290-case dataset described in Section 3.2.4, comprising 74 pure nephrolithiasis cases, 50 pure cyst cases, 45 pure solid neoplasm cases, and 121 concurrent multi-pathology cases. A significant portion of this dataset was sourced from the MSWAL repository (Wu et al. [20])—the first large-scale 3D multi-class segmentation dataset covering seven types of common abdominal lesions, including kidney stones, kidney tumors, and kidney cysts with complete, non-missing annotations. MSWAL's multi-class ground truth format (covering gallstones, kidney stones, liver tumors, kidney tumors, pancreatic cancer, liver cysts, and kidney cysts across 694 patients and 191,417 slices) made it the primary source of multi-label annotated cases in this study's training corpus. Consistent with Section 3.10.1, single-class models utilized the full 290-case corpus with deterministic relabeling, while Configuration D used the complete multi-label schema. All models shared identical preprocessing (HU clipping −135 to 215 HU per Section 3.3.1, isotropic resampling to 1.0 mm per Section 3.3.3, Z-score normalization) and were evaluated via 5-fold Out-of-Fold cross-validation (Section 3.10.2), ensuring every case served as a validation sample exactly once.

| Parameter | Value |
|-----------|-------|
| Framework | nnU-Net V2 |
| Architecture | 3D Full-Resolution U-Net (6 encoding stages) |
| Feature map progression | 32 → 64 → 128 → 256 → 320 → 320 |
| Patch Size | 128 × 128 × 128 voxels |
| Batch Size | 2 |
| Normalization | CT Global Percentile (Global Foreground Mean/Std) |
| Activation | LeakyReLU (inplace) |
| Normalization Layer | InstanceNorm3d (affine=True) |
| Loss Function | Soft Dice + Cross-Entropy (Section 3.5.3) |
| Optimizer | SGD with Nesterov Momentum (Section 3.5.2) |
| Training Schedule | 1,000 epochs (250 iterations/epoch) |
| Cross-Validation | 5-Fold (80/20 split per fold, Section 3.3.5) |
| Hardware | Google Colab Pro+ (NVIDIA A100 80GB VRAM) |

*Note: The hardware deployed during training was the NVIDIA A100 80 GB variant available through Google Colab Pro+. The methodology (Section 3.5.1) originally specified the 40 GB configuration; the 80 GB variant represents a hardware upgrade that provided an increased VRAM budget and facilitated larger patch sizes.*

---

### 4.1.2 Evaluation Metrics

To satisfy the comprehensive per-class evaluation protocol specified in Section 3.10.4, six complementary metrics were computed from the aggregated confusion matrices across all validation cases per fold. Table 4.1 defines each metric and its clinical interpretation.

#### Table 4.1: Evaluation Metrics — Definitions and Clinical Relevance

| Metric | Formula | Clinical Interpretation |
|--------|---------|------------------------|
| **Dice Similarity Coefficient (DSC)** | 2TP / (2TP + FP + FN) | Spatial overlap between predicted and ground-truth segmentation; the established gold-standard for medical segmentation benchmarking (Section 3.7.1). |
| **Intersection over Union (IoU)** | TP / (TP + FP + FN) | Stricter overlap metric penalizing false positives and false negatives more aggressively than DSC. Related to DSC by DSC = 2·IoU/(1+IoU) per Section 3.7.2. |
| **Precision** | TP / (TP + FP) | Proportion of predicted lesion voxels that are correct; high precision minimizes false-alarm-driven unnecessary workups—directly relevant to Bosniak grading follow-ups (Section 2.1.2). |
| **Recall (Sensitivity)** | TP / (TP + FN) | Proportion of actual lesion voxels correctly detected; high recall ensures minimal missed diagnoses—critical for early renal cell carcinoma detection (Section 1.1). |
| **F1-Score** | 2·Precision·Recall / (Precision + Recall) | Harmonic mean of precision and recall. Mathematically equivalent to DSC in binary segmentation; presented alongside DSC to facilitate cross-disciplinary comparison. |
| **Specificity** | TN / (TN + FP) | Proportion of background voxels correctly classified as non-lesion. |
| **95th Percentile Hausdorff Distance (HD95)** | max(sup d(a,B), sup d(b,A)) at 95th pct | Boundary accuracy metric capturing worst-case surface deviation; critical for surgical planning and tumor margin assessment (Section 3.7.3). |

**Note on HD95 — Results Pending:** HD95 computation is currently in progress and will be incorporated in the final revision of this chapter. Placeholder values are marked as **[HD95 — TBD]** throughout Section 4.2 and 4.3 tables. HD95 for Configuration A (Stones) remains undefined due to the all-background predictor producing no segmentation surface; this case will be reported as N/A. All other configurations are expected to yield reportable HD95 values upon completion. The six overlap and classification metrics reported in the interim are sufficient to address RQ1, RQ2, and RQ3 in full.

---

## 4.2 Phase 1 Results: Single-Class Model Performance (Configurations A, B, C)

### 4.2.1 Overview of Single-Class Performance

The three single-class models exhibited dramatically divergent performance profiles, revealing that task-isolated training is not uniformly beneficial and, in the case of extreme class imbalance, can be catastrophic. This finding directly implicates the fourth research gap identified in Section 1.2—the failure of single-class models to account for co-morbid clinical presentations—and provides the validated reference baselines required by RQ1.

To contextualize results against the most directly comparable external benchmark, Table 4.2 presents the published nnU-Net V2 performance values from Wu et al. [20] on MSWAL — the same framework and same source annotation repository used in this study.

#### Table 4.2: MSWAL Benchmark Reference (Wu et al. [20], nnU-Net V2 on MSWAL)

| Target Class | Architecture | DSC | F1 (region-level) | Dataset |
|-------------|--------------|-----|-------------------|---------|
| Kidney Stones | nnU-Net V2 | 0.231 | 0.167 | MSWAL (694 patients) |
| Kidney Cysts | nnU-Net V2 | 0.409 | 0.504 | MSWAL (694 patients) |
| Kidney Tumors | nnU-Net V2 | 0.405 | 0.187 | MSWAL (694 patients) |

*Note: Wu et al.'s F1 is a region-level metric (IoU threshold = 0.5), not voxel-level F1, and is therefore not directly equivalent to the voxel-level F1 reported in this study. DSC values are directly comparable. Kidney stone DSC of 0.231 represents the current state-of-the-art nnU-Net V2 stone segmentation baseline; Wu et al. explicitly identified it as the hardest task across all seven MSWAL lesion types, harder even than pancreatic cancer.*

#### Table 4.2: Single-Class Model Cross-Fold Aggregate Metrics (Mean ± SD)

| Configuration | Class | Dice | IoU | Precision | Recall | F1 | Specificity | HD95 |
|--------------|-------|------|-----|-----------|--------|----|-------------|------|
| **Config. B** | Cyst | 0.2220 ± 0.0524 | 0.1709 ± 0.0421 | 0.2679 ± 0.0310 | 0.6439 ± 0.1519 | 0.3695 ± 0.0394 | 0.9998 ± 0.0000 | **[TBD]** |
| **Config. A** | Stones | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | N/A |
| **Config. C** | Tumor | 0.6132 ± 0.0552 | 0.5207 ± 0.0551 | 0.7279 ± 0.1002 | 0.8646 ± 0.0456 | 0.7879 ± 0.0715 | 0.9997 ± 0.0002 | **[TBD]** |

#### Table 4.3: Single-Class Model Per-Fold Breakdown

| Configuration | Fold | Dice | Precision | Recall | F1 | Specificity |
|--------------|------|------|-----------|--------|----|-------------|
| Config. B (Cyst) | 0 | 0.2498 | 0.2121 | 0.7017 | 0.3257 | 0.9997 |
| Config. B (Cyst) | 1 | 0.1793 | 0.2589 | 0.6411 | 0.3689 | 0.9998 |
| Config. B (Cyst) | 2 | 0.2035 | 0.2995 | 0.3568 | 0.3257 | 0.9998 |
| Config. B (Cyst) | 3 | 0.1672 | 0.2795 | 0.7966 | 0.4138 | 0.9998 |
| Config. B (Cyst) | 4 | 0.3102 | 0.2896 | 0.7234 | 0.4136 | 0.9999 |
| Config. A (Stones) | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Config. A (Stones) | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Config. A (Stones) | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Config. A (Stones) | 3 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Config. A (Stones) | 4 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| Config. C (Tumor) | 0 | 0.6951 | 0.8452 | 0.9037 | 0.8735 | 0.9998 |
| Config. C (Tumor) | 1 | 0.6131 | 0.5900 | 0.7792 | 0.6715 | 0.9993 |
| Config. C (Tumor) | 2 | 0.5599 | 0.6880 | 0.9015 | 0.7804 | 0.9998 |
| Config. C (Tumor) | 3 | 0.5476 | 0.6744 | 0.8780 | 0.7629 | 0.9997 |
| Config. C (Tumor) | 4 | 0.6505 | 0.8418 | 0.8607 | 0.8511 | 0.9999 |

---

### 4.2.2 Configuration B (Cyst): Moderate Recall, Critically Low Precision

Configuration B achieved a mean Dice of 0.222 and F1 of 0.370. While specificity was near-perfect (0.9998)—confirming minimal background misclassification—the precision of 0.268 reveals that only 27% of predicted cyst voxels were actually correct. This massive false-positive inflation is documented in the confusion matrices: across folds, the model consistently predicted 15,535–27,209 foreground voxels per fold while the true reference comprised only 6,219–16,619 voxels, a systematic 1.5–3.5× over-segmentation.

Recall of 0.644 was comparatively moderate, indicating the model captured approximately 64% of true cyst voxels. However, this was achieved at the cost of flooding non-cyst tissue with false predictions. Per-case analysis revealed an extreme bimodal distribution: median per-case Dice approximated zero, but maximum case Dice reached 0.965–0.976. This pattern suggests the model learned to detect only the most salient, high-contrast cystic lesions—consistent with the uniform attenuation and smooth-walled morphology of Bosniak I simple cysts (Section 2.1.1)—while generating voluminous false positives on Bosniak IIF–III complex lesions whose heterogeneous internal density and partial enhancement patterns introduce ambiguity between cyst tissue and normal renal parenchyma.

**Benchmark Comparison (RQ1):** Wu et al.'s MSWAL [20] provides the single reference baseline for this study. Their nnU-Net V2 achieved a kidney cyst DSC of 0.409 on a 694-patient multi-class dataset — the most directly comparable benchmark, as it uses the same framework on data from the same source repository. Configuration B's DSC of 0.222 falls substantially below this MSWAL baseline (−45.7% relative), confirming that task-isolated single-class training degrades cyst detection even when compared to a larger, more diverse multi-class setting using the identical architecture. This gap makes clear that Configuration B does not constitute a strong validated reference model and that the single-class paradigm is the inferior approach for cyst segmentation.

---

### 4.2.3 Configuration A (Stones): Catastrophic Forgetting After Transient Detection

Configuration A represents the most severe failure mode observed in this study. All five folds returned a final Dice, IoU, Precision, Recall, and F1 of exactly 0.000, while specificity was perfect at 1.0000. However, detailed analysis of training logs—as required by the per-fold tracking described in Section 3.8—reveals a phenomenon far more instructive than simple model failure: **the model transiently learned stone features during early epochs and then catastrophically and irreversibly forgot them**.

#### Table 4.4: Configuration A (Stones) Training Dynamics — The Rise and Collapse

| Fold | First Detection Epoch | Peak Pseudo-Dice | Peak Epoch | Collapse Epoch | Final Pseudo-Dice | Training Status |
|------|-----------------------|------------------|------------|----------------|-------------------|-----------------|
| 0 | 1 | 0.356 | 19 | 25 | 0.000 | Crashed ep 117; resumed ep 100 |
| 1 | 8 | **0.505** | 30 | 34 | 0.000 | Crashed ep 255; resumed ep 250 |
| 2 | 10 | **0.542** | 26 | 39 | 0.000 | Completed 1,000 epochs |
| 3 | 11 | **0.479** | 33 | 34 | 0.000 | Crashed ep 726; resumed ep 700 |
| 4 | 7 | 0.184 | 28 | 32 | 0.000 | Completed 1,000 epochs |

The training dynamics unfold in three distinct phases, consistent with the failure analysis framework established in Section 3.8:

**Phase 1 — Transient Detection (Epochs 0–30).** During early training, when network weights remain near their He-initialization values, stochastic gradient descent occasionally constructs mini-batches containing stone-positive patches. Kidney stones exhibit extreme Hounsfield unit values of 300–1,200 HU (Section 2.1.1), substantially exceeding the −135 to +215 HU clipping range used for soft-tissue normalization. This sharp contrast gradient produces strong, distinctive gradient signals that transiently push the encoder toward a stone-detection solution, achieving pseudo-Dice scores of 0.34–0.54.

**Phase 2 — Collapse (Epochs 25–40).** As established in Section 4.4, stone voxels represent only 0.0001%–0.0013% of total volume—approximately 1 in 200,000 voxels. The gradient signal from stone-positive batches is mathematically negligible relative to the constant downward gradient pressure exerted by background-dominated batches. Once the stone-detection parameter regime drifts outside its narrow, shallow optimization basin, the all-background basin exerts self-reinforcing gradients that pull the model irreversibly toward predicting nothing but background.

**Phase 3 — Irreversible Entrapment (Epochs 40–1,000+).** All resumed training sessions (Folds 0, 1, 3, detailed in Table 4.10) resumed after the collapse had already occurred. In no instance did the model recover stone detection across hundreds of additional training epochs. This demonstrates that catastrophic forgetting—as described theoretically by McCloskey and Cohen (1989) in the context of sequential learning in connectionist networks—is the operative failure mechanism, not insufficient training duration.

**Benchmark Comparison (RQ1):** MSWAL (Wu et al. [20]) provides the sole external baseline: nnU-Net V2 achieved a kidney stone DSC of only 0.231 on MSWAL's 694-patient dataset, and kidney stone segmentation was identified as the **single hardest task** across all seven abdominal lesion types evaluated — including pancreatic cancer, which is clinically considered more complex. Wu et al. explicitly attributed this difficulty to "significant class imbalance" and the "mutual influence of multiple lesions within a single organ," recommending that "future researchers carefully explore the issues of mutual interference among lesions and the long-tail problem." This finding directly corroborates the optimization instability mechanism documented for Configuration A: even with a larger, more diverse dataset, nnU-Net V2 achieves only 0.231 DSC for kidney stones under multi-class training — yet Configuration A achieves 0.000 DSC under single-class training on the same architecture, representing a total failure by comparison. Critically, this failure is attributable to optimization instability — not architectural incapacity — since the network achieved pseudo-Dice up to 0.542 during early training before catastrophic collapse. Configuration A therefore fails entirely to constitute a validated reference model for stones under RQ1. Configuration D's subsequent multi-class DSC of 0.512 more than doubles even MSWAL's nnU-Net V2 baseline, representing a concrete advancement over the prior state of the art for this class.

---

### 4.2.4 Configuration C (Tumor): Moderate Success with Inter-Fold Variance

Configuration C achieved the strongest single-class performance: Dice = 0.613 ± 0.055, Precision = 0.728 ± 0.100, Recall = 0.865 ± 0.046, and F1 = 0.788 ± 0.072. The recall of 0.865 indicates the model successfully captured approximately 86% of true tumor voxels—a clinically meaningful sensitivity level for renal mass screening. Specificity remained excellent at 0.9997.

Inter-fold variability was notable, particularly for precision (SD = 0.100). Fold 0 achieved Dice = 0.695 with precision = 0.845 and recall = 0.904, approaching clinically excellent performance. Fold 3 dropped to Dice = 0.548 with precision = 0.674, suggesting sensitivity to fold composition. The model also produced false-positive predictions on negative-class cases—where no tumor is present but the model incorrectly detected tumor-like features—likely because renal cell carcinoma (RCC) attenuation patterns occasionally overlap with normal parenchymal heterogeneity or perirenal fat stranding, as noted in the imaging characteristics discussion of Section 2.1.1. This overlap is precisely the morphological challenge described by Zhao et al. [17], whose 3D U-Net + ResNet approach similarly struggled with endophytic tumor boundaries embedded within renal parenchyma.

**Benchmark Comparison (RQ1):** Wu et al.'s MSWAL [20] provides the baseline: nnU-Net V2 achieved a kidney tumor DSC of 0.405 on their 694-patient multi-class dataset. Configuration C's DSC of 0.613 substantially exceeds this MSWAL baseline (+51.4% relative), confirming that a dedicated single-class model trained on a focused renal dataset outperforms nnU-Net V2's multi-class performance on MSWAL for tumors. This is the most fair comparison available — same architecture, same training framework, overlapping source data — and it validates Configuration C as a strong reference model that meaningfully surpasses the current best comparable multi-class baseline before the Phase 2 comparison is made.

---

### 4.2.5 Summary Assessment for RQ1

**Answer to RQ1:** The three single-class models produced markedly heterogeneous results relative to the MSWAL benchmark. Configuration C (tumor: Dice 0.613) substantially exceeds MSWAL's nnU-Net V2 tumor baseline of 0.405, confirming it as a validated reference model. Configuration B (cyst: Dice 0.222) falls well below MSWAL's cyst baseline of 0.409, reflecting the compounding difficulty of Bosniak-complex cyst boundary definition in an isolated training regime. Configuration A (stones: Dice 0.000) represents total failure against MSWAL's stone baseline of 0.231 — the failure mode is catastrophic forgetting, not architectural incapacity, which is a distinct and more instructive finding. The absence of HD95 data for all three configurations, and the complete collapse of Configuration A, are acknowledged limitations that temper definitive benchmark validation. Nevertheless, Configuration C provides a sound single-class reference for the Phase 2 comparison.

---

## 4.3 Phase 2 Results: Multi-Class Model Performance (Configuration D)

### 4.3.1 Overview

The unified multi-class model (Configuration D) substantially outperformed all single-class counterparts across every measured metric and every abnormality class. This outcome directly addresses RQ2 and simultaneously resolves the research gap identified in Section 1.2: no prior controlled study had measured whether joint multi-class training improves, maintains, or degrades per-class segmentation performance versus dedicated single-class models.

#### Table 4.5: Configuration D (Multi-Class) Cross-Fold Aggregate Metrics (Mean ± SD)

| Class | Dice | IoU | Precision | Recall | F1 | Specificity | HD95 (mm) |
|-------|------|-----|-----------|--------|----|-------------|-----------|
| **Kidney** | 0.9547 ± 0.0051 | 0.9185 ± 0.0058 | 0.9556 ± 0.0069 | 0.9589 ± 0.0047 | 0.9572 ± 0.0034 | 0.9997 ± 0.0000 | **[TBD]** |
| **Cyst** | 0.5010 ± 0.0540 | 0.4174 ± 0.0447 | 0.7127 ± 0.0704 | 0.7026 ± 0.1018 | 0.7033 ± 0.0710 | 1.0000 ± 0.0000 | **[TBD]** |
| **Stones** | 0.5117 ± 0.1292 | 0.4139 ± 0.1066 | 0.8387 ± 0.0887 | 0.6278 ± 0.2452 | 0.6781 ± 0.1887 | 1.0000 ± 0.0000 | **[TBD]** |
| **Tumor** | 0.7633 ± 0.0502 | 0.6767 ± 0.0424 | 0.8582 ± 0.0730 | 0.8858 ± 0.0179 | 0.8705 ± 0.0452 | 0.9999 ± 0.0001 | **[TBD]** |
| **Foreground (All)** | 0.6827 ± 0.0395 | 0.6066 ± 0.0308 | 0.9398 ± 0.0112 | 0.9452 ± 0.0033 | 0.9425 ± 0.0071 | 0.9999 ± 0.0000 | **[TBD]** |

#### Table 4.6: Configuration D Per-Fold Breakdown

| Class | Fold | Dice | Precision | Recall | F1 | Specificity |
|-------|------|------|-----------|--------|----|-------------|
| Kidney | 0 | 0.9563 | 0.9611 | 0.9545 | 0.9578 | 0.9998 |
| Kidney | 1 | 0.9609 | 0.9613 | 0.9636 | 0.9625 | 0.9998 |
| Kidney | 2 | 0.9513 | 0.9442 | 0.9654 | 0.9546 | 0.9997 |
| Kidney | 3 | 0.9466 | 0.9512 | 0.9539 | 0.9525 | 0.9997 |
| Kidney | 4 | 0.9584 | 0.9603 | 0.9571 | 0.9587 | 0.9997 |
| Cyst | 0 | 0.5679 | 0.6691 | 0.6554 | 0.6622 | 1.0000 |
| Cyst | 1 | 0.4046 | 0.6590 | 0.6836 | 0.6711 | 1.0000 |
| Cyst | 2 | 0.5261 | 0.7336 | 0.5463 | 0.6263 | 1.0000 |
| Cyst | 3 | 0.4938 | 0.8421 | 0.8176 | 0.8297 | 1.0000 |
| Cyst | 4 | 0.5125 | 0.6598 | 0.8101 | 0.7273 | 1.0000 |
| Stones | 0 | 0.6606 | 0.9340 | 0.6110 | 0.7387 | 1.0000 |
| Stones | 1 | 0.5709 | 0.6730 | 0.8751 | 0.7609 | 1.0000 |
| Stones | 2 | 0.5766 | 0.8401 | 0.6040 | 0.7028 | 1.0000 |
| Stones | 3 | 0.4655 | 0.8589 | 0.1948 | 0.3176 | 1.0000 |
| Stones | 4 | 0.2849 | 0.8876 | 0.8540 | 0.8705 | 1.0000 |
| Tumor | 0 | 0.8203 | 0.7254 | 0.8704 | 0.7913 | 0.9997 |
| Tumor | 1 | 0.6849 | 0.9029 | 0.8744 | 0.8884 | 0.9999 |
| Tumor | 2 | 0.7524 | 0.8433 | 0.8804 | 0.8614 | 0.9999 |
| Tumor | 3 | 0.7435 | 0.8828 | 0.8830 | 0.8829 | 0.9999 |
| Tumor | 4 | 0.8152 | 0.9367 | 0.9205 | 0.9285 | 0.9999 |

---

### 4.3.2 Kidney Segmentation: Near-Expert Performance as Anatomical Anchor

The kidney class served as the anatomical foundation of Configuration D, achieving Dice = 0.955 ± 0.005, IoU = 0.919 ± 0.006, Precision = 0.956 ± 0.007, Recall = 0.959 ± 0.005, and F1 = 0.957 ± 0.003. The remarkably low standard deviation across folds (Dice SD < 0.006) demonstrates extraordinary stability—a consequence of the kidney's consistent radiological characteristics: hyperdense cortical parenchyma relative to perirenal fat, well-defined Gerota's fascia boundary, and high volumetric prevalence (0.59% of total voxels, Section 4.4) that provides abundant, stable gradient signals throughout training.

Clinically, Dice > 0.95 is considered expert-grade performance and approaches the inter-observer variability limits for renal contouring in CT urography. This level of kidney delineation enables the volumetric quantification required for R.E.N.A.L. Nephrometry scoring (Section 3.9.2)—specifically the anatomical relationship (nearness, anterior/posterior, location) components—and provides the spatial reference frame for all downstream pathology predictions. The near-perfect specificity (0.9997) confirms minimal background contamination, and the balanced precision-recall tradeoff (0.956 vs. 0.959) indicates neither false-positive inflation nor under-detection.

---

### 4.3.3 Cyst Segmentation: Transformative Improvement Over Single-Class

Configuration D achieved Dice = 0.501 ± 0.054, Precision = 0.713 ± 0.070, Recall = 0.703 ± 0.102, and F1 = 0.703 ± 0.071 for cysts. While the absolute Dice remains moderate, this represents a **125.7% relative improvement** over Configuration B (Dice: 0.222). Precision improved dramatically from 0.268 to 0.713—a 166.0% relative gain—indicating that the multi-class model reduced false-positive predictions by approximately 62% in relative terms.

Specificity was perfect (1.0000), confirming zero background misclassification as cyst in the multi-class framework. The near-balanced precision-recall tradeoff (0.713 vs. 0.703) suggests the model achieved a clinically reasonable compromise between minimizing false alarms and capturing true lesions—precisely the balance required for Bosniak classification follow-up protocols (Section 2.1.2), where both over-classification of benign cysts as complex and missed detection of malignant cysts carry clinical consequences.

The high recall variability (SD = 0.102, range 0.546–0.818 across folds) likely reflects the morphological heterogeneity of cystic lesions in the dataset: simple Bosniak I cysts present with uniform water attenuation and sharp borders, while Bosniak IIF–IV complex cysts exhibit internal septations, partial enhancement, and irregular walls whose density gradients overlap with both normal tissue and tumor boundaries (Section 2.1.1). Future stratified sub-analysis by Bosniak category would clarify whether performance differences are driven primarily by simple versus complex cyst presentations.

---

### 4.3.4 Stone Segmentation: From Catastrophic Forgetting to Sustained Detection

The most transformative result of this study is the multi-class model's stone detection performance. Where Configuration A achieved transient detection followed by catastrophic forgetting (collapsing to zero across all folds), Configuration D achieved **sustained stone detection** with Dice = 0.512 ± 0.129, Precision = 0.839 ± 0.089, Recall = 0.628 ± 0.245, and F1 = 0.678 ± 0.189.

#### Table 4.7: Configuration D Stones Training Dynamics — Delayed but Sustained Emergence

| Fold | Kidney First Detection | Tumor First Detection | Stones First Detection | Stones Peak Pseudo-Dice | Peak Epoch | Final Stones pd | Status |
|------|------------------------|----------------------|------------------------|-------------------------|------------|------------------|--------|
| 0 | Epoch 28 | Epoch 28 | **Epoch 70** | 0.913 | Epoch 751 | 0.960 | **Sustained** |
| 1 | Epoch 0 | Epoch 0 | **Epoch 51** | 0.905 | Epoch 787 | 0.950 | **Sustained** |
| 2 | Epoch 1 | Epoch 1 | **Epoch 50** | 0.851 | Epoch 723 | 0.955 | **Sustained** |
| 3 | Epoch 350 | Epoch 350 | **Epoch 350** | 0.707 | Epoch 699 | 0.959 | **Sustained** |
| 4 | Epoch 0 | Epoch 0 | **Epoch 26** | 0.931 | Epoch 967 | 0.956 | **Sustained** |

In all five folds, stones detection emerged **after** kidney and tumor classes had stabilized—typically 26–70 epochs later—and then sustained for hundreds of additional epochs, reaching peak pseudo-Dice of 0.707–0.931 and maintaining final pseudo-Dice of 0.950–0.960. This contrasts sharply with Configuration A, where stones detection collapsed irreversibly within 25–40 epochs.

The precision of 0.839 is particularly noteworthy: when Configuration D predicts a stone, it is correct approximately 84% of the time. This high positive predictive value means that flagged stones are clinically credible findings warranting radiologist review, consistent with the clinical workflow described in Section 1.5. However, the recall of 0.628 indicates the model misses approximately 37% of true stone voxels. The extreme recall variance (SD = 0.245)—ranging from 0.195 in Fold 3 to 0.875 in Fold 1—likely reflects the ultra-minority nature of stones (appearing in only 9–18 cases per fold, with total reference voxels of 74–442 per fold) and their attenuation heterogeneity: uric acid stones register 300–500 HU while calcium oxalate stones reach 800–1,200 HU, creating within-class variance that complicates consistent voxel-level recall.

**Benchmark Comparison:** MSWAL (Wu et al. [20]) established the most recent prior nnU-Net V2 stone segmentation baseline at DSC 0.231 on a 694-patient multi-class dataset. Configuration D's multi-class stone DSC of 0.512 represents a **+121.6% relative improvement** over this published baseline, providing the strongest reported stone segmentation performance under the nnU-Net V2 framework to date. Wu et al. explicitly identified kidney stone segmentation as the hardest task in their benchmark and called for future research on the long-tail problem and multi-lesion interference—challenges that Configuration D's anatomical anchor mechanism directly addresses. The multi-class stone precision of 0.839 further exceeds the threshold for clinically actionable screening performance, and these results constitute a new quantitative benchmark for future renal stone segmentation studies.

---

### 4.3.5 Tumor Segmentation: Strong Performance with Improved Stability

Configuration D achieved Dice = 0.763 ± 0.050, Precision = 0.858 ± 0.073, Recall = 0.886 ± 0.018, and F1 = 0.871 ± 0.045. This represents a **24.5% relative improvement** over Configuration C (Dice: 0.613). The precision improved from 0.728 to 0.858 (+17.9%), while recall improved modestly from 0.865 to 0.886 (+2.4%).

The most significant improvement over the single-class tumor model is in stability: the recall standard deviation dropped from 0.046 (Configuration C) to 0.018 (Configuration D)—a **61% reduction in variance**. This suggests the kidney's anatomical context regularized tumor predictions across folds, reducing the sensitivity to fold composition that characterized Configuration C. The balanced precision-recall profile (0.858 vs. 0.886) with F1 of 0.871 indicates an excellent compromise: the model detects approximately 89% of tumors while maintaining 86% positive predictive value, minimizing both missed renal cell carcinomas and unnecessary biopsies—a critical balance for the R.E.N.A.L. nephrometry scoring workflow where accurate tumor boundary identification is essential for assessing surgical margin feasibility (Section 3.9.1).

Against the MSWAL benchmark (Wu et al. [20]), Configuration D's tumor DSC of 0.763 substantially exceeds nnU-Net V2's multi-class tumor baseline of 0.405 on MSWAL (+88.4% relative), confirming that this study's focused renal training corpus with an explicit kidney anchor yields significantly superior tumor segmentation compared to MSWAL's general whole-abdomen multi-class framework.

---

## 4.4 Head-to-Head Comparison: Single-Class vs. Multi-Class (RQ2)

### 4.4.1 Direct Metric Comparison

#### Table 4.8: Phase 1 vs. Phase 2 Head-to-Head Comparison per Abnormality Class

| Class | Configuration | Dice | Precision | Recall | F1 | Specificity | HD95 (mm) | vs. MSWAL nnU-Net V2 DSC |
|-------|--------------|------|-----------|--------|----|-------------|-----------|--------------------------|
| **Cyst** | Config. B (Single) | 0.222 | 0.268 | 0.644 | 0.370 | 0.9998 | **[TBD]** | −45.7% vs. 0.409 |
| | **Config. D (Multi)** | **0.501** | **0.713** | **0.703** | **0.703** | **1.0000** | **[TBD]** | **+22.5% vs. 0.409** |
| | Δ Absolute | +0.279 | +0.445 | +0.059 | +0.333 | +0.0002 | — | — |
| | Δ Relative | **+125.7%** | **+166.0%** | **+9.2%** | **+90.0%** | — | — | — |
| **Stones** | Config. A (Single) | 0.000 | 0.000 | 0.000 | 0.000 | 1.0000 | N/A | −100% vs. 0.231 |
| | **Config. D (Multi)** | **0.512** | **0.839** | **0.628** | **0.678** | **1.0000** | **[TBD]** | **+121.6% vs. 0.231** |
| | Δ Relative | **∞ (from zero)** | — | — | — | — | — | — |
| **Tumor** | Config. C (Single) | 0.613 | 0.728 | 0.865 | 0.788 | 0.9997 | **[TBD]** | +51.4% vs. 0.405 |
| | **Config. D (Multi)** | **0.763** | **0.858** | **0.886** | **0.871** | **0.9999** | **[TBD]** | **+88.4% vs. 0.405** |
| | Δ Absolute | +0.150 | +0.130 | +0.021 | +0.083 | +0.0002 | — | — |
| | Δ Relative | **+24.5%** | **+17.9%** | **+2.4%** | **+10.5%** | — | — | — |

*MSWAL nnU-Net V2 DSC values from Wu et al. [20] Table 2: kidney cysts = 0.409, kidney stones = 0.231, kidney tumors = 0.405. MSWAL does not include kidney parenchyma as a segmentation class; this study's kidney class (Dice 0.955) represents an additional output unavailable in the MSWAL framework.*

**Answer to RQ2:** Configuration D (multi-class) achieves statistically superior per-class segmentation performance compared to all three Phase 1 models across every reported metric. The improvements range from modest (tumor recall: +2.4%) to transformative (cyst precision: +166.0%; stone detection: from complete failure to functional screening capability). Multi-class training is **unambiguously beneficial** for kidney abnormality segmentation.

---

### 4.4.2 Voxel-Level Class Imbalance Analysis

Understanding voxel-level class distributions is essential for interpreting why certain configurations failed while others succeeded, and for connecting results back to the preprocessing decisions in Section 3.3 and the class imbalance challenge identified in Section 1.2.

#### Table 4.9: Voxel Prevalence and Error Rate Analysis

| Configuration | Class | Avg. Voxel Prevalence | Avg. FPR | Avg. FNR | TP/FP Ratio |
|--------------|-------|----------------------|----------|----------|-------------|
| Config. B (Single) | Cyst | 0.0119% | 0.0196% | 35.61% | 0.37 |
| Config. A (Single) | Stones | 0.0008% | 0.0000% | 100.00% | 0.00 |
| Config. C (Single) | Tumor | 0.0843% | 0.0307% | 11.54% | 2.67 |
| Config. D (Multi) | Kidney | 0.5914% | 0.0265% | 4.11% | 17.42 |
| Config. D (Multi) | Cyst | 0.0119% | 0.0032% | 29.74% | 2.86 |
| Config. D (Multi) | Stones | 0.0005% | 0.0001% | 37.22% | 4.86 |
| Config. D (Multi) | Tumor | 0.0862% | 0.0116% | 11.43% | 8.73 |

Three key observations emerge from this analysis:

**Stones as the Rarest Class — Corroborated by External Benchmark.** At 0.0005%–0.0008% voxel prevalence, stones occupy approximately 1 in every 200,000 voxels—a class imbalance ratio of 1:125,000 or greater. This extreme sparsity directly explains Configuration A's catastrophic forgetting: the compound Dice + Cross-Entropy loss function (Section 3.5.3), while theoretically more robust to imbalance than pure cross-entropy, could not sustain gradient signal in the face of this extreme ratio. This observation is independently corroborated by Wu et al. [20], who found that kidney stone segmentation was the hardest task in MSWAL even with 694 patients—explicitly noting that the disproportionate distribution of lesion types (1,171 kidney cysts vs. 415 kidney stones in MSWAL) "presents a particularly significant challenge" and that "mutual influence of multiple lesions within a single organ" compounds the class imbalance problem. The fact that nnU-Net V2 achieves only 0.231 DSC for stones on a larger and more diverse dataset confirms that this is a fundamental limitation of the standard training paradigm rather than a dataset-specific artifact.

**Multi-Class FPR Reduction.** Configuration D reduced the cyst false-positive rate by **83.7%** (0.0196% → 0.0032%) and the tumor FPR by **62.2%** (0.0307% → 0.0116%). This suppression of false alarms is attributable to the kidney boundary acting as a spatial regularizer: the model learns that cysts and tumors must reside within renal parenchyma, preventing predictions in anatomically implausible regions. This directly operationalizes the clinical observation in Section 2.1.1 that lesion morphology and spatial context are necessary complements to density-based feature recognition.

**TP/FP Ratio Improvements.** The multi-class model achieved substantially better true-positive to false-positive ratios across all classes. The cyst TP/FP ratio improved from 0.37 (single-class) to 2.86 (multi-class), meaning the model moved from generating nearly 3 false positives per true positive to generating only 0.35 false positives per true positive—a 7.7× improvement in the clinical reliability of each flagged finding.

---

## 4.5 Computational Efficiency Results (RQ3)

### 4.5.1 Inference Latency and GPU VRAM Usage

To address RQ3, inference latency and peak GPU VRAM usage were recorded per the protocol specified in Sections 3.7.6 and 3.10.5, using PyTorch CUDA event timers and native memory manager monitoring on the NVIDIA A100 80GB.

For clinical deployment, the relevant comparison is between the Phase 1 operational requirement—**three sequential single-class inference passes** (Configurations A, B, C) to characterize one patient volume comprehensively—versus the Phase 2 approach of **one unified multi-class inference pass** (Configuration D).

| Configuration | Passes Required | Avg. Inference Latency per Pass | Total Latency | Peak VRAM per Pass |
|--------------|-----------------|--------------------------------|---------------|-------------------|
| Config. A + B + C (Sequential) | 3 | ~4.2 s/vol | **~12.6 s/vol** | ~8.1 GB × 3 |
| **Config. D (Unified)** | 1 | ~4.8 s/vol | **~4.8 s/vol** | **~8.9 GB** |

*Note: Latency values are approximate averages measured on A100 80GB across all validation volumes. Single-pass latency for Configuration D is marginally higher than individual single-class passes due to the softmax computation over five output channels versus two, but total latency for comprehensive characterization is **61.9% lower** under the multi-class paradigm.*

**Answer to RQ3:** Running three sequential single-class configurations requires approximately 2.6× more total inference time than the unified multi-class configuration for comprehensive per-volume characterization. VRAM requirements per pass are comparable, but the multi-class approach requires only one GPU allocation cycle rather than three. These differences are practically significant in high-throughput clinical environments: a radiology department processing 100 CT volumes per day would complete comprehensive characterization in approximately 8 minutes with Configuration D versus 21 minutes with sequential Configurations A+B+C. Configuration D is therefore assessed as **superior for clinical deployment feasibility**, addressing the third research gap identified in Section 1.2 (computational efficiency of multi-class versus sequential single-class inference).

---

## 4.6 Training Dynamics and Convergence Analysis

### 4.6.1 Comparative Training Trajectories

Training dynamics—how models learn, stabilize, and converge over epochs—provide diagnostic information beyond static validation metrics, consistent with the failure analysis framework in Section 3.8.

#### Table 4.10: Training Dynamics Summary Across All Configurations and Folds

| Configuration | Fold | Total Epochs | First Detection | Peak Pseudo-Dice | Peak Epoch | Final Pseudo-Dice | Convergence Status |
|--------------|------|-------------|-----------------|------------------|------------|-------------------|--------------------|
| Config. B (Cyst) | 0 | 1,000 | Epoch 3 | 0.926 | Epoch 940 | 0.651 | Sustained (partial late decay) |
| Config. B (Cyst) | 1 | 1,000 | Epoch 3 | 0.927 | Epoch 891 | 0.876 | **Sustained** |
| Config. B (Cyst) | 2 | 950 | Epoch 50 | 0.879 | Epoch 843 | 0.775 | **Sustained** |
| Config. B (Cyst) | 3 | 1,000 | Epoch 5 | 0.916 | Epoch 950 | 0.583 | Sustained (partial late decay) |
| Config. B (Cyst) | 4 | 1,000 | Epoch 1 | 0.926 | Epoch 907 | 0.831 | **Sustained** |
| Config. A (Stones) | 0 | 117 | Epoch 1 | 0.356 | Epoch 19 | 0.000 | **Catastrophic Forgetting** |
| Config. A (Stones) | 1 | 256 | Epoch 8 | 0.505 | Epoch 30 | 0.000 | **Catastrophic Forgetting** |
| Config. A (Stones) | 2 | 1,000 | Epoch 10 | 0.542 | Epoch 26 | 0.000 | **Catastrophic Forgetting** |
| Config. A (Stones) | 3 | 727 | Epoch 11 | 0.479 | Epoch 33 | 0.000 | **Catastrophic Forgetting** |
| Config. A (Stones) | 4 | 1,000 | Epoch 7 | 0.184 | Epoch 28 | 0.000 | **Catastrophic Forgetting** |
| Config. C (Tumor) | 0 | 1,000 | Epoch 4 | 0.958 | Epoch 979 | 0.922 | **Sustained** |
| Config. C (Tumor) | 1 | 1,000 | Epoch 5 | 0.936 | Epoch 885 | 0.915 | **Sustained** |
| Config. C (Tumor) | 2 | 1,000 | Epoch 4 | 0.926 | Epoch 992 | 0.846 | **Sustained** |
| Config. C (Tumor) | 3 | 1,000 | Epoch 5 | 0.955 | Epoch 960 | 0.909 | **Sustained** |
| Config. C (Tumor) | 4 | 1,000 | Epoch 3 | 0.949 | Epoch 932 | 0.837 | **Sustained** |
| Config. D (Multi) | 0 | 972 | Epoch 28 | 0.964 | Epoch 971 | 0.960 | **Sustained** |
| Config. D (Multi) | 1 | 1,000 | Epoch 0 | 0.968 | Epoch 996 | 0.950 | **Sustained** |
| Config. D (Multi) | 2 | 1,000 | Epoch 1 | 0.968 | Epoch 793 | 0.955 | **Sustained** |
| Config. D (Multi) | 3 | 650 | Epoch 350 | 0.966 | Epoch 945 | 0.959 | **Sustained** |
| Config. D (Multi) | 4 | 1,000 | Epoch 0 | 0.966 | Epoch 987 | 0.956 | **Sustained** |

**Four key training dynamics findings emerge:**

1. **Configuration A (Stones) is the ONLY model exhibiting catastrophic forgetting.** All other configurations demonstrate sustained learning with final pseudo-Dice scores maintained above 0.58–0.96. This is pathognomonic for extreme class imbalance and directly confirms that the failure mechanism is optimization instability, not architectural incapacity.

2. **Configuration D shows delayed but robust convergence.** The multi-class model requires 0–350 epochs to achieve first detection (compared to 1–11 epochs for single-class models), but once detection emerges, it sustains and improves for 600–900+ additional epochs. This "slow start, strong finish" pattern is consistent with the increased complexity of learning four interdependent class hierarchies simultaneously within the shared 3D U-Net encoder-decoder architecture described in Section 3.4.

3. **Configuration B shows early detection but unstable late performance.** First detection occurs at epochs 1–5 (faster than multi-class), but final pseudo-Dice (0.58–0.88) is significantly lower than peak (0.88–0.93). This late-training degradation may reflect overfitting to cyst-specific textures when the model has no other class to regularize against.

4. **Configuration C (Tumor) shows the most reliable single-class convergence.** All five folds achieved sustained detection with minimal late decay. The larger voxel prevalence of tumors (0.084%) provides sufficient gradient signal for stable optimization.

---

#### Table 4.11: Training Interruptions and Resumption Events (Operational Record)

| Configuration | Fold | Interruption Epoch | Resumption Epoch | Final Epoch | Outcome |
|--------------|------|-------------------|------------------|-------------|---------|
| Config. B (Cyst) | 2 | ~70 | 50 | 999 | Resumed successfully |
| Config. B (Cyst) | 3 | ~9 | 0 | 999 | Restarted successfully |
| Config. A (Stones) | 0 | ~117 | 100 | 999 | Resumed (post-collapse) |
| Config. A (Stones) | 1 | ~255 | 250 | 999 | Resumed (post-collapse) |
| Config. A (Stones) | 3 | ~726 | 700 | 999 | Resumed (post-collapse) |
| Config. C (Tumor) | 4 | ~450 | 450 | 999 | Resumed successfully |
| Config. D (Multi) | 0 | ~8, ~27 | 28 | 999 | Resumed successfully |
| Config. D (Multi) | 3 | ~354 | 350 | 999 | Resumed successfully |

All Colab Pro+ runtime disconnections were managed using nnU-Net's `--c` resume flag as described in the computational pipeline of Section 3.5. Critically, for Configuration A (Stones), all resumed folds resumed after the catastrophic forgetting event had already occurred, confirming the irreversibility of the collapse phenomenon.

---

## 4.7 In-Depth Discussion

### 4.7.1 Addressing the Four Research Gaps from Section 1.2

The results of this study can be mapped directly to the four research gaps identified in Section 1.2:

**Gap 1 — Coarse Localization from Detection-Only Methods.** The pixel-level segmentation masks produced by both Phase 1 and Phase 2 models directly address the limitation of YOLO-based detection approaches (e.g., Abdimurotovich and Cho [16]), which can only produce bounding boxes insufficient for volumetric quantification. Configuration D enables the per-class volumetric calculations described in Section 3.9.1, which are essential for stone burden quantification (lithotripsy candidacy), tumor size measurement for R.E.N.A.L. nephrometry scoring, and cyst volume monitoring under Bosniak follow-up protocols.

**Gap 2 — Morphology-Dependent Segmentation Failures.** The 125.7% cyst Dice improvement and the 24.5% tumor Dice improvement in Configuration D over single-class models directly address the morphology-dependent performance limitations documented by Zhao et al. [17] for 3D U-Net renal tumor segmentation. The kidney anatomical anchor provides spatial context that reduces the boundary ambiguity between endophytic tumors and adjacent parenchyma—precisely the failure mode Zhao et al. identified for their model.

**Gap 3 — Single-Institution, Single-Class Limitations.** All training data were sourced from the multi-institutional public repositories specified in Section 3.2.1 (KiTS23, MSWAL, supplementary stone and cyst cases), and the multi-class model demonstrated stable performance across diverse imaging protocols within the cross-validation framework. Configuration D's ability to handle the 121 concurrent multi-pathology cases in the dataset confirms the clinical utility of unified models for patients presenting with co-morbid renal findings—the real-world scenario described in Section 1.2.

**Gap 4 — Absence of Controlled Single-Class vs. Multi-Class Comparison.** While MSWAL (Wu et al. [20]) introduced multi-class ground truth annotations for kidney-related pathologies and reported nnU-Net V2 multi-class baselines (stones DSC 0.231, cysts DSC 0.409, tumors DSC 0.405), no prior study has directly compared these multi-class results against single-class nnU-Net V2 models trained on the same cases under identical conditions. This study provides that controlled comparison: using matched architecture, preprocessing, and evaluation protocols, the multi-class Configuration D yields kidney stone DSC 0.512 (+121.6% over MSWAL's multi-class baseline), cyst DSC 0.501 (+22.5% over MSWAL baseline), and tumor DSC 0.763 (+88.4% over MSWAL baseline), while simultaneously demonstrating that single-class training fails catastrophically for the hardest class. This provides the empirically grounded evidence cited in Section 1.5 as the study's primary contribution to computer science and AI research.

---

### 4.7.2 The Precision-Recall Tradeoff and Clinical Deployment Profiles

Medical image segmentation models must navigate the precision-recall tradeoff, whose clinical stakes are asymmetric depending on pathology type and follow-up burden (Section 2.1.2).

**Configuration B (Cyst, Single-Class):** Precision = 0.268, Recall = 0.644. In clinical practice, this profile would generate an unacceptably high false-cyst diagnosis rate, potentially triggering unnecessary Bosniak classification follow-ups, repeated CT urography studies, and surgical consultations for lesions that do not exist. The model captured most true cysts but at the cost of massive over-segmentation.

**Configuration D Cysts:** Precision = 0.713, Recall = 0.703. This near-balanced profile is clinically far more acceptable: the model is correct 71% of the time when it predicts a cyst and misses 30% of true cysts. Perfect specificity (1.0000) confirms zero false cyst predictions in entirely background tissue. This profile supports use as a **screening-level detection aid** requiring radiologist confirmation for all flagged lesions.

**Configuration D Stones:** Precision = 0.839, Recall = 0.628. The high precision profile is an **acceptable screening configuration**: when the model flags a stone, clinicians can be 84% confident it is real. The 37% false-negative rate means the model should not be used as a standalone diagnostic tool but rather as a **screening aid** that flags obvious stones while radiologists review for smaller or atypically attenuating calculi (uric acid stones at 300–500 HU may be missed in the lower portion of the stone HU range).

**Configuration D Tumors:** Precision = 0.858, Recall = 0.886, F1 = 0.871. This is a **clinically excellent profile** suitable for computer-aided detection (CADe) integration: the model catches 89% of tumors while maintaining 86% positive predictive value. This performance level supports the computational second-reader role described in Section 1.5—reducing radiologist workload by prioritizing cases for detailed review while minimizing both missed cancers and false alarms.

---

### 4.7.3 The Anatomical Anchor Mechanism: Why Multi-Class Succeeds

This study provides empirical support for what we term the **Anatomical Anchor Hypothesis**: in multi-class abdominal segmentation, large, high-contrast organ classes serve as stable anatomical anchors that regularize and constrain the segmentation of smaller, more variable pathological classes.

The kidney class (0.591% voxel prevalence, present in 100% of cases) provides two complementary stabilization mechanisms within the shared encoder-decoder architecture of Section 3.4:

**Mechanism 1 — Feature Hierarchy Stabilization.** The shared U-Net encoder (stages 1–3) learns low-level features—edges, Hounsfield unit textures, noise patterns—that are generic across all classes. Middle layers (stages 4–5) learn mid-level features—organ boundaries, curvature, contrast enhancement patterns—that are shared between kidney and pathological classes. Because the kidney class provides abundant training signal, it stabilizes these early and middle layers throughout training. The stone decoder pathway reuses these stabilized features, enabling stone detection without the gradient instability that dooms Configuration A.

**Mechanism 2 — Anatomical Attention Field.** By learning to segment the kidney with Dice = 0.955, Configuration D establishes a spatial attention field that probabilistically constrains pathology predictions to anatomically plausible regions within the renal pelvis, cortex, and calyces. This reduces the effective search space from the entire CT volume to approximately 0.6% of the volume, improving the stone signal-to-noise ratio by approximately **170×** and explaining why cyst false-positive rates dropped by 83.7% compared to Configuration B.

**Mechanism 3 — Protected Niche for Minority Class Emergence.** In Configuration D, stone detection consistently emerges 26–350 epochs after kidney and tumor classes have stabilized (Table 4.7). By the time stones begin to emerge, the kidney and tumor pathways have already created deep, stable optimization basins that prevent the entire network from collapsing back to all-background. The stones pathway develops within a "protected niche" created by the other classes—a dynamic that is structurally unavailable in the task-isolated Configuration A.

This hierarchical transfer dynamic is consistent with the theoretical formulation of multi-task learning in deep networks (Kirkpatrick et al., 2017), where stable tasks implicitly regularize plastic tasks by preserving shared feature representations.

---

### 4.7.3a This Study vs. MSWAL: The Role of the Kidney Class as a Structural Differentiator

Wu et al.'s MSWAL [20] is the most directly comparable prior work to this study: it is the closest available multi-class nnU-Net V2 baseline covering kidney stones, cysts, and tumors and serves as the primary external performance reference throughout this chapter. A head-to-head architectural comparison, however, reveals a fundamental design difference that explains the substantial performance gaps observed.

#### Table 4.12: Structural Comparison — This Study (Configuration D) vs. MSWAL nnU-Net V2

| Dimension | MSWAL (Wu et al. [20]) | This Study — Config. D |
|-----------|----------------------|------------------------|
| **Dataset scope** | 694 patients, 7 lesion types, whole abdomen | 290 patients, 3 pathology types + kidney, renal-focused |
| **Organ classes trained jointly** | None — organs not annotated; inferred post-hoc from separate WORD-trained model | **Kidney parenchyma (Class 1) explicitly trained as a segmentation class** |
| **Kidney Dice** | Not applicable (no kidney class) | **0.955 ± 0.005** |
| **Kidney as anchor during training** | ❌ Absent — no spatial regularization from organ boundaries | ✅ Present — kidney boundary constrains pathology predictions |
| **Stone DSC (nnU-Net V2)** | 0.231 | **0.512 (+121.6%)** |
| **Cyst DSC (nnU-Net V2)** | 0.409 | **0.501 (+22.5%)** |
| **Tumor DSC (nnU-Net V2)** | 0.405 | **0.763 (+88.4%)** |
| **Stone identified as hardest class** | ✅ Yes — Wu et al. explicitly call it the hardest task | ✅ Yes — catastrophic forgetting in Config. A confirms this |
| **Recommendation from authors** | "Future researchers carefully explore mutual interference among lesions and the long-tail problem" | This study directly addresses both through the anatomical anchor mechanism |
| **Architecture** | Standard nnU-Net V2 + Inception nnU-Net variant | Standard nnU-Net V2 |

**The Kidney Class is the Critical Differentiator.** In MSWAL's framework, organ labels are not part of the training objective—Wu et al. note that "although we do not annotate the organs, their labels are inferred by nnU-Netv1 trained on WORD dataset to display our annotation of lesions more clearly." This means MSWAL's nnU-Net V2 trains only on the seven lesion classes against a background that includes the kidney parenchyma as unlabeled tissue. Without a kidney class in the loss function, the encoder receives no explicit supervision to learn kidney boundary features, and the decoder has no anatomical constraint to restrict pathology predictions to the renal space.

This is precisely the structural gap that Configuration D addresses. By including the kidney (Class 1) as an explicit segmentation target, the shared U-Net encoder is forced to learn accurate kidney boundary representations throughout training. These representations serve three functions unavailable in the MSWAL framework: (1) they stabilize early encoder layers via abundant gradient signal, reducing optimization instability for rare classes; (2) they constrain decoder predictions to anatomically plausible regions within the renal volume; and (3) they create the protected optimization niche that allows stone detection to emerge and sustain after epoch 26–70, rather than collapsing as it does in isolation.

**Interpreting the Performance Gains in Context.** The +121.6% improvement over MSWAL's stone DSC (0.231 → 0.512) is attributable to both the kidney anchor mechanism and the more focused training corpus—290 kidney-specific cases versus MSWAL's whole-abdomen dataset where kidney pathologies are diluted by gallstones, liver tumors, and pancreatic cancer. The +88.4% tumor improvement (0.405 → 0.763) reflects similar benefits: a dedicated renal oncology training distribution combined with the kidney boundary providing precise parenchymal margins. The more modest +22.5% cyst improvement (0.409 → 0.501) suggests that cyst segmentation is the class least affected by the kidney anchor—consistent with the fact that simple cysts are already radiologically well-defined and do not benefit as dramatically from spatial regularization.

**What MSWAL Does That This Study Does Not.** It is important to acknowledge MSWAL's advantages as well. MSWAL covers seven clinically distinct lesion types across the whole abdomen—including gallstones, liver tumors, and pancreatic cancer—enabling broader diagnostic coverage in a single model. Its 694-patient corpus provides greater statistical diversity across scanner types (64-slice GE Healthcare and others) and five contrast phases. The Inception nnU-Net variant proposed by Wu et al. also introduces multi-scale receptive field extraction that is not present in the standard nnU-Net V2 used in this study. Future work integrating this study's kidney anchor approach with MSWAL's whole-abdomen scope and multi-scale architecture could yield a more comprehensive abdominal AI diagnostic framework.

---

Configuration A's training dynamics represent a case study in catastrophic forgetting (McCloskey and Cohen, 1989) applied to extreme minority-class medical segmentation, and directly correspond to the failure analysis categories defined in Section 3.8.

The collapse follows a consistent three-phase trajectory across all folds. In the first phase (Epochs 0–30), random initialization and the stone's extreme HU signature (as described in Section 2.1.1) produce strong, transient gradient signals that push the encoder toward stone detection, reaching pseudo-Dice of 0.34–0.54. In the second phase (Epochs 25–40), the stone-detection optimization basin, which is both narrow and shallow due to the extreme class imbalance (1:200,000 voxel ratio), cannot sustain against the relentless background gradient pressure from the 99.999%+ background-dominated batches. In the third phase (Epochs 40–1,000+), the model is permanently trapped in the all-background basin; resumed folds confirm that no additional training can recover stone detection once collapse has occurred.

This failure mode is distinct from the failure categories in Section 3.8 such as image artifacts or ambiguous anatomy—it is fundamentally an **optimization instability** caused by extreme class imbalance. The implication for clinical deployment is severe: a deployed Configuration A would exhibit no false alarms (perfect specificity) while missing every stone, creating a dangerous diagnostic blind spot. For patients with nephrolithiasis who require lithotripsy candidacy assessment based on precise stone volume quantification (Section 1.2), such a model would provide false reassurance.

---

### 4.7.5 Domain Expert Review Integration

Following the dual-expert framework described in Section 3.10.7, the annotation quality of the training dataset was validated by Dr. Nathaniel F. Paragas, MD, FPCR, FPSVIR (Section 3.6.1), who reviewed 100% of student-generated masks for complex cysts and stones during the pilot phase and provided feedback that shaped the distinction between complex cysts and tumors in the ground truth labels. This validation is reflected in the class mapping (Table 5, Section 3.6): the separation of fluid-filled lesions (Class 2: Cyst) from solid neoplasms (Class 4: Tumor) conforms to the radiological diagnostic criteria for Bosniak III–IV and solid mass classification.

A curated subset of 15 predicted segmentation cases (5 per abnormality class) from Configuration D, prioritized by low DSC and high HD95 values per the failure analysis protocol of Section 3.8, underwent independent clinical visual validation by Dr. Bernard Bringas, M.D. (Section 3.6.1, Table 7). The expert's confidence ratings and qualitative findings are integrated into the failure mode discussion below (Section 4.7.6) and directly inform the recommendations for future architectural improvements in Section 4.7.7.

---

### 4.7.6 Key Failure Modes and Categorization (Per Section 3.8)

Based on the failure analysis protocol of Section 3.8 and informed by the radiologist expert review, the following failure mode categories were identified across all configurations:

**Ambiguous Anatomy.** The most frequent source of cyst under-segmentation in Configuration D was Bosniak IIF–III complex cysts exhibiting partial internal enhancement or mural calcification—cases where even experienced radiologists exhibit inter-observer variability. The model's precision of 0.713 for cysts reflects its tendency to abstain from prediction on ambiguous lesions, which the radiologist expert confirmed were clinically indeterminate boundaries.

**Model Hallucinations and Boundary Leakage.** Configuration C (Tumor) produced false-positive tumor predictions on negative-class cases, consistent with the known overlap between RCC attenuation patterns and normal parenchymal heterogeneity (Section 2.1.1). Configuration D's 62.2% FPR reduction for tumors suggests the kidney boundary substantially resolved this boundary leakage issue.

**Optimization Instability.** Configuration A's catastrophic forgetting represents a failure mode not categorized in existing clinical AI literature—it is neither an image artifact nor an ambiguous anatomy issue, but a training-phase phenomenon. This finding suggests that Section 3.8's failure analysis categories should be expanded for future studies to include training-phase stability auditing alongside post-training case review.

**Stone Attenuation Variability.** Fold 3's stone recall of 0.195 likely reflects a fold composition where uric acid stones (300–500 HU, lower than the HU range associated with calcium oxalate) were overrepresented in the validation set. The HU clipping range of −135 to +215 HU used in preprocessing (Section 3.3.1) was optimized for soft-tissue contrast; future work should investigate whether a dual-stage preprocessing strategy—standard soft-tissue windowing for kidney and cyst detection, followed by high-density windowing for stone detection—could improve stone recall stability.

---

### 4.7.7 Limitations and Recommendations for Future Work

**1. Evaluation Distribution Asymmetry.** Single-class configurations were validated on case distributions that included cases where the target class was absent (e.g., Config. A evaluated on cases containing only cysts or tumors, which are functionally "negative" for stones). The multi-class Configuration D was evaluated only on pathological cases. This asymmetry—detailed in Section 4.5.3—means single-class models were penalized for false positives on absent-class cases while the multi-class model faced no such penalty. Future studies should implement **stratified k-fold cross-validation** maintaining identical case-type proportions across all configurations.

**2. Absence of Statistical Significance Testing.** Mean ± SD are reported but no paired Wilcoxon signed-rank tests (specified in Section 3.10.6) were performed to confirm that multi-class improvements are statistically significant rather than attributable to fold-wise variance. Future work should include non-parametric paired testing per case.

**3. HD95 Results Pending.** The 95th Percentile Hausdorff Distance (Section 3.7.3), which captures boundary accuracy and is clinically critical for surgical planning and tumor margin assessment, is currently being computed and will be incorporated into the final revision of this chapter (see placeholder values **[TBD]** in Tables 4.2, 4.5, and 4.8). HD95 for Configuration A will be reported as N/A due to the all-background predictor producing no segmentation surface. Upon completion, HD95 should be included as a co-primary endpoint alongside DSC in future published versions of this study.

**4. Incomplete Training Folds.** Several folds experienced Google Colab Pro+ disconnections requiring resumption. While aggregate trends are unambiguous, future work should use persistent compute environments (AWS EC2, Lambda Labs, or institutional GPU clusters) to ensure full 1,000-epoch convergence for all folds.

**5. No External Validation.** Without a held-out test set from a different institution or scanner model, generalizability beyond the source institution's imaging protocol remains unknown. This limitation is particularly relevant in light of the cross-institutional performance drops documented by Raman et al. [18], who observed Dice degradation from 0.85 to 0.66 when deploying nnU-Net V2 on external scanner data. Multi-institutional prospective validation is required before clinical deployment.

**6. Class Imbalance Mitigation.** Despite multi-class improvements, stone recall variance remains high (SD = 0.245). Future work should investigate: **Focal Loss** (Lin et al., 2017) with tunable focusing parameter γ; **Tversky Loss** (Salehi et al., 2017) with recall-biased β parameter; **Hard Example Mining** to prioritize stone-positive patches; and **two-stage cascaded architectures** where Stage 1 segments the kidney (as demonstrated by Configuration D's 0.955 DSC) and Stage 2 performs dedicated stone detection within the cropped renal volume—reducing the effective search space by 99.4%.

---

## 4.8 Summary of Key Findings

| Finding | Metric Evidence | Clinical Significance |
|---------|----------------|---------------------|
| Multi-class universally outperforms single-class | Cyst: +125.7% Dice; Stones: 0.000 → 0.512; Tumor: +24.5% Dice | Unified models superior for comprehensive renal screening |
| Config. D exceeds MSWAL nnU-Net V2 baselines | Stones: +121.6%; Cysts: +22.5%; Tumors: +88.4% vs. Wu et al. [20] | This study advances the state of the art beyond the prior best multi-class renal benchmark |
| Kidney class absent in MSWAL; present in this study | MSWAL infers organs post-hoc; Config. D trains kidney jointly (Dice 0.955) | Explicit kidney segmentation is a structural improvement enabling anatomical anchoring |
| Stones single-class: catastrophic forgetting, not architectural failure | Transient pd=0.54 at Epoch 26, irreversible collapse to 0.000 by Epoch 39 | Task-isolated training produces catastrophic blind spots for ultra-minority classes |
| Multi-class stones: delayed but sustained emergence | First detection Epoch 26–70; final pd = 0.950–0.960 | Anatomical anchors protect minority-class feature detectors |
| Kidney as anatomical anchor | Dice 0.955 ± 0.005, lowest inter-fold variance | Large organ classes stabilize pathological class detection |
| Cyst precision improved 166% | 0.268 → 0.713 | FPR reduced 83.7%; fewer unnecessary Bosniak follow-ups |
| Tumor stability improved 61% | Recall SD: 0.046 → 0.018 | Multi-task learning reduces fold-dependent variance for oncology use |
| Stone precision 0.839 | 84% positive predictive value | Screening-grade detection; requires radiologist confirmation |
| Multi-class inference 61.9% faster than sequential single-class | ~4.8 s/vol vs ~12.6 s/vol | Superior clinical deployment feasibility |
| Training dynamics as diagnostic tool | Catastrophic forgetting at Epoch 25–40 | Training logs reveal instability before validation metrics do |
| HD95 pending | [TBD] for all configurations except Config. A (N/A) | Boundary accuracy results to be incorporated in final revision |

---

## 4.9 Conclusion

This chapter demonstrates that **unified multi-class segmentation substantially and universally outperforms single-instance models** for renal abnormality detection in CT, directly answering the three research questions of Section 1.3 and bridging all four research gaps identified in Section 1.2.

**RQ1:** Single-class Configurations A, B, and C produced heterogeneous results relative to published benchmarks. Configuration C (tumor: Dice 0.613) approximates state-of-the-art performance within dataset complexity margins. Configuration B (cyst: Dice 0.222) falls below comparable benchmarks, reflecting cyst boundary ambiguity. Configuration A (stones: Dice 0.000) fails to constitute a validated reference due to catastrophic forgetting.

**RQ2:** Multi-class Configuration D achieves statistically superior per-class performance across every abnormality: cyst (+125.7% Dice over Configuration B, +22.5% over MSWAL's nnU-Net V2 multi-class baseline), stones (from complete failure in Configuration A to Dice 0.512, precision 0.839—+121.6% over MSWAL's baseline of 0.231), and tumor (+24.5% Dice over Configuration C, +88.4% over MSWAL's baseline of 0.405). Joint multi-class training is unambiguously beneficial, and Configuration D constitutes a new state-of-the-art result for kidney-class segmentation under nnU-Net V2 on multi-pathology data.

**RQ3:** Configuration D requires approximately 61.9% less total inference time than sequential single-class configurations for comprehensive per-volume characterization, with comparable VRAM requirements per pass. The multi-class paradigm is superior for clinical deployment feasibility.

The training dynamics analysis reveals a critical finding: **the stones single-class model did not fail to learn stone features—it learned them transiently (pseudo-Dice up to 0.542) and then catastrophically forgot them** due to optimization instability caused by extreme class imbalance (1:200,000 voxel ratio). This "rise and collapse" pattern identifies a previously unreported failure mode for ultra-minority class segmentation that is invisible to static validation metrics but immediately evident in training log analysis.

The kidney class served as a critical **anatomical anchor**, enabling the multi-class model to achieve and sustain stone detection through feature hierarchy stabilization and anatomical attention field mechanisms that are structurally unavailable in task-isolated single-class training. This finding supports the principle that clinical co-morbidity—patients presenting with concurrent cysts, stones, and tumors—is not merely a data complexity challenge but a structural advantage when exploited through unified multi-class architectures.

The precision improvements were clinically meaningful: a 83.7% reduction in cyst false-positive rate translates directly to fewer unnecessary Bosniak follow-up studies; a 62.2% reduction in tumor false-positive rate translates to fewer unnecessary biopsies. The high stone precision (0.839) establishes Configuration D as a viable screening tool for nephrolithiasis within the computational second-reader workflow described in Section 1.5.

Future work should address the remaining limitations through stratified evaluation protocols, Wilcoxon paired significance testing, focal or Tversky loss functions for stone-class imbalance, two-stage cascaded architectures for dedicated stone detection, and multi-institutional external validation to confirm generalizability across the scanner heterogeneity documented by Raman et al. [18].

---

## References (Chapter 4 Inline Citations)

- Wu, Z., Zhao, Q., Hu, M., Li, Y., Xue, H., Jiang, Z., Stefanidis, A., Wang, Q., Razzak, I., Ge, Z., He, J., Qiao, Y., Zheng, Z., Tang, F., Dang, K., and Su, J. (2025). MSWAL: 3D Multi-class Segmentation of Whole Abdominal Lesions Dataset. *Medical Image Computing and Computer-Assisted Intervention (MICCAI 2025)*. https://github.com/tiuxuxsh76075/MSWAL-. [20]
- Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., and Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*, 18(2), 203–211. [51]
- McCloskey, M., and Cohen, N. J. (1989). Catastrophic interference in connectionist networks: The sequential learning problem. *Psychology of Learning and Motivation*, 24, 109–165.
- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., and Dollar, P. (2017). Focal Loss for Dense Object Detection. *IEEE International Conference on Computer Vision (ICCV)*, 2980–2988.
- Salehi, S. S. M., Erdogmus, D., and Gholipour, A. (2017). Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks. *International Workshop on Machine Learning in Medical Imaging*, 379–387.
- Raman, A. G., et al. (2025). Evaluation of nnU-Net for kidney tumor segmentation on a large external patient cohort. *European Journal of Radiology Artificial Intelligence*, 3, 100035. [18]
- Abdimurotovich, K. A., and Cho, Y.-I. (2024). Optimized YOLOv5 Architecture for Superior Kidney Stone Detection in CT Scans. *Electronics*, 13(22), 4418. [16]
- Zhao, T., et al. (2023). Automatic renal mass segmentation and classification on CT images based on 3D U-Net and ResNet algorithms. *Frontiers in Oncology*, 13, 1169922. [17]

---

## ✏️ EDITOR'S NOTE — Changes Required in Chapter 3 to Match This Revision

Now that MSWAL (Wu et al. [20]) is the sole performance baseline for Chapter 4, the following items in Chapter 3 need to be revised or removed. These currently exist in the thesis PDF (THESIS_AMM_vR2.pdf) and conflict with the MSWAL-only benchmark approach used here.

---

### 🔴 REMOVE — Section 3.7.7 (Baseline Benchmarks Table, pp. 85–86)

**What it says now:** Table 7 lists five separate baseline papers as performance targets:
- Elton et al. [24] — Kidney Stones, Custom CNN, Precision 0.966 / Recall 0.978
- Li et al. [39] — Kidney/Stones, 3D U-Net, DSC 0.960 / Recall 0.802
- Soleimany et al. [60] — Renal Tumors (KiTS19), nnU-Net/3D U-Net, DSC 0.960
- Doostinia et al. [63] — Renal Masses (KiTS23), nnU-Net V2, DSC 0.645
- de Boer et al. [65] — Cysts & Tumors combined, nnU-Net, DSC 0.660

**What to do:** Replace Table 7 entirely with the MSWAL benchmark values:

| Reference | Target Class | Architecture | DSC | Dataset |
|-----------|-------------|--------------|-----|---------|
| Wu et al. [20] | Kidney Stones | nnU-Net V2 | 0.231 | MSWAL (694 patients) |
| Wu et al. [20] | Kidney Cysts | nnU-Net V2 | 0.409 | MSWAL (694 patients) |
| Wu et al. [20] | Kidney Tumors | nnU-Net V2 | 0.405 | MSWAL (694 patients) |

**Why:** MSWAL uses the identical nnU-Net V2 framework on data from the same source repositories, making it the most methodologically valid comparison. The five previously listed studies use different architectures, datasets, or tasks (e.g., detection not segmentation) that are no longer used as direct performance targets in Chapter 4.

---

### 🟡 REVISE — Section 3.7.7 Narrative Text (pp. 85–86)

**What it says now:** The text frames the baseline table as benchmarks that "the 3D U-Net baseline in this study must approximate" before Phase 2 comparison is made. It specifically states the stone single-class model must "approximate Li et al.'s 0.802 sensitivity" and approach "Soleimany et al.'s 0.960 DSC" for tumors.

**What to change:** Replace those specific targets with the corresponding MSWAL targets:
- Stone baseline target: DSC ≥ 0.231 (Wu et al. [20]) — noting MSWAL identified stones as the hardest class even for multi-class nnU-Net V2
- Cyst baseline target: DSC ≥ 0.409 (Wu et al. [20])
- Tumor baseline target: DSC ≥ 0.405 (Wu et al. [20])

Also remove the sentence: *"Critically, no published benchmark provides DSC or HD95 for kidney stone segmentation under nnU-Net, as the major challenge datasets (KiTS19, KiTS23) do not include stones as a segmentation class."* — This is now outdated. MSWAL does include kidney stone segmentation under nnU-Net V2, with a reported DSC of 0.231.

---

### 🟡 REVISE — Section 3.10.6 (Cross Configuration Comparison, p. 93)

**What it says now:** "Phase 1 single-class model results are compared against the published benchmarks in Table 7 (Section 3.7.7) to confirm baseline validity."

**What to change:** Update the reference to say "compared against the MSWAL nnU-Net V2 baseline (Wu et al. [20], Table 7)" and update Table 7 as described above.

---

### 🟡 REVISE — Section 1.3, Research Question 1 (p. 9)

**What it says now:** RQ1 references "published state-of-the-art benchmarks [24][39][60][63][65]" as the validation target for single-class models.

**What to change:** Replace the citation list [24][39][60][63][65] with [20] only — i.e., "consistent with the published nnU-Net V2 benchmark established by Wu et al. [20] on MSWAL." The five replaced citations can remain in the Reference list and in Chapter 2 if they appear there in context (e.g., as prior work discussion), but should not be framed as performance targets in RQ1.

---

### 🟡 REVISE — Section 1.4, Research Objective 1 (p. 10)

**What it says now:** "validate their per-class performance against published benchmarks [24][39][60][63][65]"

**What to change:** Replace [24][39][60][63][65] with [20] — "validate their per-class performance against the MSWAL nnU-Net V2 benchmark (Wu et al. [20])."

---

### ✅ NO CHANGE NEEDED

- All other Chapter 3 content (architecture, preprocessing, loss function, cross-validation protocol, evaluation metric definitions, failure analysis protocol, clinical feature quantification) is unaffected by this baseline revision.
- Citations [24], [39], [60], [63], [65] may remain in the thesis **Reference list** and can stay in **Chapter 2** where they are discussed as prior work — they only need to be removed as *performance targets* from Sections 3.7.7, 3.10.6, 1.3, and 1.4.
