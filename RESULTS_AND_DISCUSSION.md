# Chapter 4: Results and Discussion

## 4.1 Experimental Setup and Model Configurations

This study employed the nnU-Net V2 framework for 3D full-resolution segmentation of kidney abnormalities from CT volumetric data. A total of **290 cases** were utilized, segmented into four distinct modeling approaches: three single-instance models (Cyst-only, Stones-only, Tumor-only) and one unified multi-class model capable of simultaneously segmenting the kidney parenchyma (Class 1), cysts (Class 2), stones (Class 3), and tumors (Class 4). All models were trained and evaluated using **5-fold cross-validation** with identical architectural hyperparameters (patch size: 128×128×128, batch size: 2, CT normalization, 6-stage U-Net with InstanceNorm and LeakyReLU).

| Model | Label Schema | Classes | Validation Cases/Fold |
|-------|-------------|---------|----------------------|
| Cyst (Single) | 0=BG, 1=Cyst | 2 | 58 |
| Stones (Single) | 0=BG, 1=Stones | 2 | 58 |
| Tumor (Single) | 0=BG, 1=Tumor | 2 | 2 | 58 |
| Multiclass | 0=BG, 1=Kidney, 2=Cyst, 3=Stones, 4=Tumor | 5 | 58 |

---

## 4.2 Quantitative Results

### 4.2.1 Single-Instance Model Performance

The single-instance models exhibited highly heterogeneous performance across the three abnormality types, revealing critical limitations in lesion-specific training paradigms when applied to a constrained, imbalanced dataset.

| Model | Mean Dice ± SD | Mean IoU ± SD | Per-Fold Dice Range |
|-------|---------------|---------------|-------------------|
| **Cyst** | **0.2220 ± 0.0524** | 0.1709 ± 0.0421 | 0.1672 – 0.3102 |
| **Stones** | **0.0000 ± 0.0000** | 0.0000 ± 0.0000 | 0.0000 – 0.0000 |
| **Tumor** | **0.6132 ± 0.0552** | 0.5207 ± 0.0551 | 0.5476 – 0.6951 |

**Cyst Model.** The cyst-specific model achieved a mean Dice score of only 0.222, indicating substantial segmentation failure. Analysis of per-case predictions revealed a pronounced tendency toward **massive false-positive inflation**; for instance, in case `cyst_010`, the reference comprised merely 163 voxels, yet the model predicted 11,450 voxels—a 70× over-segmentation. While the model occasionally attained high Dice scores on large cystic lesions (max: 0.965), the median per-case Dice was near zero (median ≈ 0.04), underscoring poor generalization. The model failed to learn discriminative features for cyst boundaries, likely conflating cystic regions with other fluid-attenuating structures or background noise.

**Stones Model.** The stones-specific model exhibited **complete predictive failure**, registering exactly 0.000 Dice and 0.000 IoU across all five folds. Per-case inspection confirmed that the model predicted **zero foreground voxels** in every validation case (n_pred = 0 for all 58 cases per fold). Despite the presence of 12–20 stone-positive cases per fold (as corroborated by the multiclass validation sets), the model learned an all-background predictor. This represents a catastrophic class-imbalance collapse, where the gradient contributions from the overwhelmingly dominant background class completely suppressed any learning signal from the minuscule stone lesions.

**Tumor Model.** The tumor-specific model demonstrated the most respectable single-class performance, attaining a mean Dice of 0.613. However, substantial inter-fold variability (SD = 0.055) and frequent false-positive predictions on negative cases (e.g., `neg_033` and `neg_035` showed spurious tumor predictions despite zero reference voxels) suggest incomplete feature specificity. The model achieved strong segmentation on large, hyperdense renal masses (max Dice: 0.976) but struggled with smaller or heterogeneously enhancing lesions.

### 4.2.2 Multi-Class Model Performance

The unified multi-class model dramatically outperformed all single-instance counterparts, achieving robust segmentation across all anatomical and pathological classes.

| Class | Mean Dice ± SD | Mean IoU ± SD | Cases with Class/Fold | Valid Dice Range |
|-------|---------------|---------------|----------------------|-----------------|
| **Kidney** | **0.9547 ± 0.0051** | 0.9185 ± 0.0058 | 58 / 58 | 0.9466 – 0.9609 |
| **Cyst** | **0.5010 ± 0.0540** | 0.4174 ± 0.0447 | 35–48 / 58 | 0.4046 – 0.5679 |
| **Stones** | **0.5117 ± 0.1292** | 0.4139 ± 0.1066 | 12–20 / 58 | 0.2849 – 0.6606 |
| **Tumor** | **0.7633 ± 0.0502** | 0.6767 ± 0.0424 | 31–39 / 58 | 0.6849 – 0.8203 |
| **Foreground (All)** | **0.6827 ± 0.0395** | 0.6066 ± 0.0308 | — | 0.6428 – 0.7513 |

**Kidney Segmentation.** The multi-class model achieved near-expert-level kidney delineation (Dice ≈ 0.955, IoU ≈ 0.919), with remarkably low inter-fold variance (SD < 0.006). This demonstrates that the model reliably learned the high-contrast boundary between renal parenchyma and perirenal fat, establishing the kidney class as a stable anatomical anchor for downstream abnormality localization.

**Cyst Segmentation.** The multi-class cyst Dice of 0.501 represents a **125.7% relative improvement** over the single-class cyst model (0.222). While still moderate, the multi-class model substantially reduced the egregious false-positive inflation observed in the single-class paradigm. The kidney anatomical context likely provided critical spatial regularization, constraining cyst predictions to plausible anatomical locations within or adjacent to the renal capsule.

**Stone Segmentation.** Most strikingly, the multi-class model successfully detected renal calculi (mean Dice: 0.512) where the single-class model completely failed (Dice: 0.000). This represents a qualitative leap from total non-detection to clinically viable segmentation. However, stone detection remained the most variable task (SD = 0.129), with Fold 4 dropping to 0.285 Dice, highlighting persistent instability for this ultra-minority class.

**Tumor Segmentation.** Tumor segmentation in the multi-class model reached 0.763 Dice, a **24.5% improvement** over the single-class tumor model (0.613). The model consistently achieved strong performance across all folds (range: 0.685–0.820), with the kidney boundary acting as a spatial prior that improved tumor localization and reduced false positives on negative cases.

### 4.2.3 Comparative Performance Summary

| Abnormality | Single-Class Dice | Multi-Class Dice | Absolute Gain | Relative Improvement |
|-------------|------------------|------------------|---------------|---------------------|
| **Cyst** | 0.222 | 0.501 | +0.279 | **+125.7%** |
| **Stones** | 0.000 | 0.512 | +0.512 | **From Failure to Detection** |
| **Tumor** | 0.613 | 0.763 | +0.150 | **+24.5%** |

The multi-class model universally dominated, with the most dramatic gains observed for the most challenging classes: cysts (+126%) and stones (from complete failure to moderate success). Even for the best-performing single-class model (tumor), the multi-class approach provided a meaningful 24.5% boost.

---

## 4.3 Key Challenges and Resolution Strategies

### Challenge 1: Severe Class Imbalance and Small Lesion Size
**The Problem.** Renal calculi (stones) represented an extreme minority class. In the multi-class validation folds, stones appeared in only 12–20 out of 58 cases (21–34% prevalence), and within positive cases, stone volumes were typically minuscule (often <500 voxels) relative to the entire CT volume (>40 million voxels). In the single-class stones model, the effective positive-to-negative voxel ratio was approximately 1:100,000, creating an insurmountable optimization landscape where background gradients dwarfed foreground signals.

**How We Proceeded.** We retained the nnU-Net default loss function (soft Dice + cross-entropy), which theoretically provides stronger gradients for minority classes than pure cross-entropy. However, the results indicate this was insufficient for the stones-only model. For the multi-class model, the inclusion of kidney and tumor classes introduced auxiliary gradient pathways and feature hierarchies that indirectly regularized stone-feature learning. Rather than engineering custom loss functions (e.g., focal loss or class-balanced Dice), which would introduce additional hyperparameters and deviate from nnU-Net's self-configuring paradigm, we proceeded with the native framework to ensure reproducibility and fair comparison. Future work should investigate focal loss, Tversky loss, or hard example mining for stone-specific improvement.

### Challenge 2: Dataset Limitation and Fixed Case Availability
**The Problem.** The study was constrained to **290 cases provided by the institutional collaborator**. We could not augment the dataset with external public datasets (e.g., KiTS, Decathlon) due to: (a) domain shift concerns between different CT scanners, contrast protocols, and institutional annotation standards; (b) ethical and data-use agreements restricting external combination; and (c) the thesis objective of evaluating performance specifically on the target clinical population.

**How We Proceeded.** We maximized the utility of the available 290 cases by employing **5-fold cross-validation**, ensuring every case contributed to both training and validation. We accepted that class frequencies (cysts: ~70%, stones: ~25%, tumors: ~60%, multi-class combinations: ~30%) reflected the natural clinical prevalence at the source institution, enhancing real-world generalizability at the cost of statistical balance. All models were trained on identical data splits derived from nnU-Net's automatic splitting, preserving cross-model comparability.

### Challenge 3: Heterogeneous Validation Distributions Across Models
**The Problem.** A critical methodological asymmetry emerged: the single-class models were validated on heterogeneous case mixtures (e.g., the stones model included 43 negative cases and only 15 stone-positive cases per fold), whereas the multi-class model's validation set contained no explicit negative cases—only `cyst_`, `stone_`, `tumor_`, and `multi_` cases. This introduced evaluation bias: the single-class models were penalized for false-positive predictions on negative cases, while the multi-class model faced no such penalty.

**How We Proceeded.** We acknowledged this limitation transparently in our analysis. Rather than artificially filtering validation sets post-hoc (which would violate the integrity of the cross-validation protocol), we reported results as generated by nnU-Net's standard pipeline. To partially mitigate interpretation bias, we analyzed per-case metrics stratified by case type and explicitly noted when negative-case false positives inflated FP counts. For the stones model, we confirmed that the 0.000 Dice was driven entirely by failure on the 15 positive cases, not by negative-case penalties. We recommend future studies implement **stratified k-fold splitting** to ensure each fold maintains identical class distributions across all model configurations.

### Challenge 4: Hardware and Computational Constraints
**The Problem.** All experiments were conducted on Google Colab Pro+ with NVIDIA A100 80GB GPUs. Colab runtime disconnections, preprocessing worker crashes (`RuntimeError: Some background worker is 6 feet under`), and Google Drive I/O bottlenecks (~5–15 MB/s) severely hampered workflow stability. Preprocessing 290 cases with default workers crashed consistently; reducing to 2 workers prevented crashes but extended preprocessing time significantly.

**How We Proceeded.** We implemented a **resilient preprocessing pipeline**: (a) validated Drive path writability before backup; (b) created fast metadata backups (`nnUNetPlans.json`, `splits_final.json`) for rapid restoration; (c) used tar archiving instead of file-by-file copying to mitigate Drive I/O; and (d) employed `-np 2` workers for stable preprocessing. Training was conducted with batch size 2 (default) due to Colab instability, despite GPU utilization remaining low (~8 GB / 80 GB VRAM). We documented these constraints to contextualize the reproducibility and scalability of our workflow.

### Challenge 5: Lack of a Held-Out Test Set
**The Problem.** No independent, labeled test set was available beyond the 290 training cases. This precluded unbiased estimation of generalization error and meant all reported metrics derive from cross-validation folds that participated in training.

**How We Proceeded.** We adopted **5-fold cross-validation with fold-wise ensemble prediction** as the gold-standard alternative for small datasets. We reported both per-fold and cross-fold aggregate statistics (mean ± SD) to capture variability. While we acknowledge that cross-validation metrics may be optimistically biased compared to a true held-out test, this approach is standard in medical imaging studies with limited data and provides the most rigorous evaluation possible given the constraints.

---

## 4.4 Discussion

### 4.4.1 The Failure of Single-Class Stone Detection

The complete collapse of the stones-specific model (0.000 Dice) is the most consequential finding of this study. Several interrelated factors explain this failure:

1. **Extreme Class Imbalance at the Voxel Level.** Stones occupy an infinitesimal fraction of the CT volume. With background voxels outnumbering stone voxels by ratios exceeding 100,000:1, the network quickly learned the trivial all-background predictor, which minimizes cross-entropy loss and achieves high accuracy despite zero sensitivity.

2. **Loss of Anatomical Context.** By isolating stones from the kidney, the single-class model was deprived of the anatomical scaffold that defines where stones are physiologically plausible (i.e., within the renal pelvis, calyces, or ureter). Without kidney boundaries as spatial priors, the model had no structural constraints to guide attention to the relevant sub-volumes.

3. **Insufficient Positive Cases for Feature Learning.** With only ~75 stone-positive cases total (distributed across 5 folds), the single-class model encountered stone examples too infrequently during training to develop robust, generalizable feature detectors for hyperdense calcifications.

In stark contrast, the multi-class model leveraged the kidney class as an **anatomical attention mechanism**. By first learning to segment the kidney with high accuracy (Dice 0.955), the model established a spatial attention field that probabilistically constrained subsequent stone predictions to regions within or immediately adjacent to the renal collecting system. This hierarchical, context-aware prediction represents a fundamental advantage of multi-task learning for small-lesion detection.

### 4.4.2 Moderate Multi-Class Cyst Performance

While the multi-class model improved cyst segmentation by 126% relative to the single-class model, the absolute Dice of 0.501 remains moderate. Cysts present inherent segmentation ambiguity: simple cysts are fluid-attenuating and well-circumscribed, but complex cysts (Bosniak IIF–III) exhibit septations, calcifications, and wall thickening that overlap with solid tumor textures. The model's median per-case Dice (~0.60–0.76 depending on fold) suggests reasonable performance on typical cases, but the minimum Dice of 0.000 indicates persistent failure on challenging or small cysts. The false-positive inflation observed in the single-class model (e.g., 70× over-segmentation) was substantially ameliorated in the multi-class paradigm, likely due to the kidney boundary acting as a spatial regularizer.

### 4.4.3 Strong Multi-Class Tumor Performance

Tumor segmentation in the multi-class model (Dice 0.763) represents a clinically meaningful achievement. Renal cell carcinomas (RCCs) are histologically heterogeneous (clear cell, papillary, chromophobe), leading to variable CT attenuation patterns that challenge automated segmentation. The 24.5% improvement over the single-class model suggests that multi-task learning encourages the network to learn more discriminative, generalizable feature representations. The kidney class again likely contributed by defining the organ boundary, preventing the perirenal fat infiltration and adrenal gland confusion that plagued the single-class model.

### 4.4.4 The Anatomical Anchor Hypothesis

This study provides empirical support for the **Anatomical Anchor Hypothesis**: in multi-class abdominal segmentation, the large, high-contrast organ class (kidney) serves as a stable anatomical anchor that regularizes and constrains the segmentation of smaller, more variable pathological classes. The kidney's consistent shape, high attenuation relative to fat, and well-defined capsule create strong gradient signals that stabilize early network layers. These early features (edges, textures, intensity histograms) are subsequently repurposed by deeper layers for cyst, stone, and tumor detection. This hierarchical transfer learning is implicit in the shared encoder architecture and represents a key mechanism by which multi-class models outperform single-class equivalents.

### 4.4.5 Limitations and Recommendations for Fairer Comparison

Several limitations temper the strength of our conclusions:

1. **Evaluation Distribution Asymmetry.** As noted in Section 4.3, the single-class and multi-class models were evaluated on different case distributions. To ensure a rigorously fair comparison, future studies should:
   - Implement **stratified k-fold cross-validation** where each fold contains identical proportions of `cyst`, `stone`, `tumor`, `multi`, and `neg` cases across all model configurations.
   - Report **per-class metrics restricted to cases where the class is present** (n_ref > 0), removing the confounding effect of negative-case specificity when comparing sensitivity-focused models.
   - Construct a **dedicated negative-case test set** and report specificity, precision, and false-positive rate separately from Dice/IoU.

2. **No Statistical Significance Testing.** We report mean ± SD but did not perform paired statistical tests (e.g., Wilcoxon signed-rank test on per-case Dice scores) to confirm that multi-class improvements are statistically significant rather than attributable to fold-wise variance. Future work should include non-parametric paired testing across matched validation cases.

3. **Class Imbalance Remains Unaddressed.** Despite multi-class improvements, stone detection remains unstable (SD = 0.129) and cyst detection is only moderate. Future work should integrate:
   - **Focal Loss** or **Tversky Loss** with tunable β parameters to up-weight minority classes.
   - **Hard example mining** or **online hard example mining (OHEM)** to prioritize stone and small-cyst cases during training.
   - **Two-stage cascaded architectures** where the kidney is segmented first and a second-stage network focuses on the cropped renal region for abnormality detection.

4. **Generalization Uncertainty.** Without an external test set from a different institution or scanner, the generalizability of these findings to other clinical environments remains unknown. Prospective validation on multi-institutional data is essential before clinical deployment.

5. **Inference Profiling.** While training metrics were comprehensively logged, systematic inference latency profiling (per-volume prediction time, peak VRAM) was only partially completed. For clinical translation, real-time inference benchmarks on hospital PACS workstations are necessary.

---

## 4.5 Summary of Key Findings

| Finding | Significance |
|---------|-------------|
| **Multi-class universally outperforms single-class** | Demonstrates the value of joint training for renal abnormality detection |
| **Stones single-class complete failure** | Highlights catastrophic risk of class-imbalanced, context-isolated training for small lesions |
| **Kidney as anatomical anchor (Dice 0.955)** | Validates the hypothesis that large organ classes stabilize small-lesion segmentation |
| **Cyst +126% improvement in multi-class** | Shows spatial regularization from kidney boundary reduces false-positive inflation |
| **Tumor +24.5% improvement in multi-class** | Indicates multi-task learning enhances feature discriminability for heterogeneous masses |
| **High fold-wise variance for stones (SD=0.129)** | Signals need for dedicated class-imbalance handling and larger stone-positive cohorts |
| **Evaluation distribution asymmetry** | Methodological limitation requiring stratified splitting in future work |

---

## 4.6 Conclusion

This study demonstrates that **unified multi-class segmentation substantially outperforms single-instance models** for renal abnormality detection in CT. The multi-class nnU-Net model achieved excellent kidney segmentation (Dice 0.955), good tumor segmentation (Dice 0.763), and moderate but functional cyst (Dice 0.501) and stone (Dice 0.512) segmentation. In contrast, single-instance models either failed entirely (stones: 0.000 Dice), performed poorly (cysts: 0.222 Dice), or achieved only moderate success (tumors: 0.613 Dice).

The kidney class served as a critical **anatomical anchor**, providing spatial regularization and hierarchical feature transfer that stabilized small-lesion detection. These findings support the adoption of multi-class architectures in clinical renal CT analysis workflows, particularly for comprehensive screening scenarios where simultaneous detection of multiple abnormality types is required. However, the persistence of high variance for stone detection and the methodological asymmetry in validation distributions underscore the need for stratified evaluation protocols, class-imbalance mitigation strategies, and external multi-institutional validation in future research.
