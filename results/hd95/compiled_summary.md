# HD95, Diameter, and Density Results Summary

This document summarizes the evaluation metrics across all datasets and folds.

Metrics are extracted from the `hd95_results` folder.

For each dataset and label, statistics are computed for **All** cases, **Positive** cases (non-`neg_`), and **Negative** cases (`neg_`).

**Note:** HD95 (Hausdorff Distance 95th percentile), Diameter (ESD in mm), and Density (HU) are reported. Empty values are excluded from statistics.


## Dataset: Tumor

Total cases: 290

Folds: fold_0, fold_1, fold_2, fold_3, fold_4


### Label 1


#### All Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 270 | 48.92 | 3.04 | 87.86 | 0.0 | 657.44 |
| DIAMETER MM | 290 | 26.95 | 21.88 | 29.36 | 0.0 | 131.49 |
| DENSITY HU | 181 | 74.79 | 71.74 | 42.94 | -7.99 | 211.07 |

#### Positive Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 163 | 81.03 | 22.34 | 101.0 | 1.33 | 657.44 |
| DIAMETER MM | 165 | 44.58 | 38.14 | 26.34 | 0.0 | 131.49 |
| DENSITY HU | 163 | 78.35 | 75.35 | 41.92 | 2.84 | 211.07 |

#### Negative Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 107 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DIAMETER MM | 125 | 3.68 | 0.0 | 11.48 | 0.0 | 71.96 |
| DENSITY HU | 18 | 42.59 | 36.27 | 39.49 | -7.99 | 122.8 |

## Dataset: Stones

Total cases: 290

Folds: fold_0, fold_1, fold_2, fold_3, fold_4


### Label 1


#### All Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 215 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DIAMETER MM | 290 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DENSITY HU | - | - | - | - | - | - |

#### Positive Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | - | - | - | - | - | - |
| DIAMETER MM | 75 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DENSITY HU | - | - | - | - | - | - |

#### Negative Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 215 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DIAMETER MM | 215 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DENSITY HU | - | - | - | - | - | - |

## Dataset: Cyst

Total cases: 290

Folds: fold_0, fold_1, fold_2, fold_3, fold_4


### Label 1


#### All Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 174 | 128.57 | 127.28 | 76.67 | 0.0 | 336.98 |
| DIAMETER MM | 290 | 25.06 | 22.46 | 15.52 | 0.0 | 67.94 |
| DENSITY HU | 284 | 20.71 | 16.64 | 19.73 | -16.28 | 146.75 |

#### Positive Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 169 | 132.37 | 129.55 | 74.47 | 1.61 | 336.98 |
| DIAMETER MM | 170 | 29.06 | 27.42 | 15.01 | 0.0 | 67.94 |
| DENSITY HU | 169 | 20.39 | 17.94 | 14.89 | -14.72 | 76.23 |

#### Negative Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 5 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| DIAMETER MM | 120 | 19.39 | 17.16 | 14.48 | 0.0 | 62.37 |
| DENSITY HU | 115 | 21.18 | 13.62 | 25.29 | -16.28 | 146.75 |

## Dataset: MultiClass

Total cases: 290

Folds: fold_0, fold_1, fold_2, fold_3, fold_4


### Label 1


#### All Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 290 | 7.36 | 1.31 | 39.29 | 0.69 | 561.17 |
| DIAMETER MM | 290 | 85.9 | 85.88 | 8.95 | 54.96 | 112.25 |
| DENSITY HU | 290 | 95.2 | 102.2 | 49.01 | 7.8 | 224.66 |

#### Positive Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 290 | 7.36 | 1.31 | 39.29 | 0.69 | 561.17 |
| DIAMETER MM | 290 | 85.9 | 85.88 | 8.95 | 54.96 | 112.25 |
| DENSITY HU | 290 | 95.2 | 102.2 | 49.01 | 7.8 | 224.66 |

### Label 2


#### All Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 235 | 28.07 | 2.5 | 47.97 | 0.0 | 221.13 |
| DIAMETER MM | 290 | 13.97 | 10.56 | 14.36 | 0.0 | 62.98 |
| DENSITY HU | 211 | 27.16 | 21.89 | 20.98 | -11.78 | 132.2 |

#### Positive Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 235 | 28.07 | 2.5 | 47.97 | 0.0 | 221.13 |
| DIAMETER MM | 290 | 13.97 | 10.56 | 14.36 | 0.0 | 62.98 |
| DENSITY HU | 211 | 27.16 | 21.89 | 20.98 | -11.78 | 132.2 |

### Label 3


#### All Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 273 | 5.76 | 0.0 | 25.78 | 0.0 | 220.74 |
| DIAMETER MM | 290 | 2.06 | 0.0 | 4.56 | 0.0 | 28.42 |
| DENSITY HU | 70 | 330.87 | 279.89 | 207.82 | 74.94 | 997.9 |

#### Positive Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 273 | 5.76 | 0.0 | 25.78 | 0.0 | 220.74 |
| DIAMETER MM | 290 | 2.06 | 0.0 | 4.56 | 0.0 | 28.42 |
| DENSITY HU | 70 | 330.87 | 279.89 | 207.82 | 74.94 | 997.9 |

### Label 4


#### All Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 281 | 17.33 | 2.0 | 56.43 | 0.0 | 592.51 |
| DIAMETER MM | 290 | 23.66 | 19.01 | 28.41 | 0.0 | 132.75 |
| DENSITY HU | 168 | 84.01 | 79.15 | 40.37 | -2.34 | 217.45 |

#### Positive Cases

| Metric | Count | Mean | Median | Std | Min | Max |
|--------|-------|------|--------|-----|-----|-----|
| HD95 | 281 | 17.33 | 2.0 | 56.43 | 0.0 | 592.51 |
| DIAMETER MM | 290 | 23.66 | 19.01 | 28.41 | 0.0 | 132.75 |
| DENSITY HU | 168 | 84.01 | 79.15 | 40.37 | -2.34 | 217.45 |