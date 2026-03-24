# 🧠 Time-Dependent Survival Analysis for Brain Tumors with Explainable ML

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-TCGA-orange)
![XAI](https://img.shields.io/badge/XAI-SurvSHAP(t)-purple)

> Explainable survival analysis of brain tumors using Random Survival Forest (RSF) and Gradient Boosting Machine (GBM), with time-dependent feature attribution via SurvSHAP(t). Demonstrates how ML overcomes the proportional hazards assumption violated in Cox regression, using TCGA brain tumor data.

---

## Overview

Cox proportional hazards (PH) regression remains the standard in clinical survival analysis, but carries a fundamental constraint: it assumes each variable's effect on survival is **constant over time**. In brain tumor research, this assumption is routinely violated — the prognostic impact of tumor grade peaks in the early years and diminishes thereafter, while treatment effects may fade or evolve over the follow-up period.

This project addresses that limitation through a three-stage pipeline:

**Stage 1 — Establish the baseline and quantify its limits.**
A penalized Cox PH model is fitted on TCGA brain tumor data. Schoenfeld residual tests formally identify which variables violate the proportional hazards assumption (Age, Era_Before2005, Tx_None), providing a concrete benchmark for what Cox cannot model.

**Stage 2 — Train assumption-free ML survival models.**
Random Survival Forest (RSF) and Gradient Boosting Machine (GBM) are trained and tuned via 5-fold cross-validated grid search. Both models are evaluated against the Cox baseline on C-index, Integrated Brier Score (IBS), and time-dependent AUC at 1, 2, and 3 years.

**Stage 3 — Explain what the models learned, and when.**
A layered XAI pipeline moves from global to local to time-dependent explanations:
- **PFI + SHAP + ALE** reveal which features drive survival across the entire cohort
- **SHAP Waterfall + c-ICE** decompose predictions for individual high- and low-risk patients
- **SurvSHAP(t)** tracks how each feature's contribution evolves over time — directly answering the four clinical questions below and visually confirming the Cox violations detected in Stage 1

### Clinical Questions

| # | Question | Method |
|---|----------|--------|
| 1 | Which prognostic factors most critically determine brain tumor survival? | PFI + SHAP Summary + ALE |
| 2 | What drives the prediction difference between high- and low-risk patients? | SHAP Waterfall + c-ICE |
| 3 | Does the impact of tumor grade differ between short- and long-term survival? | SurvSHAP(t) |
| 4 | Does the treatment effect persist over time? | SurvSHAP(t) |


---

## Dataset

**Source**: [TCGA](https://portal.gdc.cancer.gov/) brain tumor cohort, accessed via GDC Data Portal.

Inclusion criteria: primary brain tumors with index date set to diagnosis. Patients with missing survival time, non-positive follow-up, or unclassifiable tumor grade were excluded. The final cohort consists of **1,163 patients** (659 events, 56.7%), split 80/20 into train (N=930) / test (N=233) with stratification by event status.

**Baseline characteristics (N=1,163):**

| Variable | Category | N (%) |
|----------|----------|-------|
| Age (years) | Mean ± SD | 51.0 ± 15.3 |
| | Median (IQR) | 52 (39–62) |
| Survival time (days) | Median (IQR) | 486 (232–876) |
| Sex | Male | 677 (58.2%) |
| Tumor grade | G4 | 651 (56.0%) |
| | G3 | 264 (22.7%) |
| | G2 | 248 (21.3%) |
| Treatment | Standard | 1,076 (92.5%) |
| | Single | 62 (5.3%) |
| | Supportive/None | 25 (2.1%) |
| Diagnosis era | After 2005 | 880 (75.7%) |
| Primary site | Brain NOS | 706 (60.7%) |
| | Cerebrum | 436 (37.5%) |
| Vital status | Dead | 659 (56.7%) |

**Median survival by grade:** G2 = 3,978 days / G3 = 1,578 days / G4 = 442 days (log-rank p < 0.001)

**Features used for modeling (14):** Age, Sex, Prior_cancer, Grade_G2/G3/G4, Site_BrainNOS/Cerebrum/Other, Era_After2005/Before2005, Tx_Single/Standard/None

> **Tx_None** (N=25, 2.1%) includes patients with no treatment record and those receiving supportive/palliative care only. This group is heterogeneous; SurvSHAP(t) results for this variable should be interpreted with caution.

Raw data (`clinical.tsv`) is not included due to TCGA data use policy. Download from the [GDC Portal](https://portal.gdc.cancer.gov/) and place in the project root.

---

## Methods

### Preprocessing — `NB01`
Grade assigned from WHO morphology codes: `9440/3`, `9442/3`, `9474/3` → G4; anaplastic histology → G3; `9400/3`, `9450/3`, `9382/3` → G2. Morphology codes `9680/3` and `8720/3` excluded. Age missing values (N=61, 5.2%) imputed with training-set median. Patients with unclassifiable grade (N=3) excluded. Final cohort: 1,163 patients.

### EDA — `NB02`
Kaplan-Meier curves stratified by grade, treatment, site, era, sex, and age group. Log-rank tests (all p < 0.05 except site). Baseline characteristics table (Table 1).

### Cox PH Baseline — `NB03`
Univariate Cox → VIF screening (all VIF < 10; Grade_G4 max VIF=5.89) → multivariate model with 10 features. Penalizer tuned via 5-fold Grid Search CV over `[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]`; optimal `penalizer=0.01` (CV C-index=0.783±0.016). Schoenfeld residual test used to assess proportional hazards assumption.

### ML Survival Models — `NB04`

| Model | Class | Best params | CV C-index |
|-------|-------|-------------|------------|
| RSF | `RandomSurvivalForest` | n_estimators=200, max_features=0.5, min_samples_leaf=15 | 0.785 ± 0.016 |
| GBM | `GradientBoostingSurvivalAnalysis` | n_estimators=200, lr=0.1, max_depth=2, subsample=0.6 | 0.785 ± 0.016 |

Grid search over 36 (RSF) and 81 (GBM) combinations × 5-fold CV. Evaluation: C-index, IBS, iAUC, AUC at 1/2/3 years.

### XAI Pipeline — `NB05`, `NB06`, `NB07`

| Notebook | Method | Scope | Clinical question |
|----------|--------|-------|-------------------|
| NB05 | Permutation Feature Importance (PFI, N=10 repeats) | Global | Q1 |
| NB05 | SHAP Summary (Beeswarm) | Global | Q1 |
| NB05 | ALE Plot | Global | Q1 |
| NB06 | SHAP Waterfall | Local | Q2 |
| NB06 | c-ICE Plot | Local | Q2 |
| NB07 | SurvSHAP(t) | Time-dependent | Q3, Q4 |

For SurvSHAP(t) (NB07), 8 representative patients computed: high-risk (idx=48, score=793.3), low-risk (idx=126, score=22.3), grade G4/G3/G2, and treatment-specific (Tx_Standard idx=7, Tx_Single idx=34, Tx_None idx=190). Treatment patients selected exclusively from those who actually received that treatment to avoid counterfactual instability.

---

## Results Summary

| Model | C-index (Test) | C-index (5-Fold CV) | IBS | iAUC | AUC 1yr | AUC 2yr | AUC 3yr |
|-------|----------------|---------------------|-----|------|---------|---------|---------|
| Cox PH | 0.800 | 0.783 ± 0.015 | 0.136 | 0.858 | 0.845 | 0.844 | 0.866 |
| RSF | 0.783 | 0.785 ± 0.016 | 0.138 | 0.840 | 0.829 | 0.838 | 0.863 |
| GBM | 0.787 | 0.785 ± 0.016 | 0.134 | 0.844 | 0.833 | 0.840 | 0.876 |

All models tuned via 5-fold stratified CV on the training set (N=930). Final metrics on held-out test set (N=233). Train–test C-index gaps of 0.011–0.017 confirm no overfitting.

**Optimal hyperparameters:**
- Cox PH: `penalizer = 0.01`
- RSF: `n_estimators=200, max_features=0.5, min_samples_leaf=15`
- GBM: `n_estimators=200, learning_rate=0.1, max_depth=2, subsample=0.6`

**Proportional hazards violations (Schoenfeld test, p < 0.05):** Age (p=0.0004), Era_Before2005 (p=0.0125), Tx_None (p=0.0001)

**Key findings from XAI:**
- Grade_G4 and Age are the top prognostic factors across all methods (PFI + SHAP + combined ranking consistent)
- Grade_G4 SHAP(t) peaks negatively within the first 900 days then recovers — time-varying effect confirmed, violating Cox assumption
- Grade_G2 shows sustained positive SHAP(t) over long-term follow-up — asymmetric pattern to G4
- Tx_Standard SHAP(t) peaks at ~500–700 days then declines — treatment effect is early-concentrated
- Non-flat SurvSHAP(t) curves for Age, Era_Before2005, and Tx_None visually confirm all three Schoenfeld violations

---

## Repository Structure

```
.
├── NB01_preprocessing.ipynb      # Data loading, cleaning, feature engineering
├── NB02_eda.ipynb                 # EDA, KM curves, Table 1
├── NB03_cox.ipynb                 # Cox PH baseline, VIF, Schoenfeld test
├── NB04_ml_models.ipynb           # RSF + GBM training, tuning, evaluation
├── NB05_global_xai.ipynb          # PFI + SHAP Summary + ALE
├── NB06_local_xai.ipynb           # SHAP Waterfall + c-ICE
├── NB07_survshap.ipynb            # SurvSHAP(t) time-dependent explanations
├── NB08_export.ipynb              # Paper figure export (300dpi PNG + TIFF)
├── outputs/
│   ├── models/                    # Serialized model and data pkl files
│   ├── paper_figures/             # Publication-ready figures
│   └── paper_tables/              # Publication-ready tables
├── clinical.tsv                   # Raw TCGA data (not included)
└── requirements.txt
```

Each notebook saves its outputs as `.pkl` files under `outputs/models/`, which subsequent notebooks load directly — no manual data passing required.

---

## Installation & Usage

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-survival-xai.git
cd brain-tumor-survival-xai
pip install -r requirements.txt
```

Place `clinical.tsv` (downloaded from GDC Portal) in the project root, then run notebooks in order `NB01` → `NB08`.

**Core dependencies:**

| Package | Version | Role |
|---------|---------|------|
| scikit-survival | ≥ 0.21 | RSF, GBM, C-index, IBS, AUC |
| survshap | 0.4.2 | SurvSHAP(t) |
| shap | ≥ 0.42 | SHAP values, Waterfall |
| lifelines | ≥ 0.27 | Cox PH, KM curves |
| statsmodels | ≥ 0.14 | VIF, LOWESS, Schoenfeld |

---

## Implementation Notes

- **SurvSHAP(t)**: `calculation_method='sampling'` used throughout. `treeshap` removed due to incompatibility with survshap 0.4.2. B=50 perturbations recommended (default in NB07).
- **Treatment SHAP(t)**: Each treatment variable's time-dependent SHAP is computed only for patients who actually received that treatment (median risk score within group). Using a mixed-treatment patient for Tx_None SHAP produces unstable counterfactual estimates — this is the corrected design.
- **Tx_None RSF instability**: RSF produces numerically unstable SHAP(t) for Tx_None due to group heterogeneity. Y-axis clipped to 1st–99th percentile in Figure 21. GBM results are preferred for this variable.

---

## Citation

```bibtex
@article{'',
  title   = {Time-Dependent Survival Analysis for Brain Tumors with Explainable ML},
  author  = {Sunwoo Jung},
  journal = {''},
  year    = {2026}
}
```
> Soon to be filled after final modification.
---

## License

MIT License — see [LICENSE](LICENSE) for details.
