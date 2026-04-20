# 🧠 Time-Dependent Survival Analysis for Brain Tumors with XAI

![Python](https://img.shields.io/badge/Python-3.13.9-blue)
![lifelines](https://img.shields.io/badge/lifelines-0.30.3-blue)
![scikit--survival](https://img.shields.io/badge/scikit--survival-0.27.0-blue)
![SHAP](https://img.shields.io/badge/SHAP-0.51.0-orange)
![SurvSHAP](https://img.shields.io/badge/SurvSHAP(t)-0.4.2-purple)

> Survival analysis of brain tumor patients (n=1,163) comparing Cox PH, Random Survival Forest (RSF), and Gradient Boosting Machine (GBM). Applies multi-layered XAI methods — PFI, SHAP, ALE, c-ICE, and SurvSHAP(t) — to provide time-dependent feature attribution and clinically interpretable prognostic insights. Demonstrates how ML complements Cox regression where the proportional hazards assumption is violated.

---

## Overview

Cox proportional hazards (PH) regression remains the standard in clinical survival analysis, but carries a fundamental constraint: it assumes each variable's effect on survival is **constant over time**. In brain tumor research, this assumption is routinely violated — the prognostic impact of tumor grade peaks in the early years and diminishes thereafter, while treatment effects may fade or evolve over the follow-up period.

This project addresses that limitation through a three-stage pipeline:

**Stage 1 — Establish the baseline and quantify its limits.**
A penalized Cox PH model is fitted on brain tumor data. Schoenfeld residual tests formally identify which variables violate the proportional hazards assumption (Age, Era_Before2005, Tx_None), providing a concrete benchmark for what Cox cannot model.

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

**Source**: (https://portal.gdc.cancer.gov/) brain tumor cohort, accessed via GDC Data Portal.

**Inclusion criteria:** Primary brain tumors with index date set to diagnosis (`cases.index_date = 'Diagnosis'`).

**Exclusion criteria:** Patients with survival time ≤ 0 days, missing survival variables, unclassifiable tumor grade, or non-brain tumor histology (morphology codes 9680/3, 8720/3).

The final cohort consists of **1,163 patients** (659 events, 56.7%), split 80/20 into train (N=930) / test (N=233) with stratification by event status.

**Baseline characteristics (N=1,163):**

| Variable | Category | N (%) |
|----------|----------|-------|
| Age (years) | Mean ± SD | 51.0 ± 15.3 |
| | Median (IQR) | 52 (39–62) |
| Survival time (days) | Mean ± SD | 726.2 ± 810.5 |
| | Median (IQR) | 486 (232–876) |
| Sex | Male | 677 (58.2%) |
| | Female | 486 (41.8%) |
| Tumor grade | G4 | 651 (56.0%) |
| | G3 | 264 (22.7%) |
| | G2 | 248 (21.3%) |
| Treatment | Standard | 1,076 (92.5%) |
| | Single | 62 (5.3%) |
| | Supportive/None | 25 (2.1%) |
| Diagnosis era | After 2005 | 880 (75.7%) |
| | Before 2005 | 283 (24.3%) |
| Primary site | Brain NOS | 706 (60.7%) |
| | Cerebrum | 436 (37.5%) |
| | Other Specific | 21 (1.8%) |
| Prior malignancy | No | 1,115 (95.9%) |
| | Yes | 48 (4.1%) |
| Vital status | Dead | 659 (56.7%) |
| | Alive (censored) | 504 (43.3%) |

**Median survival by grade:** G2 = 3,978 days / G3 = 1,578 days / G4 = 442 days (log-rank p < 0.001)

**Features used for modeling (14):** Age, Sex, Prior_cancer, Grade_G2/G3/G4, Site_BrainNOS/Cerebrum/Other, Era_After2005/Before2005, Tx_Single/Standard/None

> **Note:** Age missing values (< 5%) were imputed with training-set median. `Tx_None` (N=25, 2.1%) includes patients with no treatment record and those receiving supportive/palliative care only. This group is heterogeneous; SurvSHAP(t) results for this variable should be interpreted with caution.

Raw data (`clinical.tsv`) is not included due to TCGA data use policy. Download from the [GDC Portal](https://portal.gdc.cancer.gov/) and place in the project root.

---

## Methods

### Preprocessing — `NB01`
Grade assigned from WHO morphology codes: `9440/3`, `9442/3`, `9474/3` → G4; anaplastic histology → G3; `9400/3`, `9450/3`, `9382/3` → G2. Morphology codes `9680/3` and `8720/3` excluded. Age missing values imputed with training-set median. Patients with unclassifiable grade excluded. Final cohort: 1,163 patients.

### EDA — `NB02`
Kaplan-Meier curves stratified by grade, treatment, site, era, sex, and age group. Log-rank tests (all p < 0.05). Baseline characteristics table (Table 1).

### Cox PH Baseline — `NB03`
Univariate Cox → VIF screening (all VIF < 10; Grade_G4 max VIF = 5.89) → multivariate model with 10 features. Penalizer tuned via 5-fold Grid Search CV over `[0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]`; optimal `penalizer = 0.01` (CV C-index = 0.783 ± 0.016). Schoenfeld residual test used to assess proportional hazards assumption.

**Independent prognostic factors (multivariable Cox):**

| Variable | HR | 95% CI | p-value |
|----------|-----|--------|---------|
| Grade G4 | 5.987 | 3.643–9.840 | < 0.001 |
| Grade G3 | 2.547 | 1.718–3.777 | < 0.001 |
| Tx_Single | 2.188 | 1.570–3.051 | < 0.001 |
| Age | 1.042 | 1.034–1.049 | < 0.001 |
| Sex | 1.237 | 1.037–1.475 | 0.018 |

**Proportional hazards violations (Schoenfeld test, p < 0.05):** Age (p = 0.0004), Era_Before2005 (p = 0.0125), Tx_None (p = 0.0001)

### ML Survival Models — `NB04`

| Model | Class | Best params | CV C-index |
|-------|-------|-------------|------------|
| RSF | `RandomSurvivalForest` | n_estimators=200, max_features=0.5, min_samples_leaf=15 | 0.785 ± 0.016 |
| GBM | `GradientBoostingSurvivalAnalysis` | n_estimators=200, lr=0.1, max_depth=2, subsample=0.6 | 0.785 ± 0.016 |

Grid search over 36 (RSF) and 81 (GBM) combinations × 5-fold CV.

### XAI Pipeline — `NB05`, `NB06`, `NB07`

| Notebook | Method | Scope | Clinical question |
|----------|--------|-------|-------------------|
| NB05 | Permutation Feature Importance (PFI, N=10 repeats) | Global | Q1 |
| NB05 | SHAP Summary (Beeswarm) | Global | Q1 |
| NB05 | ALE Plot | Global | Q1 |
| NB06 | SHAP Waterfall | Local | Q2 |
| NB06 | c-ICE Plot | Local | Q2 |
| NB07 | SurvSHAP(t) | Time-dependent | Q3, Q4 |

---

## Results Summary

| Model | C-index (Test) | C-index (5-Fold CV) | IBS | iAUC | AUC 1yr | AUC 2yr | AUC 3yr |
|-------|----------------|---------------------|-----|------|---------|---------|---------|
| Cox PH | **0.800** | 0.783 ± 0.015 | 0.136 | **0.858** | 0.845 | 0.844 | 0.866 |
| RSF | 0.783 | 0.785 ± 0.016 | 0.138 | 0.840 | 0.829 | 0.838 | 0.863 |
| GBM | 0.787 | 0.785 ± 0.016 | **0.134** | 0.844 | 0.833 | 0.840 | 0.876 |

All models tuned via 5-fold stratified CV on the training set (N=930). Final metrics on held-out test set (N=233). Overall predictive performance was comparable across the three models; Cox PH achieved the highest C-index and iAUC, while GBM showed marginally superior calibration (IBS).

**Key findings from XAI:**
- Grade_G4 and Age are the top prognostic factors across all methods (PFI + SHAP consistent)
- Grade_G4 SurvSHAP(t) peaks negatively within the first 500 days then recovers — time-varying effect confirmed, consistent with Schoenfeld violation
- Grade_G2 shows sustained positive SurvSHAP(t) over long-term follow-up — protective effect strongest at 1–3 years
- Tx_Standard SurvSHAP(t) peaks at ~500 days then declines — treatment effect is early-concentrated
- Non-flat SurvSHAP(t) curves for Age, Era_Before2005, and Tx_None visually confirm all three Schoenfeld violations

---

## Repository Structure
```
AI-DynaInfo/
├── NB01.ipynb                     # Data loading, cleaning, feature engineering
├── NB02.ipynb                     # EDA, KM curves, Table 1
├── NB03.ipynb                     # Cox PH baseline, VIF, Schoenfeld test
├── NB04.ipynb                     # RSF + GBM training, tuning, evaluation
├── NB05.ipynb                     # PFI + SHAP Summary + ALE
├── NB06.ipynb                     # SHAP Waterfall + c-ICE
├── NB07.ipynb                     # SurvSHAP(t) time-dependent explanations
├── NB08.ipynb                     # Paper figure export (300dpi PNG)
├── outputs/
│   ├── figures/                   # Generated visualizations (34 figures)
│   ├── tables/                    # Generated CSV tables (14 tables)
│   └── models/                    # Serialized model and data pkl files
├── data/
│   └── README.md                  # Data download instructions
├── report/
│   └── report.ipynb               # Final paper (Korean)
├── .gitignore
└── requirements.txt
```

Each notebook saves its outputs as `.pkl` files under `outputs/models/`, which subsequent notebooks load directly — no manual data passing required.

---

## Installation & Usage
```bash
git clone https://github.com/Sunwooxxi/AI-DynaInfo.git
cd AI-DynaInfo
pip install -r requirements.txt
```

Place `clinical.tsv` (downloaded from GDC Portal) in the `data/` directory, then run notebooks in order `NB01` → `NB08`.

**Core dependencies:**

| Package | Version | Role |
|---------|---------|------|
| scikit-survival | 0.27.0 | RSF, GBM, C-index, IBS, AUC |
| survshap | 0.4.2 | SurvSHAP(t) |
| shap | 0.51.0 | SHAP values, Waterfall |
| lifelines | 0.30.3 | Cox PH, KM curves |
| numpy | 2.3.5 | Numerical computation |
| pandas | 2.3.3 | Data manipulation |
| scikit-learn | 1.8.0 | Preprocessing, CV |

---

## Implementation Notes

- **SurvSHAP(t)**: `calculation_method='sampling'` used throughout. B=50 perturbations (default).
- **SHAP scale difference**: RSF SHAP values are on cumulative hazard function (CHF) scale; GBM SHAP values are on log hazard ratio scale. Direct numerical comparison across models is not meaningful — interpret within-model feature rankings only.
- **Tx_None instability**: RSF produces numerically unstable SurvSHAP(t) for Tx_None due to group heterogeneity (N=25). GBM results are preferred for this variable.
- **Random seed**: Fixed at 42 across all notebooks for reproducibility.

---

## Citation
```bibtex
@article{,
  title   = {Time-Dependent Survival Analysis for Brain Tumors with XAI},
  author  = {Jung, Sunwoo},
  journal = {unpublished},
  year    = {2026}
}
```
