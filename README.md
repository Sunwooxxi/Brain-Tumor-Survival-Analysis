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

- **Source**: [TCGA](https://portal.gdc.cancer.gov/) brain tumor cohort
- **Filter**: Primary tumors, index date = Diagnosis
- **Final cohort**: N = XXX patients after exclusion criteria
- **Outcome**: Overall survival (days to death / last follow-up)
- **Features**: 14 variables (see Table 1 in paper)

> **Note on Tx_None**: This category includes patients with no treatment record **or** supportive/palliative care only, and should be interpreted with caution due to group heterogeneity.

---

## Methods

### Preprocessing (`NB01`)
- Grade classification: WHO morphology codes (`9440/3`, `9442/3`, `9474/3` → G4; Anaplastic → G3; `9400/3`, `9450/3`, `9382/3` → G2)
- Missing values: Age imputed with median (< 5% missing); patients with unclassifiable Grade excluded
- One-hot encoding for Grade, Site, Era, Treatment

### Baseline — Cox PH (`NB03`)
- Univariate → multivariate Cox regression
- VIF multicollinearity check (threshold: VIF < 10)
- penalizer tuned via 5-fold Grid Search CV
- Schoenfeld residual test for proportional hazards assumption

### ML Survival Models (`NB04`)
- **RSF**: `sksurv.ensemble.RandomSurvivalForest`
- **GBM**: `sksurv.ensemble.GradientBoostingSurvivalAnalysis`
- Hyperparameter tuning: 5-fold stratified Grid Search CV (C-index + IBS monitoring)
- Evaluation: C-index, IBS, iAUC, clinical time-point AUC (1yr / 2yr / 3yr)

### XAI Pipeline

| NB | Method | Library |
|----|--------|---------|
| NB05 | Permutation Feature Importance (PFI) | sksurv |
| NB05 | SHAP Summary (Beeswarm) | shap |
| NB05 | ALE Plot | alibi / custom |
| NB06 | SHAP Waterfall | shap |
| NB06 | c-ICE Plot | custom |
| NB07 | SurvSHAP(t) | survshap 0.4.2 |

---

## Results Summary

| Model | C-index | IBS | iAUC |
|-------|---------|-----|------|
| Cox PH | — | — | — |
| RSF | — | — | — |
| GBM | — | — | — |

> Values to be filled after final run.

**Key findings:**
- Grade_G4 and Age are the strongest prognostic factors across all models
- Grade_G4 effect is concentrated in the first 900 days, then diminishes (time-varying) — violating Cox's proportional hazards assumption
- Grade_G2 shows sustained protective effect over long-term follow-up
- Tx_Standard effect peaks at ~500–700 days then declines
- Schoenfeld test violations (Age, Era_Before2005, Tx_None) are visually confirmed by non-flat SurvSHAP(t) curves

---

## Repository Structure

```
.
├── NB01_preprocessing.ipynb      # Data loading, cleaning, feature engineering
├── NB02_eda.ipynb                 # EDA, KM curves, baseline statistics
├── NB03_cox.ipynb                 # Cox PH baseline + Schoenfeld test
├── NB04_ml_models.ipynb           # RSF + GBM training + evaluation
├── NB05_global_xai.ipynb          # PFI + SHAP Summary + ALE
├── NB06_local_xai.ipynb           # Waterfall + c-ICE
├── NB07_survshap.ipynb            # SurvSHAP(t) time-dependent explanations
├── NB08_export.ipynb              # Paper figure export (300dpi PNG + TIFF)
├── outputs/
│   ├── models/                    # Saved model pkl files
│   ├── paper_figures/             # Final figures for publication
│   └── paper_tables/              # Final tables for publication
├── clinical.tsv                   # Raw TCGA data (not included — see Data section)
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-survival-xai.git
cd brain-tumor-survival-xai

pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
lifelines
scikit-survival
shap
survshap==0.4.2
joblib
statsmodels
Pillow
```

---

## Usage

Run notebooks in order:

```bash
jupyter notebook NB01_preprocessing.ipynb
jupyter notebook NB02_eda.ipynb
jupyter notebook NB03_cox.ipynb
jupyter notebook NB04_ml_models.ipynb
jupyter notebook NB05_global_xai.ipynb
jupyter notebook NB06_local_xai.ipynb
jupyter notebook NB07_survshap.ipynb
jupyter notebook NB08_export.ipynb
```

Each notebook saves intermediate results as `.pkl` files in `outputs/models/`, which are loaded by subsequent notebooks.

> **Data**: Raw TCGA data (`clinical.tsv`) is not included due to data use restrictions. Download from [GDC Portal](https://portal.gdc.cancer.gov/) and place in the root directory.

---

## Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-survival | ≥ 0.21 | RSF, GBM, evaluation metrics |
| survshap | 0.4.2 | SurvSHAP(t) time-dependent SHAP |
| shap | ≥ 0.42 | SHAP values, Waterfall plots |
| lifelines | ≥ 0.27 | Cox PH, KM curves |

---

## Notes

- **SurvSHAP(t) method**: `calculation_method='sampling'` is used throughout (treeshap removed for compatibility with survshap 0.4.2)
- **Treatment representative patients**: For STEP 4 in NB07, each treatment variable's SurvSHAP(t) is computed from a patient who *actually received* that treatment (median risk score within group), avoiding unreliable counterfactual estimates
- **Tx_None interpretation**: Heterogeneous group (no record + supportive care); SurvSHAP(t) results for this variable should be interpreted with caution

---

## Citation

If you use this code, please cite:

```bibtex
@article{yourname2025,
  title   = {Explainable Machine Learning for Brain Tumor Survival Prediction:
             Time-Dependent Feature Contributions via SurvSHAP(t)},
  author  = {YOUR NAME},
  journal = {JOURNAL NAME},
  year    = {2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
