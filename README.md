# 🧠 AI-Based Brain Tumor Survival Analysis: A Comparative Modeling Study

> Survival analysis pipeline combining statistical and machine learning approaches to identify key prognostic factors and improve survival prediction performance.

---

## Overview

This project presents a comprehensive survival analysis of a brain tumor cohort, focusing on both **interpretability** and **predictive performance**.  

The study integrates:
- Traditional statistical modeling (Cox Proportional Hazards)
- Model validation and diagnostics (PH assumption, VIF)
- Machine learning–based survival models (RSF, Gradient Boosting)
- Ensemble learning for performance optimization  

The goal is to **identify key risk factors** and build a **robust prognostic prediction system**.

---

## Objectives

- Perform rigorous **data preprocessing and cohort definition**
- Identify **independent prognostic factors** using Cox regression
- Validate model assumptions (Proportional Hazards)
- Compare multiple survival models:
  - Cox PH
  - Stratified Cox
  - Random Survival Forest (RSF)
  - Gradient Boosting Survival
- Build an **ensemble model** for improved prediction
- Evaluate models using:
  - C-index
  - IBS (Integrated Brier Score)
  - Time-dependent AUC

---

## Dataset & Preprocessing

- **Cohort size:** 1,167 patients  
- **Event (death):** 660 patients  

### Key preprocessing steps:
- Defined survival variables (`time`, `event`)
- Removed duplicates (longest follow-up retained)
- Imputed missing values (median for age)
- Feature engineering:
  - Age groups (Young / Middle / Old)
  - Treatment combinations (Standard / Single / Supportive)
  - Era grouping (Before vs After 2005)
- Reconstructed **tumor grade (G2/G3/G4)** using clinical information

---

## tatistical Survival Analysis

### 1. Kaplan–Meier & Log-rank Test
- Significant survival differences observed in:
  - Tumor grade
  - Age group
  - Treatment
  - Tumor origin

### 2. Univariate Cox Analysis
- Strong risk factors:
  - Grade 4 (HR ≈ 11.78)
  - Age ≥ 60 (HR ≈ 7.13)
- Protective factor:
  - Standard treatment (HR ≈ 0.31)

### 3. Multicollinearity Check (VIF)
- Severe collinearity between:
  - `hist_group` and `grade_final`
- Solution:
  - Removed `hist_group`
  - Retained clinically meaningful `grade_final`

---

## Multivariate Cox Model

### Key findings (Adjusted HR):

| Factor | Effect |
|------|--------|
| **Grade 4** | ~7.91× higher risk |
| **Age ≥ 60** | ~3.47× higher risk |
| **Standard Treatment** | ~55% risk reduction |

### Model performance:
- **C-index:** 0.77

### Key insight:
> The most critical independent predictors are **tumor grade, age, and treatment**.

---

## Proportional Hazards (PH) Assumption

- Tested using **Schoenfeld residuals**
- Violations detected in:
  - Age group
  - Era group
  - Supportive treatment

### Solution:
- Applied **Stratified Cox model** (age + era)

---

## Machine Learning Survival Models

### 1. Random Survival Forest (RSF)

- **C-index:** 0.786  
- **IBS:** 0.124  

#### Key insight:
- Age is the most important predictor
- Nonlinear threshold around **~65 years**

---

### 2. Gradient Boosting Survival

- **C-index:** 0.778  
- **IBS:** 0.118 (best calibration)

#### Key insight:
- **Grade 4 dominates prediction (~57% importance)**
- Strong long-term prediction performance

---

## Ensemble Model (RSF + GB)

### Method:
- Min-Max scaling
- 50:50 weighted averaging

### Performance:

| Metric | Value |
|------|------|
| **C-index** | **0.791** |
| **IBS** | 0.121 |
| **Mean AUC** | 0.875 |

### Key result:
- Best overall performance
- Strong **long-term survival prediction (AUC ≈ 0.96 at 5 years)**

---

## Key Insights

- **Tumor grade (G4)** is the most dominant risk factor
- **Age effect is nonlinear**, with a critical threshold around ~65
- **Standard treatment significantly improves survival**
- Machine learning models outperform Cox in prediction
- Ensemble approach provides the most robust performance

---

## Conclusion

This study demonstrates that:

- Traditional survival models are effective for **interpretability**
- Machine learning models significantly improve **prediction accuracy**
- Combining both approaches leads to a **clinically meaningful and high-performance prognostic system**

---

## Project Structure


---

## Tech Stack

- Python (pandas, numpy)
- lifelines (Cox, KM, PH test)
- scikit-survival / sklearn
- matplotlib / seaborn

---

## Future Work

- Time-varying Cox model
- Deep learning survival models (DeepSurv)
- External validation with independent dataset
