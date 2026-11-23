# Engineering Graduate Salary Prediction: A Binary Classification Study Using ML & Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellowgreen.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Accelerated-red.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## Project Links & Resources

### Interactive Notebook

* **Engineering Graduate Salary Prediction Notebook:**
**Colab URL:** https://drive.google.com/file/d/1a1KXA3X8mWq-vsQjbveUFbQjFWXmmY7C/view?usp=sharing


### Documentation & Data

* **Project Presentation:** https://docs.google.com/presentation/d/1rZrM9z54SF7UZT3zji0wF_sHl3smegdj/edit?usp=sharing&ouid=116368063423906871146&rtpof=true&sd=true

* **Dataset (`salary.csv`):** https://drive.google.com/file/d/1kNAqoLfBQ5EccRd1Xr0o-Xw3E5BHM5rr/view?usp=sharing

* **Dataset Documentation:** https://docs.google.com/spreadsheets/d/1GV66XG5OOmQAJUm6Vue9uZ00B_mEEtqA/edit?usp=sharing&ouid=116368063423906871146&rtpof=true&sd=true

---

## Project Overview

I have done this Engineering Graduate Salary Prediction project using Python in collaboration with **United Network of Professionals (UNP)** as part of my data science internship.

The primary goal of this project is to build a predictive **binary classification** system that categorizes engineering graduates into **High-salary** or **Low-salary** brackets (split at the median salary of ₹3,00,000), and benchmark 9 classical machine learning algorithms against a deep Artificial Neural Network (ANN) to identify the most accurate and reliable model for talent acquisition.

The dataset encompasses **2,998 engineering graduate profiles** across demographics, academic history, cognitive aptitude scores, domain-specific engineering module scores, and psychometric traits sourced from an Indian campus placement dataset.

### Key Project Highlights
* **Binary Target Setup:** Converted continuous graduate salaries into balanced `Low` (≤ ₹3,00,000) and `High` (> ₹3,00,000) classes via median split.
* **Feature Engineering & Hygiene:** Handled `-1` sentinel values, grouped granular engineering specializations, and converted zero-inflated module scores into 10 ordinal bins (`0–9`).
* **Comprehensive Benchmarking:** Evaluated 9 classification algorithms with `GridSearchCV` 5-fold cross-validation alongside 4 deep ANN architectures across Adam and SGD optimizers.

The steps, methodology, and experimental results are presented below.

---

## Team & Institutional Affiliation

* **Academic Institution:** Bhavans Vivekananda College of Science, Humanities and Commerce
* **Program:** BSc (Honours) in Data Science — Data Science Internship Program
* **Collaborating Organization:** United Network of Professionals (UNP)
* **Project Group:** Group 6
* **Team Members:**
  * **Akhiranand Boddani**
  * **Uppaluru S V N Santhosha Roopa**
  * **Deepthi Hansika**
  * **Akshay Kumar**

---

## Dataset Description

The dataset contains **2,998 engineering graduate candidate records** (reduced to 2,997 after dropping one invalid row) across **34 original columns**, with **zero missing or null values** in the raw data.

* **Total Records (Raw):** 2,998 candidate profiles
* **Total Records (After Cleaning):** 2,897 rows × 19 features
* **Target Variable:** `SalaryLabel` — a binary label derived from the continuous `Salary` column via median split:
  * `Low` → Salary ≤ ₹3,00,000 (median)
  * `High` → Salary > ₹3,00,000

### Feature Groups

| Group | Features |
| :--- | :--- |
| **Demographics** | `Gender`, `Age`, `CollegeTier`, `CollegeCityTier` |
| **Academic History** | `10percentage`, `12percentage`, `collegeGPA`, `Degree`, `Specialization` |
| **Aptitude Scores** | `English`, `Logical`, `Quant`, `Domain` |
| **Engineering Test Modules** | `ComputerProgramming`, `ElectronicsAndSemicon`, `ComputerScience`, `MechanicalEngg`, `ElectricalEngg`, `TelecomEngg`, `CivilEngg` |
| **Psychometrics** | `conscientiousness`, `agreeableness`, `extraversion`, `nueroticism`, `openess_to_experience` |
| **Target** | `Salary` → converted to `SalaryLabel` (`Low` / `High`) |

---

## Project Methodology & Step-by-Step Execution

### 1. Data Inspection, Cleaning & Sentinel Value Treatment

* **Initial Audit:** Loaded `salary.csv` (**2,998 records**, **34 columns**) and confirmed **0 null values** across all columns.
* **Sentinel Value & Scale Fixes:** Cleaned invalid values, standardized categories, and engineered `Age` (`GraduationYear - DOB.year`) before dropping non-predictive identifier columns (`ID`, `CollegeID`, `CollegeCityID`, `DOB`, `10board`, `12board`, `12graduation`, `CollegeState`, `GraduationYear`, `CollegeCityTier`):

| Issue Found | Action Taken | Result |
| :--- | :--- | :--- |
| Invalid `GraduationYear` = 0 | Row dropped | 2,998 → 2,997 rows |
| `collegeGPA` on 0–10 scale (9 rows) | Rescaled × 9.5 to match 0–100 scale | min 49.07 – max 99.93 |
| `Domain` score = −1 (not attempted) | Replaced with median of valid scores | median ≈ 0.636 |
| Engineering test modules with −1 (not attempted) | Replaced with 0 (e.g., 649 missing `ComputerProgramming`, 2,971 missing `CivilEngg`) | 7 module columns cleaned |
| `Gender` values (`'m'`/`'f'`) | Standardized to `'MALE'`/`'FEMALE'` | Consistent labels |
| `Degree` values (MCA, M.Tech, M.Sc.) | Consolidated to `'MASTERS'` vs `'B.Tech/B.E.'` | 2 clean categories |
| Continuous `Salary` column | Median-split into `SalaryLabel` (`Low` / `High`) | Split at ₹3,00,000 |

### 2. Exploratory Data Analysis (EDA) & Feature Engineering

* **EDA Findings:** Count plots, KDE curves, and boxplots revealed that core cognitive aptitude scores (`English`, `Logical`, `Quant`) follow near-normal Gaussian distributions, `Age` is right-skewed toward recent graduates, and engineering test-module scores are **heavily zero-inflated** because candidates typically attempt only their own specialization module.
* **Specialization Grouping:** Consolidated dozens of granular `Specialization` entries into 8 broad categories (`CS_IT`, `COMP_APPL`, `ELECTRONICS`, `MECH`, `ELECTRICAL`, `TELECOM`, `CIVIL`, `OTHER`) under `GroupedSpecialization`.
* **Engineering Module Score Binning:** Converted all 7 zero-inflated continuous module scores into **ordinal bins** (`*Binned` suffix, dropping raw columns) using `pd.cut`, where **Bin 0** represents unattempted modules (`0`) and **Bins 1–9** represent 100-point score bands (`100–900`):

```python
pd.cut(
    bins = [-inf, 0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
)
```

### 3. Outlier Removal, Multicollinearity & Final Encoding

* **Outlier Detection & Removal (IQR):** Applied Interquartile Range filtering (`Q1 - 1.5 × IQR` to `Q3 + 1.5 × IQR`) across core numerical features, flagging **144 values** (`Age`: 56, `10percentage`: 17, `12percentage`: 1, `collegeGPA`: 23, `English`: 12, `Logical`: 13, `Quant`: 22, `Domain`: 0). Removing affected rows in a single pass yielded **2,897 clean records**.
* **Correlation & VIF Analysis:** Pearson and Phi-K matrices showed an expected `0.64` correlation between `10percentage` and `12percentage` with no pair exceeding `0.85`. Variance Inflation Factor (VIF) scores confirmed all genuine numeric predictors sat comfortably below `2.0` (higher VIFs appeared only in mutually exclusive `GroupedSpecialization` dummy columns: `ELECTRICAL` at `26.16`, `CS_IT` at `24.45`, `MECH` at `11.43`).
* **Feature Selection & Encoding:** Dropped low-correlation psychometric columns (`conscientiousness`, `agreeableness`, `extraversion`, `openess_to_experience`, retaining `nueroticism`), dummy-encoded `Gender` (`Gender_MALE`), ordinal-encoded `Degree` (`B.Tech/B.E. = 0`, `MASTERS = 1`), cast `*Binned` columns to integers, and encoded `SalaryLabel` (`Low = 0`, `High = 1`). Exported `cleaned_salary.csv` (**2,897 rows × 19 features** + target).

### 4. Validation, Model Benchmarking & Deep Learning (ANN)

* **Multi-Split Validation & Scaling:** Evaluated all baseline classifiers across **5 train/test splits** (80/20, 75/25, 70/30, 65/35, and 60/40) using `StandardScaler` fitted strictly on training folds. Tracked **Accuracy, Precision, Recall, F1-Score, and ROC-AUC**.
* **9 Classical ML Classifiers & GridSearchCV:** Benchmarked **Logistic Regression, Gaussian Naïve Bayes, KNN, Linear SVC, Decision Tree, Random Forest, AdaBoost, Gradient Boosting, and XGBoost**, followed by 5-fold `GridSearchCV` (`cv=5`, optimizing `accuracy`) on the 80/20 split:

| Model | Parameters Tuned |
| :--- | :--- |
| **Logistic Regression** | `C`: `[0.01, 0.1, 1.0, 10.0]`; `penalty`: `['l1', 'l2']` |
| **Naïve Bayes** | `var_smoothing`: `[1e-9, 1e-8, 1e-7, 1e-5]` |
| **KNN** | `n_neighbors`: `[5, 7, 9]`; `weights`: `['uniform', 'distance']`; `metric`: `['manhattan', 'euclidean', 'cosine', 'haversine']` |
| **Linear SVM** | `penalty`: `['l1', 'l2']`; `C`: `[0.1, 1.0, 10.0]` |
| **Decision Tree** | `max_depth`: `[10, 15, 20]`; `max_leaf_nodes`: `[10, 20, 50, 100]` |
| **Random Forest** | `n_estimators`: `[100, 150, 200]`; `max_depth`: `[10, 15, 20]` |
| **AdaBoost** | `n_estimators`: `[100, 150, 200]`; `learning_rate`: `[0.01, 0.1, 1.0]` |
| **Gradient Boosting** | `n_estimators`: `[100, 150, 200]`; `learning_rate`: `[0.01, 0.05, 0.1]` |
| **XGBoost** | `n_estimators`: `[100, 150, 200]`; `max_depth`: `[3, 5, 7]` |

* **Artificial Neural Network (ANN) Sweeps:** Built `TensorFlow/Keras` models (Binary Cross-Entropy loss, Sigmoid output) testing four architectures—**Shallow** (`Input → 10 → 5 → 1`), **Medium** (`Input → 64 → 32 → 16 → 1`), **Deep Pyramid** (`Input → 32 → 32 → 16 → 16 → 8 → 1`), and **Very Deep** (`Input → 128 → 64 → 32 → 16 → 8 → 1`)—across **Adam** and **SGD** optimizers at **20, 50, and 100 epochs**.
* **Best Overall Model:** **Logistic Regression** (`C=1.0`, `penalty='l2'`) achieved the highest test accuracy (**~69.8%**) with the smallest train-test gap across all splits, outperforming both tree ensembles and the top ANN (`128 → 64 → 32 → 16 → 8 → 1`, Adam, 100 epochs, batch size 64).

---

## Key Insights & Discussion

1. **Best-Performing Model (Logistic Regression):**
   * Despite the complexity of ensemble methods, Logistic Regression consistently achieved the highest test accuracy (~69.8%) and the smallest train-test gap across all 5 splits — signaling strong generalization and a well-suited linear decision boundary for this feature space.
2. **Tree-Based Overfitting:**
   * Unconstrained Decision Trees, Random Forests, and XGBoost achieved perfect training accuracy (`1.0000`) but dropped to test accuracies between `0.60` and `0.67`. `GridSearchCV` pruning improved generalization, though tree models still trailed linear models on unseen data.
3. **ANN vs. Classical ML:**
   * The deep ANN (`128 → 64 → 32 → 16 → 8 → 1`, Adam, 100 epochs) reached competitive accuracy levels but did not surpass regularized Logistic Regression — confirming that classical linear models remain strong baselines on moderate-sized tabular datasets.
4. **Key Predictive Features:**
   * Cognitive aptitude scores (`English`, `Logical`, `Quant`) and academic performance (`10percentage`, `12percentage`, `collegeGPA`) were the strongest differentiators of salary bracket, while binning zero-inflated engineering module scores added predictive signal without sparsity issues.

---

## Future Scope

* **Additional Candidate & Job Features:** Incorporate job location, company tier, prior work experience, and industry sector to strengthen predictive accuracy.
* **Class Imbalance Handling:** Apply SMOTE or class-weight adjustments to maintain strong recall if applied to imbalanced salary distributions.
* **Model Explainability (SHAP):** Use SHAP values to quantify how each feature contributes to individual salary predictions for transparent HR decision-making.
* **Multi-Class Salary Brackets:** Expand beyond binary High/Low classification into quartile-based brackets (e.g., Very Low, Low, High, Very High) for finer salary estimation.
* **API Deployment:** Package the trained Logistic Regression model as a REST API (Flask or FastAPI) for integration into campus placement portals.

---

## Acknowledgments

* **United Network of Professionals (UNP):** For providing internship guidance, project mentoring, and structured technical support throughout the program.
* **Bhavans Vivekananda College of Science, Humanities and Commerce:** For academic encouragement and project enablement.
