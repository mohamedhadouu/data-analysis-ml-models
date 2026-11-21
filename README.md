# Interactive Data Cleaning and ML-Based Missing Value Imputation

A data analysis pipeline on a **healthcare (stroke) dataset of about 43,400 patients**: exploratory analysis, interactive data cleaning, and **filling missing values with machine learning models** that are compared and selected automatically.

![Python](https://img.shields.io/badge/Python-3-blue) ![pandas](https://img.shields.io/badge/pandas-EDA-150458) ![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## Dataset

Patient records with 12 columns: `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `stroke` (target) and `id`.

Missing values: **1,462 in `bmi`** and **13,292 in `smoking_status`**.

## Pipeline

1. **Exploratory data analysis**
   - Missing-value overview with `missingno`
   - Distributions of categorical variables (count plots), also split by a target chosen by the user (e.g. `stroke`)
   - Distributions of numerical variables (histograms, KDE, violin plots) and correlation heatmap
2. **Interactive cleaning**
   - The user enters a valid range for numerical values, and outliers are removed
   - The user enters invalid categories to exclude (e.g. `Others` for gender)
3. **ML-based imputation**
   - **Numerical columns** (`bmi`, `avg_glucose_level`): KNN regressor, random forest regressor and linear regression are compared, and the best model fills the missing values
   - **Categorical column** (`smoking_status`): random forest, decision tree and logistic regression are compared, and the best model fills the missing values
4. **Visual check** of the completed dataset with `missingno`

## How to run

```bash
git clone https://github.com/mohamedhadouu/data-analysis-ml-models.git
cd data-analysis-ml-models
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn missingno
jupyter notebook Application_Final.ipynb
```

Replace the Windows file path in `pd.read_csv(...)` with the path to your `dataset.csv`. Some cells ask for input with `input()` (target variable, valid ranges, categories to exclude).

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · Seaborn · Matplotlib · missingno

## Next steps

- Use classification metrics (accuracy, F1) instead of R² to select the categorical imputation model
- Use more predictor features for numerical imputation (currently only `age`)
- Compare with `KNNImputer` and `IterativeImputer` from scikit-learn
- Train a stroke prediction model on the cleaned data

## Author

**Mohamed Hadou**, AI / ML Engineer · [LinkedIn](https://www.linkedin.com/in/mohamed-hadou/) · [GitHub](https://github.com/mohamedhadouu)
