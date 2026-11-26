# AgriPak Farms — Low-Yield Risk Prediction

**Internship task:** Machine Learning Internship — Task 01, Week 1  
**Goal:** Identify fields that may be at risk of low yield before harvest.

## 1. Project objective

This project builds a first Logistic Regression baseline to predict the `low_yield_risk` target from the supplied AgriPak crop dataset. The Week 1 work focuses on data checks, a reproducible stratified split, baseline evaluation, and a first leakage sanity check.

## 2. Dataset overview

- File: `data/crop_data.csv`
- Rows: 50,000
- Columns: 10
- Target: `low_yield_risk`
- Target classes: 0 = 32,500 (65.0%); 1 = 17,500 (35.0%)
- Missing values in supplied file: 0
- Duplicate rows removed: 0

The supplied CSV contains these columns:

`region, soil_type, crop, rainfall_mm, temperature_c, fertilizer_used, irrigation_used, weather_condition, days_to_harvest, low_yield_risk`

## 3. Repository structure

```text
agripak-low-yield-prediction/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
├── data/
│   ├── crop_data.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── reports/
│   ├── baseline_metrics.json
│   └── leakage_correlation.csv
├── src/
│   └── run_pipeline.py
└── notebooks/
    └── README.md
```

## 4. Setup and run

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
python src/run_pipeline.py
```

Run the command from the repository root. The script loads `data/crop_data.csv`, cleans the data, creates the split, trains the model, and writes the reports.

## 5. Data cleaning decisions

The pipeline applies the following checks and transformations:

1. Removes whitespace from column names and text values.
2. Converts blank text values to missing values.
3. Converts the known numeric measurement columns (`rainfall_mm`, `temperature_c`, `days_to_harvest`) to numeric values where possible.
4. Removes duplicate rows.
5. Removes rows with a missing target because supervised training requires a known label.
6. Fills missing numeric feature values with the training-set median.
7. Fills missing categorical feature values with the training-set most frequent value.
8. One-hot encodes categorical features and scales numeric features.
9. Fits preprocessing only on the training data through a scikit-learn pipeline, avoiding fitting transformations on the test set.

**Observation for this supplied file:** it has no missing values and no duplicate rows, so the missing-value and duplicate-removal steps did not need to alter any records in this run.

## 6. Train-test split

- Training data: 40,000 rows (80%)
- Test data: 10,000 rows (20%)
- Method: stratified split
- Random seed: 42

The fixed random seed and stratification make the split reproducible and preserve the target-class proportions in both sets.

## 7. Logistic Regression baseline results

The model was trained on the training set and evaluated on the held-out test set.

| Metric | Test score |
|---|---:|
| Accuracy | **0.9187 (91.87%)** |
| Precision | **0.8884 (88.84%)** |
| Recall | **0.8780 (87.80%)** |
| F1-score | **0.8832 (88.32%)** |

These are the results from the supplied dataset and this fixed train-test split. They are baseline results, not a guarantee of performance on new farms or future seasons.

## 8. Leakage sanity check

The pipeline calculates the absolute Pearson correlation between each original numeric / one-hot encoded feature and the encoded target. Any feature with absolute correlation **greater than 0.90** is flagged for manual review.

**Result:** 0 feature(s) exceeded the 0.90 threshold.

No feature was flagged by this threshold in the current dataset. The strongest absolute correlations were:

| Encoded feature | Correlation with target |
|---|---:|
| `rainfall_mm` | -0.6364 |\n| `fertilizer_used` | -0.3342 |\n| `irrigation_used` | -0.2662 |\n| `temperature_c` | -0.0685 |\n| `soil_type_Chalky` | -0.0076 |\n
This means the scan did not identify a very high-correlation feature. It does **not** prove that the dataset is completely free of leakage: correlation is only a first screening check. Each feature should still be reviewed to confirm that it would genuinely be available before harvest and was not calculated using post-harvest information.

## 9. Output files

- `data/processed/train.csv` — training split with target column.
- `data/processed/test.csv` — test split with target column.
- `reports/baseline_metrics.json` — evaluation metrics and split settings.
- `reports/leakage_correlation.csv` — feature correlations and threshold flags.

## 10. Limitations and next steps

This is a first-week baseline. Further work should include checking the meaning and timing of every input feature with the project mentor, reviewing class-specific errors, comparing additional models only after the data is trusted, and validating on genuinely unseen farms or seasons if such data is available.
