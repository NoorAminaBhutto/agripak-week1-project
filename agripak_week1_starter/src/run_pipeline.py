"""Week 1 AgriPak baseline pipeline."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.preprocessing import FunctionTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "crop_data.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
TARGET = "low_yield_risk"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Columns expected to contain measurements/counts. Conversion is safe:
# values that cannot be parsed become missing and are imputed using train data.
KNOWN_NUMERIC = [
    "rainfall_mm", "temperature_c", "days_to_harvest",
]

def load_and_basic_clean(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the original crop_data.csv file inside the repository's data/ folder."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' was not found. Available columns: {df.columns.tolist()}"
        )
    before = len(df)
    df = df.drop_duplicates().copy()
    print(f"Loaded {before:,} rows; removed {before - len(df):,} duplicate rows.")

    # Normalize whitespace in text values and convert common numeric columns.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        df[col] = df[col].replace("", np.nan)

    for col in KNOWN_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_target = int(df[TARGET].isna().sum())
    if missing_target:
        print(f"Dropping {missing_target:,} rows with missing target labels.")
        df = df.dropna(subset=[TARGET]).copy()

    if df[TARGET].nunique() < 2:
        raise ValueError("The target must contain at least two classes for classification.")
    return df

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
    ])

def leakage_scan(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Correlation scan across numeric and one-hot encoded feature columns."""
    y_encoded = LabelEncoder().fit_transform(y.astype(str))
    encoded = pd.get_dummies(X, dummy_na=False, dtype=float)
    rows = []
    for col in encoded.columns:
        values = pd.to_numeric(encoded[col], errors="coerce")
        if values.nunique(dropna=True) < 2:
            corr = np.nan
        else:
            corr = values.corr(pd.Series(y_encoded, index=values.index))
        rows.append({
            "encoded_feature": str(col),
            "correlation_with_target": float(corr) if pd.notna(corr) else np.nan,
            "absolute_correlation": float(abs(corr)) if pd.notna(corr) else np.nan,
            "flag_above_0.90": bool(pd.notna(corr) and abs(corr) > 0.90),
        })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("absolute_correlation", ascending=False, na_position="last")
    return result

def main() -> int:
    df = load_and_basic_clean(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratification requires enough rows in each class to place examples in both splits.
    class_counts = y.value_counts()
    if class_counts.min() < 2:
        raise ValueError(
            "Stratified splitting needs at least 2 rows in every target class. "
            f"Class counts: {class_counts.to_dict()}"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    train_df = X_train.copy()
    train_df[TARGET] = y_train
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    model = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Binary metrics use the positive class selected by sklearn's sorted class order.
    # For multiclass targets, weighted averages are reported.
    average = "binary" if y.nunique() == 2 else "weighted"
    positive_label = sorted(y.unique(), key=lambda v: str(v))[-1] if average == "binary" else None
    metric_args = {"average": average, "zero_division": 0}
    if positive_label is not None:
        metric_args["pos_label"] = positive_label

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, **metric_args)),
        "recall": float(recall_score(y_test, predictions, **metric_args)),
        "f1": float(f1_score(y_test, predictions, **metric_args)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "target_column": TARGET,
    }
    with open(REPORTS_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    corr_report = leakage_scan(X, y)
    corr_report.to_csv(REPORTS_DIR / "leakage_correlation.csv", index=False)
    flagged = corr_report[corr_report["flag_above_0.90"]]

    print("\nBASELINE TEST METRICS")
    for name in ("accuracy", "precision", "recall", "f1"):
        print(f"{name.capitalize():<10}: {metrics[name]:.4f}")
    print(f"\nSaved split files in: {PROCESSED_DIR}")
    print(f"Saved baseline metrics in: {REPORTS_DIR / 'baseline_metrics.json'}")
    print(f"Saved leakage scan in: {REPORTS_DIR / 'leakage_correlation.csv'}")
    print("\nLEAKAGE SCAN")
    if flagged.empty:
        print("No encoded feature had absolute correlation above 0.90.")
    else:
        print(flagged[["encoded_feature", "correlation_with_target"]].to_string(index=False))
        print("Review each flagged feature's meaning and whether it is available before harvest.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
