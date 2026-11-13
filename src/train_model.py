from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.inspection import permutation_importance

from data_utils import load_dataset, clean_column_names

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
PROCESSED_DIR = ROOT / "data" / "processed"
for folder in [MODEL_DIR, OUTPUT_DIR, FIG_DIR, PROCESSED_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "Machine failure"
# These are labels describing failure modes. Excluding them prevents target leakage.
LEAKAGE_COLUMNS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
ID_COLUMNS = ["UDI", "Product ID"]

def main():
    df = clean_column_names(load_dataset())
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found. Found columns: {list(df.columns)}")

    # Save a clean working copy for reproducibility.
    df.to_csv(PROCESSED_DIR / "ai4i_clean.csv", index=False)

    # Summary and data quality report.
    quality = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()},
        "columns": list(df.columns),
    }
    (OUTPUT_DIR / "data_quality.json").write_text(json.dumps(quality, indent=2))

    # Exploratory charts.
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET)
    plt.title("Machine Failure Class Distribution")
    plt.xlabel("Machine failure (0 = No, 1 = Yes)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "target_distribution.png", dpi=160)
    plt.close()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    plt.figure(figsize=(11, 8))
    sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", center=0, annot=False)
    plt.title("Correlation Heatmap (Numeric Columns)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    sensor_cols = [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
    ]
    existing_sensor_cols = [c for c in sensor_cols if c in df.columns]
    for col in existing_sensor_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df, x=TARGET, y=col)
        plt.title(f"{col} by Machine Failure")
        plt.tight_layout()
        plt.savefig(FIG_DIR / (col.lower().replace(" ", "_").replace("[","").replace("]","").replace("/","_") + ".png"), dpi=160)
        plt.close()

    # Exclude identifiers and failure-mode labels to avoid leakage.
    drop_cols = [c for c in ID_COLUMNS + LEAKAGE_COLUMNS if c in df.columns]
    X = df.drop(columns=[TARGET] + drop_cols)
    y = df[TARGET].astype(int)

    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1
        ),
    }

    results = []
    fitted = {}
    reports = {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, prob)),
            "average_precision": float(average_precision_score(y_test, prob)),
        }
        results.append(metrics)
        fitted[name] = pipe
        reports[name] = classification_report(y_test, pred, zero_division=0, output_dict=True)

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No failure", "Failure"],
                    yticklabels=["No failure", "Failure"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix — {name}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / (name.lower().replace(" ", "_") + "_confusion_matrix.png"), dpi=160)
        plt.close()

    metrics_df = pd.DataFrame(results).sort_values(["recall", "f1"], ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    (OUTPUT_DIR / "classification_reports.json").write_text(json.dumps(reports, indent=2))

    # Select best model by failure recall, with F1 as tie-breaker.
    best_name = metrics_df.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    (MODEL_DIR / "model_metadata.json").write_text(json.dumps({
        "best_model": best_name,
        "features": list(X.columns),
        "target": TARGET,
        "dropped_columns": drop_cols,
        "threshold": 0.5,
        "note": "Threshold 0.5 is the default. Adjust using validation data and business costs."
    }, indent=2))

    # Permutation importance: model-agnostic importance on held-out data.
    try:
        pi = permutation_importance(
            best_model, X_test, y_test, n_repeats=5, random_state=42,
            scoring="average_precision", n_jobs=-1
        )
        importance = pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std
        }).sort_values("importance_mean", ascending=False)
        importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
        plt.figure(figsize=(8, 5))
        sns.barplot(data=importance.head(10), x="importance_mean", y="feature")
        plt.title("Top 10 Permutation Importances (Average Precision)")
        plt.xlabel("Decrease in average precision when shuffled")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "feature_importance.png", dpi=160)
        plt.close()
    except Exception as exc:
        (OUTPUT_DIR / "feature_importance_error.txt").write_text(str(exc))

    # Unsupervised anomaly detector on numeric sensor columns (separate from failure classifier).
    anomaly_cols = [c for c in existing_sensor_cols if c in df.columns]
    if anomaly_cols:
        anomaly_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", IsolationForest(contamination=0.03, random_state=42)),
        ])
        anomaly_pipeline.fit(df[anomaly_cols])
        joblib.dump({"pipeline": anomaly_pipeline, "features": anomaly_cols},
                    MODEL_DIR / "anomaly_detector.joblib")
        anomaly_labels = anomaly_pipeline.predict(df[anomaly_cols])
        anomaly_scores = anomaly_pipeline.decision_function(df[anomaly_cols])
        anomaly_df = df[anomaly_cols].copy()
        anomaly_df["anomaly_label"] = np.where(anomaly_labels == -1, "Anomaly", "Typical")
        anomaly_df["anomaly_score"] = anomaly_scores
        anomaly_df.to_csv(OUTPUT_DIR / "anomaly_results.csv", index=False)

    # Hold-out prediction file for review.
    prediction_df = X_test.copy()
    prediction_df["actual_failure"] = y_test
    prediction_df["predicted_failure"] = best_model.predict(X_test)
    prediction_df["failure_probability"] = best_model.predict_proba(X_test)[:, 1]
    prediction_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    summary = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "failure_rate_percent": round(float(y.mean() * 100), 3),
        "best_model": best_name,
        "best_model_metrics": results[[r["model"] for r in results].index(best_name)],
        "feature_columns_used": list(X.columns),
        "dropped_columns_to_prevent_leakage": drop_cols,
    }
    (OUTPUT_DIR / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nTRAINING COMPLETED")
    print(f"Dataset rows: {len(df)}")
    print(f"Best model (selected by recall then F1): {best_name}")
    print(metrics_df.to_string(index=False))
    print(f"\nOutputs saved in: {OUTPUT_DIR}")
    print(f"Model saved in: {MODEL_DIR / 'best_model.joblib'}")

if __name__ == "__main__":
    main()
