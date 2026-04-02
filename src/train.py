"""
train.py

Trains a Random Forest classifier to predict NFL run vs. pass plays.

Usage:
    python src/train.py

Outputs:
    models/model.pkl     - Trained model (for inference)
    models/metrics.json  - Evaluation metrics on held-out test split
"""

import json
import os
import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from preprocess import FEATURE_COLUMNS, preprocess_data

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/nfl_plays.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/model.pkl")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "../models/metrics.json")


def train() -> RandomForestClassifier:
    # Load and preprocess
    df = pd.read_csv(DATA_PATH)
    X, y = preprocess_data(df)

    print(f"Dataset: {len(X):,} plays  |  runs: {(y == 0).sum():,}  passes: {(y == 1).sum():,}")

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["run", "pass"], output_dict=True)
    print("\n" + classification_report(y_test, y_pred, target_names=["run", "pass"]))

    # Feature importances
    importances = dict(zip(FEATURE_COLUMNS, model.feature_importances_.round(4)))
    print("Feature importances:")
    for feat, score in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat:<28} {score:.4f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Save metrics
    metrics = {
        "accuracy": round(report["accuracy"], 4),
        "run":  {k: round(v, 4) for k, v in report["run"].items()  if k != "support"},
        "pass": {k: round(v, 4) for k, v in report["pass"].items() if k != "support"},
        "feature_importances": importances,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved  -> {MODEL_PATH}")
    print(f"Metrics saved -> {METRICS_PATH}")

    return model


if __name__ == "__main__":
    train()
