"""
train_probe.py

Trains the final logistic regression probe on the 4-source combined
dataset (ToxicChat, harmful-prompts, BeaverTails, oasst1). WildJailbreak
and Civil Comments are deliberately excluded — see README for why.

Usage: python train_probe.py
Requires: embeddings/labels already extracted (see extract_embeddings.py)
Output: <OUTPUT_DIR>/trained_probe.joblib
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_DIR = os.environ.get("OUTPUT_DIR", "./data")
OUTPUT_PATH = os.path.join(DATA_DIR, "trained_probe.joblib")

SOURCES = ["toxic", "harmfulprompts", "beavertails", "oasst1"]
C = 0.1  # selected via local sweep — see README Evaluation section


def load_combined():
    embeddings, labels = [], []
    for name in SOURCES:
        emb = np.load(os.path.join(DATA_DIR, f"{name}_embeddings.npy"))
        lab = np.load(os.path.join(DATA_DIR, f"{name}_labels.npy"))
        assert emb.shape[0] == lab.shape[0], f"{name}: embeddings/labels length mismatch"
        embeddings.append(emb)
        labels.append(lab)
    X = np.concatenate(embeddings, axis=0)
    y = np.concatenate(labels, axis=0)
    return X, y


if __name__ == "__main__":
    X, y = load_combined()
    print(f"Combined dataset: {X.shape}, toxic={y.sum()}, safe={len(y)-y.sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=C, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Held-out test accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Safe", "Toxic"]))

    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")
