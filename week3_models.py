"""
Week 3: Model Building
KNN (non-parametric), Logistic Regression (parametric),
Random Forest (ensemble), Neural Network (deep learning)
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pickle
import time
import json

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

t_start = time.time()

# ---------------------------------------------------------
# Load features
# ---------------------------------------------------------
X = sp.load_npz("X_tfidf.npz")
y = pd.read_csv("y_labels.csv")["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

models = {
    "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "LogReg": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1, random_state=42),
    "NeuralNet": MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42, early_stopping=True),
}

results = {}
trained_models = {}

for name, model in models.items():
    t0 = time.time()
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    elapsed = time.time() - t0
    print(f"{name} -> Acc: {acc:.4f}  Prec: {prec:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}  ({elapsed:.1f}s)")
    print(classification_report(y_test, preds, target_names=["Fake", "Real"]))

    results[name] = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "confusion_matrix": cm.tolist(),
        "train_time_sec": elapsed,
    }
    trained_models[name] = model

    if probs is not None:
        fpr, tpr, _ = roc_curve(y_test, probs)
        results[name]["roc_auc"] = float(auc(fpr, tpr))
        results[name]["fpr"] = fpr.tolist()
        results[name]["tpr"] = tpr.tolist()

# ---------------------------------------------------------
# Save results + models for Week 4
# ---------------------------------------------------------
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

with open("trained_models.pkl", "wb") as f:
    pickle.dump(trained_models, f)

np.save("y_test.npy", y_test)
sp.save_npz("X_test.npz", X_test)

print(f"\nTotal Week 3 time: {time.time()-t_start:.1f}s")
print("\nSummary:")
for name, r in results.items():
    print(f"  {name:15s} Acc={r['accuracy']:.4f}  F1={r['f1']:.4f}")
