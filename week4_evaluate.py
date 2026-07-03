"""
Week 4: Evaluate models, visualize results
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("results.json") as f:
    results = json.load(f)

model_names = list(results.keys())
colors = {"KNN": "#3498db", "LogReg": "#9b59b6", "RandomForest": "#2ecc71", "NeuralNet": "#e67e22"}

# ---------------------------------------------------------
# 1. Bar chart: Accuracy / Precision / Recall / F1 comparison
# ---------------------------------------------------------
metrics = ["accuracy", "precision", "recall", "f1"]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(metrics))
width = 0.2
for i, name in enumerate(model_names):
    vals = [results[name][m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=name, color=colors[name])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([m.capitalize() for m in metrics])
ax.set_ylim(0.8, 1.0)
ax.set_ylabel("Score")
ax.set_title("Model Performance Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/model_comparison.png")
plt.close()

# ---------------------------------------------------------
# 2. Confusion matrices (2x2 grid)
# ---------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(9, 8))
for ax, name in zip(axes.flat, model_names):
    cm = np.array(results[name]["confusion_matrix"])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name} (Acc: {results[name]['accuracy']:.3f})")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Fake", "Real"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Fake", "Real"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/confusion_matrices.png")
plt.close()

# ---------------------------------------------------------
# 3. ROC curves
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
for name in model_names:
    if "fpr" in results[name]:
        ax.plot(results[name]["fpr"], results[name]["tpr"],
                 label=f"{name} (AUC={results[name]['roc_auc']:.3f})", color=colors[name])
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("outputs/roc_curves.png")
plt.close()

# ---------------------------------------------------------
# 4. Training time comparison
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))
times = [results[n]["train_time_sec"] for n in model_names]
ax.bar(model_names, times, color=[colors[n] for n in model_names])
ax.set_ylabel("Training Time (seconds)")
ax.set_title("Training Time Comparison")
plt.tight_layout()
plt.savefig("outputs/training_time.png")
plt.close()

print("Saved evaluation charts to outputs/")
print("\nFinal summary table:")
print(f"{'Model':<15}{'Accuracy':<10}{'Precision':<11}{'Recall':<9}{'F1':<8}{'ROC-AUC':<9}{'Time(s)'}")
for name in model_names:
    r = results[name]
    print(f"{name:<15}{r['accuracy']:<10.4f}{r['precision']:<11.4f}{r['recall']:<9.4f}{r['f1']:<8.4f}{r.get('roc_auc',0):<9.4f}{r['train_time_sec']:.1f}")
