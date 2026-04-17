"""
Level 3 - Task 2: Support Vector Machine (SVM) for Classification
Dataset: Iris Dataset (multi-class, using 2 features for boundary visualization)
Objectives:
  - Train SVM with linear and RBF kernels
  - Visualize decision boundary
  - Evaluate with accuracy, precision, recall, AUC
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix,
                              roc_curve, auc)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

# ─────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────
df = pd.read_csv("datasets/Data Set For Task/1) iris.csv")

print("=" * 60)
print("STEP 1: Loading & Preprocessing")
print("=" * 60)
print(f"Shape: {df.shape}")

le = LabelEncoder()
X = df.drop(columns=['species'])
y = le.fit_transform(df['species'])
class_names = le.classes_
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 2. Train SVM - Linear & RBF Kernels
# ─────────────────────────────────────────
kernels = ['linear', 'rbf']
results = {}

print("\n" + "=" * 60)
print("STEP 2: Comparing Kernels")
print("=" * 60)

for kernel in kernels:
    svm = SVC(kernel=kernel, probability=True, random_state=42, C=1.0)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    cv = cross_val_score(svm, X_scaled, y, cv=5, scoring='accuracy')

    results[kernel] = {
        'model': svm,
        'y_pred': y_pred,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'cv_mean': cv.mean(),
        'cv_std': cv.std()
    }

    print(f"\n  Kernel: {kernel.upper()}")
    print(f"    Accuracy : {results[kernel]['accuracy']:.4f}")
    print(f"    Precision: {results[kernel]['precision']:.4f}")
    print(f"    Recall   : {results[kernel]['recall']:.4f}")
    print(f"    F1-Score : {results[kernel]['f1']:.4f}")
    print(f"    5-CV Acc : {results[kernel]['cv_mean']:.4f} ± {results[kernel]['cv_std']:.4f}")

# Best kernel
best_kernel = max(results, key=lambda k: results[k]['accuracy'])
print(f"\n  Best Kernel: {best_kernel.upper()} (Accuracy={results[best_kernel]['accuracy']:.4f})")

# ─────────────────────────────────────────
# 3. Best Model Details
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 3: Best Model ({best_kernel.upper()}) - Full Report")
print("=" * 60)
print(classification_report(y_test, results[best_kernel]['y_pred'], target_names=class_names))

# ─────────────────────────────────────────
# 4. ROC Curves (One-vs-Rest)
# ─────────────────────────────────────────
best_model = results[best_kernel]['model']
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = best_model.predict_proba(X_test)

fpr, tpr, roc_auc_vals = {}, {}, {}
for i in range(3):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc_vals[i] = auc(fpr[i], tpr[i])

print(f"\nROC AUC per class:")
for i, cls in enumerate(class_names):
    print(f"  {cls}: {roc_auc_vals[i]:.4f}")

# ─────────────────────────────────────────
# 5. Decision Boundary (2 features: petal_length & petal_width)
# ─────────────────────────────────────────
X_2d = X_scaled[:, 2:4]  # petal_length, petal_width
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

svm_2d = SVC(kernel=best_kernel, probability=True, random_state=42)
svm_2d.fit(X_train_2d, y_train_2d)

# ─────────────────────────────────────────
# 6. Visualizations
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('SVM Classification - Iris Dataset', fontsize=14, fontweight='bold')

# Decision Boundary
h = 0.02
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

colors_map = ['#FFAAAA', '#AAFFAA', '#AAAAFF']
pred_colors = ['red', 'green', 'blue']
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
for i, (cls, col) in enumerate(zip(class_names, pred_colors)):
    mask = y == i
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=col, label=cls, edgecolors='k', linewidths=0.5, s=60)
axes[0].set_xlabel('petal_length (scaled)')
axes[0].set_ylabel('petal_width (scaled)')
axes[0].set_title(f'Decision Boundary\n(Kernel={best_kernel.upper()}, 2 features)')
axes[0].legend()

# ROC Curves
colors_roc = ['blue', 'red', 'green']
for i, col in zip(range(3), colors_roc):
    axes[1].plot(fpr[i], tpr[i], color=col, lw=2,
                 label=f'{class_names[i]} (AUC={roc_auc_vals[i]:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', lw=1.5)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title(f'ROC Curves (One-vs-Rest)\nKernel={best_kernel.upper()}')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

# Kernel Comparison Bar Chart
metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_labels))
width = 0.35
bars1 = [results['linear'][m] for m in ['accuracy', 'precision', 'recall', 'f1']]
bars2 = [results['rbf'][m] for m in ['accuracy', 'precision', 'recall', 'f1']]

axes[2].bar(x - width/2, bars1, width, label='Linear', color='steelblue', edgecolor='black')
axes[2].bar(x + width/2, bars2, width, label='RBF', color='coral', edgecolor='black')
axes[2].set_xlabel('Metric')
axes[2].set_ylabel('Score')
axes[2].set_title('Linear vs RBF Kernel Comparison')
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics_labels)
axes[2].set_ylim([0.9, 1.02])
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/level3_task2_svm.png', dpi=150, bbox_inches='tight')
print("\n📊 Plot saved.")
print("\n" + "=" * 60)
print("SVM COMPLETE")
print("=" * 60)
