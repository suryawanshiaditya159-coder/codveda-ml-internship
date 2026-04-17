"""
Level 2 - Task 2: Decision Trees for Classification
Dataset: Iris Dataset
Objectives:
  - Train a decision tree on iris dataset
  - Visualize the tree structure
  - Prune to prevent overfitting
  - Evaluate with accuracy and F1-score
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              confusion_matrix)

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv("datasets/Data Set For Task/1) iris.csv")

print("=" * 60)
print("STEP 1: Dataset Overview")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nClass distribution:\n{df['species'].value_counts()}")

# ─────────────────────────────────────────
# 2. Preprocess
# ─────────────────────────────────────────
le = LabelEncoder()
X = df.drop(columns=['species'])
y = le.fit_transform(df['species'])
class_names = le.classes_
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 3. Train Unpruned Decision Tree
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Unpruned Decision Tree")
print("=" * 60)

dt_unpruned = DecisionTreeClassifier(random_state=42)
dt_unpruned.fit(X_train, y_train)

y_pred_unp = dt_unpruned.predict(X_test)
acc_unp = accuracy_score(y_test, y_pred_unp)
f1_unp = f1_score(y_test, y_pred_unp, average='weighted')

print(f"Max Depth (unpruned): {dt_unpruned.get_depth()}")
print(f"Num Leaves          : {dt_unpruned.get_n_leaves()}")
print(f"Accuracy            : {acc_unp:.4f}")
print(f"F1-Score (weighted) : {f1_unp:.4f}")

# ─────────────────────────────────────────
# 4. Pruning - Test Various max_depth
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Pruning - Comparing max_depth values")
print("=" * 60)

depths = range(1, 10)
train_accs, test_accs = [], []

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
    test_accs.append(accuracy_score(y_test, dt.predict(X_test)))

best_depth = depths[np.argmax(test_accs)]
print(f"\nDepth | Train Acc | Test Acc")
print("-" * 32)
for d, tr, te in zip(depths, train_accs, test_accs):
    marker = " ← best" if d == best_depth else ""
    print(f"  {d:2d}  |   {tr:.4f}  |  {te:.4f}{marker}")

# ─────────────────────────────────────────
# 5. Best Pruned Tree
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print(f"STEP 4: Pruned Tree (max_depth={best_depth})")
print("=" * 60)

dt_pruned = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
dt_pruned.fit(X_train, y_train)
y_pred_p = dt_pruned.predict(X_test)
acc_p = accuracy_score(y_test, y_pred_p)
f1_p = f1_score(y_test, y_pred_p, average='weighted')

cv_scores = cross_val_score(dt_pruned, X, y, cv=5, scoring='accuracy')

print(f"Accuracy            : {acc_p:.4f}")
print(f"F1-Score (weighted) : {f1_p:.4f}")
print(f"5-Fold CV Accuracy  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_p, target_names=class_names)}")
print(f"\nTree Rules:\n{export_text(dt_pruned, feature_names=feature_names)}")

# ─────────────────────────────────────────
# 6. Visualizations
# ─────────────────────────────────────────
fig = plt.figure(figsize=(20, 12))
fig.suptitle('Decision Tree Classification - Iris Dataset', fontsize=16, fontweight='bold')

# Tree visualization
ax1 = fig.add_subplot(2, 2, (1, 2))
plot_tree(dt_pruned, feature_names=feature_names, class_names=class_names,
          filled=True, rounded=True, fontsize=10, ax=ax1)
ax1.set_title(f'Decision Tree Structure (max_depth={best_depth})', fontsize=13)

# Depth vs Accuracy
ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(list(depths), train_accs, 'b-o', label='Train Accuracy', linewidth=2)
ax2.plot(list(depths), test_accs, 'r-o', label='Test Accuracy', linewidth=2)
ax2.axvline(x=best_depth, color='green', linestyle='--', lw=1.5, label=f'Best depth={best_depth}')
ax2.set_xlabel('Max Depth')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy vs Tree Depth\n(Overfitting Analysis)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Confusion Matrix
ax3 = fig.add_subplot(2, 2, 4)
cm = confusion_matrix(y_test, y_pred_p)
im = ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
ax3.set_title('Confusion Matrix (Pruned Tree)')
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')
ax3.set_xticks(range(3)); ax3.set_yticks(range(3))
ax3.set_xticklabels(class_names, rotation=15)
ax3.set_yticklabels(class_names)
for i in range(3):
    for j in range(3):
        ax3.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=14)
plt.colorbar(im, ax=ax3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/level2_task2_decision_tree.png', dpi=150, bbox_inches='tight')
print("\n📊 Plot saved.")
print("\n" + "=" * 60)
print("DECISION TREE COMPLETE")
print("=" * 60)
