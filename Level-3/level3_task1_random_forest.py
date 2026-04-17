"""
Level 3 - Task 1: Random Forest Classifier
Dataset: Churn Prediction Dataset
Objectives:
  - Train Random Forest & tune hyperparameters
  - Evaluate with cross-validation and classification metrics
  - Feature importance analysis
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report, confusion_matrix)

# ─────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────
df = pd.read_csv("datasets/Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv")

print("=" * 60)
print("STEP 1: Loading & Preprocessing")
print("=" * 60)

le = LabelEncoder()
for col in ['State', 'International plan', 'Voice mail plan', 'Churn']:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Churn'])
y = df['Churn']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 2. Baseline Random Forest
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Baseline Random Forest")
print("=" * 60)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred_base):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_base):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_base):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_base):.4f}")

# ─────────────────────────────────────────
# 3. Hyperparameter Tuning
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Hyperparameter Tuning (GridSearchCV)")
print("=" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=3, scoring='f1', n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV F1-Score: {grid_search.best_score_:.4f}")

# ─────────────────────────────────────────
# 4. Best Model Evaluation
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Best Model Evaluation")
print("=" * 60)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

cv_scores = cross_val_score(best_rf, X, y, cv=5, scoring='f1')

print(f"Accuracy           : {acc:.4f}")
print(f"Precision          : {prec:.4f}")
print(f"Recall             : {rec:.4f}")
print(f"F1-Score           : {f1:.4f}")
print(f"5-Fold CV F1 Mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ─────────────────────────────────────────
# 5. Feature Importance
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Feature Importance Analysis")
print("=" * 60)

importances = best_rf.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df = feat_imp_df.sort_values('Importance', ascending=False)

print(feat_imp_df.to_string(index=False))
print(f"\nTop 3 most important features:")
for _, row in feat_imp_df.head(3).iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# ─────────────────────────────────────────
# 6. Visualizations
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Random Forest Classifier - Customer Churn', fontsize=14, fontweight='bold')

# Feature Importance
top15 = feat_imp_df.head(15)
axes[0].barh(top15['Feature'][::-1], top15['Importance'][::-1],
             color='steelblue', edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Importance Score')
axes[0].set_title('Top 15 Feature Importances')
axes[0].grid(True, alpha=0.3, axis='x')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, interpolation='nearest', cmap=plt.cm.Oranges)
axes[1].set_title('Confusion Matrix')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')
axes[1].set_xticks([0, 1]); axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['No Churn', 'Churn'])
axes[1].set_yticklabels(['No Churn', 'Churn'])
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
plt.colorbar(im, ax=axes[1])

# n_estimators vs OOB error approximation (cross-val)
n_trees = [10, 50, 100, 150, 200]
cv_f1s = []
for n in n_trees:
    rf_tmp = RandomForestClassifier(n_estimators=n, **{k: v for k, v in grid_search.best_params_.items() if k != 'n_estimators'}, random_state=42, n_jobs=-1)
    scores = cross_val_score(rf_tmp, X, y, cv=3, scoring='f1')
    cv_f1s.append(scores.mean())

axes[2].plot(n_trees, cv_f1s, 'g-o', lw=2, markersize=8)
axes[2].set_xlabel('Number of Trees')
axes[2].set_ylabel('CV F1-Score')
axes[2].set_title('F1-Score vs Number of Trees')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/level3_task1_random_forest.png', dpi=150, bbox_inches='tight')
print("\n📊 Plot saved.")
print("\n" + "=" * 60)
print("RANDOM FOREST COMPLETE")
print("=" * 60)
