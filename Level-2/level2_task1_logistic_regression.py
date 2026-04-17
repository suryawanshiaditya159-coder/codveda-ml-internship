"""
Level 2 - Task 1: Logistic Regression for Binary Classification
Dataset: Churn Prediction Dataset
Objectives:
  - Load and preprocess dataset
  - Train logistic regression model
  - Interpret coefficients and odds ratio
  - Evaluate with accuracy, precision, recall, ROC curve
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)

# ─────────────────────────────────────────
# 1. Load & Preprocess
# ─────────────────────────────────────────
df = pd.read_csv("datasets/Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv")

print("=" * 60)
print("STEP 1: Loading & Preprocessing")
print("=" * 60)
print(f"Shape: {df.shape}")

# Encode categoricals
le = LabelEncoder()
for col in ['State', 'International plan', 'Voice mail plan', 'Churn']:
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Churn'])
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
print(f"Churn rate: {y.mean()*100:.1f}%")

# ─────────────────────────────────────────
# 2. Train Logistic Regression
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Training Logistic Regression")
print("=" * 60)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")

# ─────────────────────────────────────────
# 3. Coefficients & Odds Ratio
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Coefficients & Odds Ratios")
print("=" * 60)

feature_names = df.drop(columns=['Churn']).columns.tolist()
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_[0],
    'Odds Ratio': np.exp(model.coef_[0])
}).sort_values('Odds Ratio', ascending=False)

print(coef_df.to_string(index=False))
print("\nInterpretation: Odds Ratio > 1 → increases churn probability")

# ─────────────────────────────────────────
# 4. Evaluate Model
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print(f"AUC-ROC  : {roc_auc:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ─────────────────────────────────────────
# 5. Visualizations
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Logistic Regression - Customer Churn Prediction', fontsize=14, fontweight='bold')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axes[0].set_title('Confusion Matrix')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')
axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['No Churn', 'Churn'])
axes[0].set_yticklabels(['No Churn', 'Churn'])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max()/2 else 'black', fontsize=16)
plt.colorbar(im, ax=axes[0])

# ROC Curve
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
axes[1].plot([0, 1], [0, 1], 'r--', lw=1.5, label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc='lower right')
axes[1].grid(True, alpha=0.3)

# Top Feature Odds Ratios
top10 = coef_df.head(10)
colors = ['green' if v > 1 else 'red' for v in top10['Odds Ratio']]
axes[2].barh(top10['Feature'], top10['Odds Ratio'], color=colors, edgecolor='black', linewidth=0.5)
axes[2].axvline(x=1, color='black', linestyle='--', lw=1.5)
axes[2].set_xlabel('Odds Ratio')
axes[2].set_title('Top 10 Feature Odds Ratios\n(>1 increases churn risk)')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/level2_task1_logistic_regression.png', dpi=150, bbox_inches='tight')
print("\n📊 Plot saved.")
print("\n" + "=" * 60)
print("LOGISTIC REGRESSION COMPLETE")
print("=" * 60)
