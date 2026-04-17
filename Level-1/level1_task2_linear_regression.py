"""
Level 1 - Task 2: Build a Simple Linear Regression Model
Dataset: House Prices Dataset
Objectives:
  - Load and preprocess the dataset
  - Train a linear regression model using scikit-learn
  - Interpret model coefficients
  - Evaluate using R-squared and MSE
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
col_names = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
    'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]
df = pd.read_csv(
    "datasets/Data Set For Task/4) house Prediction Data Set.csv",
    sep=r'\s+', header=None, names=col_names
)

print("=" * 60)
print("STEP 1: Dataset Overview")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nDescriptive Stats:\n{df.describe()}")

# ─────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)
print(f"Missing values: {df.isnull().sum().sum()}")

X = df.drop(columns=['MEDV'])
y = df['MEDV']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────
# 3. Train Linear Regression Model
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Training Linear Regression Model")
print("=" * 60)

model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully!")

# ─────────────────────────────────────────
# 4. Interpret Coefficients
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Model Coefficients")
print("=" * 60)
coef_df = pd.DataFrame({'Feature': col_names[:-1], 'Coefficient': model.coef_})
coef_df = coef_df.sort_values('Coefficient', ascending=False)
print(coef_df.to_string(index=False))
print(f"\nIntercept: {model.intercept_:.4f}")
print("\nInterpretation:")
print("  Positive coef → feature increases house price")
print("  Negative coef → feature decreases house price")
print(f"  Top positive: {coef_df.iloc[0]['Feature']} ({coef_df.iloc[0]['Coefficient']:.3f})")
print(f"  Top negative: {coef_df.iloc[-1]['Feature']} ({coef_df.iloc[-1]['Coefficient']:.3f})")

# ─────────────────────────────────────────
# 5. Evaluate Model
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE) : {mse:.4f}")
print(f"Root MSE (RMSE)          : {rmse:.4f}")
print(f"R-squared (R²)           : {r2:.4f}")
print(f"\nR² = {r2:.4f} means the model explains {r2*100:.1f}% of variance in house prices.")

# ─────────────────────────────────────────
# 6. Visualizations
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Linear Regression - House Price Prediction', fontsize=14, fontweight='bold')

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.6, color='steelblue', edgecolors='k', linewidths=0.3)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Price (MEDV)')
axes[0].set_ylabel('Predicted Price')
axes[0].set_title(f'Actual vs Predicted\nR² = {r2:.4f}')
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6, color='coral', edgecolors='k', linewidths=0.3)
axes[1].axhline(y=0, color='black', linestyle='--', lw=2)
axes[1].set_xlabel('Predicted Price')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

# Plot 3: Feature Coefficients
colors = ['green' if c > 0 else 'red' for c in coef_df['Coefficient']]
axes[2].barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor='black', linewidth=0.5)
axes[2].axvline(x=0, color='black', linestyle='-', lw=1)
axes[2].set_xlabel('Coefficient Value')
axes[2].set_title('Feature Coefficients\n(Green=Positive, Red=Negative)')
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/level1_task2_linear_regression.png', dpi=150, bbox_inches='tight')
print("\n📊 Plot saved.")
print("\n" + "=" * 60)
print("LINEAR REGRESSION COMPLETE")
print("=" * 60)
