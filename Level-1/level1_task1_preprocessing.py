"""
Level 1 - Task 1: Data Preprocessing for Machine Learning
Dataset: Churn Prediction Dataset
Objectives:
  - Handle missing data
  - Encode categorical variables
  - Normalize/standardize numerical features
  - Split dataset into train/test sets
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv("datasets/Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv")

print("=" * 60)
print("STEP 1: Dataset Overview")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")

# ─────────────────────────────────────────
# 2. Handle Missing Data
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Missing Data Analysis")
print("=" * 60)
print(f"Missing values per column:\n{df.isnull().sum()}")

# Fill numerical missing values with median (robust to outliers)
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  Filled '{col}' missing values with median: {df[col].median():.2f}")

# Fill categorical missing values with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)
        print(f"  Filled '{col}' missing values with mode: {df[col].mode()[0]}")

print(f"\nMissing values after handling: {df.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 3. Encode Categorical Variables
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Encoding Categorical Variables")
print("=" * 60)

le = LabelEncoder()
categorical_cols = ['State', 'International plan', 'Voice mail plan', 'Churn']

for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])
        print(f"  Label-encoded '{col}'")

print(f"\nDataset after encoding (first 3 rows):\n{df.head(3)}")

# ─────────────────────────────────────────
# 4. Feature / Target Split
# ─────────────────────────────────────────
X = df.drop(columns=['Churn'])
y = df['Churn']

# ─────────────────────────────────────────
# 5. Normalize / Standardize Numerical Features
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Standardizing Numerical Features")
print("=" * 60)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(f"Before scaling - 'Account length' stats:\n  mean={df['Account length'].mean():.2f}, std={df['Account length'].std():.2f}")
print(f"After scaling  - 'Account length' stats:\n  mean={X_scaled['Account length'].mean():.4f}, std={X_scaled['Account length'].std():.4f}")

# ─────────────────────────────────────────
# 6. Train / Test Split
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Train/Test Split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Total samples   : {len(df)}")
print(f"Training samples: {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Testing samples : {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"\nClass distribution in train:\n{y_train.value_counts()}")
print(f"\nClass distribution in test:\n{y_test.value_counts()}")

# Save preprocessed data
X_train.to_csv("datasets/X_train.csv", index=False)
X_test.to_csv("datasets/X_test.csv", index=False)
y_train.to_csv("datasets/y_train.csv", index=False)
y_test.to_csv("datasets/y_test.csv", index=False)

print("\n✅ Preprocessed data saved to datasets/X_train.csv, X_test.csv, y_train.csv, y_test.csv")
print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
