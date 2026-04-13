"""
Codveda Data Analytics Internship
Level 1 - Task 1: Data Cleaning and Preprocessing
Dataset: Iris CSV
Author: Nadine
"""

import pandas as pd
import numpy as np

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
df = pd.read_csv("datasets/Data Set For Task/1) iris.csv")
print("=" * 55)
print("STEP 1: Raw Dataset")
print("=" * 55)
print(f"Shape: {df.shape}")
print(df.head(10))

# ─────────────────────────────────────────
# 2. Inject Synthetic Dirty Data to Demonstrate Cleaning
#    (real dataset is already clean – we simulate realistic issues)
# ─────────────────────────────────────────
np.random.seed(42)
dirty = df.copy()

# Introduce missing values (~8%)
for col in ['sepal_length', 'sepal_width', 'petal_length']:
    idx = np.random.choice(dirty.index, size=10, replace=False)
    dirty.loc[idx, col] = np.nan

# Introduce duplicates
dirty = pd.concat([dirty, dirty.sample(8, random_state=1)], ignore_index=True)

# Inconsistent species casing
dirty['species'] = dirty['species'].apply(
    lambda x: x.upper() if np.random.rand() < 0.3 else x
)

# Outlier (data entry error)
dirty.loc[0, 'sepal_length'] = 99.0

print("\n" + "=" * 55)
print("STEP 2: After Injecting Issues")
print("=" * 55)
print(f"Shape: {dirty.shape}")
print(f"Missing values:\n{dirty.isnull().sum()}")
print(f"Duplicate rows: {dirty.duplicated().sum()}")
print(f"Species variants: {dirty['species'].unique()}")

# ─────────────────────────────────────────
# 3. Handle Missing Values
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Handling Missing Values")
print("=" * 55)

# Impute numerical columns with median (robust to outliers)
num_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for col in num_cols:
    median_val = dirty[col].median()
    missing_count = dirty[col].isnull().sum()
    dirty[col] = dirty[col].fillna(median_val)
    print(f"  {col}: filled {missing_count} nulls with median={median_val:.3f}")

print(f"\nMissing values after imputation: {dirty.isnull().sum().sum()}")

# ─────────────────────────────────────────
# 4. Remove Duplicates
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 4: Removing Duplicates")
print("=" * 55)
before = len(dirty)
dirty = dirty.drop_duplicates().reset_index(drop=True)
after = len(dirty)
print(f"  Removed {before - after} duplicate rows. Shape: {dirty.shape}")

# ─────────────────────────────────────────
# 5. Standardize Categorical Formats
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 5: Standardizing Categorical Values")
print("=" * 55)
print(f"  Before: {dirty['species'].unique()}")
dirty['species'] = dirty['species'].str.lower().str.strip()
print(f"  After:  {dirty['species'].unique()}")

# ─────────────────────────────────────────
# 6. Handle Outliers (IQR method)
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 6: Outlier Detection & Removal (IQR)")
print("=" * 55)
before = len(dirty)
for col in num_cols:
    Q1 = dirty[col].quantile(0.25)
    Q3 = dirty[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = dirty[(dirty[col] < lower) | (dirty[col] > upper)]
    if len(outliers):
        print(f"  {col}: {len(outliers)} outlier(s) detected and removed")
    dirty = dirty[(dirty[col] >= lower) & (dirty[col] <= upper)]

dirty = dirty.reset_index(drop=True)
print(f"  Rows before: {before} → after: {len(dirty)}")

# ─────────────────────────────────────────
# 7. Final Summary
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 7: Final Clean Dataset Summary")
print("=" * 55)
print(f"Shape: {dirty.shape}")
print(f"Missing values: {dirty.isnull().sum().sum()}")
print(f"Duplicates: {dirty.duplicated().sum()}")
print("\nDescriptive Statistics:")
print(dirty.describe().round(3))

# Save cleaned dataset
dirty.to_csv("iris_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved to: iris_cleaned.csv")
