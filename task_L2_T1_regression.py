"""
Codveda Data Analytics Internship
Level 2 - Task 1: Regression Analysis
Dataset: House Prediction Data Set (Boston Housing)
Author: Nadine
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────
# 1. Load & Parse Dataset
#    (Boston housing – header is in row 0 as data)
# ─────────────────────────────────────────
COLUMNS = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
           'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

raw = pd.read_csv("datasets/Data Set For Task/4) house Prediction Data Set.csv", header=None)
data_rows = []
for _, row in raw.iterrows():
    values = str(row.iloc[0]).split()
    if len(values) == 14:
        data_rows.append([float(v) for v in values])

df = pd.DataFrame(data_rows, columns=COLUMNS)

print("=" * 55)
print("STEP 1: Dataset Overview")
print("=" * 55)
print(f"Shape: {df.shape}")
print(f"\nFeature descriptions:")
print("  CRIM   – per capita crime rate")
print("  RM     – avg rooms per dwelling")
print("  LSTAT  – % lower-status population")
print("  MEDV   – Median home value ($1000s) [TARGET]")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nDescriptive Stats:\n{df.describe().round(2)}")

# ─────────────────────────────────────────
# 2. Feature & Target Selection
# ─────────────────────────────────────────
# Use all features for multivariate regression (more interesting than simple)
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# ─────────────────────────────────────────
# 3. Train/Test Split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────
# 4. Scale + Fit Linear Regression
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_sc, y_train)

# ─────────────────────────────────────────
# 5. Evaluation
# ─────────────────────────────────────────
y_pred = model.predict(X_test_sc)

r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "=" * 55)
print("STEP 5: Model Evaluation")
print("=" * 55)
print(f"  R² Score  : {r2:.4f}")
print(f"  MSE       : {mse:.4f}")
print(f"  RMSE      : {rmse:.4f}")
print(f"  MAE       : {mae:.4f}")

print("\nFeature Coefficients (standardized):")
coeff_df = pd.DataFrame({
    'Feature': COLUMNS[:-1],
    'Coefficient': model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)
print(coeff_df.to_string(index=False))

# ─────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('#f8f9fa')

# Plot 1: Actual vs Predicted
ax = axes[0]
ax.set_facecolor('#f0f0f0')
ax.scatter(y_test, y_pred, alpha=0.6, color='#4C72B0', edgecolors='white', s=50)
lims = [min(y_test.min(), y_pred.min())-2, max(y_test.max(), y_pred.max())+2]
ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Fit')
ax.set_xlabel('Actual MEDV ($1000s)')
ax.set_ylabel('Predicted MEDV ($1000s)')
ax.set_title(f'Actual vs Predicted\nR² = {r2:.4f}', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_pred
ax = axes[1]
ax.set_facecolor('#f0f0f0')
ax.scatter(y_pred, residuals, alpha=0.6, color='#55A868', edgecolors='white', s=50)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted MEDV ($1000s)')
ax.set_ylabel('Residuals')
ax.set_title('Residual Plot', fontweight='bold')
ax.grid(alpha=0.3)

# Plot 3: Feature Coefficients
ax = axes[2]
ax.set_facecolor('#f0f0f0')
colors = ['#C44E52' if c < 0 else '#4C72B0' for c in coeff_df['Coefficient']]
bars = ax.barh(coeff_df['Feature'], coeff_df['Coefficient'], color=colors, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (Standardized)')
ax.set_title('Feature Coefficients', fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.suptitle('House Price Regression Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('L2_T1_Regression.png', dpi=150, bbox_inches='tight')
print("\n✅ Regression plots saved to: L2_T1_Regression.png")

print("\n" + "=" * 55)
print("KEY INSIGHTS")
print("=" * 55)
print(f"• Model explains {r2*100:.1f}% of variance in house prices")
print("• LSTAT (lower-status %) negatively impacts prices most")
print("• RM (avg rooms) is the strongest positive predictor")
print(f"• Predictions are off by ~${rmse:.1f}k on average (RMSE)")
