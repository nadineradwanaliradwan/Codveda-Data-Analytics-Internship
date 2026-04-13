"""
Codveda Data Analytics Internship
Level 1 - Task 2: Exploratory Data Analysis (EDA)
Dataset: Iris CSV
Author: Nadine
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# 1. Load & Quick Inspection
# ─────────────────────────────────────────
df = pd.read_csv("iris_cleaned.csv")

print("=" * 55)
print("STEP 1: Dataset Overview")
print("=" * 55)
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")

# ─────────────────────────────────────────
# 2. Summary Statistics
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 2: Summary Statistics")
print("=" * 55)
print(df.describe().round(3))

print("\nMode per column:")
print(df.mode().iloc[0])

print("\nSpecies distribution:")
print(df['species'].value_counts())

# ─────────────────────────────────────────
# 3. Correlation Analysis
# ─────────────────────────────────────────
print("\n" + "=" * 55)
print("STEP 3: Correlation Matrix (Numerical Features)")
print("=" * 55)
num_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
corr = df[num_cols].corr()
print(corr.round(3))

print("\nTop correlations:")
corr_pairs = corr.unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1.0].drop_duplicates()
print(corr_pairs.head(5).round(3))

# ─────────────────────────────────────────
# 4. Visualizations
# ─────────────────────────────────────────
palette = {'setosa': '#4C72B0', 'versicolor': '#55A868', 'virginica': '#C44E52'}
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#f8f9fa')

# --- Plot 1: Histograms ---
for i, col in enumerate(num_cols, 1):
    ax = fig.add_subplot(4, 3, i)
    ax.set_facecolor('#f0f0f0')
    for species, color in palette.items():
        subset = df[df['species'] == species][col]
        ax.hist(subset, bins=15, alpha=0.6, color=color, label=species, edgecolor='white')
    ax.set_title(f'Distribution of {col.replace("_", " ").title()}', fontsize=10, fontweight='bold')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

# --- Plot 2: Boxplots per species ---
for i, col in enumerate(num_cols, 5):
    ax = fig.add_subplot(4, 3, i)
    ax.set_facecolor('#f0f0f0')
    data_by_species = [df[df['species'] == s][col].dropna().values for s in palette]
    bp = ax.boxplot(data_by_species, patch_artist=True, labels=list(palette.keys()),
                    medianprops=dict(color='black', linewidth=2))
    for patch, color in zip(bp['boxes'], palette.values()):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title(f'Boxplot: {col.replace("_", " ").title()}', fontsize=10, fontweight='bold')
    ax.set_ylabel(col)
    ax.grid(axis='y', alpha=0.3)

# --- Plot 3: Correlation Heatmap ---
ax = fig.add_subplot(4, 3, 9)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
ax.set_title('Feature Correlation Heatmap', fontsize=10, fontweight='bold')

# --- Plot 4: Petal Length vs Petal Width Scatter ---
ax = fig.add_subplot(4, 3, 10)
ax.set_facecolor('#f0f0f0')
for species, color in palette.items():
    subset = df[df['species'] == species]
    ax.scatter(subset['petal_length'], subset['petal_width'],
               color=color, label=species, alpha=0.7, s=40, edgecolors='white')
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_title('Petal Length vs Petal Width', fontsize=10, fontweight='bold')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# --- Plot 5: Sepal Length vs Sepal Width Scatter ---
ax = fig.add_subplot(4, 3, 11)
ax.set_facecolor('#f0f0f0')
for species, color in palette.items():
    subset = df[df['species'] == species]
    ax.scatter(subset['sepal_length'], subset['sepal_width'],
               color=color, label=species, alpha=0.7, s=40, edgecolors='white')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Sepal Width')
ax.set_title('Sepal Length vs Sepal Width', fontsize=10, fontweight='bold')
ax.legend(fontsize=7)
ax.grid(alpha=0.3)

# --- Plot 6: Species Count ---
ax = fig.add_subplot(4, 3, 12)
ax.set_facecolor('#f0f0f0')
counts = df['species'].value_counts()
bars = ax.bar(counts.index, counts.values, color=list(palette.values()), edgecolor='white', width=0.5)
ax.set_title('Sample Count per Species', fontsize=10, fontweight='bold')
ax.set_ylabel('Count')
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha='center', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.suptitle('Iris Dataset – Exploratory Data Analysis', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('L1_T2_EDA.png', dpi=150, bbox_inches='tight')
print("\n✅ EDA plots saved to: L1_T2_EDA.png")

print("\n" + "=" * 55)
print("KEY INSIGHTS")
print("=" * 55)
print("• Petal length & petal width are highly correlated (r ≈ 0.96)")
print("• Setosa is clearly separable from the other two species on petal features")
print("• Virginica has the largest petals; Setosa the smallest")
print("• Sepal width shows least variation across species")
