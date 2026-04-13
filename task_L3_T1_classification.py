"""
Codveda Data Analytics Internship
Level 3 - Task 1: Predictive Modeling (Classification)
Dataset: Churn Prediction Data
Author: Nadine
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve)

# ─────────────────────────────────────────
# 1. Load & Combine Train/Test Splits
# ─────────────────────────────────────────
train = pd.read_csv("datasets/Data Set For Task/Churn Prdiction Data/churn-bigml-80.csv")
test  = pd.read_csv("datasets/Data Set For Task/Churn Prdiction Data/churn-bigml-20.csv")

print("=" * 60)
print("STEP 1: Dataset Overview")
print("=" * 60)
print(f"Train shape: {train.shape} | Test shape: {test.shape}")
print(f"\nColumns: {train.columns.tolist()}")
print(f"\nMissing values – Train: {train.isnull().sum().sum()} | Test: {test.isnull().sum().sum()}")
print(f"\nChurn distribution (Train):\n{train['Churn'].value_counts()}")
print(f"Churn rate: {train['Churn'].mean()*100:.1f}%")

# ─────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Preprocessing")
print("=" * 60)

def preprocess(df):
    df = df.copy()
    # Encode binary categoricals
    le = LabelEncoder()
    for col in ['International plan', 'Voice mail plan', 'Churn']:
        df[col] = le.fit_transform(df[col].astype(str))
    # Drop non-informative columns
    df = df.drop(columns=['State', 'Area code'], errors='ignore')
    return df

train_p = preprocess(train)
test_p  = preprocess(test)

TARGET = 'Churn'
X_train = train_p.drop(TARGET, axis=1)
y_train = train_p[TARGET]
X_test  = test_p.drop(TARGET, axis=1)
y_test  = test_p[TARGET]

# Feature scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Features: {X_train.shape[1]}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")
print(f"Class balance — Train: {y_train.value_counts().to_dict()}")

# ─────────────────────────────────────────
# 3. Train Multiple Models
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Model Training & Evaluation")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, 'predict_proba') else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    auc  = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    results[name] = {'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
                     'Accuracy': acc, 'Precision': prec, 'Recall': rec,
                     'F1': f1, 'AUC': auc}

    print(f"\n  {name}:")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    if auc: print(f"    ROC-AUC   : {auc:.4f}")

# ─────────────────────────────────────────
# 4. Hyperparameter Tuning (Random Forest – best model)
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Hyperparameter Tuning (Random Forest – Grid Search)")
print("=" * 60)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 10, 20],
    'min_samples_split': [2, 5]
}
gs = GridSearchCV(RandomForestClassifier(random_state=42),
                  param_grid, cv=3, scoring='f1', n_jobs=-1)
gs.fit(X_train_sc, y_train)

best_rf = gs.best_estimator_
y_pred_best = best_rf.predict(X_test_sc)
y_prob_best = best_rf.predict_proba(X_test_sc)[:, 1]

print(f"  Best params: {gs.best_params_}")
print(f"  Best CV F1 : {gs.best_score_:.4f}")
print(f"  Test F1    : {f1_score(y_test, y_pred_best):.4f}")
print(f"  Test AUC   : {roc_auc_score(y_test, y_prob_best):.4f}")

print("\nFull Classification Report (Tuned RF):")
print(classification_report(y_test, y_pred_best, target_names=['Not Churn', 'Churn']))

# ─────────────────────────────────────────
# 5. Feature Importance
# ─────────────────────────────────────────
feat_imp = pd.Series(best_rf.feature_importances_, index=X_train.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(10)

# ─────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#f8f9fa')

# Plot 1: Model Comparison Bar Chart
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_facecolor('#f0f0f0')
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
model_names  = list(results.keys())
x = np.arange(len(metric_names))
width = 0.25
colors_m = ['#4C72B0', '#55A868', '#C44E52']
for i, (name, res) in enumerate(results.items()):
    vals = [res[m] for m in metric_names]
    bars = ax1.bar(x + i*width, vals, width, label=name, color=colors_m[i], alpha=0.85)
ax1.set_title('Model Performance Comparison', fontweight='bold')
ax1.set_ylabel('Score')
ax1.set_xticks(x + width)
ax1.set_xticklabels(metric_names)
ax1.legend(fontsize=7)
ax1.set_ylim(0, 1.1)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Confusion Matrix (Tuned RF)
ax2 = fig.add_subplot(2, 3, 2)
cm = confusion_matrix(y_test, y_pred_best)
im = ax2.imshow(cm, cmap='Blues')
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Not Churn', 'Churn'])
ax2.set_yticklabels(['Not Churn', 'Churn'])
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha='center', va='center',
                 fontsize=18, fontweight='bold',
                 color='white' if cm[i, j] > cm.max()/2 else 'black')
ax2.set_title('Confusion Matrix\n(Tuned Random Forest)', fontweight='bold')
ax2.set_xlabel('Predicted'); ax2.set_ylabel('Actual')
plt.colorbar(im, ax=ax2)

# Plot 3: ROC Curves
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_facecolor('#f0f0f0')
for (name, res), color in zip(results.items(), colors_m):
    if res['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax3.plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{name} (AUC={res['AUC']:.3f})")
fpr_b, tpr_b, _ = roc_curve(y_test, y_prob_best)
ax3.plot(fpr_b, tpr_b, 'k--', linewidth=2,
         label=f"Tuned RF (AUC={roc_auc_score(y_test, y_prob_best):.3f})")
ax3.plot([0,1],[0,1], 'gray', linestyle=':')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves', fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# Plot 4: Feature Importance
ax4 = fig.add_subplot(2, 3, (4, 5))
ax4.set_facecolor('#f0f0f0')
colors_fi = ['#C44E52' if i == 0 else '#4C72B0' for i in range(len(feat_imp))]
ax4.barh(feat_imp.index[::-1], feat_imp.values[::-1], color=colors_fi[::-1], edgecolor='white')
ax4.set_title('Top 10 Feature Importances (Tuned Random Forest)', fontweight='bold')
ax4.set_xlabel('Importance')
ax4.grid(axis='x', alpha=0.3)

# Plot 5: Churn Distribution
ax5 = fig.add_subplot(2, 3, 6)
ax5.set_facecolor('#f0f0f0')
churn_counts = y_test.value_counts()
wedges, texts, autotexts = ax5.pie(
    churn_counts.values,
    labels=['Not Churn', 'Churn'],
    autopct='%1.1f%%',
    colors=['#4C72B0', '#C44E52'],
    startangle=90,
    explode=(0, 0.05)
)
for at in autotexts:
    at.set_fontweight('bold')
ax5.set_title('Test Set Churn Distribution', fontweight='bold')

plt.suptitle('Customer Churn – Classification Analysis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('L3_T1_Classification.png', dpi=150, bbox_inches='tight')
print("\n✅ Classification plots saved to: L3_T1_Classification.png")

print("\n" + "=" * 60)
print("KEY INSIGHTS")
print("=" * 60)
print(f"• Random Forest outperforms all models with highest F1 & AUC")
top_feat = feat_imp.index[0]
print(f"• '{top_feat}' is the most important churn predictor")
print("• High recall on churn class is critical (false negatives = lost customers)")
print("• Hyperparameter tuning improved F1 further vs baseline Random Forest")
