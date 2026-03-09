# E-Commerce Project - Step 3: ML Model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve, f1_score,
                              precision_recall_curve)
import joblib

OUT = 'outputs/'

import os
os.makedirs(OUT, exist_ok=True)

print("=" * 55)
print("  STEP 3: ML Model - Late Delivery Prediction")
print("=" * 55)

# ── Upload الـ Master Features ──
print("\n master_features.csv...")
df = pd.read_csv(OUT + 'master_features.csv')
print(f" Loaded: {df.shape[0]:,} orders | {df.shape[1]} features")
print(f" Late: {df['is_late'].sum():.0f} ({df['is_late'].mean()*100:.1f}%) | On Time: {(df['is_late']==0).sum():.0f}")

# ──Choose the Features ──
feature_cols = [
    'approval_days', 'estimated_delivery_days',
    'purchase_hour', 'purchase_month', 'is_weekend',
    'total_payment', 'max_installments',
    'num_items', 'total_price', 'total_freight',
    'freight_ratio', 'num_sellers', 'num_categories',
    'customer_state_enc'
]
pay_cols = [c for c in df.columns if c.startswith('pay_')]
feature_cols += pay_cols

df_model = df[feature_cols + ['is_late']].dropna()
X = df_model[feature_cols]
y = df_model['is_late'].astype(int)

print(f"\n Training Data: {len(X):,} rows | {len(feature_cols)} features")

# ── Train/Test Split ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"🔀 Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── Model training ──
print("\n جاري تدريب الموديلات...")

models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'), True),
    'Random Forest':       (RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1), False),
    'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42), False),
}

results = {}
for name, (model, use_scale) in models.items():
    Xtr = X_train_sc if use_scale else X_train
    Xte = X_test_sc  if use_scale else X_test
    model.fit(Xtr, y_train)
    y_prob = model.predict_proba(Xte)[:, 1]

    # threshold
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores_thresh = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh = thresholds[np.argmax(f1_scores_thresh[:-1])] if len(thresholds) > 0 else 0.5
    y_pred = (y_prob >= best_thresh).astype(int)

    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'auc': roc_auc_score(y_test, y_prob),
        'f1':  f1_score(y_test, y_pred),
        'threshold': best_thresh,
        'use_scale': use_scale
    }
    print(f"  [{name}] AUC={results[name]['auc']:.4f} | F1={results[name]['f1']:.4f} | Threshold={best_thresh:.2f}")

# ── The best model──
best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
print(f"\n Best Model: {best_name} (AUC={best['auc']:.4f})")
print(f"\n Classification Report:")
print(classification_report(y_test, best['y_pred'], target_names=['On Time', 'Late']))

# ── رسم النتايج ──
print(" Results are being plotted...")

DARK='#0f172a'; CARD='#1e293b'; TEXT='#e2e8f0'; SUB='#94a3b8'
PURP='#6366f1'; GREEN='#10b981'; AMBER='#f59e0b'; RED='#ef4444'; BLUE='#3b82f6'

fig = plt.figure(figsize=(18, 5), facecolor=DARK)
fig.suptitle(f' ML Results: Late Delivery Prediction | Best: {best_name} (AUC={best["auc"]:.3f})',
             color=TEXT, fontsize=13, fontweight='bold')
axes = [fig.add_subplot(1, 3, i+1) for i in range(3)]

# 1. Confusion Matrix
cm = confusion_matrix(y_test, best['y_pred'])
axes[0].imshow(cm, cmap='Blues')
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['On Time','Late'], color=TEXT)
axes[0].set_yticklabels(['On Time','Late'], color=TEXT)
axes[0].set_xlabel('Predicted', color=SUB)
axes[0].set_ylabel('Actual', color=SUB)
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                     color='white' if cm[i,j] > cm.max()/2 else 'black', fontsize=12)
axes[0].set_title('Confusion Matrix', color=TEXT, fontweight='bold')
axes[0].set_facecolor(CARD)

# 2. ROC Curves
colors = [PURP, GREEN, AMBER]
for (name, res), col in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    axes[1].plot(fpr, tpr, color=col, linewidth=2, label=f"{name} ({res['auc']:.3f})")
axes[1].plot([0,1],[0,1],'--', color='#475569')
axes[1].set_title('ROC Curves - All Models', color=TEXT, fontweight='bold')
axes[1].set_xlabel('False Positive Rate', color=SUB)
axes[1].set_ylabel('True Positive Rate', color=SUB)
axes[1].legend(fontsize=9, facecolor=CARD, labelcolor=TEXT)
axes[1].set_facecolor(CARD)
axes[1].tick_params(colors=SUB)

# 3. Feature Importance
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
axes[2].barh(range(len(importances)), importances.values[::-1], color=GREEN, edgecolor='none')
axes[2].set_yticks(range(len(importances)))
axes[2].set_yticklabels(importances.index[::-1], color=SUB, fontsize=9)
axes[2].set_title('Top 12 Feature Importances', color=TEXT, fontweight='bold')
axes[2].set_facecolor(CARD)
axes[2].tick_params(colors=SUB)

for ax in axes:
    for spine in ax.spines.values(): spine.set_edgecolor('#334155')

plt.tight_layout()
plt.savefig(OUT + 'ML_Results.png', dpi=150, bbox_inches='tight', facecolor=DARK)

# ── Save Model──
best_model_obj = results[best_name]['model']
joblib.dump(best_model_obj, OUT + 'best_model.pkl')
joblib.dump(scaler,         OUT + 'scaler.pkl')

print(f"\n The image was saved in: output/ML_Results.png")
print(f" The image was saved in: outputs/best_model.pkl")


print("\n" + "=" * 55)
print("   ملخص نتايج الموديلات")
print("=" * 55)
for name, res in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
    print(f"  {name:25s} AUC={res['auc']:.4f} | F1={res['f1']:.4f}")

print("\n" + "=" * 55)
print("   outputs/")
print("=" * 55)
