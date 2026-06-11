import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

# Load Phase 4 train output
df = pd.read_parquet('phase4_risk_scores_train.parquet')

# Load Year 3 ground truth
holdout = pd.read_parquet('phase2_feature_matrix_test.parquet')[
    ['lsoa21cd', 'severity_weighted_count']
].rename(columns={'severity_weighted_count': 'sev_y3'})

df = df.merge(holdout, on='lsoa21cd', how='left')

# Build the three targets from Year 3 severity
df['actual_high_demand_y3']     = (df['sev_y3'] >= df['sev_y3'].quantile(0.85)).astype(int)
df['actual_elevated_demand_y3'] = (df['sev_y3'] >= df['sev_y3'].quantile(0.50)).astype(int)
df['actual_any_demand_y3']      = (df['sev_y3'] > 0).astype(int)

# Extract the continuous risk score (prediction) and binary target (ground truth)
y_true = df['actual_high_demand_year3']
y_scores = df['risk_score_scaled']

#ROC-AUC and Youden's J
def youden_threshold(y_true, y_scores):
    """
    Compute the optimal threshold via Youden's J statistic.
    J = TPR - FPR; maximised where sensitivity + specificity is jointly highest.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    j = tpr - fpr
    idx = np.argmax(j)
    return thresholds[idx], fpr, tpr, thresholds, auc, idx

cut_tier1, fpr1, tpr1, thr1, auc1, idx1 = youden_threshold(
    df['actual_high_demand_y3'], y_scores
)
cut_tier2, fpr2, tpr2, thr2, auc2, idx2 = youden_threshold(
    df['actual_elevated_demand_y3'], y_scores
)
cut_tier3, fpr3, tpr3, thr3, auc3, idx3 = youden_threshold(
    df['actual_any_demand_y3'], y_scores
)

# Enforce strict ordering: cut_tier1 > cut_tier2 > cut_tier3
cut_tier2 = min(cut_tier2, cut_tier1 - 0.1)
cut_tier3 = min(cut_tier3, cut_tier2 - 0.1)

# Output the results
print("=" * 60)
print("ROC-AUC SCORES")
print(f"  Tier 1 (High Risk vs rest):           AUC = {auc1:.4f}")
print(f"  Tier 2 (Elevated or above vs rest):   AUC = {auc2:.4f}")
print(f"  Tier 3 (Any demand vs near-zero):     AUC = {auc3:.4f}")
print()
print("YOUDEN'S J — OPTIMAL CUT-POINTS")
print(f"  Tier 1 / Tier 2 boundary:  score ≥ {cut_tier1:.2f}")
print(f"  Tier 2 / Tier 3 boundary:  score ≥ {cut_tier2:.2f}")
print(f"  Tier 3 / Tier 4 boundary:  score ≥ {cut_tier3:.2f}")
print("=" * 60)

def assign_supervised_tier(score):
    if score >= cut_tier1:
        return 'Tier 1: High Risk'
    elif score >= cut_tier2:
        return 'Tier 2: Elevated Risk'
    elif score >= cut_tier3:
        return 'Tier 3: Moderate Risk'
    else:
        return 'Tier 4: Low Risk'

df['supervised_priority_tier'] = df['risk_score_scaled'].apply(assign_supervised_tier)
df['predicted_high_risk'] = (df['risk_score_scaled'] >= cut_tier1).astype(int)

# Generate a Confusion Matrix
cm = confusion_matrix(df['actual_high_demand_y3'], df['predicted_high_risk'])
print("CONFUSION MATRIX")
print(f"  True Negatives  (correctly assigned to Tier 2–4): {cm[0, 0]:>5}")
print(f"  False Positives (assigned Tier 1, not genuinely): {cm[0, 1]:>5}")
print(f"  False Negatives (missed genuine High Risk LSOAs): {cm[1, 0]:>5}")
print(f"  True Positives  (correctly identified High Risk):  {cm[1, 1]:>5}")

# Tier size summary
tier_counts = df['supervised_priority_tier'].value_counts().sort_index()
print("\nTIER DISTRIBUTION")
for tier, count in tier_counts.items():
    pct = count / len(df) * 100
    print(f"  {tier:<35} {count:>5} LSOAs  ({pct:.1f}%)")

# Plots 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Phase 5 — Supervised Operational Threshold Validation', fontsize=14, y=1.01)

roc_configs = [
    (fpr1, tpr1, auc1, idx1, thr1, cut_tier1, 'darkorange',
     'Tier 1: High Risk vs rest'),
    (fpr2, tpr2, auc2, idx2, thr2, cut_tier2, 'steelblue',
     'Tier 2: Elevated or above vs rest'),
    (fpr3, tpr3, auc3, idx3, thr3, cut_tier3, 'seagreen',
     'Tier 3: Any demand vs near-zero'),
]

for ax, (fpr, tpr, auc, idx, thr, cut, colour, label) in zip(axes, roc_configs):
    ax.plot(fpr, tpr, color=colour, lw=2,
            label=f'AUC = {auc:.3f}')
    ax.plot([0, 1], [0, 1], color='grey', lw=1.5, linestyle='--',
            label='Random (AUC = 0.50)')
    ax.scatter(fpr[idx], tpr[idx], color='red', s=90, zorder=5,
               label=f'Youden cut-point: {cut:.2f}')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate (1 − Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(label)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('phase5_supervised_roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# Risk score distribution coloured by supervised tier
tier_colours = {
    'Tier 1: High Risk':      '#d62728',
    'Tier 2: Elevated Risk':  '#ff7f0e',
    'Tier 3: Moderate Risk':  '#2ca02c',
    'Tier 4: Low Risk':       '#1f77b4',
}

fig2, ax2 = plt.subplots(figsize=(12, 5))
for tier, colour in tier_colours.items():
    subset = df.loc[df['supervised_priority_tier'] == tier, 'risk_score_scaled']
    ax2.hist(subset, bins=60, color=colour, alpha=0.75, label=tier, edgecolor='none')

for cut, label, colour in [
    (cut_tier1, f'Tier 1 cut ({cut_tier1:.1f})', '#d62728'),
    (cut_tier2, f'Tier 2 cut ({cut_tier2:.1f})', '#ff7f0e'),
    (cut_tier3, f'Tier 3 cut ({cut_tier3:.1f})', '#2ca02c'),
]:
    ax2.axvline(cut, color=colour, linestyle='--', lw=1.8, label=label)

ax2.set_xlabel('Risk Score (0–100)')
ax2.set_ylabel('Number of LSOAs')
ax2.set_title('Risk Score Distribution by Supervised Priority Tier')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('phase5_supervised_score_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Output
df.to_parquet('phase5_supervised_output.parquet', index=False)