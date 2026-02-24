#!/usr/bin/env python3
"""
NFPC Phase 1 & 2 — Complete Pipeline
Enhanced EDA + Feature Engineering + Model Training + Predictions
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ks_2samp, entropy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
import warnings
import os
import json
import gc
from collections import Counter

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

DATA_DIR = "/home/DIVYA/nfpc/IITD-Tryst-Hackathon/EDA-Phase-1"
OUT_DIR = "/home/DIVYA/nfpc/output"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

REF_DATE = pd.Timestamp('2025-06-30')

# ================================================================
# PART 1: DATA LOADING
# ================================================================
print("=" * 70)
print("PART 1: LOADING DATA")
print("=" * 70)

customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
accounts = pd.read_csv(os.path.join(DATA_DIR, "accounts.csv"))
linkage = pd.read_csv(os.path.join(DATA_DIR, "customer_account_linkage.csv"))
products = pd.read_csv(os.path.join(DATA_DIR, "product_details.csv"))
labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test_accounts.csv"))

print("Loading transactions...")
transactions = pd.concat(
    [pd.read_csv(os.path.join(DATA_DIR, f"transactions_part_{i}.csv")) for i in range(6)],
    ignore_index=True
)
print(f"Total transactions: {len(transactions):,}")

# Parse dates
for col in ['date_of_birth', 'relationship_start_date']:
    customers[col] = pd.to_datetime(customers[col], errors='coerce')
for col in ['account_opening_date', 'last_mobile_update_date', 'last_kyc_date', 'freeze_date', 'unfreeze_date']:
    accounts[col] = pd.to_datetime(accounts[col], errors='coerce')
transactions['transaction_timestamp'] = pd.to_datetime(transactions['transaction_timestamp'], errors='coerce')
labels['mule_flag_date'] = pd.to_datetime(labels['mule_flag_date'], errors='coerce')

# ================================================================
# PART 2: ENHANCED EDA WITH STATISTICAL TESTS
# ================================================================
print("\n" + "=" * 70)
print("PART 2: ENHANCED EDA WITH STATISTICAL TESTS")
print("=" * 70)

# Build merged training table
train = labels.merge(accounts, on='account_id', how='left')
train = train.merge(linkage, on='account_id', how='left')
train = train.merge(customers, on='customer_id', how='left')
train = train.merge(products, on='customer_id', how='left')

mule = train[train['is_mule'] == 1]
legit = train[train['is_mule'] == 0]

# --- EDA Section: Data Integrity & Schema Validation ---
print("\n--- Data Integrity Checks ---")
print(f"Unique customers: {customers['customer_id'].nunique()}")
print(f"Unique accounts: {accounts['account_id'].nunique()}")
print(f"Linkage entries: {len(linkage)}")
print(f"Orphan accounts (in linkage but not accounts): {len(set(linkage['account_id']) - set(accounts['account_id']))}")
print(f"Orphan customers (in linkage but not customers): {len(set(linkage['customer_id']) - set(customers['customer_id']))}")
print(f"Train/test overlap: {len(set(labels['account_id']) & set(test['account_id']))}")
print(f"All train accounts in accounts table: {labels['account_id'].isin(accounts['account_id']).all()}")
print(f"All test accounts in accounts table: {test['account_id'].isin(accounts['account_id']).all()}")
print(f"Duplicate txn IDs: {transactions['transaction_id'].duplicated().sum()}")

# --- Statistical Tests ---
print("\n--- Statistical Tests (Mann-Whitney U) ---")
stat_results = {}
numeric_test_cols = ['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance', 'num_chequebooks']
for col in numeric_test_cols:
    m_vals = mule[col].dropna()
    l_vals = legit[col].dropna()
    if len(m_vals) > 0 and len(l_vals) > 0:
        u_stat, p_val = mannwhitneyu(m_vals, l_vals, alternative='two-sided')
        ks_stat, ks_p = ks_2samp(m_vals, l_vals)
        effect_size = u_stat / (len(m_vals) * len(l_vals))  # rank-biserial
        stat_results[col] = {'mann_whitney_p': p_val, 'ks_p': ks_p, 'effect_size': effect_size}
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {col}: U-test p={p_val:.2e} {sig}, KS p={ks_p:.2e}, effect={effect_size:.4f}")

print("\n--- Chi-Square Tests (Categorical) ---")
cat_test_cols = ['account_status', 'product_family', 'nomination_flag', 'cheque_allowed',
                 'cheque_availed', 'kyc_compliant', 'rural_branch']
for col in cat_test_cols:
    ct = pd.crosstab(train['is_mule'], train[col])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p_val, dof, expected = chi2_contingency(ct)
        cramers_v = np.sqrt(chi2 / (len(train) * (min(ct.shape) - 1)))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {col}: chi2={chi2:.2f}, p={p_val:.2e} {sig}, Cramer's V={cramers_v:.4f}")

print("\n--- Chi-Square Tests (Customer KYC & Banking Flags) ---")
cust_cat_cols = ['pan_available', 'aadhaar_available', 'passport_available',
                 'mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
                 'demat_flag', 'credit_card_flag', 'fastag_flag']
for col in cust_cat_cols:
    ct = pd.crosstab(train['is_mule'], train[col])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p_val, dof, expected = chi2_contingency(ct)
        cramers_v = np.sqrt(chi2 / (len(train) * (min(ct.shape) - 1)))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {col}: chi2={chi2:.2f}, p={p_val:.2e} {sig}, Cramer's V={cramers_v:.4f}")

# --- Enhanced Visualizations ---
print("\n--- Generating Enhanced Plots ---")

# 1. Class distribution with annotation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
mule_counts = labels['is_mule'].value_counts()
bars = axes[0].bar(['Legitimate\n(n=23,760)', 'Mule\n(n=263)'], mule_counts.values,
                     color=['#27ae60', '#c0392b'], edgecolor='white', linewidth=1.5)
axes[0].set_title('Class Distribution — Severe Imbalance (1:90)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Number of Accounts', fontsize=11)
for bar, val in zip(bars, mule_counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                  f'{val:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[0].set_ylim(0, mule_counts.max() * 1.15)

# Log scale version
bars2 = axes[1].bar(['Legitimate', 'Mule'], mule_counts.values,
                      color=['#27ae60', '#c0392b'], edgecolor='white', linewidth=1.5)
axes[1].set_yscale('log')
axes[1].set_title('Class Distribution (Log Scale)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Number of Accounts (log)', fontsize=11)
for bar, val in zip(bars2, mule_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                  f'{val:,}\n({val/len(labels)*100:.2f}%)', ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_class_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()

# 2. Alert reason distribution
fig, ax = plt.subplots(figsize=(12, 5))
alert_counts = labels[labels['is_mule']==1]['alert_reason'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(alert_counts)))
bars = ax.barh(range(len(alert_counts)), alert_counts.values, color=colors, edgecolor='white')
ax.set_yticks(range(len(alert_counts)))
ax.set_yticklabels(alert_counts.index, fontsize=9)
ax.set_xlabel('Count', fontsize=11)
ax.set_title('Alert Reasons for Mule Accounts', fontsize=13, fontweight='bold')
for bar, val in zip(bars, alert_counts.values):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, str(val),
             va='center', fontweight='bold', fontsize=9)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '02_alert_reasons.png'), dpi=200, bbox_inches='tight')
plt.close()

# 3. Account status comparison (THE strongest signal)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
status_ct = pd.crosstab(train['is_mule'], train['account_status'], normalize='index')
status_ct.index = ['Legitimate', 'Mule']
status_ct.plot(kind='bar', ax=axes[0], color=['#27ae60', '#c0392b'], edgecolor='white')
axes[0].set_title('Account Status Distribution', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Proportion', fontsize=11)
axes[0].set_xticklabels(['Legitimate', 'Mule'], rotation=0, fontsize=11)
axes[0].legend(['Active', 'Frozen'], fontsize=10)

# Freeze rate comparison
freeze_data = pd.DataFrame({
    'Legitimate': [train.loc[train['is_mule']==0, 'freeze_date'].notna().mean()],
    'Mule': [train.loc[train['is_mule']==1, 'freeze_date'].notna().mean()]
}, index=['Ever Frozen'])
freeze_data.T.plot(kind='bar', ax=axes[1], color='#e74c3c', edgecolor='white', legend=False)
axes[1].set_title('Account Freeze Rate (⚠ Potential Leakage)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Proportion', fontsize=11)
axes[1].set_xticklabels(['Legitimate', 'Mule'], rotation=0, fontsize=11)
for i, v in enumerate(freeze_data.values[0]):
    axes[1].text(i, v + 0.02, f'{v:.1%}', ha='center', fontweight='bold', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '03_account_status_freeze.png'), dpi=200, bbox_inches='tight')
plt.close()

# 4. Balance distributions with box plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
balance_cols = ['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance']
for idx, col in enumerate(balance_cols):
    ax = axes[idx // 2][idx % 2]
    data = [legit[col].dropna().clip(-50000, 200000), mule[col].dropna().clip(-50000, 200000)]
    bp = ax.boxplot(data, labels=['Legitimate', 'Mule'], patch_artist=True,
                     boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red', linewidth=2))
    bp['boxes'][1].set_facecolor('#ffcccc')
    ax.set_title(f'{col}', fontsize=11, fontweight='bold')
    ax.set_ylabel('INR', fontsize=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.suptitle('Balance Distributions: Mule vs Legitimate', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '04_balance_boxplots.png'), dpi=200, bbox_inches='tight')
plt.close()

# 5. Account opening patterns
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    subset = train[train['is_mule'] == is_m]
    year_counts = subset['account_opening_date'].dt.year.value_counts(normalize=True).sort_index()
    axes[0].plot(year_counts.index, year_counts.values, label=label, color=color, marker='o', linewidth=2)
axes[0].set_title('Account Opening Year (Normalized)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Proportion')
axes[0].legend()

# Mule rate by account age bucket
train['acct_age_years'] = (REF_DATE - train['account_opening_date']).dt.days / 365.25
age_bins = pd.cut(train['acct_age_years'], bins=[0, 1, 2, 3, 5, 10, 20, 50])
mule_by_age = train.groupby(age_bins, observed=True)['is_mule'].agg(['mean', 'count'])
bars = axes[1].bar(range(len(mule_by_age)), mule_by_age['mean'].values, color='#c0392b', edgecolor='white')
axes[1].set_xticks(range(len(mule_by_age)))
axes[1].set_xticklabels([str(x) for x in mule_by_age.index], rotation=45, fontsize=9)
axes[1].set_title('Mule Rate by Account Age', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Mule Rate')
for bar, val, n in zip(bars, mule_by_age['mean'].values, mule_by_age['count'].values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                  f'{val:.3f}\n(n={n:,})', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '05_account_opening.png'), dpi=200, bbox_inches='tight')
plt.close()

# 6. Customer demographics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Age
train['age'] = (REF_DATE - train['date_of_birth']).dt.days / 365.25
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    subset = train[train['is_mule'] == is_m]['age'].dropna()
    axes[0].hist(subset, bins=40, alpha=0.5, density=True, label=label, color=color)
axes[0].set_title('Age Distribution', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Age (Years)')
axes[0].legend()

# Relationship tenure
train['rel_years'] = (REF_DATE - train['relationship_start_date']).dt.days / 365.25
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    subset = train[train['is_mule'] == is_m]['rel_years'].dropna()
    axes[1].hist(subset, bins=40, alpha=0.5, density=True, label=label, color=color)
axes[1].set_title('Relationship Tenure', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Years')
axes[1].legend()

# KYC document heatmap
kyc_data = pd.DataFrame({
    'Legit': [train.loc[train['is_mule']==0, c].eq('Y').mean() for c in ['pan_available', 'aadhaar_available', 'passport_available']],
    'Mule': [train.loc[train['is_mule']==1, c].eq('Y').mean() for c in ['pan_available', 'aadhaar_available', 'passport_available']]
}, index=['PAN', 'Aadhaar', 'Passport'])
sns.heatmap(kyc_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=axes[2], vmin=0, vmax=1, linewidths=1)
axes[2].set_title('KYC Document Availability', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '06_customer_demographics.png'), dpi=200, bbox_inches='tight')
plt.close()

# 7. Digital banking flags heatmap
fig, ax = plt.subplots(figsize=(10, 6))
flag_cols = ['mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
             'demat_flag', 'credit_card_flag', 'fastag_flag',
             'nomination_flag', 'cheque_availed', 'kyc_compliant', 'rural_branch']
flag_data = pd.DataFrame({
    'Legitimate': [train.loc[train['is_mule']==0, c].eq('Y').mean() for c in flag_cols],
    'Mule': [train.loc[train['is_mule']==1, c].eq('Y').mean() for c in flag_cols]
}, index=[c.replace('_flag', '').replace('_', ' ').title() for c in flag_cols])
flag_data['Difference'] = flag_data['Mule'] - flag_data['Legitimate']
sns.heatmap(flag_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax, linewidths=1)
ax.set_title('Account & Customer Flags: Mule vs Legitimate', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '07_flags_heatmap.png'), dpi=200, bbox_inches='tight')
plt.close()

# ================================================================
# PART 3: TRANSACTION-LEVEL ANALYSIS (Enhanced)
# ================================================================
print("\n" + "=" * 70)
print("PART 3: TRANSACTION-LEVEL ANALYSIS")
print("=" * 70)

# Join labels to transactions
train_acct_ids = set(labels['account_id'])
test_acct_ids = set(test['account_id'])
all_acct_ids = train_acct_ids | test_acct_ids

txn_labeled = transactions[transactions['account_id'].isin(train_acct_ids)].copy()
txn_labeled = txn_labeled.merge(labels[['account_id', 'is_mule']], on='account_id', how='left')

# Time features
txn_labeled['hour'] = txn_labeled['transaction_timestamp'].dt.hour
txn_labeled['dow'] = txn_labeled['transaction_timestamp'].dt.dayofweek
txn_labeled['day_of_month'] = txn_labeled['transaction_timestamp'].dt.day
txn_labeled['month'] = txn_labeled['transaction_timestamp'].dt.month
txn_labeled['year'] = txn_labeled['transaction_timestamp'].dt.year

print(f"Labeled txns: {len(txn_labeled):,}")

# 8. Channel usage radar chart
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
channel_ct = pd.crosstab(txn_labeled['is_mule'], txn_labeled['channel'], normalize='index')
top_channels = channel_ct.sum().nlargest(12).index
channel_sub = channel_ct[top_channels]

# Bar chart comparison
x = np.arange(len(top_channels))
w = 0.35
axes[0].bar(x - w/2, channel_sub.loc[0], w, label='Legitimate', color='#27ae60', edgecolor='white')
axes[0].bar(x + w/2, channel_sub.loc[1], w, label='Mule', color='#c0392b', edgecolor='white')
axes[0].set_xticks(x)
axes[0].set_xticklabels(top_channels, rotation=45, fontsize=9)
axes[0].set_title('Top 12 Channel Usage', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Proportion')
axes[0].legend()

# Difference chart
channel_diff = (channel_ct.loc[1] - channel_ct.loc[0]).sort_values()
colors = ['#c0392b' if v > 0 else '#27ae60' for v in channel_diff.values]
axes[1].barh(range(len(channel_diff)), channel_diff.values, color=colors, edgecolor='white')
axes[1].set_yticks(range(len(channel_diff)))
axes[1].set_yticklabels(channel_diff.index, fontsize=8)
axes[1].axvline(x=0, color='black', linewidth=0.8)
axes[1].set_title('Channel Over/Under-representation in Mule Accounts', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Proportion Difference (Mule - Legit)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '08_channel_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()

# 9. Temporal patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# Hour of day
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    h = txn_labeled[txn_labeled['is_mule']==is_m]['hour'].value_counts(normalize=True).sort_index()
    axes[0][0].plot(h.index, h.values, label=label, color=color, linewidth=2)
axes[0][0].set_title('Hour of Day Distribution', fontsize=12, fontweight='bold')
axes[0][0].set_xlabel('Hour')
axes[0][0].legend()
axes[0][0].axvspan(0, 6, alpha=0.1, color='gray', label='Night hours')

# Day of week
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    d = txn_labeled[txn_labeled['is_mule']==is_m]['dow'].value_counts(normalize=True).sort_index()
    axes[0][1].plot(d.index, d.values, label=label, color=color, linewidth=2, marker='o')
axes[0][1].set_title('Day of Week Distribution', fontsize=12, fontweight='bold')
axes[0][1].set_xticks(range(7))
axes[0][1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
axes[0][1].legend()
axes[0][1].axvspan(5, 6.5, alpha=0.1, color='orange', label='Weekend')

# Day of month
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    dm = txn_labeled[txn_labeled['is_mule']==is_m]['day_of_month'].value_counts(normalize=True).sort_index()
    axes[1][0].plot(dm.index, dm.values, label=label, color=color, linewidth=1.5)
axes[1][0].set_title('Day of Month Distribution', fontsize=12, fontweight='bold')
axes[1][0].set_xlabel('Day')
axes[1][0].legend()
axes[1][0].axvspan(1, 5, alpha=0.1, color='blue', label='Salary window')
axes[1][0].axvspan(25, 31, alpha=0.1, color='blue')

# Monthly trend over years
monthly_trend = txn_labeled.groupby([txn_labeled['transaction_timestamp'].dt.to_period('Q'), 'is_mule']).size().reset_index(name='count')
monthly_trend['period'] = monthly_trend['transaction_timestamp'].astype(str)
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    subset = monthly_trend[monthly_trend['is_mule'] == is_m]
    if is_m == 0:
        axes[1][1].plot(range(len(subset)), subset['count'].values / subset['count'].max(), label=label, color=color, linewidth=2)
    else:
        axes[1][1].plot(range(len(subset)), subset['count'].values / subset['count'].max(), label=label, color=color, linewidth=2)
axes[1][1].set_title('Quarterly Transaction Volume (Normalized)', fontsize=12, fontweight='bold')
axes[1][1].legend()
plt.suptitle('Temporal Patterns: Mule vs Legitimate', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '09_temporal_patterns.png'), dpi=200, bbox_inches='tight')
plt.close()

# 10. Amount distribution with log scale
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    amounts = txn_labeled[txn_labeled['is_mule']==is_m]['amount'].clip(1, 10000000)
    axes[0].hist(np.log10(amounts[amounts > 0]), bins=80, alpha=0.5, density=True, label=label, color=color)
axes[0].set_title('Transaction Amount (Log10 Scale)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('log10(Amount in INR)')
axes[0].legend()

# CDF comparison
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    amounts = txn_labeled[txn_labeled['is_mule']==is_m]['amount'].clip(0, 200000).sort_values()
    axes[1].plot(amounts.values, np.linspace(0, 1, len(amounts)), label=label, color=color, linewidth=2)
axes[1].set_title('CDF of Transaction Amount', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Amount (INR)')
axes[1].set_ylabel('Cumulative Proportion')
axes[1].legend()
axes[1].axvline(x=50000, color='gray', linestyle='--', alpha=0.5, label='50K threshold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '10_amount_distribution.png'), dpi=200, bbox_inches='tight')
plt.close()

# 11. Structuring detection
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Amount histogram near 50K
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    amounts = txn_labeled[(txn_labeled['is_mule']==is_m) & (txn_labeled['amount'] > 30000) & (txn_labeled['amount'] < 60000)]['amount']
    axes[0].hist(amounts, bins=60, alpha=0.5, density=True, label=label, color=color)
axes[0].axvline(x=50000, color='red', linestyle='--', linewidth=2, label='50K Threshold')
axes[0].set_title('Transaction Amounts Near 50K (Structuring)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Amount (INR)')
axes[0].legend()

# Round amount analysis
round_data = pd.DataFrame({
    'Legitimate': [
        txn_labeled[txn_labeled['is_mule']==0]['amount'].apply(lambda x: x % 1000 == 0).mean(),
        txn_labeled[txn_labeled['is_mule']==0]['amount'].apply(lambda x: x % 5000 == 0).mean(),
        txn_labeled[txn_labeled['is_mule']==0]['amount'].apply(lambda x: x % 10000 == 0).mean(),
        txn_labeled[txn_labeled['is_mule']==0]['amount'].apply(lambda x: x % 50000 == 0).mean(),
    ],
    'Mule': [
        txn_labeled[txn_labeled['is_mule']==1]['amount'].apply(lambda x: x % 1000 == 0).mean(),
        txn_labeled[txn_labeled['is_mule']==1]['amount'].apply(lambda x: x % 5000 == 0).mean(),
        txn_labeled[txn_labeled['is_mule']==1]['amount'].apply(lambda x: x % 10000 == 0).mean(),
        txn_labeled[txn_labeled['is_mule']==1]['amount'].apply(lambda x: x % 50000 == 0).mean(),
    ]
}, index=['Round 1K', 'Round 5K', 'Round 10K', 'Round 50K'])
round_data.plot(kind='bar', ax=axes[1], color=['#27ae60', '#c0392b'], edgecolor='white')
axes[1].set_title('Round Amount Transaction Rates', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Proportion')
axes[1].set_xticklabels(round_data.index, rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '11_structuring.png'), dpi=200, bbox_inches='tight')
plt.close()

# 12. Counterparty analysis
print("\n--- Counterparty Analysis ---")
cp_stats = txn_labeled.groupby(['account_id', 'is_mule']).agg(
    n_counterparties=('counterparty_id', 'nunique'),
    n_txn=('transaction_id', 'count')
).reset_index()
cp_stats['cp_per_txn'] = cp_stats['n_counterparties'] / cp_stats['n_txn']

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    subset = cp_stats[cp_stats['is_mule']==is_m]['n_counterparties'].clip(0, 200)
    axes[0].hist(subset, bins=50, alpha=0.5, density=True, label=label, color=color)
axes[0].set_title('Unique Counterparties per Account', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Count')
axes[0].legend()

# Counterparty concentration
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    subset = cp_stats[cp_stats['is_mule']==is_m]['cp_per_txn']
    axes[1].hist(subset, bins=50, alpha=0.5, density=True, label=label, color=color)
axes[1].set_title('Counterparty Diversity (unique CP / total txns)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Ratio')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '12_counterparty.png'), dpi=200, bbox_inches='tight')
plt.close()

# 13. MCC code analysis
print("\n--- MCC Code Analysis ---")
mcc_mule_rate = txn_labeled.groupby('mcc_code').agg(
    total=('is_mule', 'count'),
    mule_txns=('is_mule', 'sum')
).reset_index()
mcc_mule_rate['mule_rate'] = mcc_mule_rate['mule_txns'] / mcc_mule_rate['total']
mcc_mule_rate = mcc_mule_rate[mcc_mule_rate['total'] >= 100].sort_values('mule_rate', ascending=False)

fig, ax = plt.subplots(figsize=(12, 6))
top_mccs = mcc_mule_rate.head(15)
bars = ax.bar(range(len(top_mccs)), top_mccs['mule_rate'].values, color='#c0392b', edgecolor='white')
ax.set_xticks(range(len(top_mccs)))
ax.set_xticklabels(top_mccs['mcc_code'].values, rotation=45)
ax.set_title('Top 15 MCC Codes by Mule Transaction Rate', fontsize=13, fontweight='bold')
ax.set_ylabel('Proportion of Txns from Mule Accounts')
ax.axhline(y=labels['is_mule'].mean(), color='blue', linestyle='--', label=f'Overall mule rate ({labels["is_mule"].mean():.3f})')
ax.legend()
for bar, val, n in zip(bars, top_mccs['mule_rate'].values, top_mccs['total'].values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}\n(n={n:,})', ha='center', va='bottom', fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '13_mcc_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()

# 14. Branch-level analysis
print("\n--- Branch-Level Analysis ---")
branch_stats = train.groupby('branch_code').agg(
    n_accounts=('account_id', 'count'),
    n_mules=('is_mule', 'sum'),
    mule_rate=('is_mule', 'mean')
).reset_index()
branch_stats = branch_stats[branch_stats['n_accounts'] >= 3]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(branch_stats['mule_rate'], bins=50, color='#3498db', edgecolor='white')
axes[0].axvline(x=labels['is_mule'].mean(), color='red', linestyle='--', linewidth=2,
                 label=f'Overall rate: {labels["is_mule"].mean():.3f}')
axes[0].set_title('Mule Rate Distribution Across Branches', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Branch Mule Rate')
axes[0].set_ylabel('Count')
axes[0].legend()

# Scatter: branch size vs mule rate
axes[1].scatter(branch_stats['n_accounts'], branch_stats['mule_rate'],
                 alpha=0.3, s=20, c='#3498db')
high_risk = branch_stats[branch_stats['mule_rate'] > 0.3]
axes[1].scatter(high_risk['n_accounts'], high_risk['mule_rate'],
                 alpha=0.8, s=60, c='#c0392b', label=f'High risk (>30%) - {len(high_risk)} branches')
axes[1].axhline(y=labels['is_mule'].mean(), color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('Branch Size vs Mule Rate', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Number of Accounts')
axes[1].set_ylabel('Mule Rate')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '14_branch_analysis.png'), dpi=200, bbox_inches='tight')
plt.close()

# 15. Velocity and burst analysis
print("\n--- Velocity Analysis ---")
txn_sorted = txn_labeled.sort_values(['account_id', 'transaction_timestamp'])
txn_sorted['time_diff_hrs'] = txn_sorted.groupby('account_id')['transaction_timestamp'].diff().dt.total_seconds() / 3600

velocity = txn_sorted.groupby(['account_id', 'is_mule']).agg(
    med_gap_hrs=('time_diff_hrs', 'median'),
    min_gap_hrs=('time_diff_hrs', 'min'),
    std_gap_hrs=('time_diff_hrs', 'std')
).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    subset = velocity[velocity['is_mule']==is_m]['med_gap_hrs'].clip(0, 2000)
    axes[0].hist(subset, bins=60, alpha=0.5, density=True, label=label, color=color)
axes[0].set_title('Median Hours Between Transactions', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Hours')
axes[0].legend()

for is_m, label, color in [(0, 'Legitimate', '#27ae60'), (1, 'Mule', '#c0392b')]:
    subset = velocity[velocity['is_mule']==is_m]['min_gap_hrs'].clip(0, 50)
    axes[1].hist(subset, bins=60, alpha=0.5, density=True, label=label, color=color)
axes[1].set_title('Minimum Hours Between Any Two Transactions', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Hours')
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '15_velocity.png'), dpi=200, bbox_inches='tight')
plt.close()

# Correlation heatmap
print("\n--- Correlation Heatmap ---")
num_cols = ['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance',
            'num_chequebooks', 'loan_count', 'cc_count', 'od_count', 'ka_count', 'sa_count', 'is_mule']
corr = train[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlations', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '16_correlations.png'), dpi=200, bbox_inches='tight')
plt.close()

del txn_labeled, txn_sorted
gc.collect()

# ================================================================
# PART 4: FEATURE ENGINEERING
# ================================================================
print("\n" + "=" * 70)
print("PART 4: FEATURE ENGINEERING")
print("=" * 70)

# We need features for ALL accounts (train + test)
all_accounts = pd.concat([labels[['account_id']], test[['account_id']]], ignore_index=True)
print(f"Total accounts to featurize: {len(all_accounts):,}")

# Get all transactions for these accounts
txn_all = transactions[transactions['account_id'].isin(set(all_accounts['account_id']))].copy()
txn_all['transaction_timestamp'] = pd.to_datetime(txn_all['transaction_timestamp'])
txn_all['hour'] = txn_all['transaction_timestamp'].dt.hour
txn_all['dow'] = txn_all['transaction_timestamp'].dt.dayofweek
txn_all['day_of_month'] = txn_all['transaction_timestamp'].dt.day
txn_all['month'] = txn_all['transaction_timestamp'].dt.month
print(f"Transactions for featurization: {len(txn_all):,}")

# --- Feature Group 1: Basic Transaction Stats ---
print("  Computing basic transaction stats...")
basic_stats = txn_all.groupby('account_id').agg(
    txn_count=('transaction_id', 'count'),
    total_amount=('amount', 'sum'),
    mean_amount=('amount', 'mean'),
    median_amount=('amount', 'median'),
    std_amount=('amount', 'std'),
    max_amount=('amount', 'max'),
    min_amount=('amount', 'min'),
    p25_amount=('amount', lambda x: x.quantile(0.25)),
    p75_amount=('amount', lambda x: x.quantile(0.75)),
    p95_amount=('amount', lambda x: x.quantile(0.95)),
    n_unique_counterparties=('counterparty_id', 'nunique'),
    n_unique_mcc=('mcc_code', 'nunique'),
    first_txn=('transaction_timestamp', 'min'),
    last_txn=('transaction_timestamp', 'max')
).reset_index()
basic_stats['amount_range'] = basic_stats['max_amount'] - basic_stats['min_amount']
basic_stats['iqr_amount'] = basic_stats['p75_amount'] - basic_stats['p25_amount']
basic_stats['cv_amount'] = basic_stats['std_amount'] / basic_stats['mean_amount'].clip(lower=0.01)
basic_stats['active_span_days'] = (basic_stats['last_txn'] - basic_stats['first_txn']).dt.days
basic_stats['txn_per_day'] = basic_stats['txn_count'] / basic_stats['active_span_days'].clip(lower=1)
basic_stats['cp_per_txn'] = basic_stats['n_unique_counterparties'] / basic_stats['txn_count']

# --- Feature Group 2: Credit/Debit Features ---
print("  Computing credit/debit features...")
cd_stats = txn_all.groupby(['account_id', 'txn_type']).agg(
    cd_count=('transaction_id', 'count'),
    cd_amount=('amount', 'sum'),
    cd_mean=('amount', 'mean'),
    cd_max=('amount', 'max')
).reset_index()

credit_stats = cd_stats[cd_stats['txn_type'] == 'C'].rename(columns={
    'cd_count': 'credit_count', 'cd_amount': 'credit_total', 'cd_mean': 'credit_mean', 'cd_max': 'credit_max'
}).drop('txn_type', axis=1)
debit_stats = cd_stats[cd_stats['txn_type'] == 'D'].rename(columns={
    'cd_count': 'debit_count', 'cd_amount': 'debit_total', 'cd_mean': 'debit_mean', 'cd_max': 'debit_max'
}).drop('txn_type', axis=1)

basic_stats = basic_stats.merge(credit_stats, on='account_id', how='left')
basic_stats = basic_stats.merge(debit_stats, on='account_id', how='left')
basic_stats['credit_count'] = basic_stats['credit_count'].fillna(0)
basic_stats['debit_count'] = basic_stats['debit_count'].fillna(0)
basic_stats['credit_total'] = basic_stats['credit_total'].fillna(0)
basic_stats['debit_total'] = basic_stats['debit_total'].fillna(0)
basic_stats['passthrough_ratio'] = basic_stats['debit_total'] / basic_stats['credit_total'].clip(lower=1)
basic_stats['net_flow'] = basic_stats['credit_total'] - basic_stats['debit_total']
basic_stats['credit_debit_ratio'] = basic_stats['credit_count'] / (basic_stats['credit_count'] + basic_stats['debit_count']).clip(lower=1)

# --- Feature Group 3: Channel Features ---
print("  Computing channel features...")
channel_counts = txn_all.groupby(['account_id', 'channel']).size().unstack(fill_value=0)
channel_total = channel_counts.sum(axis=1)
channel_rates = channel_counts.div(channel_total, axis=0)
channel_rates.columns = ['ch_' + c + '_rate' for c in channel_rates.columns]

# Channel entropy
def calc_entropy(row):
    probs = row[row > 0].values
    if len(probs) <= 1:
        return 0
    return entropy(probs)

channel_rates['channel_entropy'] = channel_counts.div(channel_total, axis=0).apply(calc_entropy, axis=1)
channel_rates['n_channels_used'] = (channel_counts > 0).sum(axis=1)
channel_rates = channel_rates.reset_index()

# Key channel features
key_channels = ['UPC', 'UPD', 'IPM', 'NTD', 'FTD', 'ATW', 'CHQ', 'END', 'P2A', 'MCR']
for ch in key_channels:
    col = f'ch_{ch}_rate'
    if col not in channel_rates.columns:
        channel_rates[col] = 0

basic_stats = basic_stats.merge(channel_rates[['account_id', 'channel_entropy', 'n_channels_used'] +
                                                [f'ch_{ch}_rate' for ch in key_channels]], on='account_id', how='left')

# --- Feature Group 4: Temporal Features ---
print("  Computing temporal features...")
temporal = txn_all.groupby('account_id').agg(
    night_rate=('hour', lambda x: ((x >= 0) & (x < 6)).mean()),
    weekend_rate=('dow', lambda x: (x >= 5).mean()),
    salary_window_rate=('day_of_month', lambda x: ((x <= 5) | (x >= 25)).mean()),
).reset_index()

# Hour entropy
hour_counts = txn_all.groupby(['account_id', 'hour']).size().unstack(fill_value=0)
hour_total = hour_counts.sum(axis=1)
hour_probs = hour_counts.div(hour_total, axis=0)
temporal['hour_entropy'] = hour_probs.apply(calc_entropy, axis=1).values

basic_stats = basic_stats.merge(temporal, on='account_id', how='left')

# --- Feature Group 5: Velocity Features ---
print("  Computing velocity features...")
txn_sorted = txn_all.sort_values(['account_id', 'transaction_timestamp'])
txn_sorted['time_diff'] = txn_sorted.groupby('account_id')['transaction_timestamp'].diff().dt.total_seconds() / 3600

vel = txn_sorted.groupby('account_id')['time_diff'].agg(
    med_gap_hrs='median',
    min_gap_hrs='min',
    std_gap_hrs='std',
    p10_gap_hrs=lambda x: x.quantile(0.1) if len(x) > 1 else np.nan
).reset_index()
basic_stats = basic_stats.merge(vel, on='account_id', how='left')

# --- Feature Group 6: Structuring Features ---
print("  Computing structuring features...")
struct = txn_all.groupby('account_id').agg(
    near_50k_rate=('amount', lambda x: ((x >= 45000) & (x < 50000)).mean()),
    near_10k_rate=('amount', lambda x: ((x >= 9000) & (x < 10000)).mean()),
    round_1k_rate=('amount', lambda x: (x % 1000 == 0).mean()),
    round_5k_rate=('amount', lambda x: (x % 5000 == 0).mean()),
    round_10k_rate=('amount', lambda x: (x % 10000 == 0).mean()),
    round_50k_rate=('amount', lambda x: (x % 50000 == 0).mean()),
    negative_rate=('amount', lambda x: (x < 0).mean()),
).reset_index()
basic_stats = basic_stats.merge(struct, on='account_id', how='left')

# --- Feature Group 7: Burst Features ---
print("  Computing burst features...")
txn_all['date'] = txn_all['transaction_timestamp'].dt.date
daily = txn_all.groupby(['account_id', 'date']).agg(
    daily_count=('transaction_id', 'count'),
    daily_amount=('amount', 'sum')
).reset_index()

burst = daily.groupby('account_id').agg(
    mean_daily_txn=('daily_count', 'mean'),
    max_daily_txn=('daily_count', 'max'),
    std_daily_txn=('daily_count', 'std'),
    mean_daily_amount=('daily_amount', 'mean'),
    max_daily_amount=('daily_amount', 'max'),
    n_active_days=('date', 'count')
).reset_index()
burst['burstiness'] = burst['max_daily_txn'] / burst['mean_daily_txn'].clip(lower=0.01)
burst['amount_burstiness'] = burst['max_daily_amount'] / burst['mean_daily_amount'].clip(lower=0.01)

basic_stats = basic_stats.merge(burst, on='account_id', how='left')

# --- Feature Group 8: MCC Features ---
print("  Computing MCC features...")
# High-risk MCCs identified in EDA
high_risk_mccs = [6011, 5933, 6051, 6012, 4814]
for mcc in high_risk_mccs:
    col_name = f'mcc_{mcc}_rate'
    mcc_count = txn_all[txn_all['mcc_code'] == mcc].groupby('account_id').size()
    total_txn = txn_all.groupby('account_id').size()
    basic_stats[col_name] = basic_stats['account_id'].map(mcc_count / total_txn).fillna(0)

# MCC entropy
mcc_counts = txn_all.groupby(['account_id', 'mcc_code']).size().unstack(fill_value=0)
mcc_total = mcc_counts.sum(axis=1)
mcc_probs = mcc_counts.div(mcc_total, axis=0)
basic_stats['mcc_entropy'] = basic_stats['account_id'].map(
    pd.Series(mcc_probs.apply(calc_entropy, axis=1))
).fillna(0)

del txn_all, txn_sorted, daily
gc.collect()

# --- Feature Group 9: Account-Level Features ---
print("  Computing account-level features...")
acct_features = accounts.copy()
acct_features['acct_age_days'] = (REF_DATE - acct_features['account_opening_date']).dt.days
acct_features['acct_age_months'] = acct_features['acct_age_days'] / 30.44
acct_features['was_frozen'] = acct_features['freeze_date'].notna().astype(int)
acct_features['was_unfrozen'] = acct_features['unfreeze_date'].notna().astype(int)
acct_features['had_mobile_update'] = acct_features['last_mobile_update_date'].notna().astype(int)
acct_features['days_since_mobile_update'] = (REF_DATE - acct_features['last_mobile_update_date']).dt.days
acct_features['days_since_kyc'] = (REF_DATE - acct_features['last_kyc_date']).dt.days
acct_features['frozen_duration'] = (acct_features['unfreeze_date'] - acct_features['freeze_date']).dt.days

# Encode categoricals
for col in ['account_status', 'product_family', 'nomination_flag', 'cheque_allowed',
            'cheque_availed', 'kyc_compliant', 'rural_branch']:
    acct_features[col + '_enc'] = LabelEncoder().fit_transform(acct_features[col].fillna('MISSING'))

# Balance volatility
acct_features['balance_range'] = acct_features[['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance']].max(axis=1) - \
                                  acct_features[['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance']].min(axis=1)
acct_features['balance_std'] = acct_features[['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance']].std(axis=1)

acct_cols_to_use = ['account_id', 'avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance',
                     'num_chequebooks', 'acct_age_days', 'acct_age_months', 'was_frozen', 'was_unfrozen',
                     'had_mobile_update', 'days_since_mobile_update', 'days_since_kyc', 'frozen_duration',
                     'account_status_enc', 'product_family_enc', 'nomination_flag_enc', 'cheque_allowed_enc',
                     'cheque_availed_enc', 'kyc_compliant_enc', 'rural_branch_enc',
                     'balance_range', 'balance_std', 'product_code', 'currency_code']
basic_stats = basic_stats.merge(acct_features[acct_cols_to_use], on='account_id', how='left')

# --- Feature Group 10: Customer-Level Features ---
print("  Computing customer-level features...")
cust_features = customers.copy()
cust_features['age'] = (REF_DATE - cust_features['date_of_birth']).dt.days / 365.25
cust_features['rel_years'] = (REF_DATE - cust_features['relationship_start_date']).dt.days / 365.25
cust_features['pin_mismatch'] = (cust_features['customer_pin'] != cust_features['permanent_pin']).astype(int)

for col in ['pan_available', 'aadhaar_available', 'passport_available',
            'mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag',
            'demat_flag', 'credit_card_flag', 'fastag_flag']:
    cust_features[col + '_enc'] = (cust_features[col] == 'Y').astype(int)

# KYC completeness score
cust_features['kyc_score'] = (cust_features['pan_available_enc'] +
                               cust_features['aadhaar_available_enc'] +
                               cust_features['passport_available_enc'])
cust_features['digital_score'] = (cust_features['mobile_banking_flag_enc'] +
                                   cust_features['internet_banking_flag_enc'] +
                                   cust_features['atm_card_flag_enc'] +
                                   cust_features['credit_card_flag_enc'] +
                                   cust_features['fastag_flag_enc'] +
                                   cust_features['demat_flag_enc'])

# Number of accounts per customer
acct_per_cust = linkage.groupby('customer_id')['account_id'].count().reset_index(name='n_accounts_customer')
cust_features = cust_features.merge(acct_per_cust, on='customer_id', how='left')

# Product details
cust_features = cust_features.merge(products, on='customer_id', how='left')

cust_cols = ['customer_id', 'age', 'rel_years', 'pin_mismatch',
             'pan_available_enc', 'aadhaar_available_enc', 'passport_available_enc',
             'mobile_banking_flag_enc', 'internet_banking_flag_enc', 'atm_card_flag_enc',
             'demat_flag_enc', 'credit_card_flag_enc', 'fastag_flag_enc',
             'kyc_score', 'digital_score', 'n_accounts_customer',
             'loan_sum', 'loan_count', 'cc_sum', 'cc_count', 'od_sum', 'od_count',
             'ka_sum', 'ka_count', 'sa_sum', 'sa_count']

# Join through linkage
cust_via_link = linkage.merge(cust_features[cust_cols], on='customer_id', how='left')
basic_stats = basic_stats.merge(cust_via_link.drop('customer_id', axis=1), on='account_id', how='left')

# --- Feature Group 11: Derived Ratios ---
print("  Computing derived ratios...")
basic_stats['txn_to_balance_ratio'] = basic_stats['total_amount'] / basic_stats['avg_balance'].clip(lower=1)
basic_stats['max_txn_to_balance'] = basic_stats['max_amount'] / basic_stats['avg_balance'].clip(lower=1)
basic_stats['acct_age_txn_ratio'] = basic_stats['txn_count'] / basic_stats['acct_age_days'].clip(lower=1)

# Drop helper columns
basic_stats = basic_stats.drop(['first_txn', 'last_txn'], axis=1, errors='ignore')

print(f"\nFinal feature matrix: {basic_stats.shape}")
print(f"Features: {basic_stats.shape[1] - 1}")  # minus account_id

# ================================================================
# PART 5: MODEL TRAINING (Phase 2)
# ================================================================
print("\n" + "=" * 70)
print("PART 5: MODEL TRAINING")
print("=" * 70)

# Prepare train/test
train_features = basic_stats[basic_stats['account_id'].isin(train_acct_ids)].copy()
test_features = basic_stats[basic_stats['account_id'].isin(test_acct_ids)].copy()

train_features = train_features.merge(labels[['account_id', 'is_mule']], on='account_id', how='left')

feature_cols = [c for c in train_features.columns if c not in ['account_id', 'is_mule']]
X_train = train_features[feature_cols].values
y_train = train_features['is_mule'].values
X_test = test_features[feature_cols].values

print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Mule rate: {y_train.mean():.4f}")
print(f"Features: {len(feature_cols)}")

# Handle inf/nan
X_train = np.nan_to_num(X_train, nan=0, posinf=1e10, neginf=-1e10)
X_test = np.nan_to_num(X_test, nan=0, posinf=1e10, neginf=-1e10)

# --- LightGBM with Stratified K-Fold ---
print("\n--- LightGBM Training ---")
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': (y_train == 0).sum() / max((y_train == 1).sum(), 1),
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42
}

lgb_oof = np.zeros(len(X_train))
lgb_test_preds = np.zeros(len(X_test))
lgb_models = []
lgb_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    model = lgb.train(
        lgb_params, lgb_train,
        num_boost_round=2000,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )

    lgb_oof[val_idx] = model.predict(X_val)
    lgb_test_preds += model.predict(X_test) / n_splits
    lgb_models.append(model)

    fold_auc = roc_auc_score(y_val, lgb_oof[val_idx])
    lgb_aucs.append(fold_auc)
    print(f"  Fold {fold+1}: AUC = {fold_auc:.6f}")

overall_lgb_auc = roc_auc_score(y_train, lgb_oof)
print(f"\n  Overall LightGBM OOF AUC: {overall_lgb_auc:.6f}")
print(f"  Mean fold AUC: {np.mean(lgb_aucs):.6f} +/- {np.std(lgb_aucs):.6f}")

# --- XGBoost ---
print("\n--- XGBoost Training ---")
xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': (y_train == 0).sum() / max((y_train == 1).sum(), 1),
    'tree_method': 'hist',
    'seed': 42,
    'verbosity': 0
}

xgb_oof = np.zeros(len(X_train))
xgb_test_preds = np.zeros(len(X_test))
xgb_models = []
xgb_aucs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)

    model = xgb.train(
        xgb_params, dtrain,
        num_boost_round=2000,
        evals=[(dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )

    xgb_oof[val_idx] = model.predict(dval)
    xgb_test_preds += model.predict(xgb.DMatrix(X_test)) / n_splits
    xgb_models.append(model)

    fold_auc = roc_auc_score(y_val, xgb_oof[val_idx])
    xgb_aucs.append(fold_auc)
    print(f"  Fold {fold+1}: AUC = {fold_auc:.6f}")

overall_xgb_auc = roc_auc_score(y_train, xgb_oof)
print(f"\n  Overall XGBoost OOF AUC: {overall_xgb_auc:.6f}")
print(f"  Mean fold AUC: {np.mean(xgb_aucs):.6f} +/- {np.std(xgb_aucs):.6f}")

# --- Ensemble ---
print("\n--- Ensemble (Average) ---")
ensemble_oof = 0.5 * lgb_oof + 0.5 * xgb_oof
ensemble_test = 0.5 * lgb_test_preds + 0.5 * xgb_test_preds
ensemble_auc = roc_auc_score(y_train, ensemble_oof)
print(f"  Ensemble OOF AUC: {ensemble_auc:.6f}")

# Choose best approach
best_test_preds = ensemble_test if ensemble_auc >= max(overall_lgb_auc, overall_xgb_auc) else \
    (lgb_test_preds if overall_lgb_auc >= overall_xgb_auc else xgb_test_preds)
best_name = "Ensemble" if ensemble_auc >= max(overall_lgb_auc, overall_xgb_auc) else \
    ("LightGBM" if overall_lgb_auc >= overall_xgb_auc else "XGBoost")
print(f"\n  Best approach: {best_name}")

# 17. Feature importance
print("\n--- Feature Importance ---")
importance = lgb_models[0].feature_importance(importance_type='gain')
feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
feat_imp = feat_imp.sort_values('importance', ascending=False)
print(feat_imp.head(30).to_string())

fig, ax = plt.subplots(figsize=(10, 12))
top30 = feat_imp.head(30)
ax.barh(range(len(top30)), top30['importance'].values, color='#3498db', edgecolor='white')
ax.set_yticks(range(len(top30)))
ax.set_yticklabels(top30['feature'].values, fontsize=9)
ax.invert_yaxis()
ax.set_title('Top 30 Feature Importances (LightGBM Gain)', fontsize=13, fontweight='bold')
ax.set_xlabel('Gain')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '17_feature_importance.png'), dpi=200, bbox_inches='tight')
plt.close()

# 18. Model evaluation plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ROC curve
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_train, ensemble_oof)
axes[0].plot(fpr, tpr, color='#c0392b', linewidth=2, label=f'Ensemble AUC = {ensemble_auc:.4f}')
fpr_l, tpr_l, _ = roc_curve(y_train, lgb_oof)
axes[0].plot(fpr_l, tpr_l, color='#2980b9', linewidth=1.5, alpha=0.7, label=f'LightGBM AUC = {overall_lgb_auc:.4f}')
fpr_x, tpr_x, _ = roc_curve(y_train, xgb_oof)
axes[0].plot(fpr_x, tpr_x, color='#27ae60', linewidth=1.5, alpha=0.7, label=f'XGBoost AUC = {overall_xgb_auc:.4f}')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_title('ROC Curve (OOF)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_train, ensemble_oof)
ap = average_precision_score(y_train, ensemble_oof)
axes[1].plot(recall, precision, color='#c0392b', linewidth=2, label=f'AP = {ap:.4f}')
axes[1].set_title('Precision-Recall Curve (OOF)', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend()

# Score distribution
axes[2].hist(ensemble_oof[y_train == 0], bins=50, alpha=0.5, density=True, label='Legitimate', color='#27ae60')
axes[2].hist(ensemble_oof[y_train == 1], bins=50, alpha=0.5, density=True, label='Mule', color='#c0392b')
axes[2].set_title('Prediction Score Distribution (OOF)', fontsize=13, fontweight='bold')
axes[2].set_xlabel('Predicted Probability')
axes[2].legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '18_model_evaluation.png'), dpi=200, bbox_inches='tight')
plt.close()

# ================================================================
# PART 6: SUSPICIOUS WINDOW DETECTION
# ================================================================
print("\n" + "=" * 70)
print("PART 6: SUSPICIOUS WINDOW DETECTION")
print("=" * 70)

# For accounts predicted as mule (>0.5), find the suspicious activity window
# Strategy: Use rolling transaction velocity to find burst periods
print("Reloading transactions for suspicious window detection...")
txn_test = transactions[transactions['account_id'].isin(test_acct_ids)].copy()
txn_test['transaction_timestamp'] = pd.to_datetime(txn_test['transaction_timestamp'])

suspicious_windows = {}
high_risk_accounts = test_features[best_test_preds > 0.3]['account_id'].values

for acct_id in high_risk_accounts:
    acct_txns = txn_test[txn_test['account_id'] == acct_id].sort_values('transaction_timestamp')
    if len(acct_txns) < 5:
        continue

    # Compute daily transaction counts
    acct_txns['date'] = acct_txns['transaction_timestamp'].dt.date
    daily = acct_txns.groupby('date').agg(
        n_txn=('transaction_id', 'count'),
        total_amount=('amount', 'sum'),
        max_amount=('amount', 'max')
    ).reset_index()

    if len(daily) < 3:
        continue

    # Find the period with highest activity (rolling 30-day window)
    daily['date'] = pd.to_datetime(daily['date'])
    daily = daily.set_index('date').resample('D').sum().fillna(0).reset_index()

    # Rolling 30-day transaction sum
    daily['rolling_30d_txn'] = daily['n_txn'].rolling(30, min_periods=1).sum()
    daily['rolling_30d_amount'] = daily['total_amount'].rolling(30, min_periods=1).sum()

    # Find peak activity period
    peak_idx = daily['rolling_30d_txn'].idxmax()
    peak_date = daily.loc[peak_idx, 'date']

    # Define window as the contiguous period of above-average activity around peak
    threshold = daily['n_txn'].mean() * 1.5
    active_mask = daily['n_txn'] > 0

    # Find start: go back from peak until activity drops
    start_date = daily.loc[max(0, peak_idx - 90):peak_idx, 'date']
    start_date = start_date[daily.loc[start_date.index, 'n_txn'] > 0]
    if len(start_date) > 0:
        s = start_date.iloc[0]
    else:
        s = peak_date - pd.Timedelta(days=30)

    # Find end: go forward from peak
    end_date = daily.loc[peak_idx:min(len(daily)-1, peak_idx + 90), 'date']
    end_date = end_date[daily.loc[end_date.index, 'n_txn'] > 0]
    if len(end_date) > 0:
        e = end_date.iloc[-1]
    else:
        e = peak_date + pd.Timedelta(days=30)

    suspicious_windows[acct_id] = (s, e)

print(f"Suspicious windows computed for {len(suspicious_windows)} accounts")

# ================================================================
# PART 7: GENERATE PREDICTIONS
# ================================================================
print("\n" + "=" * 70)
print("PART 7: GENERATING PREDICTIONS")
print("=" * 70)

submission = pd.DataFrame({
    'account_id': test_features['account_id'].values,
    'is_mule': best_test_preds
})

submission['suspicious_start'] = ''
submission['suspicious_end'] = ''

for idx, row in submission.iterrows():
    acct_id = row['account_id']
    if acct_id in suspicious_windows:
        s, e = suspicious_windows[acct_id]
        submission.loc[idx, 'suspicious_start'] = s.strftime('%Y-%m-%dT%H:%M:%S') if pd.notna(s) else ''
        submission.loc[idx, 'suspicious_end'] = e.strftime('%Y-%m-%dT%H:%M:%S') if pd.notna(e) else ''

submission.to_csv(os.path.join(OUT_DIR, 'predictions.csv'), index=False)
print(f"Predictions saved: {os.path.join(OUT_DIR, 'predictions.csv')}")
print(f"  Total predictions: {len(submission)}")
print(f"  Predicted mule (>0.5): {(submission['is_mule'] > 0.5).sum()}")
print(f"  With suspicious windows: {(submission['suspicious_start'] != '').sum()}")
print(f"  Mean prediction: {submission['is_mule'].mean():.6f}")
print(f"  Prediction distribution:")
for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
    print(f"    > {thresh}: {(submission['is_mule'] > thresh).sum()}")

# Save feature importance
feat_imp.to_csv(os.path.join(OUT_DIR, 'feature_importance.csv'), index=False)

# ================================================================
# PART 8: SHAP ANALYSIS
# ================================================================
print("\n" + "=" * 70)
print("PART 8: SHAP EXPLAINABILITY")
print("=" * 70)

try:
    import shap
    explainer = shap.TreeExplainer(lgb_models[0])
    shap_values = explainer.shap_values(X_train[:1000])

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    fig, ax = plt.subplots(figsize=(10, 12))
    shap.summary_plot(shap_vals, X_train[:1000], feature_names=feature_cols, show=False, max_display=25)
    plt.title('SHAP Feature Importance (Top 25)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, '19_shap_summary.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("SHAP analysis complete")
except Exception as e:
    print(f"SHAP analysis failed: {e}")

print("\n" + "=" * 70)
print("ALL DONE!")
print(f"Outputs saved to: {OUT_DIR}")
print(f"Plots saved to: {PLOT_DIR}")
print("=" * 70)
