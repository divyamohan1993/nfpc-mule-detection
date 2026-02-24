"""
Phase 1 EDA - National Fraud Prevention Challenge
Comprehensive Exploratory Data Analysis for Mule Account Detection
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import json
import os

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

DATA_DIR = r"r:/national-fraud-prevention-challenge/IITD-Tryst-Hackathon/EDA-Phase-1"
OUT_DIR = r"r:/national-fraud-prevention-challenge/eda_output"
PLOT_DIR = os.path.join(OUT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

results = {}  # Store all findings as JSON-serializable dict

# ============================================================
# SECTION 1: LOAD AND INSPECT DATA
# ============================================================
print("=" * 60)
print("SECTION 1: LOADING AND INSPECTING DATA")
print("=" * 60)

customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
accounts = pd.read_csv(os.path.join(DATA_DIR, "accounts.csv"))
linkage = pd.read_csv(os.path.join(DATA_DIR, "customer_account_linkage.csv"))
products = pd.read_csv(os.path.join(DATA_DIR, "product_details.csv"))
labels = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test_accounts.csv"))

print("Loading transactions (6 parts)...")
txn_parts = []
for i in range(6):
    part = pd.read_csv(os.path.join(DATA_DIR, f"transactions_part_{i}.csv"))
    txn_parts.append(part)
    print(f"  Part {i}: {len(part):,} rows")
transactions = pd.concat(txn_parts, ignore_index=True)
del txn_parts
print(f"Total transactions: {len(transactions):,}")

# Parse dates
for col in ['date_of_birth', 'relationship_start_date']:
    customers[col] = pd.to_datetime(customers[col], errors='coerce')
for col in ['account_opening_date', 'last_mobile_update_date', 'last_kyc_date', 'freeze_date', 'unfreeze_date']:
    accounts[col] = pd.to_datetime(accounts[col], errors='coerce')
transactions['transaction_timestamp'] = pd.to_datetime(transactions['transaction_timestamp'], errors='coerce')
labels['mule_flag_date'] = pd.to_datetime(labels['mule_flag_date'], errors='coerce')

# Shape summary
datasets = {
    'customers': customers, 'accounts': accounts, 'linkage': linkage,
    'products': products, 'labels': labels, 'test': test, 'transactions': transactions
}

print("\n--- Dataset Shapes ---")
shape_info = {}
for name, df in datasets.items():
    shape_info[name] = {'rows': df.shape[0], 'cols': df.shape[1]}
    print(f"  {name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
results['shapes'] = shape_info

# Missing values
print("\n--- Missing Values ---")
missing_info = {}
for name, df in datasets.items():
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing_info[name] = {k: int(v) for k, v in missing.items()}
        print(f"\n  {name}:")
        for col, count in missing.items():
            print(f"    {col}: {count:,} ({count/len(df)*100:.1f}%)")
results['missing_values'] = missing_info

# Dtypes summary
print("\n--- Data Types ---")
for name, df in datasets.items():
    print(f"\n  {name}: {dict(df.dtypes.value_counts())}")

# ============================================================
# SECTION 2: TARGET VARIABLE ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: TARGET VARIABLE ANALYSIS")
print("=" * 60)

mule_counts = labels['is_mule'].value_counts()
mule_rate = labels['is_mule'].mean()
print(f"Total training accounts: {len(labels):,}")
print(f"Legitimate (0): {mule_counts[0]:,} ({mule_counts[0]/len(labels)*100:.2f}%)")
print(f"Mule (1):       {mule_counts[1]:,} ({mule_counts[1]/len(labels)*100:.2f}%)")
print(f"Mule rate:      {mule_rate:.4f} ({mule_rate*100:.2f}%)")
print(f"Imbalance ratio: 1:{mule_counts[0]//mule_counts[1]}")

results['target'] = {
    'total': int(len(labels)),
    'legitimate': int(mule_counts[0]),
    'mule': int(mule_counts[1]),
    'mule_rate': float(mule_rate),
    'imbalance_ratio': f"1:{mule_counts[0]//mule_counts[1]}"
}

# Plot class distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
mule_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Class Distribution (Count)')
axes[0].set_xticklabels(['Legitimate', 'Mule'], rotation=0)
axes[0].set_ylabel('Count')
for i, v in enumerate(mule_counts.values):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

mule_counts.plot(kind='pie', ax=axes[1], autopct='%1.2f%%', colors=['#2ecc71', '#e74c3c'],
                  labels=['Legitimate', 'Mule'], startangle=90)
axes[1].set_title('Class Distribution (Proportion)')
axes[1].set_ylabel('')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '01_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Mule flag date analysis
mule_labels = labels[labels['is_mule'] == 1]
print(f"\nMule flag date range: {mule_labels['mule_flag_date'].min()} to {mule_labels['mule_flag_date'].max()}")
print(f"\nAlert reasons:")
print(mule_labels['alert_reason'].value_counts())
results['alert_reasons'] = mule_labels['alert_reason'].value_counts().to_dict()

print(f"\nFlagged by branch - unique branches: {mule_labels['flagged_by_branch'].nunique()}")

fig, ax = plt.subplots(figsize=(10, 5))
mule_labels['mule_flag_date'].dt.to_period('M').value_counts().sort_index().plot(kind='bar', ax=ax)
ax.set_title('Mule Flag Date Distribution (Monthly)')
ax.set_ylabel('Count')
ax.set_xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '02_mule_flag_dates.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# SECTION 3: MERGE DATA FOR ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: MERGING DATASETS")
print("=" * 60)

# Build master training table
train = labels.merge(accounts, on='account_id', how='left')
train = train.merge(linkage, on='account_id', how='left')
train = train.merge(customers, on='customer_id', how='left')
train = train.merge(products, on='customer_id', how='left')

print(f"Merged training set: {train.shape}")
print(f"Merge coverage - accounts match: {train['account_status'].notna().sum()}/{len(train)}")
print(f"Merge coverage - customers match: {train['date_of_birth'].notna().sum()}/{len(train)}")

mule = train[train['is_mule'] == 1]
legit = train[train['is_mule'] == 0]
print(f"Mule accounts: {len(mule):,}, Legitimate accounts: {len(legit):,}")

# ============================================================
# SECTION 4: ACCOUNT-LEVEL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: ACCOUNT-LEVEL ANALYSIS")
print("=" * 60)

# 4a. Account status
print("\n--- Account Status ---")
status_ct = pd.crosstab(train['is_mule'], train['account_status'], normalize='index')
print(status_ct)
results['account_status'] = status_ct.to_dict()

fig, ax = plt.subplots(figsize=(8, 4))
status_ct.plot(kind='bar', ax=ax)
ax.set_title('Account Status by Class')
ax.set_xticklabels(['Legitimate', 'Mule'], rotation=0)
ax.set_ylabel('Proportion')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '03_account_status.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4b. Product family
print("\n--- Product Family ---")
pf_ct = pd.crosstab(train['is_mule'], train['product_family'], normalize='index')
print(pf_ct)

fig, ax = plt.subplots(figsize=(8, 4))
pf_ct.plot(kind='bar', ax=ax)
ax.set_title('Product Family by Class')
ax.set_xticklabels(['Legitimate', 'Mule'], rotation=0)
ax.set_ylabel('Proportion')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '04_product_family.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4c. Balance metrics
print("\n--- Balance Metrics (Mule vs Legit) ---")
balance_cols = ['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance']
for col in balance_cols:
    m_mean = mule[col].mean()
    l_mean = legit[col].mean()
    m_med = mule[col].median()
    l_med = legit[col].median()
    print(f"  {col}:")
    print(f"    Legit  - Mean: {l_mean:,.2f}, Median: {l_med:,.2f}")
    print(f"    Mule   - Mean: {m_mean:,.2f}, Median: {m_med:,.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(balance_cols):
    ax = axes[idx // 2][idx % 2]
    data_legit = legit[col].clip(-50000, 200000)
    data_mule = mule[col].clip(-50000, 200000)
    ax.hist(data_legit, bins=80, alpha=0.5, label='Legitimate', density=True, color='#2ecc71')
    ax.hist(data_mule, bins=80, alpha=0.5, label='Mule', density=True, color='#e74c3c')
    ax.set_title(f'{col} Distribution')
    ax.legend()
    ax.set_xlabel('Amount (INR)')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '05_balance_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4d. Account opening date patterns
print("\n--- Account Opening Date ---")
train['acct_age_days'] = (pd.Timestamp('2025-06-30') - train['account_opening_date']).dt.days

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
legit['account_opening_date'].dt.year.value_counts().sort_index().plot(kind='bar', ax=axes[0], alpha=0.7, color='#2ecc71', label='Legitimate')
mule['account_opening_date'].dt.year.value_counts().sort_index().plot(kind='bar', ax=axes[0], alpha=0.7, color='#e74c3c', label='Mule')
axes[0].set_title('Account Opening Year Distribution')
axes[0].legend()

# Mule rate by opening year
yearly = train.groupby(train['account_opening_date'].dt.year)['is_mule'].agg(['mean', 'count'])
yearly['mean'].plot(kind='bar', ax=axes[1], color='#e74c3c')
axes[1].set_title('Mule Rate by Account Opening Year')
axes[1].set_ylabel('Mule Rate')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '06_account_opening.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4e. KYC and flags
print("\n--- KYC and Service Flags ---")
flag_cols = ['nomination_flag', 'cheque_allowed', 'cheque_availed', 'kyc_compliant', 'rural_branch']
flag_results = {}
for col in flag_cols:
    ct = pd.crosstab(train['is_mule'], train[col], normalize='index')
    flag_results[col] = ct.to_dict()
    print(f"  {col}:")
    print(f"    Legit: {ct.loc[0].to_dict()}")
    print(f"    Mule:  {ct.loc[1].to_dict()}")

fig, axes = plt.subplots(1, 5, figsize=(22, 4))
for idx, col in enumerate(flag_cols):
    ct = pd.crosstab(train['is_mule'], train[col], normalize='index')
    ct.plot(kind='bar', ax=axes[idx])
    axes[idx].set_title(col)
    axes[idx].set_xticklabels(['Legit', 'Mule'], rotation=0)
    axes[idx].legend(title=col)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '07_account_flags.png'), dpi=150, bbox_inches='tight')
plt.close()

# 4f. Freeze/Unfreeze analysis
print("\n--- Freeze Analysis ---")
train['was_frozen'] = train['freeze_date'].notna().astype(int)
frozen_ct = pd.crosstab(train['is_mule'], train['was_frozen'], normalize='index')
print(frozen_ct)

# 4g. Last mobile update
print("\n--- Mobile Update Analysis ---")
train['had_mobile_update'] = train['last_mobile_update_date'].notna().astype(int)
mobile_ct = pd.crosstab(train['is_mule'], train['had_mobile_update'], normalize='index')
print(mobile_ct)

# ============================================================
# SECTION 5: CUSTOMER-LEVEL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: CUSTOMER-LEVEL ANALYSIS")
print("=" * 60)

# 5a. Age analysis
ref_date = pd.Timestamp('2025-06-30')
train['age'] = (ref_date - train['date_of_birth']).dt.days / 365.25

print("\n--- Age Distribution ---")
print(f"  Legit: Mean={legit['date_of_birth'].apply(lambda x: (ref_date - x).days / 365.25 if pd.notna(x) else np.nan).mean():.1f}, Median={legit['date_of_birth'].apply(lambda x: (ref_date - x).days / 365.25 if pd.notna(x) else np.nan).median():.1f}")
print(f"  Mule:  Mean={mule['date_of_birth'].apply(lambda x: (ref_date - x).days / 365.25 if pd.notna(x) else np.nan).mean():.1f}, Median={mule['date_of_birth'].apply(lambda x: (ref_date - x).days / 365.25 if pd.notna(x) else np.nan).median():.1f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(train.loc[train['is_mule']==0, 'age'].dropna(), bins=50, alpha=0.5, label='Legitimate', density=True, color='#2ecc71')
ax.hist(train.loc[train['is_mule']==1, 'age'].dropna(), bins=50, alpha=0.5, label='Mule', density=True, color='#e74c3c')
ax.set_title('Age Distribution by Class')
ax.set_xlabel('Age (Years)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '08_age_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5b. Relationship tenure
train['relationship_years'] = (ref_date - train['relationship_start_date']).dt.days / 365.25
print(f"\n--- Relationship Tenure ---")
for label, subset in [('Legit', legit), ('Mule', mule)]:
    tenure = (ref_date - subset['relationship_start_date']).dt.days / 365.25
    print(f"  {label}: Mean={tenure.mean():.1f}y, Median={tenure.median():.1f}y")

# 5c. KYC document analysis
print("\n--- KYC Documents ---")
kyc_cols = ['pan_available', 'aadhaar_available', 'passport_available']
for col in kyc_cols:
    ct = pd.crosstab(train['is_mule'], train[col], normalize='index')
    print(f"  {col}: Legit Y={ct.loc[0].get('Y', 0):.3f}, Mule Y={ct.loc[1].get('Y', 0):.3f}")

# 5d. Digital banking flags
print("\n--- Digital Banking Flags ---")
digital_cols = ['mobile_banking_flag', 'internet_banking_flag', 'atm_card_flag', 'demat_flag', 'credit_card_flag', 'fastag_flag']
for col in digital_cols:
    ct = pd.crosstab(train['is_mule'], train[col], normalize='index')
    print(f"  {col}: Legit Y={ct.loc[0].get('Y', 0):.3f}, Mule Y={ct.loc[1].get('Y', 0):.3f}")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for idx, col in enumerate(digital_cols):
    ax = axes[idx // 3][idx % 3]
    ct = pd.crosstab(train['is_mule'], train[col], normalize='index')
    ct.plot(kind='bar', ax=ax)
    ax.set_title(col)
    ax.set_xticklabels(['Legit', 'Mule'], rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '09_digital_banking_flags.png'), dpi=150, bbox_inches='tight')
plt.close()

# 5e. Product holdings
print("\n--- Product Holdings ---")
prod_cols = ['loan_count', 'cc_count', 'od_count', 'ka_count', 'sa_count']
for col in prod_cols:
    legit_v = train.loc[train['is_mule']==0, col]
    mule_v = train.loc[train['is_mule']==1, col]
    print(f"  {col}: Legit mean={legit_v.mean():.3f}, Mule mean={mule_v.mean():.3f}")

# 5f. Multiple accounts per customer
print("\n--- Multi-Account Customers ---")
acct_per_cust = linkage.groupby('customer_id')['account_id'].count().reset_index()
acct_per_cust.columns = ['customer_id', 'num_accounts']
train_multi = train.merge(acct_per_cust, on='customer_id', how='left')
print(f"  Legit avg accounts: {train_multi.loc[train_multi['is_mule']==0, 'num_accounts'].mean():.3f}")
print(f"  Mule avg accounts:  {train_multi.loc[train_multi['is_mule']==1, 'num_accounts'].mean():.3f}")

# 5g. PIN code mismatch (geographic anomaly signal)
print("\n--- PIN Code Mismatch (customer_pin vs permanent_pin) ---")
train['pin_mismatch'] = (train['customer_pin'] != train['permanent_pin']).astype(int)
print(f"  Legit pin mismatch rate: {train.loc[train['is_mule']==0, 'pin_mismatch'].mean():.4f}")
print(f"  Mule pin mismatch rate:  {train.loc[train['is_mule']==1, 'pin_mismatch'].mean():.4f}")

# ============================================================
# SECTION 6: TRANSACTION-LEVEL ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: TRANSACTION-LEVEL ANALYSIS")
print("=" * 60)

# Join labels to transactions
train_acct_ids = set(labels['account_id'])
txn_labeled = transactions[transactions['account_id'].isin(train_acct_ids)].copy()
txn_labeled = txn_labeled.merge(labels[['account_id', 'is_mule']], on='account_id', how='left')
print(f"Labeled transactions: {len(txn_labeled):,}")
print(f"  Legitimate: {(txn_labeled['is_mule']==0).sum():,}")
print(f"  Mule:       {(txn_labeled['is_mule']==1).sum():,}")

# 6a. Transaction volume per account
print("\n--- Transaction Volume per Account ---")
txn_counts = txn_labeled.groupby(['account_id', 'is_mule']).size().reset_index(name='txn_count')
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = txn_counts[txn_counts['is_mule'] == val]['txn_count']
    print(f"  {label}: Mean={subset.mean():.1f}, Median={subset.median():.1f}, Std={subset.std():.1f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(txn_counts.loc[txn_counts['is_mule']==0, 'txn_count'].clip(0, 1000), bins=80, alpha=0.5, density=True, label='Legitimate', color='#2ecc71')
ax.hist(txn_counts.loc[txn_counts['is_mule']==1, 'txn_count'].clip(0, 1000), bins=80, alpha=0.5, density=True, label='Mule', color='#e74c3c')
ax.set_title('Transaction Count per Account')
ax.set_xlabel('Number of Transactions')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '10_txn_volume.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6b. Transaction amount distribution
print("\n--- Transaction Amount Distribution ---")
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = txn_labeled[txn_labeled['is_mule'] == val]['amount']
    print(f"  {label}: Mean={subset.mean():.2f}, Median={subset.median():.2f}, Std={subset.std():.2f}")
    print(f"         Min={subset.min():.2f}, Max={subset.max():.2f}")
    print(f"         P25={subset.quantile(0.25):.2f}, P75={subset.quantile(0.75):.2f}, P95={subset.quantile(0.95):.2f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(txn_labeled.loc[txn_labeled['is_mule']==0, 'amount'].clip(-5000, 100000), bins=100, alpha=0.5, density=True, label='Legitimate', color='#2ecc71')
ax.hist(txn_labeled.loc[txn_labeled['is_mule']==1, 'amount'].clip(-5000, 100000), bins=100, alpha=0.5, density=True, label='Mule', color='#e74c3c')
ax.set_title('Transaction Amount Distribution')
ax.set_xlabel('Amount (INR)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '11_txn_amount_dist.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6c. Channel analysis
print("\n--- Channel Usage ---")
channel_mule = pd.crosstab(txn_labeled['is_mule'], txn_labeled['channel'], normalize='index')
print("Top channels (Mule):")
print(channel_mule.loc[1].sort_values(ascending=False).head(10))
print("\nTop channels (Legit):")
print(channel_mule.loc[0].sort_values(ascending=False).head(10))

# Channel difference
channel_diff = (channel_mule.loc[1] - channel_mule.loc[0]).sort_values(ascending=False)
print("\nChannel over-representation in Mule accounts (mule_rate - legit_rate):")
print(channel_diff.head(10))
print(channel_diff.tail(10))

fig, ax = plt.subplots(figsize=(14, 6))
channel_diff.plot(kind='bar', ax=ax, color=[('#e74c3c' if v > 0 else '#2ecc71') for v in channel_diff.values])
ax.set_title('Channel Usage Difference (Mule - Legitimate)')
ax.set_ylabel('Proportion Difference')
ax.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '12_channel_diff.png'), dpi=150, bbox_inches='tight')
plt.close()

# 6d. Debit vs Credit
print("\n--- Debit vs Credit ---")
txn_type_ct = pd.crosstab(txn_labeled['is_mule'], txn_labeled['txn_type'], normalize='index')
print(txn_type_ct)

# 6e. Temporal patterns
print("\n--- Temporal Patterns ---")
txn_labeled['hour'] = txn_labeled['transaction_timestamp'].dt.hour
txn_labeled['dow'] = txn_labeled['transaction_timestamp'].dt.dayofweek
txn_labeled['month'] = txn_labeled['transaction_timestamp'].dt.month

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Hour of day
for val, label, color in [(0, 'Legitimate', '#2ecc71'), (1, 'Mule', '#e74c3c')]:
    hour_dist = txn_labeled[txn_labeled['is_mule']==val]['hour'].value_counts(normalize=True).sort_index()
    axes[0].plot(hour_dist.index, hour_dist.values, label=label, color=color)
axes[0].set_title('Transaction Hour Distribution')
axes[0].set_xlabel('Hour of Day')
axes[0].legend()

# Day of week
for val, label, color in [(0, 'Legitimate', '#2ecc71'), (1, 'Mule', '#e74c3c')]:
    dow_dist = txn_labeled[txn_labeled['is_mule']==val]['dow'].value_counts(normalize=True).sort_index()
    axes[1].plot(dow_dist.index, dow_dist.values, label=label, color=color, marker='o')
axes[1].set_title('Transaction Day-of-Week Distribution')
axes[1].set_xlabel('Day (0=Mon)')
axes[1].legend()

# Monthly
for val, label, color in [(0, 'Legitimate', '#2ecc71'), (1, 'Mule', '#e74c3c')]:
    month_dist = txn_labeled[txn_labeled['is_mule']==val]['month'].value_counts(normalize=True).sort_index()
    axes[2].plot(month_dist.index, month_dist.values, label=label, color=color, marker='o')
axes[2].set_title('Transaction Month Distribution')
axes[2].set_xlabel('Month')
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '13_temporal_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# SECTION 7: MULE BEHAVIOR PATTERN ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 7: MULE BEHAVIOR PATTERN ANALYSIS")
print("=" * 60)

# 7a. Pattern 1: Dormant Activation
print("\n--- Pattern 1: Dormant Activation ---")
# For each account, compute time between first and second transaction burst
acct_txn_stats = txn_labeled.groupby(['account_id', 'is_mule']).agg(
    first_txn=('transaction_timestamp', 'min'),
    last_txn=('transaction_timestamp', 'max'),
    txn_count=('transaction_id', 'count')
).reset_index()

acct_txn_stats['active_span_days'] = (acct_txn_stats['last_txn'] - acct_txn_stats['first_txn']).dt.days
acct_txn_stats['txn_per_day'] = acct_txn_stats['txn_count'] / acct_txn_stats['active_span_days'].clip(lower=1)

# Accounts with opening date much before first txn
acct_txn_stats = acct_txn_stats.merge(accounts[['account_id', 'account_opening_date']], on='account_id', how='left')
acct_txn_stats['dormant_days'] = (acct_txn_stats['first_txn'] - acct_txn_stats['account_opening_date']).dt.days
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = acct_txn_stats[acct_txn_stats['is_mule'] == val]
    print(f"  {label} - Avg dormant days before first txn: {subset['dormant_days'].mean():.1f}")
    print(f"  {label} - Median dormant days: {subset['dormant_days'].median():.1f}")

# 7b. Pattern 2: Structuring (near threshold amounts)
print("\n--- Pattern 2: Structuring (Near 50K Threshold) ---")
txn_labeled['near_50k'] = ((txn_labeled['amount'] >= 45000) & (txn_labeled['amount'] < 50000)).astype(int)
txn_labeled['near_10k'] = ((txn_labeled['amount'] >= 9000) & (txn_labeled['amount'] < 10000)).astype(int)
structuring = txn_labeled.groupby(['account_id', 'is_mule']).agg(
    near_50k_count=('near_50k', 'sum'),
    near_10k_count=('near_10k', 'sum'),
    total_txn=('transaction_id', 'count')
).reset_index()
structuring['near_50k_rate'] = structuring['near_50k_count'] / structuring['total_txn']
structuring['near_10k_rate'] = structuring['near_10k_count'] / structuring['total_txn']
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = structuring[structuring['is_mule'] == val]
    print(f"  {label} - Near 50K rate: {subset['near_50k_rate'].mean():.6f}")
    print(f"  {label} - Near 10K rate: {subset['near_10k_rate'].mean():.6f}")

# 7c. Pattern 3: Rapid Pass-Through
print("\n--- Pattern 3: Rapid Pass-Through ---")
# For each account, compute credit-debit pairs close in time
credit_total = txn_labeled.groupby(['account_id', 'is_mule']).apply(
    lambda x: x[x['txn_type'] == 'C']['amount'].sum(), include_groups=False
).reset_index(name='total_credit')
debit_total = txn_labeled.groupby(['account_id', 'is_mule']).apply(
    lambda x: x[x['txn_type'] == 'D']['amount'].sum(), include_groups=False
).reset_index(name='total_debit')
passthrough = credit_total.merge(debit_total, on=['account_id', 'is_mule'])
passthrough['passthrough_ratio'] = passthrough['total_debit'] / passthrough['total_credit'].clip(lower=1)
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = passthrough[passthrough['is_mule'] == val]
    print(f"  {label} - Median passthrough ratio (debit/credit): {subset['passthrough_ratio'].median():.4f}")
    print(f"  {label} - Mean passthrough ratio: {subset['passthrough_ratio'].mean():.4f}")

# 7d. Pattern 4: Fan-In / Fan-Out
print("\n--- Pattern 4: Fan-In / Fan-Out ---")
counterparty_stats = txn_labeled.groupby(['account_id', 'is_mule']).agg(
    unique_counterparties=('counterparty_id', 'nunique'),
    total_txn=('transaction_id', 'count'),
    unique_credit_cp=('counterparty_id', lambda x: x[txn_labeled.loc[x.index, 'txn_type'] == 'C'].nunique()),
    unique_debit_cp=('counterparty_id', lambda x: x[txn_labeled.loc[x.index, 'txn_type'] == 'D'].nunique())
).reset_index()
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = counterparty_stats[counterparty_stats['is_mule'] == val]
    print(f"  {label} - Avg unique counterparties: {subset['unique_counterparties'].mean():.1f}")
    print(f"  {label} - Avg unique credit counterparties: {subset['unique_credit_cp'].mean():.1f}")
    print(f"  {label} - Avg unique debit counterparties: {subset['unique_debit_cp'].mean():.1f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(counterparty_stats.loc[counterparty_stats['is_mule']==0, 'unique_counterparties'].clip(0, 200), bins=50, alpha=0.5, density=True, label='Legitimate', color='#2ecc71')
ax.hist(counterparty_stats.loc[counterparty_stats['is_mule']==1, 'unique_counterparties'].clip(0, 200), bins=50, alpha=0.5, density=True, label='Mule', color='#e74c3c')
ax.set_title('Unique Counterparties per Account')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '14_counterparties.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7e. Pattern 5: Geographic Anomaly
print("\n--- Pattern 5: Geographic Anomaly ---")
# Compare customer_pin, permanent_pin, and branch_pin
train['cust_branch_pin_mismatch'] = (train['customer_pin'].astype(str) != train['branch_pin'].astype(str)).astype(int)
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = train[train['is_mule'] == val]
    print(f"  {label} - Customer-Branch PIN mismatch: {subset['cust_branch_pin_mismatch'].mean():.4f}")
    print(f"  {label} - Customer-Permanent PIN mismatch: {subset['pin_mismatch'].mean():.4f}")

# 7f. Pattern 6: New Account High Value
print("\n--- Pattern 6: New Account High Value ---")
train['acct_age_months'] = (ref_date - train['account_opening_date']).dt.days / 30.44
new_accounts = train[train['acct_age_months'] <= 12]  # opened within last year
old_accounts = train[train['acct_age_months'] > 12]
print(f"  New accounts (<1yr) mule rate: {new_accounts['is_mule'].mean():.4f} (n={len(new_accounts)})")
print(f"  Old accounts (>1yr) mule rate: {old_accounts['is_mule'].mean():.4f} (n={len(old_accounts)})")

# Merge txn volume with account age
new_acct_txn = acct_txn_stats.merge(train[['account_id', 'acct_age_months', 'is_mule']].drop_duplicates(), on=['account_id', 'is_mule'])

fig, ax = plt.subplots(figsize=(10, 5))
age_bins = pd.cut(new_acct_txn['acct_age_months'], bins=10)
mule_by_age = new_acct_txn.groupby(age_bins, observed=True)['is_mule'].mean()
mule_by_age.plot(kind='bar', ax=ax, color='#e74c3c')
ax.set_title('Mule Rate by Account Age (Months)')
ax.set_ylabel('Mule Rate')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '15_mule_rate_by_age.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7g. Pattern 7: Income Mismatch
print("\n--- Pattern 7: Income Mismatch ---")
# Compare avg transaction amount to account balance
acct_txn_amounts = txn_labeled.groupby(['account_id', 'is_mule']).agg(
    avg_txn_amount=('amount', 'mean'),
    max_txn_amount=('amount', 'max'),
    total_txn_amount=('amount', 'sum')
).reset_index()
acct_txn_amounts = acct_txn_amounts.merge(accounts[['account_id', 'avg_balance']], on='account_id', how='left')
acct_txn_amounts['txn_to_balance_ratio'] = acct_txn_amounts['total_txn_amount'] / acct_txn_amounts['avg_balance'].clip(lower=1)
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = acct_txn_amounts[acct_txn_amounts['is_mule'] == val]
    print(f"  {label} - Median txn-to-balance ratio: {subset['txn_to_balance_ratio'].median():.2f}")
    print(f"  {label} - Mean max txn amount: {subset['max_txn_amount'].mean():.2f}")

# 7h. Pattern 8: Post-Mobile-Change Spike
print("\n--- Pattern 8: Post-Mobile-Change Spike ---")
mobile_updates = train[train['last_mobile_update_date'].notna()][['account_id', 'is_mule', 'last_mobile_update_date']]
print(f"  Accounts with mobile update: {len(mobile_updates)}")
print(f"  Legit with mobile update: {(mobile_updates['is_mule']==0).sum()}")
print(f"  Mule with mobile update: {(mobile_updates['is_mule']==1).sum()}")
if len(mobile_updates) > 0:
    # Check txn volume in 30 days after mobile update
    mobile_updates_expanded = mobile_updates.merge(
        txn_labeled[['account_id', 'transaction_timestamp', 'amount']],
        on='account_id'
    )
    mobile_updates_expanded['days_after_update'] = (
        mobile_updates_expanded['transaction_timestamp'] - mobile_updates_expanded['last_mobile_update_date']
    ).dt.days
    post_update = mobile_updates_expanded[
        (mobile_updates_expanded['days_after_update'] >= 0) &
        (mobile_updates_expanded['days_after_update'] <= 30)
    ]
    post_update_stats = post_update.groupby(['account_id', 'is_mule']).agg(
        post_txn_count=('amount', 'count'),
        post_txn_sum=('amount', 'sum')
    ).reset_index()
    for label, val in [('Legit', 0), ('Mule', 1)]:
        subset = post_update_stats[post_update_stats['is_mule'] == val]
        if len(subset) > 0:
            print(f"  {label} - Avg txns 30d post-mobile-update: {subset['post_txn_count'].mean():.1f}")
            print(f"  {label} - Avg amount 30d post-mobile-update: {subset['post_txn_sum'].mean():.2f}")

# 7i. Pattern 9: Round Amount Patterns
print("\n--- Pattern 9: Round Amount Patterns ---")
txn_labeled['is_round_1k'] = (txn_labeled['amount'] % 1000 == 0).astype(int)
txn_labeled['is_round_5k'] = (txn_labeled['amount'] % 5000 == 0).astype(int)
txn_labeled['is_round_10k'] = (txn_labeled['amount'] % 10000 == 0).astype(int)
round_stats = txn_labeled.groupby(['account_id', 'is_mule']).agg(
    round_1k_rate=('is_round_1k', 'mean'),
    round_5k_rate=('is_round_5k', 'mean'),
    round_10k_rate=('is_round_10k', 'mean')
).reset_index()
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = round_stats[round_stats['is_mule'] == val]
    print(f"  {label} - Round 1K rate: {subset['round_1k_rate'].mean():.4f}")
    print(f"  {label} - Round 5K rate: {subset['round_5k_rate'].mean():.4f}")
    print(f"  {label} - Round 10K rate: {subset['round_10k_rate'].mean():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
round_summary = pd.DataFrame({
    'Legitimate': [round_stats[round_stats['is_mule']==0][c].mean() for c in ['round_1k_rate', 'round_5k_rate', 'round_10k_rate']],
    'Mule': [round_stats[round_stats['is_mule']==1][c].mean() for c in ['round_1k_rate', 'round_5k_rate', 'round_10k_rate']]
}, index=['Round 1K', 'Round 5K', 'Round 10K'])
round_summary.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
ax.set_title('Round Amount Transaction Rates')
ax.set_ylabel('Proportion')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '16_round_amounts.png'), dpi=150, bbox_inches='tight')
plt.close()

# 7j. Pattern 11: Salary Cycle Exploitation
print("\n--- Pattern 11: Salary Cycle Exploitation ---")
txn_labeled['day_of_month'] = txn_labeled['transaction_timestamp'].dt.day
# Salary days typically 1-5 or 25-31
txn_labeled['is_salary_window'] = ((txn_labeled['day_of_month'] <= 5) | (txn_labeled['day_of_month'] >= 25)).astype(int)
salary_pattern = txn_labeled.groupby(['account_id', 'is_mule']).agg(
    salary_window_rate=('is_salary_window', 'mean')
).reset_index()
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = salary_pattern[salary_pattern['is_mule'] == val]
    print(f"  {label} - Salary window txn rate: {subset['salary_window_rate'].mean():.4f}")

# 7k. Pattern 12: Branch-Level Collusion
print("\n--- Pattern 12: Branch-Level Collusion ---")
branch_mule = train.groupby('branch_code').agg(
    total_accounts=('account_id', 'count'),
    mule_count=('is_mule', 'sum'),
    mule_rate=('is_mule', 'mean')
).reset_index()
branch_mule = branch_mule[branch_mule['total_accounts'] >= 5]  # min 5 accounts
high_mule_branches = branch_mule[branch_mule['mule_rate'] > 0.3].sort_values('mule_rate', ascending=False)
print(f"  Branches with >30% mule rate (min 5 accounts): {len(high_mule_branches)}")
print(f"  Top branches:")
print(high_mule_branches.head(10).to_string())

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(branch_mule['mule_rate'], bins=50, color='#3498db', edgecolor='white')
ax.axvline(x=branch_mule['mule_rate'].mean(), color='red', linestyle='--', label=f"Mean: {branch_mule['mule_rate'].mean():.3f}")
ax.set_title('Mule Rate Distribution Across Branches')
ax.set_xlabel('Mule Rate')
ax.set_ylabel('Number of Branches')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '17_branch_mule_rate.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# SECTION 8: ADVANCED ANALYTICS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 8: ADVANCED ANALYTICS")
print("=" * 60)

# 8a. Transaction velocity features per account
print("\n--- Transaction Velocity ---")
txn_labeled_sorted = txn_labeled.sort_values(['account_id', 'transaction_timestamp'])
txn_labeled_sorted['time_diff'] = txn_labeled_sorted.groupby('account_id')['transaction_timestamp'].diff().dt.total_seconds() / 3600

velocity_stats = txn_labeled_sorted.groupby(['account_id', 'is_mule']).agg(
    median_hours_between_txn=('time_diff', 'median'),
    min_hours_between_txn=('time_diff', 'min'),
    std_hours_between_txn=('time_diff', 'std')
).reset_index()
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = velocity_stats[velocity_stats['is_mule'] == val]
    print(f"  {label} - Median hours between txns: {subset['median_hours_between_txn'].median():.2f}")
    print(f"  {label} - Min hours between txns (median): {subset['min_hours_between_txn'].median():.4f}")

# 8b. Night transactions
print("\n--- Night Transactions (0:00-6:00) ---")
txn_labeled['is_night'] = ((txn_labeled['hour'] >= 0) & (txn_labeled['hour'] < 6)).astype(int)
night_stats = txn_labeled.groupby(['account_id', 'is_mule'])['is_night'].mean().reset_index(name='night_rate')
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = night_stats[night_stats['is_mule'] == val]
    print(f"  {label} - Avg night txn rate: {subset['night_rate'].mean():.4f}")

# 8c. Weekend transactions
print("\n--- Weekend Transactions ---")
txn_labeled['is_weekend'] = (txn_labeled['dow'] >= 5).astype(int)
weekend_stats = txn_labeled.groupby(['account_id', 'is_mule'])['is_weekend'].mean().reset_index(name='weekend_rate')
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = weekend_stats[weekend_stats['is_mule'] == val]
    print(f"  {label} - Avg weekend txn rate: {subset['weekend_rate'].mean():.4f}")

# 8d. Burst detection - standard deviation of daily txn counts
print("\n--- Burst Detection ---")
txn_labeled['date'] = txn_labeled['transaction_timestamp'].dt.date
daily_counts = txn_labeled.groupby(['account_id', 'is_mule', 'date']).size().reset_index(name='daily_count')
burst_stats = daily_counts.groupby(['account_id', 'is_mule']).agg(
    mean_daily_txn=('daily_count', 'mean'),
    max_daily_txn=('daily_count', 'max'),
    std_daily_txn=('daily_count', 'std')
).reset_index()
burst_stats['burstiness'] = burst_stats['max_daily_txn'] / burst_stats['mean_daily_txn'].clip(lower=0.01)
for label, val in [('Legit', 0), ('Mule', 1)]:
    subset = burst_stats[burst_stats['is_mule'] == val]
    print(f"  {label} - Avg burstiness (max/mean daily txn): {subset['burstiness'].mean():.2f}")
    print(f"  {label} - Avg max daily txn: {subset['max_daily_txn'].mean():.1f}")

# 8e. MCC code analysis
print("\n--- MCC Code Analysis ---")
mcc_mule = txn_labeled.groupby(['is_mule', 'mcc_code']).size().reset_index(name='count')
mcc_pivot = mcc_mule.pivot(index='mcc_code', columns='is_mule', values='count').fillna(0)
mcc_pivot.columns = ['legit_count', 'mule_count']
mcc_pivot['legit_rate'] = mcc_pivot['legit_count'] / mcc_pivot['legit_count'].sum()
mcc_pivot['mule_rate'] = mcc_pivot['mule_count'] / mcc_pivot['mule_count'].sum()
mcc_pivot['diff'] = mcc_pivot['mule_rate'] - mcc_pivot['legit_rate']
print("Top MCC codes over-represented in Mule:")
print(mcc_pivot.sort_values('diff', ascending=False).head(10)[['legit_rate', 'mule_rate', 'diff']])

# 8f. Correlation heatmap for numeric features
print("\n--- Feature Correlations ---")
numeric_cols = ['avg_balance', 'monthly_avg_balance', 'quarterly_avg_balance', 'daily_avg_balance',
                'num_chequebooks', 'loan_count', 'cc_count', 'od_count', 'ka_count', 'sa_count',
                'is_mule']
corr_data = train[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Feature Correlations with is_mule')
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, '18_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()

# Print correlations with target
print("\nCorrelation with is_mule:")
target_corr = corr_data['is_mule'].drop('is_mule').sort_values(key=abs, ascending=False)
print(target_corr)

# ============================================================
# SECTION 9: DATA QUALITY ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 9: DATA QUALITY ANALYSIS")
print("=" * 60)

# 9a. Missing value patterns by class
print("\n--- Missing Values by Class ---")
for col in train.columns:
    missing_legit = train.loc[train['is_mule']==0, col].isna().mean()
    missing_mule = train.loc[train['is_mule']==1, col].isna().mean()
    if missing_legit > 0 or missing_mule > 0:
        print(f"  {col}: Legit={missing_legit:.4f}, Mule={missing_mule:.4f}")

# 9b. Potential data leakage
print("\n--- Data Leakage Analysis ---")
print("  Columns that exist ONLY for mule accounts (potential leakage):")
for col in ['mule_flag_date', 'alert_reason', 'flagged_by_branch']:
    non_null_legit = labels.loc[labels['is_mule']==0, col].notna().sum()
    non_null_mule = labels.loc[labels['is_mule']==1, col].notna().sum()
    print(f"    {col}: Legit non-null={non_null_legit}, Mule non-null={non_null_mule}")

print("\n  Freeze date leakage check:")
print(f"    Legit frozen: {train.loc[train['is_mule']==0, 'was_frozen'].mean():.4f}")
print(f"    Mule frozen:  {train.loc[train['is_mule']==1, 'was_frozen'].mean():.4f}")

# 9c. Label noise
print("\n--- Label Noise ---")
print(f"  Total mule labels: {labels['is_mule'].sum()}")
# Check if any mule accounts have no transactions
mule_accts = set(labels[labels['is_mule']==1]['account_id'])
mule_txn_counts = transactions[transactions['account_id'].isin(mule_accts)].groupby('account_id').size()
mule_no_txn = mule_accts - set(mule_txn_counts.index)
print(f"  Mule accounts with no transactions: {len(mule_no_txn)}")

# 9d. Duplicate checks
print("\n--- Duplicate Checks ---")
print(f"  Duplicate transaction_ids: {transactions['transaction_id'].duplicated().sum()}")
print(f"  Duplicate account_ids in labels: {labels['account_id'].duplicated().sum()}")
print(f"  Overlap train/test: {len(set(labels['account_id']) & set(test['account_id']))}")

# 9e. Negative amounts
print("\n--- Negative / Zero Amounts ---")
print(f"  Negative amounts: {(transactions['amount'] < 0).sum():,} ({(transactions['amount'] < 0).mean()*100:.2f}%)")
print(f"  Zero amounts: {(transactions['amount'] == 0).sum():,}")

print("\n\n" + "=" * 60)
print("EDA COMPLETE - All plots saved to:", PLOT_DIR)
print("=" * 60)
