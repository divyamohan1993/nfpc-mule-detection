"""Deep data pollution and edge case check."""
import pandas as pd
import numpy as np
from glob import glob

DATA = "/home/DIVYA/nfpc-phase2/data"

print("=== DATA POLLUTION & EDGE CASE CHECK ===")

# 1. NaN/Inf in numeric columns
print("\n--- NaN/Inf in static files ---")
for name, path in [
    ("accounts", "accounts.parquet"),
    ("customers", "customers.parquet"),
    ("demographics", "demographics.parquet"),
    ("product_details", "product_details.parquet"),
    ("branch", "branch.parquet"),
]:
    df = pd.read_parquet(f"{DATA}/{path}")
    for col in df.select_dtypes(include=[np.number]).columns:
        n_nan = int(df[col].isna().sum())
        n_inf = int(np.isinf(df[col].dropna()).sum()) if df[col].dtype in ["float64", "float32"] else 0
        if n_nan > 0 or n_inf > 0:
            pct = 100 * n_nan / len(df)
            print(f"  {name}.{col}: {n_nan} NaN ({pct:.1f}%), {n_inf} Inf")
    del df

# 2. Duplicate IDs
print("\n--- Duplicate IDs ---")
for name, path, col in [
    ("accounts", "accounts.parquet", "account_id"),
    ("train_labels", "train_labels.parquet", "account_id"),
    ("test_accounts", "test_accounts.parquet", "account_id"),
    ("customers", "customers.parquet", "customer_id"),
]:
    df = pd.read_parquet(f"{DATA}/{path}")
    dups = int(df[col].duplicated().sum())
    status = "[OK]" if dups == 0 else "[WARN]"
    print(f"  {name}.{col}: {dups} duplicates {status}")
    del df

# 3. Empty strings / string "None"
print("\n--- Empty strings / string 'None' ---")
for name, path in [
    ("accounts", "accounts.parquet"),
    ("customers", "customers.parquet"),
    ("demographics", "demographics.parquet"),
    ("train_labels", "train_labels.parquet"),
]:
    df = pd.read_parquet(f"{DATA}/{path}")
    for col in df.select_dtypes(include=["object"]).columns:
        n_empty = int((df[col] == "").sum())
        n_none_str = int((df[col] == "None").sum())
        if n_empty > 0:
            print(f"  {name}.{col}: {n_empty} empty strings")
        if n_none_str > 0:
            print(f"  {name}.{col}: {n_none_str} string 'None' (not NaN)")
    del df
print("  (none found = clean)")

# 4. Transaction edge cases (3 sampled parts)
print("\n--- Transaction edge cases (3 sampled parts) ---")
parts = sorted(glob(f"{DATA}/transactions/batch-*/part_*.parquet"))
for idx in [0, len(parts) // 2, len(parts) - 1]:
    df = pd.read_parquet(parts[idx])
    label = parts[idx].split("/")[-2] + "/" + parts[idx].split("/")[-1]

    cp_nan = int(df["counterparty_id"].isna().sum())
    cp_none = int((df["counterparty_id"] == "None").sum())
    cp_empty = int((df["counterparty_id"] == "").sum())
    print(f"  {label} counterparty: {cp_nan} NaN, {cp_none} str-None, {cp_empty} empty")

    amt = df["amount"]
    print(f"  {label} amount: min={amt.min():.0f}, max={amt.max():.0f}, "
          f"zero={int((amt == 0).sum())}, negative={int((amt < 0).sum())}")

    mcc = df["mcc_code"]
    print(f"  {label} mcc_code: min={mcc.min()}, max={mcc.max()}, "
          f"unique={mcc.nunique()}, NaN={int(mcc.isna().sum())}")

    ch = df["channel"]
    print(f"  {label} channel: unique={ch.nunique()}, NaN={int(ch.isna().sum())}")

    tt = df["txn_type"]
    print(f"  {label} txn_type: values={dict(tt.value_counts())}")

    ts = df["transaction_timestamp"]
    ts_parsed = pd.to_datetime(ts, format="mixed")
    print(f"  {label} timestamps: min={ts_parsed.min()}, max={ts_parsed.max()}")
    del df
    print()

# 5. Transactions additional edge cases
print("--- Txn additional edge cases ---")
add_parts = sorted(glob(f"{DATA}/transactions_additional/batch-*/part_*.parquet"))
df = pd.read_parquet(add_parts[0])
for col in ["ip_address", "part_transaction_type", "atm_deposit_channel_code", "transaction_sub_type"]:
    n_nan = int(df[col].isna().sum())
    n_none = int((df[col] == "None").sum())
    uniq = df[col].dropna().nunique()
    print(f"  {col}: {n_nan} NaN, {n_none} str-None, {uniq} unique values")
    if uniq < 20:
        print(f"    values: {sorted(df[col].dropna().unique().tolist())}")

lat_nan = int(df["latitude"].isna().sum())
lon_nan = int(df["longitude"].isna().sum())
bal_nan = int(df["balance_after_transaction"].isna().sum())
print(f"  latitude NaN: {lat_nan}/{len(df)} ({100*lat_nan/len(df):.1f}%)")
print(f"  longitude NaN: {lon_nan}/{len(df)} ({100*lon_nan/len(df):.1f}%)")
print(f"  balance NaN: {bal_nan}/{len(df)} ({100*bal_nan/len(df):.1f}%)")
del df

# 6. Linkage completeness
print("\n--- Linkage completeness ---")
link = pd.read_parquet(f"{DATA}/customer_account_linkage.parquet")
accts = pd.read_parquet(f"{DATA}/accounts.parquet", columns=["account_id"])
custs = pd.read_parquet(f"{DATA}/customers.parquet", columns=["customer_id"])
acct_in_link = link["account_id"].isin(accts["account_id"]).all()
cust_in_link = link["customer_id"].isin(custs["customer_id"]).all()
acct_missing = (~accts["account_id"].isin(link["account_id"])).sum()
cust_missing = (~custs["customer_id"].isin(link["customer_id"])).sum()
print(f"  All linked accounts valid: {acct_in_link}")
print(f"  All linked customers valid: {cust_in_link}")
print(f"  Accounts not in linkage: {acct_missing}")
print(f"  Customers not in linkage: {cust_missing}")
print(f"  Linkage: {len(link)}, Accounts: {len(accts)}, Customers: {len(custs)}")

# 7. Check balance columns for potential division-by-zero
print("\n--- Balance zero/negative check ---")
accts_full = pd.read_parquet(f"{DATA}/accounts.parquet")
for col in ["avg_balance", "monthly_avg_balance", "quarterly_avg_balance", "daily_avg_balance"]:
    vals = accts_full[col].dropna()
    n_zero = int((vals == 0).sum())
    n_neg = int((vals < 0).sum())
    print(f"  {col}: {n_zero} zeros, {n_neg} negatives (pipeline uses .clip(lower=1))")

print("\n=== ALL CHECKS COMPLETE ===")
