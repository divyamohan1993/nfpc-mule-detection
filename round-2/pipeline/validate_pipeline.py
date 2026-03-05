"""
NFPC Phase 2 — Pipeline-Data Compatibility Validator

Checks every column reference the pipeline makes against actual parquet schemas.
Run this BEFORE the pipeline to catch mismatches.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import sys

DATA = Path("/home/DIVYA/nfpc-phase2/data")
TXN_DIR = DATA / "transactions"
TXN_ADD_DIR = DATA / "transactions_additional"

PASS = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  [OK] {msg}")


def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def check_cols(df, required_cols, file_label):
    actual = set(df.columns)
    for col in required_cols:
        if col in actual:
            ok(f"{file_label}: '{col}' exists (dtype={df[col].dtype})")
        else:
            fail(f"{file_label}: '{col}' MISSING! Available: {sorted(actual)}")


def check_file_exists(path, label):
    if Path(path).exists():
        ok(f"{label} exists at {path}")
        return True
    else:
        fail(f"{label} NOT FOUND at {path}")
        return False


def main():
    print("=" * 60)
    print("PIPELINE-DATA COMPATIBILITY CHECK")
    print("=" * 60)

    # 1. Check all required files exist
    print("\n--- File existence ---")
    files = {
        "accounts": DATA / "accounts.parquet",
        "customers": DATA / "customers.parquet",
        "demographics": DATA / "demographics.parquet",
        "accounts-additional": DATA / "accounts-additional.parquet",
        "branch": DATA / "branch.parquet",
        "customer_account_linkage": DATA / "customer_account_linkage.parquet",
        "product_details": DATA / "product_details.parquet",
        "train_labels": DATA / "train_labels.parquet",
        "test_accounts": DATA / "test_accounts.parquet",
    }
    for label, path in files.items():
        check_file_exists(path, label)

    # 2. Check transactions directory
    print("\n--- Transactions ---")
    txn_parts = sorted(glob(str(TXN_DIR / "batch-*" / "part_*.parquet")))
    if txn_parts:
        ok(f"transactions: {len(txn_parts)} parts found")
    else:
        fail("transactions: no parts found!")

    txn_add_parts = sorted(glob(str(TXN_ADD_DIR / "batch-*" / "part_*.parquet")))
    if txn_add_parts:
        ok(f"transactions_additional: {len(txn_add_parts)} parts found")
    else:
        fail("transactions_additional: no parts found!")

    # 3. Check column names match what features.py expects

    # --- features.py Pass 1: build_txn_features ---
    print("\n--- Pass 1 columns (transactions) ---")
    # Pipeline reads: account_id, transaction_timestamp, amount, txn_type, channel, mcc_code, counterparty_id
    df = pd.read_parquet(txn_parts[0])
    check_cols(df, ["transaction_id", "account_id", "transaction_timestamp",
                     "amount", "txn_type", "channel", "mcc_code", "counterparty_id"], "txn")

    # Test datetime parsing with format="mixed"
    try:
        ts = pd.to_datetime(df["transaction_timestamp"].head(100), format="mixed")
        ok(f"txn: datetime parse with format='mixed' works ({ts.dtype})")
    except Exception as e:
        fail(f"txn: datetime parse failed: {e}")

    # Test on LAST part too (different batch might have different format)
    df_last = pd.read_parquet(txn_parts[-1])
    check_cols(df_last, ["transaction_id", "account_id", "transaction_timestamp",
                          "amount", "txn_type", "channel", "mcc_code", "counterparty_id"], "txn_last")
    try:
        ts2 = pd.to_datetime(df_last["transaction_timestamp"].head(100), format="mixed")
        ok(f"txn_last: datetime parse with format='mixed' works ({ts2.dtype})")
    except Exception as e:
        fail(f"txn_last: datetime parse failed: {e}")

    # Test on a middle part too
    df_mid = pd.read_parquet(txn_parts[len(txn_parts)//2])
    try:
        ts3 = pd.to_datetime(df_mid["transaction_timestamp"].head(100), format="mixed")
        ok(f"txn_mid: datetime parse with format='mixed' works ({ts3.dtype})")
    except Exception as e:
        fail(f"txn_mid: datetime parse failed: {e}")

    # Check txn_type values
    unique_types = df["txn_type"].dropna().unique()
    if "C" in unique_types and "D" in unique_types:
        ok(f"txn: txn_type has C/D values: {sorted(unique_types)}")
    else:
        fail(f"txn: txn_type unexpected values: {sorted(unique_types)}")

    # Check channel values
    unique_channels = df["channel"].dropna().unique()
    ok(f"txn: {len(unique_channels)} unique channels: {sorted(unique_channels)[:10]}...")
    del df, df_last, df_mid

    # --- features.py Pass 2: build_txn_additional_features ---
    print("\n--- Pass 2 columns (transactions_additional) ---")
    # Pipeline reads: transaction_id, latitude, longitude, ip_address,
    #   balance_after_transaction, part_transaction_type, atm_deposit_channel_code, transaction_sub_type
    df2 = pd.read_parquet(txn_add_parts[0])
    check_cols(df2, ["transaction_id", "latitude", "longitude", "ip_address",
                      "balance_after_transaction", "part_transaction_type",
                      "atm_deposit_channel_code", "transaction_sub_type"], "txn_add")
    del df2

    # --- features.py Pass 3: build_static_features ---
    print("\n--- Pass 3 columns (static files) ---")

    # accounts.parquet
    acct = pd.read_parquet(files["accounts"])
    check_cols(acct, ["account_id", "account_opening_date", "account_status",
                       "product_family", "branch_code", "branch_pin",
                       "avg_balance", "monthly_avg_balance", "quarterly_avg_balance",
                       "daily_avg_balance", "freeze_date", "unfreeze_date",
                       "last_mobile_update_date", "kyc_compliant", "last_kyc_date",
                       "nomination_flag", "cheque_allowed", "cheque_availed",
                       "rural_branch", "product_code", "currency_code", "num_chequebooks"], "accounts")
    # Test datetime parsing on account_opening_date
    try:
        pd.to_datetime(acct["account_opening_date"].head(100))
        ok("accounts: account_opening_date parses as datetime")
    except Exception as e:
        fail(f"accounts: account_opening_date parse error: {e}")
    # Test freeze_date
    try:
        pd.to_datetime(acct["freeze_date"].dropna().head(100))
        ok("accounts: freeze_date parses as datetime")
    except Exception as e:
        fail(f"accounts: freeze_date parse error: {e}")
    # Test last_mobile_update_date
    try:
        pd.to_datetime(acct["last_mobile_update_date"].dropna().head(100))
        ok("accounts: last_mobile_update_date parses as datetime")
    except Exception as e:
        fail(f"accounts: last_mobile_update_date parse error: {e}")
    # Test last_kyc_date
    try:
        pd.to_datetime(acct["last_kyc_date"].dropna().head(100))
        ok("accounts: last_kyc_date parses as datetime")
    except Exception as e:
        fail(f"accounts: last_kyc_date parse error: {e}")
    del acct

    # customers.parquet
    cust = pd.read_parquet(files["customers"])
    check_cols(cust, ["customer_id", "date_of_birth", "relationship_start_date",
                       "pan_available", "aadhaar_available", "passport_available",
                       "mobile_banking_flag", "internet_banking_flag", "atm_card_flag",
                       "demat_flag", "credit_card_flag", "fastag_flag",
                       "customer_pin", "permanent_pin"], "customers")
    try:
        pd.to_datetime(cust["date_of_birth"].head(100))
        ok("customers: date_of_birth parses as datetime")
    except Exception as e:
        fail(f"customers: date_of_birth parse error: {e}")
    try:
        pd.to_datetime(cust["relationship_start_date"].head(100))
        ok("customers: relationship_start_date parses as datetime")
    except Exception as e:
        fail(f"customers: relationship_start_date parse error: {e}")
    del cust

    # demographics.parquet
    demo = pd.read_parquet(files["demographics"])
    check_cols(demo, ["customer_id", "gender", "joint_account_flag", "nri_flag",
                       "address_last_update_date", "passbook_last_update_date"], "demographics")
    try:
        pd.to_datetime(demo["address_last_update_date"].dropna().head(100))
        ok("demographics: address_last_update_date parses as datetime")
    except Exception as e:
        fail(f"demographics: address_last_update_date parse error: {e}")
    try:
        pd.to_datetime(demo["passbook_last_update_date"].dropna().head(100))
        ok("demographics: passbook_last_update_date parses as datetime")
    except Exception as e:
        fail(f"demographics: passbook_last_update_date parse error: {e}")
    del demo

    # customer_account_linkage.parquet
    link = pd.read_parquet(files["customer_account_linkage"])
    check_cols(link, ["customer_id", "account_id"], "linkage")
    del link

    # product_details.parquet
    prod = pd.read_parquet(files["product_details"])
    check_cols(prod, ["customer_id", "loan_sum", "loan_count", "cc_sum", "cc_count",
                       "od_sum", "od_count", "ka_sum", "ka_count", "sa_sum", "sa_count"], "products")
    del prod

    # accounts-additional.parquet
    acct_add = pd.read_parquet(files["accounts-additional"])
    check_cols(acct_add, ["account_id", "scheme_code"], "accounts-additional")
    del acct_add

    # branch.parquet
    br = pd.read_parquet(files["branch"])
    check_cols(br, ["branch_code", "branch_employee_count", "branch_turnover",
                     "branch_asset_size", "branch_type", "branch_address",
                     "branch_pin_code", "branch_city", "branch_state"], "branch")
    del br

    # train_labels.parquet
    labels = pd.read_parquet(files["train_labels"])
    check_cols(labels, ["account_id", "is_mule", "alert_reason", "mule_flag_date"], "train_labels")
    try:
        pd.to_datetime(labels["mule_flag_date"].dropna().head(100))
        ok("train_labels: mule_flag_date parses as datetime")
    except Exception as e:
        fail(f"train_labels: mule_flag_date parse error: {e}")

    # Check mule counts
    n_mules = labels["is_mule"].sum()
    n_total = len(labels)
    ok(f"train_labels: {n_mules} mules / {n_total} total (ratio 1:{n_total//max(n_mules,1)})")
    del labels

    # test_accounts.parquet
    test = pd.read_parquet(files["test_accounts"])
    check_cols(test, ["account_id"], "test_accounts")
    ok(f"test_accounts: {len(test)} accounts")
    del test

    # 4. Verify account ID consistency
    print("\n--- Account ID consistency ---")
    acct_ids = set(pd.read_parquet(files["accounts"])["account_id"])
    train_ids = set(pd.read_parquet(files["train_labels"])["account_id"])
    test_ids = set(pd.read_parquet(files["test_accounts"])["account_id"])

    train_in_acct = len(train_ids & acct_ids)
    test_in_acct = len(test_ids & acct_ids)
    ok(f"train IDs in accounts: {train_in_acct}/{len(train_ids)}")
    ok(f"test IDs in accounts: {test_in_acct}/{len(test_ids)}")

    overlap = train_ids & test_ids
    if len(overlap) == 0:
        ok("No overlap between train and test account IDs")
    else:
        fail(f"{len(overlap)} accounts appear in BOTH train and test!")

    # Summary
    print("\n" + "=" * 60)
    print(f"RESULT: {PASS} OK, {FAIL} FAIL")
    print("=" * 60)
    if FAIL > 0:
        print("FIX FAILURES BEFORE RUNNING PIPELINE!")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED — pipeline is safe to run.")


if __name__ == "__main__":
    main()
