"""
NFPC Phase 2 — Comprehensive Data Field Validation

Scans ALL parquet files, checks every column's dtype, nulls, value ranges,
and format consistency. Reports issues before the pipeline runs.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import sys

DATA = Path("/home/DIVYA/nfpc-phase2/data")
TXN_DIR = DATA / "transactions"

PASS = 0
WARN = 0
FAIL = 0


def ok(msg):
    global PASS
    PASS += 1
    print(f"  [OK]   {msg}")


def warn(msg):
    global WARN
    WARN += 1
    print(f"  [WARN] {msg}")


def fail(msg):
    global FAIL
    FAIL += 1
    print(f"  [FAIL] {msg}")


def check_col(df, col, expected_dtype=None, nullable=True, min_val=None, max_val=None):
    if col not in df.columns:
        fail(f"Column '{col}' missing")
        return
    nulls = df[col].isna().sum()
    if nulls > 0 and not nullable:
        fail(f"'{col}' has {nulls} nulls but should not be nullable")
    elif nulls > 0:
        warn(f"'{col}' has {nulls}/{len(df)} nulls ({100*nulls/len(df):.1f}%)")
    else:
        ok(f"'{col}' no nulls")

    if expected_dtype == "datetime":
        try:
            parsed = pd.to_datetime(df[col].dropna(), format="mixed")
            ok(f"'{col}' all parseable as datetime ({len(parsed)} values)")
            # Check for mixed formats
            sample = df[col].dropna().head(20).astype(str)
            has_T = sample.str.contains("T").any()
            has_space = sample.str.contains(" ").any()
            if has_T and has_space:
                warn(f"'{col}' has MIXED datetime formats (both 'T' and space separators)")
        except Exception as e:
            fail(f"'{col}' datetime parse error: {e}")
    elif expected_dtype:
        if not str(df[col].dtype).startswith(expected_dtype):
            warn(f"'{col}' dtype is {df[col].dtype}, expected {expected_dtype}")
        else:
            ok(f"'{col}' dtype={df[col].dtype}")

    if min_val is not None:
        actual_min = df[col].min()
        if actual_min < min_val:
            warn(f"'{col}' min={actual_min}, expected >= {min_val}")
    if max_val is not None:
        actual_max = df[col].max()
        if actual_max > max_val:
            warn(f"'{col}' max={actual_max}, expected <= {max_val}")


def validate_transactions():
    print("\n" + "=" * 60)
    print("TRANSACTIONS (sampling 10 parts evenly spaced)")
    print("=" * 60)
    parts = sorted(glob(str(TXN_DIR / "batch-*" / "part_*.parquet")))
    print(f"Total parts: {len(parts)}")
    if not parts:
        fail("No transaction parts found!")
        return

    # Sample 10 evenly spaced parts
    indices = np.linspace(0, len(parts) - 1, 10, dtype=int)
    sampled = [parts[i] for i in indices]

    all_cols = None
    for p in sampled:
        print(f"\n--- {Path(p).parent.name}/{Path(p).name} ---")
        df = pd.read_parquet(p)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Dtypes:\n{df.dtypes.to_string()}")

        if all_cols is None:
            all_cols = set(df.columns)
        else:
            if set(df.columns) != all_cols:
                fail(f"Column mismatch! Expected {all_cols}, got {set(df.columns)}")

        check_col(df, "transaction_id", nullable=False)
        check_col(df, "account_id", nullable=False)
        check_col(df, "transaction_timestamp", expected_dtype="datetime")
        check_col(df, "amount", expected_dtype="float", min_val=0)
        check_col(df, "transaction_channel")
        check_col(df, "counterparty_id")

        # Check amount distribution
        amt = df["amount"].dropna()
        ok(f"'amount' range: {amt.min():.2f} to {amt.max():.2f}, mean={amt.mean():.2f}")

        del df


def validate_file(name, path, checks):
    print(f"\n{'=' * 60}")
    print(f"{name}: {path}")
    print("=" * 60)
    if not Path(path).exists():
        fail(f"{path} does not exist!")
        return
    df = pd.read_parquet(path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Dtypes:\n{df.dtypes.to_string()}")
    for col, kwargs in checks.items():
        check_col(df, col, **kwargs)
    return df


def main():
    print("NFPC Phase 2 — Data Validation Report")
    print("=" * 60)

    # 1. Train labels
    validate_file("TRAIN LABELS", DATA / "train_labels.parquet", {
        "account_id": {"nullable": False},
        "is_mule": {"nullable": False},
        "alert_reason": {"nullable": True},
        "mule_flag_date": {"expected_dtype": "datetime", "nullable": True},
    })

    # 2. Test accounts
    validate_file("TEST ACCOUNTS", DATA / "test_accounts.parquet", {
        "account_id": {"nullable": False},
    })

    # 3. Accounts
    validate_file("ACCOUNTS", DATA / "accounts.parquet", {
        "account_id": {"nullable": False},
        "account_open_date": {"expected_dtype": "datetime", "nullable": True},
        "account_type": {"nullable": True},
        "account_status": {"nullable": True},
    })

    # 4. Customers
    validate_file("CUSTOMERS", DATA / "customers.parquet", {
        "customer_id": {"nullable": False},
        "customer_since": {"expected_dtype": "datetime", "nullable": True},
        "occupation": {"nullable": True},
        "income_band": {"nullable": True},
    })

    # 5. Customer-Account mapping
    validate_file("CUST-ACCT MAPPING", DATA / "customer_account_mapping.parquet", {
        "customer_id": {"nullable": False},
        "account_id": {"nullable": False},
    })

    # 6. Demographics
    validate_file("DEMOGRAPHICS", DATA / "demographics.parquet", {
        "customer_id": {"nullable": False},
        "age": {"nullable": True, "min_val": 0, "max_val": 120},
        "gender": {"nullable": True},
    })

    # 7. Branch info
    validate_file("BRANCH INFO", DATA / "branch_info.parquet", {
        "branch_id": {"nullable": False},
        "branch_city": {"nullable": True},
        "branch_state": {"nullable": True},
    })

    # 8. Products
    validate_file("PRODUCTS", DATA / "products.parquet", {
        "account_id": {"nullable": False},
    })

    # 9. Alerts
    validate_file("ALERTS", DATA / "alerts.parquet", {
        "alert_id": {"nullable": False},
        "account_id": {"nullable": False},
        "alert_date": {"expected_dtype": "datetime", "nullable": True},
        "alert_type": {"nullable": True},
    })

    # 10. Transactions (sampled)
    validate_transactions()

    # Summary
    print("\n" + "=" * 60)
    print(f"VALIDATION SUMMARY: {PASS} OK, {WARN} WARN, {FAIL} FAIL")
    print("=" * 60)

    if FAIL > 0:
        print("ACTION REQUIRED: Fix FAIL items before running pipeline.")
        sys.exit(1)
    elif WARN > 0:
        print("Pipeline can proceed but review warnings above.")
    else:
        print("All checks passed. Pipeline is safe to run.")


if __name__ == "__main__":
    main()
