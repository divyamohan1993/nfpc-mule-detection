"""
NFPC Phase 2 — Feature Engineering (memory-efficient batch processing)

Processes 400M transactions using RUNNING STATISTICS — never stores raw lists.
Uses reservoir sampling for distributional features (percentiles, Benford, skew).

Three passes:
  1. Transaction features: behavioral stats per account from transactions/
  2. Transaction additional features: geo, IP, balance from transactions_additional/
  3. Static features: account, customer, branch, demographics, products
"""
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import random
import gc

from config import (
    DATA_DIR, TXN_DIR, TXN_ADDITIONAL_DIR, TXN_ID_MAP_PATH,
    TXN_FEATURES_PATH, TXN_ADD_FEATURES_PATH, STATIC_FEATURES_PATH,
    FEATURES_DIR, FULL_FEATURES_PATH,
    CUSTOMERS_PATH, ACCOUNTS_PATH, DEMOGRAPHICS_PATH,
    ACCOUNTS_ADDITIONAL_PATH, BRANCH_PATH, LINKAGE_PATH,
    PRODUCT_DETAILS_PATH, TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH,
    STRUCTURING_THRESHOLDS, ROUND_AMOUNTS, DORMANCY_DAYS,
    ALL_CHANNELS, log, SEED, OUTPUT_DIR,
)

RESERVOIR_SIZE = 500  # samples per account for distributional features


def _shannon_entropy(counts: np.ndarray) -> float:
    """Shannon entropy of a discrete distribution."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _hhi(counts: np.ndarray) -> float:
    """Herfindahl-Hirschman Index (concentration)."""
    total = counts.sum()
    if total == 0:
        return 0.0
    shares = counts / total
    return float(np.sum(shares ** 2))


def _benford_divergence_from_reservoir(samples: list) -> float:
    """KL divergence from Benford's Law using reservoir samples."""
    arr = np.abs(np.array(samples))
    arr = arr[arr > 0]
    if len(arr) < 10:
        return 0.0
    first_digits = (arr / 10 ** np.floor(np.log10(arr))).astype(int)
    first_digits = first_digits[(first_digits >= 1) & (first_digits <= 9)]
    if len(first_digits) < 10:
        return 0.0
    observed = np.bincount(first_digits, minlength=10)[1:].astype(float)
    observed = observed / observed.sum() + 1e-10
    benford = np.log10(1 + 1 / np.arange(1, 10))
    return float(np.sum(observed * np.log(observed / benford)))


def _reservoir_add(reservoir: list, item: float, n_seen: int, size: int = RESERVOIR_SIZE):
    """Reservoir sampling: add item to reservoir, keeping uniform sample of size."""
    if n_seen < size:
        reservoir.append(item)
    else:
        j = random.randint(0, n_seen)
        if j < size:
            reservoir[j] = item


# ═══════════════════════════════════════════════════════════════════════
# PASS 1: Transaction features (memory-efficient)
# ═══════════════════════════════════════════════════════════════════════

def _make_account_accum():
    """Create a fresh account accumulator — only fixed-size fields."""
    return {
        # Running stats for amounts (abs)
        "n_txn": 0, "n_credit": 0, "n_debit": 0,
        "sum_credit": 0.0, "sum_debit": 0.0,
        "sum_abs": 0.0, "sum_abs_sq": 0.0,
        "min_abs": float("inf"), "max_abs": 0.0,
        "cr_sum_abs": 0.0, "cr_sum_abs_sq": 0.0, "cr_max": 0.0,
        "dr_sum_abs": 0.0, "dr_sum_abs_sq": 0.0, "dr_max": 0.0,
        # Reservoir for distributional features
        "reservoir": [],
        # Temporal: histograms (fixed-size arrays)
        "hour_hist": np.zeros(24, dtype=np.int32),
        "dow_hist": np.zeros(7, dtype=np.int32),
        "month_hist": defaultdict(int),  # year-month → count
        # Temporal running stats
        "ts_min": None, "ts_max": None,
        "prev_ts": None,  # for inter-txn time tracking
        "itd_sum": 0.0, "itd_sum_sq": 0.0, "itd_min": float("inf"),
        "itd_max": 0.0, "n_itd": 0,
        # Dormancy
        "dormancy_events": 0,
        # Channel counts (fixed set)
        "channel_counts": defaultdict(int),
        # MCC counts (variable but small — just counts, no amounts)
        "mcc_counts": defaultdict(int),
        # Counterparty counts
        "n_unique_cp": 0,
        "cp_counter": defaultdict(int),
        # Structuring
        "near_threshold": {t: 0 for t in STRUCTURING_THRESHOLDS},
        # Round amounts
        "round_count": 0,
        # Reversals
        "n_negative": 0,
        # Night/business/weekend transaction counts
        "n_night": 0, "n_business": 0, "n_weekend": 0,
    }


def _update_account(acc: dict, amounts: np.ndarray, txn_types: np.ndarray,
                    channels: np.ndarray, mcc_codes: np.ndarray,
                    counterparties: np.ndarray, timestamps: pd.DatetimeIndex):
    """Update account accumulator with a batch of transactions (vectorized)."""
    n = len(amounts)
    abs_amounts = np.abs(amounts)

    # Basic counts
    n_before = acc["n_txn"]
    acc["n_txn"] += n
    credit_mask = txn_types == "C"
    debit_mask = txn_types == "D"
    acc["n_credit"] += int(credit_mask.sum())
    acc["n_debit"] += int(debit_mask.sum())
    acc["sum_credit"] += float(amounts[credit_mask].sum())
    acc["sum_debit"] += float(amounts[debit_mask].sum())

    # Running stats for abs amounts
    acc["sum_abs"] += float(abs_amounts.sum())
    acc["sum_abs_sq"] += float((abs_amounts ** 2).sum())
    batch_min = float(abs_amounts.min()) if n > 0 else float("inf")
    batch_max = float(abs_amounts.max()) if n > 0 else 0.0
    if batch_min < acc["min_abs"]:
        acc["min_abs"] = batch_min
    if batch_max > acc["max_abs"]:
        acc["max_abs"] = batch_max

    # Credit/debit running stats
    cr_abs = abs_amounts[credit_mask]
    if len(cr_abs) > 0:
        acc["cr_sum_abs"] += float(cr_abs.sum())
        acc["cr_sum_abs_sq"] += float((cr_abs ** 2).sum())
        cr_max = float(cr_abs.max())
        if cr_max > acc["cr_max"]:
            acc["cr_max"] = cr_max
    dr_abs = abs_amounts[debit_mask]
    if len(dr_abs) > 0:
        acc["dr_sum_abs"] += float(dr_abs.sum())
        acc["dr_sum_abs_sq"] += float((dr_abs ** 2).sum())
        dr_max = float(dr_abs.max())
        if dr_max > acc["dr_max"]:
            acc["dr_max"] = dr_max

    # Reservoir sampling for distributional features
    reservoir = acc["reservoir"]
    for i in range(n):
        _reservoir_add(reservoir, float(amounts[i]), n_before + i, RESERVOIR_SIZE)

    # Temporal histograms
    hours = timestamps.dt.hour.values
    dows = timestamps.dt.dayofweek.values
    np.add.at(acc["hour_hist"], hours, 1)
    np.add.at(acc["dow_hist"], dows, 1)
    for ym in timestamps.dt.to_period("M"):
        acc["month_hist"][str(ym)] += 1

    # Night/business/weekend counts
    acc["n_night"] += int(((hours >= 0) & (hours < 6)).sum())
    acc["n_business"] += int(((hours >= 9) & (hours < 17)).sum())
    acc["n_weekend"] += int((dows >= 5).sum())

    # Inter-transaction time deltas (need sorted timestamps)
    ts_sorted = timestamps.sort_values().reset_index(drop=True)
    ts_min = ts_sorted.iloc[0]
    ts_max = ts_sorted.iloc[-1]
    if acc["ts_min"] is None or ts_min < acc["ts_min"]:
        acc["ts_min"] = ts_min
    if acc["ts_max"] is None or ts_max > acc["ts_max"]:
        acc["ts_max"] = ts_max

    # Compute inter-txn deltas within this batch
    if len(ts_sorted) > 1:
        deltas = np.diff(ts_sorted.values).astype("timedelta64[s]").astype(np.float64)
        # Also delta from previous batch's last timestamp
        if acc["prev_ts"] is not None:
            first_delta = (ts_sorted.iloc[0] - acc["prev_ts"]).total_seconds()
            if first_delta >= 0:
                deltas = np.concatenate([[first_delta], deltas])

        valid = deltas[deltas >= 0]
        if len(valid) > 0:
            acc["itd_sum"] += float(valid.sum())
            acc["itd_sum_sq"] += float((valid ** 2).sum())
            v_min = float(valid.min())
            v_max = float(valid.max())
            if v_min < acc["itd_min"]:
                acc["itd_min"] = v_min
            if v_max > acc["itd_max"]:
                acc["itd_max"] = v_max
            acc["n_itd"] += len(valid)

            # Dormancy events
            acc["dormancy_events"] += int(np.sum(valid > DORMANCY_DAYS * 86400))

    acc["prev_ts"] = ts_sorted.iloc[-1] if len(ts_sorted) > 0 else acc["prev_ts"]

    # Channel counts (vectorized value_counts)
    for ch, cnt in zip(*np.unique(channels, return_counts=True)):
        acc["channel_counts"][ch] += int(cnt)

    # MCC counts
    for mcc, cnt in zip(*np.unique(mcc_codes, return_counts=True)):
        acc["mcc_counts"][mcc] += int(cnt)

    # Counterparty counts
    for cp in counterparties:
        if pd.notna(cp):
            acc["cp_counter"][cp] += 1

    # Structuring
    for threshold in STRUCTURING_THRESHOLDS:
        lower = threshold * 0.85
        upper = threshold * 1.0
        acc["near_threshold"][threshold] += int(
            ((abs_amounts >= lower) & (abs_amounts < upper)).sum()
        )

    # Round amounts
    round_set = set(ROUND_AMOUNTS)
    for amt in abs_amounts:
        if amt > 0 and int(amt) in round_set:
            acc["round_count"] += 1

    # Reversals
    acc["n_negative"] += int((amounts < 0).sum())


def _finalize_account(acct_id: str, acc: dict) -> dict:
    """Convert accumulator to feature dict."""
    row = {"account_id": acct_id}
    n = acc["n_txn"]
    total_txn = max(n, 1)

    # ── Basic volume ──────────────────────────────────
    row["n_txn"] = n
    row["n_credit"] = acc["n_credit"]
    row["n_debit"] = acc["n_debit"]
    row["credit_debit_ratio"] = acc["n_credit"] / max(acc["n_debit"], 1)
    row["sum_credit"] = acc["sum_credit"]
    row["sum_debit"] = acc["sum_debit"]
    row["net_flow"] = acc["sum_credit"] - acc["sum_debit"]
    row["credit_debit_amount_ratio"] = acc["sum_credit"] / max(abs(acc["sum_debit"]), 1)

    # ── Amount statistics (from running stats) ────────
    if n > 0:
        mean_abs = acc["sum_abs"] / n
        var_abs = max(acc["sum_abs_sq"] / n - mean_abs ** 2, 0)
        std_abs = var_abs ** 0.5
        row["amt_mean"] = mean_abs
        row["amt_std"] = std_abs
        row["amt_max"] = acc["max_abs"]
        row["amt_min"] = acc["min_abs"] if acc["min_abs"] != float("inf") else 0
        row["amt_cv"] = std_abs / max(mean_abs, 1e-6)

        # From reservoir: percentiles, skew, kurtosis, Benford
        reservoir = acc["reservoir"]
        if len(reservoir) >= 5:
            r_abs = np.abs(np.array(reservoir))
            row["amt_median"] = float(np.median(r_abs))
            row["amt_p25"] = float(np.percentile(r_abs, 25))
            row["amt_p75"] = float(np.percentile(r_abs, 75))
            row["amt_p95"] = float(np.percentile(r_abs, 95))
            row["amt_p99"] = float(np.percentile(r_abs, 99))
            row["amt_iqr"] = row["amt_p75"] - row["amt_p25"]
            from scipy.stats import skew, kurtosis
            row["amt_skew"] = float(skew(r_abs))
            row["amt_kurtosis"] = float(kurtosis(r_abs))
            row["benford_divergence"] = _benford_divergence_from_reservoir(reservoir)
        else:
            for k in ["amt_median", "amt_p25", "amt_p75", "amt_p95", "amt_p99",
                      "amt_iqr", "amt_skew", "amt_kurtosis", "benford_divergence"]:
                row[k] = 0.0
    else:
        for k in ["amt_mean", "amt_std", "amt_max", "amt_min", "amt_cv",
                  "amt_median", "amt_p25", "amt_p75", "amt_p95", "amt_p99",
                  "amt_iqr", "amt_skew", "amt_kurtosis", "benford_divergence"]:
            row[k] = 0.0

    # ── Credit/debit amount stats ─────────────────────
    for prefix, cnt, sum_abs, sum_sq, mx in [
        ("cr_amt", acc["n_credit"], acc["cr_sum_abs"], acc["cr_sum_abs_sq"], acc["cr_max"]),
        ("dr_amt", acc["n_debit"], acc["dr_sum_abs"], acc["dr_sum_abs_sq"], acc["dr_max"]),
    ]:
        if cnt > 0:
            m = sum_abs / cnt
            v = max(sum_sq / cnt - m ** 2, 0)
            row[f"{prefix}_mean"] = m
            row[f"{prefix}_std"] = v ** 0.5
            row[f"{prefix}_max"] = mx
        else:
            row[f"{prefix}_mean"] = 0.0
            row[f"{prefix}_std"] = 0.0
            row[f"{prefix}_max"] = 0.0

    # ── Temporal features ─────────────────────────────
    if acc["n_itd"] > 0 and acc["ts_min"] is not None:
        span = (acc["ts_max"] - acc["ts_min"]).total_seconds()
        row["txn_span_days"] = span / 86400

        itd_mean = acc["itd_sum"] / acc["n_itd"]
        itd_var = max(acc["itd_sum_sq"] / acc["n_itd"] - itd_mean ** 2, 0)
        itd_std = itd_var ** 0.5
        row["inter_txn_mean_sec"] = itd_mean
        row["inter_txn_std_sec"] = itd_std
        row["inter_txn_min_sec"] = acc["itd_min"] if acc["itd_min"] != float("inf") else 0
        row["inter_txn_max_sec"] = acc["itd_max"]
        row["inter_txn_cv"] = itd_std / max(itd_mean, 1e-6)

        # Burstiness (Goh-Barabási)
        row["burstiness"] = (itd_std - itd_mean) / (itd_std + itd_mean) if (itd_std + itd_mean) > 0 else 0

        # Dormancy
        row["max_gap_days"] = acc["itd_max"] / 86400
        row["dormancy_events"] = acc["dormancy_events"]

        # Txns per day
        row["txn_per_day"] = n / max(row["txn_span_days"], 1)
    else:
        for k in ["txn_span_days", "inter_txn_mean_sec", "inter_txn_std_sec",
                  "inter_txn_min_sec", "inter_txn_max_sec", "inter_txn_cv",
                  "burstiness", "max_gap_days", "dormancy_events", "txn_per_day"]:
            row[k] = 0.0

    # Hour/DOW/month features (from histograms)
    row["pct_night_txn"] = acc["n_night"] / total_txn
    row["pct_business_hours"] = acc["n_business"] / total_txn
    row["pct_weekend_txn"] = acc["n_weekend"] / total_txn
    row["hour_entropy"] = _shannon_entropy(acc["hour_hist"].astype(float))
    row["dow_entropy"] = _shannon_entropy(acc["dow_hist"].astype(float))

    # Monthly activity consistency
    month_counts = np.array(list(acc["month_hist"].values()), dtype=float)
    if len(month_counts) > 0:
        row["monthly_txn_std"] = float(np.std(month_counts))
        row["monthly_txn_cv"] = float(np.std(month_counts) / max(np.mean(month_counts), 1e-6))
        row["n_active_months"] = len(month_counts)
    else:
        row["monthly_txn_std"] = 0.0
        row["monthly_txn_cv"] = 0.0
        row["n_active_months"] = 0

    # ── Channel diversity ─────────────────────────────
    ch_counts = np.array(
        [acc["channel_counts"].get(c, 0) for c in ALL_CHANNELS], dtype=float
    )
    row["n_channels_used"] = int(np.sum(ch_counts > 0))
    row["channel_entropy"] = _shannon_entropy(ch_counts)
    row["channel_hhi"] = _hhi(ch_counts)
    row["top_channel_share"] = float(ch_counts.max() / max(ch_counts.sum(), 1))
    for ch in ["UPC", "UPD", "ATW", "CHQ", "CSD", "NTD", "IPM", "END"]:
        row[f"pct_{ch}"] = acc["channel_counts"].get(ch, 0) / total_txn

    # ── MCC diversity ─────────────────────────────────
    row["n_unique_mcc"] = len(acc["mcc_counts"])
    mcc_cnts = np.array(list(acc["mcc_counts"].values()), dtype=float)
    row["mcc_entropy"] = _shannon_entropy(mcc_cnts) if len(mcc_cnts) > 0 else 0
    row["mcc_hhi"] = _hhi(mcc_cnts) if len(mcc_cnts) > 0 else 0

    # ── Counterparty concentration ────────────────────
    cp_cnts = np.array(list(acc["cp_counter"].values()), dtype=float)
    row["n_unique_counterparties"] = len(acc["cp_counter"])
    row["cp_entropy"] = _shannon_entropy(cp_cnts) if len(cp_cnts) > 0 else 0
    row["cp_hhi"] = _hhi(cp_cnts) if len(cp_cnts) > 0 else 0
    if len(cp_cnts) > 0:
        row["top_cp_share"] = float(cp_cnts.max() / cp_cnts.sum())
        row["top3_cp_share"] = float(np.sort(cp_cnts)[-3:].sum() / cp_cnts.sum())
    else:
        row["top_cp_share"] = 0.0
        row["top3_cp_share"] = 0.0

    # ── Structuring ───────────────────────────────────
    for threshold in STRUCTURING_THRESHOLDS:
        row[f"near_{threshold}_count"] = acc["near_threshold"][threshold]
        row[f"near_{threshold}_ratio"] = acc["near_threshold"][threshold] / total_txn

    # ── Round amounts ─────────────────────────────────
    row["round_amount_count"] = acc["round_count"]
    row["round_amount_ratio"] = acc["round_count"] / total_txn

    # ── Reversals ─────────────────────────────────────
    row["n_reversals"] = acc["n_negative"]
    row["reversal_ratio"] = acc["n_negative"] / total_txn

    return row


def build_txn_features() -> pd.DataFrame:
    """Pass 1: Process all transaction parts with running statistics."""
    if TXN_FEATURES_PATH.exists():
        log.info("Loading cached transaction features from %s", TXN_FEATURES_PATH)
        return pd.read_parquet(TXN_FEATURES_PATH)

    parts = sorted(glob(str(TXN_DIR / "batch-*" / "part_*.parquet")))
    log.info("Processing %d transaction parts", len(parts))

    random.seed(SEED)
    accounts = {}  # acct_id → accumulator

    # Process in batches, also save txn_id mapping in chunks
    map_chunks = []
    MAP_CHUNK_SIZE = 50  # save mapping every N parts

    for i, part_path in enumerate(tqdm(parts, desc="Pass 1: Transactions")):
        df = pd.read_parquet(part_path)
        df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], format="mixed")

        # Collect txn_id → acct_id mapping
        map_chunks.append(df[["transaction_id", "account_id"]].copy())

        # Save mapping periodically to free memory
        if len(map_chunks) >= MAP_CHUNK_SIZE:
            chunk_df = pd.concat(map_chunks, ignore_index=True)
            chunk_path = OUTPUT_DIR / f"txn_map_chunk_{i}.parquet"
            chunk_df.to_parquet(chunk_path, index=False)
            del chunk_df
            map_chunks = []
            gc.collect()

        # Update per-account accumulators
        for acct_id, grp in df.groupby("account_id"):
            if acct_id not in accounts:
                accounts[acct_id] = _make_account_accum()
            _update_account(
                accounts[acct_id],
                grp["amount"].values,
                grp["txn_type"].values,
                grp["channel"].values,
                grp["mcc_code"].values,
                grp["counterparty_id"].values,
                grp["transaction_timestamp"],
            )
        del df

        # Periodic memory check
        if (i + 1) % 100 == 0:
            gc.collect()
            log.info("  Processed %d/%d parts, %d accounts tracked", i + 1, len(parts), len(accounts))

    # Save remaining mapping chunks
    if map_chunks:
        chunk_df = pd.concat(map_chunks, ignore_index=True)
        chunk_path = OUTPUT_DIR / f"txn_map_chunk_final.parquet"
        chunk_df.to_parquet(chunk_path, index=False)
        del chunk_df
        del map_chunks
        gc.collect()

    # Combine all mapping chunks into single file
    if not TXN_ID_MAP_PATH.exists():
        log.info("Combining transaction ID mapping chunks")
        map_files = sorted(glob(str(OUTPUT_DIR / "txn_map_chunk_*.parquet")))
        txn_map = pd.concat([pd.read_parquet(f) for f in map_files], ignore_index=True)
        txn_map.to_parquet(TXN_ID_MAP_PATH, index=False)
        log.info("Saved mapping: %d transaction IDs", len(txn_map))
        del txn_map
        # Clean up chunk files
        import os
        for f in map_files:
            os.remove(f)
        gc.collect()

    # Finalize features
    log.info("Finalizing transaction features for %d accounts", len(accounts))
    rows = []
    for acct_id, acc in tqdm(accounts.items(), desc="Finalizing features"):
        rows.append(_finalize_account(acct_id, acc))
    del accounts
    gc.collect()

    features = pd.DataFrame(rows).set_index("account_id")
    features.to_parquet(TXN_FEATURES_PATH)
    log.info("Saved transaction features: %s", features.shape)
    return features


# ═══════════════════════════════════════════════════════════════════════
# PASS 2: Transactions additional features (memory-efficient)
# ═══════════════════════════════════════════════════════════════════════

def _make_additional_accum():
    """Create a fresh accumulator for additional transaction features."""
    return {
        # Geo running stats
        "n_geo": 0,
        "lat_sum": 0.0, "lat_sum_sq": 0.0, "lat_min": float("inf"), "lat_max": -float("inf"),
        "lon_sum": 0.0, "lon_sum_sq": 0.0, "lon_min": float("inf"), "lon_max": -float("inf"),
        "geo_cells": set(),  # 0.1-degree cells for cluster count
        # IP tracking
        "ip_set": set(),
        "subnet_set": set(),
        # Balance running stats
        "n_bal": 0,
        "bal_sum": 0.0, "bal_sum_sq": 0.0,
        "bal_min": float("inf"), "bal_max": -float("inf"),
        "n_near_zero_bal": 0,
        # Balance diffs running stats
        "prev_bal": None,
        "bal_diff_sum_abs": 0.0, "n_bal_diff": 0,
        "max_neg_diff": 0.0, "neg_diff_sum": 0.0, "n_neg_diff": 0,
        # Part transaction type counts
        "pt_counts": defaultdict(int),
        # Sub type counts
        "st_counts": defaultdict(int),
        # ATM deposits
        "n_atm_deposits": 0,
        # Total txn count (for ratios)
        "n_total": 0,
    }


def _update_additional(acc: dict, grp: pd.DataFrame):
    """Update additional accumulator with a batch."""
    acc["n_total"] += len(grp)

    # Geo
    valid_geo = grp.dropna(subset=["latitude", "longitude"])
    if len(valid_geo) > 0:
        lats = valid_geo["latitude"].values
        lons = valid_geo["longitude"].values
        n = len(lats)
        acc["n_geo"] += n
        acc["lat_sum"] += float(lats.sum())
        acc["lat_sum_sq"] += float((lats ** 2).sum())
        acc["lat_min"] = min(acc["lat_min"], float(lats.min()))
        acc["lat_max"] = max(acc["lat_max"], float(lats.max()))
        acc["lon_sum"] += float(lons.sum())
        acc["lon_sum_sq"] += float((lons ** 2).sum())
        acc["lon_min"] = min(acc["lon_min"], float(lons.min()))
        acc["lon_max"] = max(acc["lon_max"], float(lons.max()))
        for lat, lon in zip(np.round(lats, 1), np.round(lons, 1)):
            acc["geo_cells"].add((lat, lon))

    # IPs
    valid_ips = grp["ip_address"].dropna()
    for ip in valid_ips:
        acc["ip_set"].add(ip)
        parts = str(ip).split(".")
        if len(parts) >= 3:
            acc["subnet_set"].add(".".join(parts[:3]))

    # Balance
    valid_bal = grp["balance_after_transaction"].dropna()
    if len(valid_bal) > 0:
        bals = valid_bal.values
        n = len(bals)
        acc["n_bal"] += n
        acc["bal_sum"] += float(bals.sum())
        acc["bal_sum_sq"] += float((bals ** 2).sum())
        acc["bal_min"] = min(acc["bal_min"], float(bals.min()))
        acc["bal_max"] = max(acc["bal_max"], float(bals.max()))
        acc["n_near_zero_bal"] += int(np.sum(np.abs(bals) < 100))

        # Balance diffs
        for b in bals:
            if acc["prev_bal"] is not None:
                d = b - acc["prev_bal"]
                acc["bal_diff_sum_abs"] += abs(d)
                acc["n_bal_diff"] += 1
                if d < 0:
                    acc["neg_diff_sum"] += d
                    acc["n_neg_diff"] += 1
                    if d < acc["max_neg_diff"]:
                        acc["max_neg_diff"] = d
            acc["prev_bal"] = b

    # Part transaction type
    for pt in grp["part_transaction_type"].dropna():
        acc["pt_counts"][pt] += 1

    # Sub type
    for st in grp["transaction_sub_type"].dropna():
        acc["st_counts"][st] += 1

    # ATM deposits
    acc["n_atm_deposits"] += int(grp["atm_deposit_channel_code"].notna().sum())


def _finalize_additional(acct_id: str, acc: dict) -> dict:
    """Convert additional accumulator to feature dict."""
    row = {"account_id": acct_id}
    n_total = max(acc["n_total"], 1)

    # ── Geographic ────────────────────────────────────
    n_geo = acc["n_geo"]
    row["n_geo_txns"] = n_geo
    row["geo_coverage_ratio"] = n_geo / n_total

    if n_geo >= 2:
        lat_mean = acc["lat_sum"] / n_geo
        lat_var = max(acc["lat_sum_sq"] / n_geo - lat_mean ** 2, 0)
        lon_mean = acc["lon_sum"] / n_geo
        lon_var = max(acc["lon_sum_sq"] / n_geo - lon_mean ** 2, 0)
        row["lat_std"] = lat_var ** 0.5
        row["lon_std"] = lon_var ** 0.5
        row["geo_spread"] = (lat_var + lon_var) ** 0.5
        lat_range = acc["lat_max"] - acc["lat_min"]
        lon_range = acc["lon_max"] - acc["lon_min"]
        row["geo_range_km"] = ((lat_range * 111) ** 2 + (lon_range * 85) ** 2) ** 0.5
        row["n_geo_clusters"] = len(acc["geo_cells"])
    else:
        for k in ["lat_std", "lon_std", "geo_spread", "geo_range_km"]:
            row[k] = 0.0
        row["n_geo_clusters"] = min(n_geo, 1)

    # ── IP ────────────────────────────────────────────
    row["n_unique_ips"] = len(acc["ip_set"])
    row["n_unique_subnets"] = len(acc["subnet_set"])
    row["ip_per_txn"] = len(acc["ip_set"]) / n_total

    # ── Balance trajectory ────────────────────────────
    n_bal = acc["n_bal"]
    if n_bal > 1:
        bal_mean = acc["bal_sum"] / n_bal
        bal_var = max(acc["bal_sum_sq"] / n_bal - bal_mean ** 2, 0)
        row["bal_mean"] = bal_mean
        row["bal_std"] = bal_var ** 0.5
        row["bal_min"] = acc["bal_min"] if acc["bal_min"] != float("inf") else 0
        row["bal_max"] = acc["bal_max"] if acc["bal_max"] != -float("inf") else 0
        row["bal_cv"] = (bal_var ** 0.5) / max(abs(bal_mean), 1e-6)
        row["bal_range"] = row["bal_max"] - row["bal_min"]
        row["n_near_zero_balance"] = acc["n_near_zero_bal"]
        row["pct_near_zero_balance"] = acc["n_near_zero_bal"] / n_bal
        row["max_drawdown"] = acc["max_neg_diff"]
        row["mean_drawdown"] = acc["neg_diff_sum"] / max(acc["n_neg_diff"], 1)
        row["bal_volatility"] = (
            acc["bal_diff_sum_abs"] / max(acc["n_bal_diff"], 1)
        ) / max(abs(bal_mean), 1e-6)
    else:
        for k in ["bal_mean", "bal_std", "bal_min", "bal_max", "bal_cv",
                  "bal_range", "n_near_zero_balance", "pct_near_zero_balance",
                  "max_drawdown", "mean_drawdown", "bal_volatility"]:
            row[k] = 0.0

    # ── Part transaction type ─────────────────────────
    total_pt = sum(acc["pt_counts"].values()) or 1
    for pt in ["CI", "BI", "IP", "IC"]:
        row[f"pct_pt_{pt}"] = acc["pt_counts"].get(pt, 0) / total_pt

    # ── Sub type ──────────────────────────────────────
    total_st = sum(acc["st_counts"].values()) or 1
    for st in ["normal", "cash", "loan"]:
        row[f"pct_st_{st}"] = acc["st_counts"].get(st, 0) / total_st
    row["n_cash_txns"] = acc["st_counts"].get("cash", 0)

    # ── ATM deposits ──────────────────────────────────
    row["n_atm_deposits"] = acc["n_atm_deposits"]

    return row


def build_txn_additional_features() -> pd.DataFrame:
    """Pass 2: Process transactions_additional with account ID lookup.

    Memory-efficient: reads txn_id mapping from disk per-batch (never holds
    the full 397M-row table in memory). Trades ~12s disk I/O per batch for
    ~20GB memory savings.
    """
    if TXN_ADD_FEATURES_PATH.exists():
        log.info("Loading cached additional features from %s", TXN_ADD_FEATURES_PATH)
        return pd.read_parquet(TXN_ADD_FEATURES_PATH)

    import pyarrow.parquet as pq

    parts = sorted(glob(str(TXN_ADDITIONAL_DIR / "batch-*" / "part_*.parquet")))
    log.info("Processing %d transactions_additional parts", len(parts))

    # Get mapping metadata (row count) without loading data
    map_meta = pq.read_metadata(TXN_ID_MAP_PATH)
    total_map_rows = map_meta.num_rows
    log.info("Mapping has %d rows (will read from disk per-batch)", total_map_rows)

    MAP_CHUNK_SIZE = 50_000_000  # 50M rows per disk read chunk
    accounts = {}

    BATCH_SIZE = 60  # Process 60 additional parts at a time (larger batch = fewer disk scans)
    for batch_start in range(0, len(parts), BATCH_SIZE):
        batch_parts = parts[batch_start:batch_start + BATCH_SIZE]
        log.info("  Processing additional parts %d-%d/%d",
                 batch_start + 1, min(batch_start + BATCH_SIZE, len(parts)), len(parts))

        # Read all txn_ids from this batch of additional parts
        batch_dfs = []
        batch_txn_ids = set()
        for part_path in batch_parts:
            df = pd.read_parquet(part_path)
            batch_dfs.append(df)
            batch_txn_ids.update(df["transaction_id"].values)

        # Read mapping FROM DISK in chunks, keeping only matching txn_ids
        # This re-reads the file per batch but never holds the full table in RAM
        partial_lookup = {}
        map_file = pq.ParquetFile(TXN_ID_MAP_PATH)
        for batch_chunk in map_file.iter_batches(
            batch_size=MAP_CHUNK_SIZE,
            columns=["transaction_id", "account_id"],
        ):
            chunk = batch_chunk.to_pandas()
            mask = chunk["transaction_id"].isin(batch_txn_ids)
            matched = chunk[mask]
            for tid, aid in zip(matched["transaction_id"].values, matched["account_id"].values):
                partial_lookup[tid] = aid
            del chunk, matched, batch_chunk
            # Early exit if all txn_ids found
            if len(partial_lookup) >= len(batch_txn_ids):
                break
        del map_file

        log.info("    Matched %d/%d txn_ids in mapping", len(partial_lookup), len(batch_txn_ids))
        del batch_txn_ids

        # Process each additional part with the partial lookup
        for df in batch_dfs:
            df["account_id"] = df["transaction_id"].map(partial_lookup)
            df = df.dropna(subset=["account_id"])
            for acct_id, grp in df.groupby("account_id"):
                if acct_id not in accounts:
                    accounts[acct_id] = _make_additional_accum()
                _update_additional(accounts[acct_id], grp)

        del batch_dfs, partial_lookup
        gc.collect()

    log.info("Finalizing additional features for %d accounts", len(accounts))
    rows = []
    for acct_id, acc in tqdm(accounts.items(), desc="Finalizing additional"):
        rows.append(_finalize_additional(acct_id, acc))
    del accounts
    gc.collect()

    features = pd.DataFrame(rows).set_index("account_id")
    features.to_parquet(TXN_ADD_FEATURES_PATH)
    log.info("Saved additional features: %s", features.shape)
    return features


# ═══════════════════════════════════════════════════════════════════════
# PASS 3: Static features (unchanged — small data, fits in memory)
# ═══════════════════════════════════════════════════════════════════════

def build_static_features() -> pd.DataFrame:
    """Build features from static tables."""
    if STATIC_FEATURES_PATH.exists():
        log.info("Loading cached static features from %s", STATIC_FEATURES_PATH)
        return pd.read_parquet(STATIC_FEATURES_PATH)

    log.info("Building static features")

    accounts = pd.read_parquet(ACCOUNTS_PATH)
    customers = pd.read_parquet(CUSTOMERS_PATH)
    demographics = pd.read_parquet(DEMOGRAPHICS_PATH)
    accounts_add = pd.read_parquet(ACCOUNTS_ADDITIONAL_PATH)
    branch = pd.read_parquet(BRANCH_PATH)
    linkage = pd.read_parquet(LINKAGE_PATH)
    products = pd.read_parquet(PRODUCT_DETAILS_PATH)

    df = accounts.set_index("account_id").copy()
    ref_date = pd.Timestamp("2025-06-30")

    df["account_opening_date"] = pd.to_datetime(df["account_opening_date"])
    df["account_age_days"] = (ref_date - df["account_opening_date"]).dt.days

    df["balance_ratio_monthly_avg"] = df["avg_balance"] / df["monthly_avg_balance"].clip(lower=1)
    df["balance_ratio_quarterly_avg"] = df["avg_balance"] / df["quarterly_avg_balance"].clip(lower=1)
    df["balance_ratio_daily_avg"] = df["avg_balance"] / df["daily_avg_balance"].clip(lower=1)
    df["balance_consistency"] = df[["monthly_avg_balance", "quarterly_avg_balance", "daily_avg_balance"]].std(axis=1)

    df["was_frozen"] = df["freeze_date"].notna().astype(int)
    df["was_unfrozen"] = df["unfreeze_date"].notna().astype(int)
    df["freeze_date"] = pd.to_datetime(df["freeze_date"])
    df["unfreeze_date"] = pd.to_datetime(df["unfreeze_date"])
    df["freeze_duration_days"] = (df["unfreeze_date"] - df["freeze_date"]).dt.days.fillna(0)

    df["last_mobile_update_date"] = pd.to_datetime(df["last_mobile_update_date"])
    df["days_since_mobile_update"] = (ref_date - df["last_mobile_update_date"]).dt.days
    df["mobile_update_after_opening"] = (df["last_mobile_update_date"] - df["account_opening_date"]).dt.days
    df["has_mobile_update"] = df["last_mobile_update_date"].notna().astype(int)

    df["last_kyc_date"] = pd.to_datetime(df["last_kyc_date"])
    df["days_since_kyc"] = (ref_date - df["last_kyc_date"]).dt.days

    df["account_status_frozen"] = (df["account_status"] == "frozen").astype(int)
    df["product_family_S"] = (df["product_family"] == "S").astype(int)
    df["product_family_K"] = (df["product_family"] == "K").astype(int)
    df["product_family_O"] = (df["product_family"] == "O").astype(int)
    for col in ["nomination_flag", "cheque_allowed", "cheque_availed", "kyc_compliant", "rural_branch"]:
        df[f"{col}_Y"] = (df[col] == "Y").astype(int)

    drop_cols = [
        "account_status", "product_code", "currency_code", "account_opening_date",
        "branch_code", "branch_pin", "product_family", "nomination_flag",
        "cheque_allowed", "cheque_availed", "kyc_compliant", "rural_branch",
        "last_mobile_update_date", "last_kyc_date", "freeze_date", "unfreeze_date",
    ]
    acct_features = df.drop(columns=drop_cols, errors="ignore")

    linkage_indexed = linkage.set_index("account_id")
    acct_features = acct_features.join(linkage_indexed["customer_id"])

    cust = customers.set_index("customer_id").copy()
    cust["date_of_birth"] = pd.to_datetime(cust["date_of_birth"])
    cust["relationship_start_date"] = pd.to_datetime(cust["relationship_start_date"])
    cust["age_years"] = (ref_date - cust["date_of_birth"]).dt.days / 365.25
    cust["relationship_years"] = (ref_date - cust["relationship_start_date"]).dt.days / 365.25
    for col in ["pan_available", "aadhaar_available", "passport_available",
                "mobile_banking_flag", "internet_banking_flag", "atm_card_flag",
                "demat_flag", "credit_card_flag", "fastag_flag"]:
        cust[f"{col}_Y"] = (cust[col] == "Y").astype(int)
    cust["n_kyc_docs"] = cust["pan_available_Y"] + cust["aadhaar_available_Y"] + cust["passport_available_Y"]
    cust["digital_score"] = (
        cust["mobile_banking_flag_Y"] + cust["internet_banking_flag_Y"]
        + cust["atm_card_flag_Y"] + cust["demat_flag_Y"]
        + cust["credit_card_flag_Y"] + cust["fastag_flag_Y"]
    )
    cust["pin_mismatch"] = (cust["customer_pin"] != cust["permanent_pin"]).astype(int)
    cust_drop = [
        "date_of_birth", "relationship_start_date",
        "pan_available", "aadhaar_available", "passport_available",
        "mobile_banking_flag", "internet_banking_flag", "atm_card_flag",
        "demat_flag", "credit_card_flag", "fastag_flag",
        "customer_pin", "permanent_pin",
    ]
    cust = cust.drop(columns=cust_drop, errors="ignore")
    acct_features = acct_features.join(cust, on="customer_id")

    demo = demographics.set_index("customer_id").copy()
    demo["gender_M"] = (demo["gender"] == "M").astype(int)
    demo["joint_account_Y"] = (demo["joint_account_flag"] == "Y").astype(int)
    demo["nri_Y"] = (demo["nri_flag"] == "Y").astype(int)
    demo["address_last_update_date"] = pd.to_datetime(demo["address_last_update_date"])
    demo["days_since_address_update"] = (ref_date - demo["address_last_update_date"]).dt.days
    demo["passbook_last_update_date"] = pd.to_datetime(demo["passbook_last_update_date"])
    demo["days_since_passbook_update"] = (ref_date - demo["passbook_last_update_date"]).dt.days
    demo_keep = ["gender_M", "joint_account_Y", "nri_Y", "days_since_address_update", "days_since_passbook_update"]
    acct_features = acct_features.join(demo[demo_keep], on="customer_id")

    prod = products.set_index("customer_id").fillna(0)
    prod["total_product_count"] = prod["loan_count"] + prod["cc_count"] + prod["od_count"] + prod["ka_count"] + prod["sa_count"]
    prod["total_outstanding"] = prod["loan_sum"] + prod["cc_sum"] + prod["od_sum"]
    acct_features = acct_features.join(prod, on="customer_id")

    scheme = accounts_add.set_index("account_id")
    scheme_dummies = pd.get_dummies(scheme["scheme_code"], prefix="scheme")
    acct_features = acct_features.join(scheme_dummies)

    acct_branch = accounts[["account_id", "branch_code"]].set_index("account_id")
    acct_features = acct_features.join(acct_branch)

    branch_feat = branch.set_index("branch_code").copy()
    branch_type_dummies = pd.get_dummies(branch_feat["branch_type"], prefix="branch_type")
    branch_feat = branch_feat.drop(columns=["branch_address", "branch_pin_code", "branch_city", "branch_state", "branch_type"], errors="ignore")
    branch_feat = branch_feat.join(branch_type_dummies)
    acct_features = acct_features.join(branch_feat, on="branch_code")

    train_labels = pd.read_parquet(TRAIN_LABELS_PATH)
    acct_branch_full = accounts[["account_id", "branch_code"]]
    train_with_branch = train_labels.merge(acct_branch_full, on="account_id")
    branch_mule_stats = train_with_branch.groupby("branch_code").agg(
        branch_n_accounts=("account_id", "count"),
        branch_n_mules=("is_mule", "sum"),
    )
    branch_mule_stats["branch_mule_rate"] = branch_mule_stats["branch_n_mules"] / branch_mule_stats["branch_n_accounts"]
    acct_features = acct_features.join(branch_mule_stats, on="branch_code")

    acct_features = acct_features.drop(columns=["customer_id", "branch_code"], errors="ignore")
    acct_features = acct_features.fillna(0)

    acct_features.to_parquet(STATIC_FEATURES_PATH)
    log.info("Saved static features: %s", acct_features.shape)
    return acct_features


# ═══════════════════════════════════════════════════════════════════════
# COMBINE ALL FEATURES
# ═══════════════════════════════════════════════════════════════════════

def build_all_features() -> pd.DataFrame:
    """Combine all feature sources into a single DataFrame."""
    if FULL_FEATURES_PATH.exists():
        log.info("Loading cached full features from %s", FULL_FEATURES_PATH)
        return pd.read_parquet(FULL_FEATURES_PATH)

    txn_feat = build_txn_features()
    txn_add_feat = build_txn_additional_features()
    static_feat = build_static_features()

    from config import GRAPH_FEATURES_PATH, TEMPORAL_FEATURES_PATH
    graph_feat = pd.read_parquet(GRAPH_FEATURES_PATH) if GRAPH_FEATURES_PATH.exists() else None
    temp_feat = pd.read_parquet(TEMPORAL_FEATURES_PATH) if TEMPORAL_FEATURES_PATH.exists() else None

    full = static_feat.join(txn_feat, how="left")
    full = full.join(txn_add_feat, how="left")
    if graph_feat is not None:
        full = full.join(graph_feat, how="left")
    if temp_feat is not None:
        full = full.join(temp_feat, how="left")

    full = full.fillna(0)
    full.to_parquet(FULL_FEATURES_PATH)
    log.info("Saved full features: %s", full.shape)
    return full


if __name__ == "__main__":
    build_all_features()
