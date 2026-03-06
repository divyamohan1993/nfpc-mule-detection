"""
NFPC Phase 2 — Feature Engineering (V2)

Four passes over data:
  Pass 1: Transaction features (running stats + MCC population stats + behavioral split)
  Pass 2: Transaction additional features (geo, IP, balance)
  Pass 3: Static features (account, customer, branch — NO label leakage)
  Pass 4: Graph features (PageRank, HITS, community detection, betweenness)

Key V2 changes vs V1:
  - REMOVED branch_mule_rate/branch_n_mules (was label leakage)
  - ADDED MCC-amount anomaly features (pattern #13)
  - ADDED behavioral change features (before/after midpoint)
  - ADDED fan-in/fan-out counterparty features
  - ADDED graph features (network centrality, community)
  - branch_code preserved for target encoding in models.py (inside CV folds)
"""
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from tqdm import tqdm
import random
import gc
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import skew, kurtosis

from config import (
    DATA_DIR, TXN_DIR, TXN_ADDITIONAL_DIR, TXN_ID_MAP_PATH,
    TXN_FEATURES_PATH, TXN_ADD_FEATURES_PATH, STATIC_FEATURES_PATH,
    GRAPH_FEATURES_PATH, FEATURES_DIR, FULL_FEATURES_PATH,
    CUSTOMERS_PATH, ACCOUNTS_PATH, DEMOGRAPHICS_PATH,
    ACCOUNTS_ADDITIONAL_PATH, BRANCH_PATH, LINKAGE_PATH,
    PRODUCT_DETAILS_PATH, TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH,
    STRUCTURING_THRESHOLDS, ROUND_AMOUNTS, DORMANCY_DAYS,
    ALL_CHANNELS, BEHAVIORAL_SPLIT_DATE, RESERVOIR_SIZE,
    log, SEED, OUTPUT_DIR, N_THREADS,
)

PREFETCH_DEPTH = 4


# ═══════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _prefetch_parquets(paths, parse_ts=False):
    def _read(p):
        df = pd.read_parquet(p)
        if parse_ts and "transaction_timestamp" in df.columns:
            df["transaction_timestamp"] = pd.to_datetime(
                df["transaction_timestamp"], format="mixed"
            )
        return df
    with ThreadPoolExecutor(max_workers=min(len(paths), PREFETCH_DEPTH)) as pool:
        return list(pool.map(_read, paths))


def _prefetch_iterator(parts, parse_ts=False, chunk_size=PREFETCH_DEPTH):
    for start in range(0, len(parts), chunk_size):
        batch = parts[start:start + chunk_size]
        dfs = _prefetch_parquets(batch, parse_ts=parse_ts)
        yield from dfs


def _shannon_entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def _hhi(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    shares = counts / total
    return float(np.sum(shares ** 2))


def _benford_divergence(samples):
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


def _reservoir_add(reservoir, item, n_seen, size=RESERVOIR_SIZE):
    if n_seen < size:
        reservoir.append(item)
    else:
        j = random.randint(0, n_seen)
        if j < size:
            reservoir[j] = item


# ═══════════════════════════════════════════════════════════════════════
# PASS 1: Transaction features
# ═══════════════════════════════════════════════════════════════════════

# Global MCC stats for anomaly detection (Pattern #13)
_global_mcc_stats = {}  # mcc_code -> {n, sum, sum_sq}

SPLIT_TS = pd.Timestamp(BEHAVIORAL_SPLIT_DATE)


def _make_account_accum():
    return {
        "n_txn": 0, "n_credit": 0, "n_debit": 0,
        "sum_credit": 0.0, "sum_debit": 0.0,
        "sum_abs": 0.0, "sum_abs_sq": 0.0,
        "min_abs": float("inf"), "max_abs": 0.0,
        "cr_sum_abs": 0.0, "cr_sum_abs_sq": 0.0, "cr_max": 0.0,
        "dr_sum_abs": 0.0, "dr_sum_abs_sq": 0.0, "dr_max": 0.0,
        "reservoir": [],  # (amount, mcc_code) tuples
        "hour_hist": np.zeros(24, dtype=np.int32),
        "dow_hist": np.zeros(7, dtype=np.int32),
        "month_hist": defaultdict(int),
        "ts_min": None, "ts_max": None,
        "prev_ts": None,
        "itd_sum": 0.0, "itd_sum_sq": 0.0,
        "itd_min": float("inf"), "itd_max": 0.0, "n_itd": 0,
        "dormancy_events": 0,
        "channel_counts": defaultdict(int),
        "mcc_counts": defaultdict(int),
        "mcc_amount_sums": defaultdict(float),  # V2: per-MCC amount tracking
        "mcc_amount_counts": defaultdict(int),
        "cp_credit": defaultdict(int),  # V2: separate credit counterparties
        "cp_debit": defaultdict(int),   # V2: separate debit counterparties
        "near_threshold": {t: 0 for t in STRUCTURING_THRESHOLDS},
        "round_count": 0,
        "n_negative": 0,
        "n_night": 0, "n_business": 0, "n_weekend": 0,
        # V2: Behavioral change (before/after midpoint)
        "early_n": 0, "early_sum_abs": 0.0,
        "early_n_credit": 0, "early_n_debit": 0,
        "late_n": 0, "late_sum_abs": 0.0,
        "late_n_credit": 0, "late_n_debit": 0,
    }


def _update_account(acc, amounts, txn_types, channels, mcc_codes,
                    counterparties, timestamps):
    n = len(amounts)
    abs_amounts = np.abs(amounts)
    n_before = acc["n_txn"]
    acc["n_txn"] += n

    credit_mask = txn_types == "C"
    debit_mask = txn_types == "D"
    acc["n_credit"] += int(credit_mask.sum())
    acc["n_debit"] += int(debit_mask.sum())
    acc["sum_credit"] += float(amounts[credit_mask].sum())
    acc["sum_debit"] += float(amounts[debit_mask].sum())

    acc["sum_abs"] += float(abs_amounts.sum())
    acc["sum_abs_sq"] += float((abs_amounts ** 2).sum())
    if n > 0:
        batch_min = float(abs_amounts.min())
        batch_max = float(abs_amounts.max())
        if batch_min < acc["min_abs"]:
            acc["min_abs"] = batch_min
        if batch_max > acc["max_abs"]:
            acc["max_abs"] = batch_max

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

    # Reservoir sampling — store (amount, mcc_code) for MCC anomaly
    reservoir = acc["reservoir"]
    for i in range(n):
        _reservoir_add(reservoir, (float(amounts[i]), int(mcc_codes[i])),
                       n_before + i, RESERVOIR_SIZE)

    # Temporal histograms
    hours = timestamps.dt.hour.values
    dows = timestamps.dt.dayofweek.values
    np.add.at(acc["hour_hist"], hours, 1)
    np.add.at(acc["dow_hist"], dows, 1)
    for ym in timestamps.dt.to_period("M"):
        acc["month_hist"][str(ym)] += 1

    acc["n_night"] += int(((hours >= 0) & (hours < 6)).sum())
    acc["n_business"] += int(((hours >= 9) & (hours < 17)).sum())
    acc["n_weekend"] += int((dows >= 5).sum())

    # Inter-transaction time deltas
    ts_sorted = timestamps.sort_values().reset_index(drop=True)
    ts_min = ts_sorted.iloc[0]
    ts_max = ts_sorted.iloc[-1]
    if acc["ts_min"] is None or ts_min < acc["ts_min"]:
        acc["ts_min"] = ts_min
    if acc["ts_max"] is None or ts_max > acc["ts_max"]:
        acc["ts_max"] = ts_max

    if len(ts_sorted) > 1:
        deltas = np.diff(ts_sorted.values).astype("timedelta64[s]").astype(np.float64)
        if acc["prev_ts"] is not None:
            first_delta = (ts_sorted.iloc[0] - acc["prev_ts"]).total_seconds()
            if first_delta >= 0:
                deltas = np.concatenate([[first_delta], deltas])
        valid = deltas[deltas >= 0]
        if len(valid) > 0:
            acc["itd_sum"] += float(valid.sum())
            acc["itd_sum_sq"] += float((valid ** 2).sum())
            v_min, v_max = float(valid.min()), float(valid.max())
            if v_min < acc["itd_min"]:
                acc["itd_min"] = v_min
            if v_max > acc["itd_max"]:
                acc["itd_max"] = v_max
            acc["n_itd"] += len(valid)
            acc["dormancy_events"] += int(np.sum(valid > DORMANCY_DAYS * 86400))

    acc["prev_ts"] = ts_sorted.iloc[-1] if len(ts_sorted) > 0 else acc["prev_ts"]

    # Channel counts
    for ch, cnt in zip(*np.unique(channels, return_counts=True)):
        acc["channel_counts"][ch] += int(cnt)

    # MCC counts + per-MCC amount tracking
    for mcc, cnt in zip(*np.unique(mcc_codes, return_counts=True)):
        acc["mcc_counts"][mcc] += int(cnt)
    for i in range(n):
        mcc = int(mcc_codes[i])
        acc["mcc_amount_sums"][mcc] += float(abs_amounts[i])
        acc["mcc_amount_counts"][mcc] += 1

    # V2: Separate credit/debit counterparties (fan-in/fan-out)
    for i in range(n):
        cp = counterparties[i]
        if pd.notna(cp):
            if credit_mask[i]:
                acc["cp_credit"][cp] += 1
            else:
                acc["cp_debit"][cp] += 1

    # Structuring
    for threshold in STRUCTURING_THRESHOLDS:
        lower = threshold * 0.85
        acc["near_threshold"][threshold] += int(
            ((abs_amounts >= lower) & (abs_amounts < threshold)).sum()
        )

    # Round amounts
    round_set = set(ROUND_AMOUNTS)
    for amt in abs_amounts:
        if amt > 0 and int(amt) in round_set:
            acc["round_count"] += 1

    acc["n_negative"] += int((amounts < 0).sum())

    # V2: Behavioral split (before/after midpoint)
    early_mask = timestamps < SPLIT_TS
    late_mask = ~early_mask
    acc["early_n"] += int(early_mask.sum())
    acc["early_sum_abs"] += float(abs_amounts[early_mask].sum())
    acc["early_n_credit"] += int((credit_mask & early_mask).sum())
    acc["early_n_debit"] += int((debit_mask & early_mask).sum())
    acc["late_n"] += int(late_mask.sum())
    acc["late_sum_abs"] += float(abs_amounts[late_mask].sum())
    acc["late_n_credit"] += int((credit_mask & late_mask).sum())
    acc["late_n_debit"] += int((debit_mask & late_mask).sum())

    # Update global MCC stats
    for i in range(n):
        mcc = int(mcc_codes[i])
        amt = float(abs_amounts[i])
        if mcc not in _global_mcc_stats:
            _global_mcc_stats[mcc] = {"n": 0, "sum": 0.0, "sum_sq": 0.0}
        s = _global_mcc_stats[mcc]
        s["n"] += 1
        s["sum"] += amt
        s["sum_sq"] += amt * amt


def _finalize_account(acct_id, acc):
    row = {"account_id": acct_id}
    n = acc["n_txn"]
    total_txn = max(n, 1)

    # Basic volume
    row["n_txn"] = n
    row["n_credit"] = acc["n_credit"]
    row["n_debit"] = acc["n_debit"]
    row["credit_debit_ratio"] = acc["n_credit"] / max(acc["n_debit"], 1)
    row["sum_credit"] = acc["sum_credit"]
    row["sum_debit"] = acc["sum_debit"]
    row["net_flow"] = acc["sum_credit"] - acc["sum_debit"]
    row["credit_debit_amount_ratio"] = acc["sum_credit"] / max(abs(acc["sum_debit"]), 1)

    # Amount statistics
    if n > 0:
        mean_abs = acc["sum_abs"] / n
        var_abs = max(acc["sum_abs_sq"] / n - mean_abs ** 2, 0)
        std_abs = var_abs ** 0.5
        row["amt_mean"] = mean_abs
        row["amt_std"] = std_abs
        row["amt_max"] = acc["max_abs"]
        row["amt_min"] = acc["min_abs"] if acc["min_abs"] != float("inf") else 0
        row["amt_cv"] = std_abs / max(mean_abs, 1e-6)

        reservoir = acc["reservoir"]
        if len(reservoir) >= 5:
            r_amounts = np.abs(np.array([x[0] for x in reservoir]))
            row["amt_median"] = float(np.median(r_amounts))
            row["amt_p25"] = float(np.percentile(r_amounts, 25))
            row["amt_p75"] = float(np.percentile(r_amounts, 75))
            row["amt_p95"] = float(np.percentile(r_amounts, 95))
            row["amt_p99"] = float(np.percentile(r_amounts, 99))
            row["amt_iqr"] = row["amt_p75"] - row["amt_p25"]
            row["amt_skew"] = float(skew(r_amounts))
            row["amt_kurtosis"] = float(kurtosis(r_amounts))
            row["benford_divergence"] = _benford_divergence([x[0] for x in reservoir])
        else:
            for k in ["amt_median", "amt_p25", "amt_p75", "amt_p95", "amt_p99",
                       "amt_iqr", "amt_skew", "amt_kurtosis", "benford_divergence"]:
                row[k] = 0.0
    else:
        for k in ["amt_mean", "amt_std", "amt_max", "amt_min", "amt_cv",
                   "amt_median", "amt_p25", "amt_p75", "amt_p95", "amt_p99",
                   "amt_iqr", "amt_skew", "amt_kurtosis", "benford_divergence"]:
            row[k] = 0.0

    # Credit/debit amount stats
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

    # Temporal features
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
        row["burstiness"] = (itd_std - itd_mean) / (itd_std + itd_mean) if (itd_std + itd_mean) > 0 else 0
        row["max_gap_days"] = acc["itd_max"] / 86400
        row["dormancy_events"] = acc["dormancy_events"]
        row["txn_per_day"] = n / max(row["txn_span_days"], 1)
    else:
        for k in ["txn_span_days", "inter_txn_mean_sec", "inter_txn_std_sec",
                   "inter_txn_min_sec", "inter_txn_max_sec", "inter_txn_cv",
                   "burstiness", "max_gap_days", "dormancy_events", "txn_per_day"]:
            row[k] = 0.0

    # Hour/DOW distributions
    row["pct_night_txn"] = acc["n_night"] / total_txn
    row["pct_business_hours"] = acc["n_business"] / total_txn
    row["pct_weekend_txn"] = acc["n_weekend"] / total_txn
    row["hour_entropy"] = _shannon_entropy(acc["hour_hist"].astype(float))
    row["dow_entropy"] = _shannon_entropy(acc["dow_hist"].astype(float))

    # Monthly consistency
    month_counts = np.array(list(acc["month_hist"].values()), dtype=float)
    if len(month_counts) > 0:
        row["monthly_txn_std"] = float(np.std(month_counts))
        row["monthly_txn_cv"] = float(np.std(month_counts) / max(np.mean(month_counts), 1e-6))
        row["n_active_months"] = len(month_counts)
    else:
        row["monthly_txn_std"] = 0.0
        row["monthly_txn_cv"] = 0.0
        row["n_active_months"] = 0

    # Channel diversity
    ch_counts = np.array([acc["channel_counts"].get(c, 0) for c in ALL_CHANNELS], dtype=float)
    row["n_channels_used"] = int(np.sum(ch_counts > 0))
    row["channel_entropy"] = _shannon_entropy(ch_counts)
    row["channel_hhi"] = _hhi(ch_counts)
    row["top_channel_share"] = float(ch_counts.max() / max(ch_counts.sum(), 1))
    for ch in ["UPC", "UPD", "ATW", "CHQ", "CSD", "NTD", "IPM", "END"]:
        row[f"pct_{ch}"] = acc["channel_counts"].get(ch, 0) / total_txn

    # MCC diversity
    row["n_unique_mcc"] = len(acc["mcc_counts"])
    mcc_cnts = np.array(list(acc["mcc_counts"].values()), dtype=float)
    row["mcc_entropy"] = _shannon_entropy(mcc_cnts) if len(mcc_cnts) > 0 else 0
    row["mcc_hhi"] = _hhi(mcc_cnts) if len(mcc_cnts) > 0 else 0

    # V2: MCC-amount anomaly (Pattern #13)
    mcc_zscores = []
    for mcc, mcc_sum in acc["mcc_amount_sums"].items():
        mcc_cnt = acc["mcc_amount_counts"][mcc]
        acct_mcc_mean = mcc_sum / mcc_cnt
        gs = _global_mcc_stats.get(mcc)
        if gs and gs["n"] >= 10:
            pop_mean = gs["sum"] / gs["n"]
            pop_var = max(gs["sum_sq"] / gs["n"] - pop_mean ** 2, 0)
            pop_std = pop_var ** 0.5
            if pop_std > 1e-6:
                z = (acct_mcc_mean - pop_mean) / pop_std
                mcc_zscores.append(z)
    if mcc_zscores:
        mcc_z = np.array(mcc_zscores)
        row["mcc_amt_zscore_mean"] = float(np.mean(mcc_z))
        row["mcc_amt_zscore_max"] = float(np.max(np.abs(mcc_z)))
        row["mcc_amt_n_outliers"] = int(np.sum(np.abs(mcc_z) > 2))
        row["mcc_amt_outlier_ratio"] = row["mcc_amt_n_outliers"] / max(len(mcc_z), 1)
    else:
        row["mcc_amt_zscore_mean"] = 0.0
        row["mcc_amt_zscore_max"] = 0.0
        row["mcc_amt_n_outliers"] = 0
        row["mcc_amt_outlier_ratio"] = 0.0

    # V2: Fan-in / fan-out (separate credit/debit counterparties)
    n_cp_credit = len(acc["cp_credit"])
    n_cp_debit = len(acc["cp_debit"])
    row["n_unique_cp_credit"] = n_cp_credit
    row["n_unique_cp_debit"] = n_cp_debit
    row["n_unique_counterparties"] = len(set(acc["cp_credit"]) | set(acc["cp_debit"]))
    row["fan_in_out_ratio"] = n_cp_credit / max(n_cp_debit, 1)
    # Counterparty concentration (combined)
    all_cp = defaultdict(int)
    for cp, cnt in acc["cp_credit"].items():
        all_cp[cp] += cnt
    for cp, cnt in acc["cp_debit"].items():
        all_cp[cp] += cnt
    cp_cnts = np.array(list(all_cp.values()), dtype=float)
    row["cp_entropy"] = _shannon_entropy(cp_cnts) if len(cp_cnts) > 0 else 0
    row["cp_hhi"] = _hhi(cp_cnts) if len(cp_cnts) > 0 else 0
    if len(cp_cnts) > 0:
        row["top_cp_share"] = float(cp_cnts.max() / cp_cnts.sum())
        row["top3_cp_share"] = float(np.sort(cp_cnts)[-3:].sum() / cp_cnts.sum())
    else:
        row["top_cp_share"] = 0.0
        row["top3_cp_share"] = 0.0

    # Structuring
    for threshold in STRUCTURING_THRESHOLDS:
        row[f"near_{threshold}_count"] = acc["near_threshold"][threshold]
        row[f"near_{threshold}_ratio"] = acc["near_threshold"][threshold] / total_txn

    # Round amounts
    row["round_amount_count"] = acc["round_count"]
    row["round_amount_ratio"] = acc["round_count"] / total_txn

    # Reversals
    row["n_reversals"] = acc["n_negative"]
    row["reversal_ratio"] = acc["n_negative"] / total_txn

    # V2: Behavioral change (before/after midpoint)
    early_n = max(acc["early_n"], 1)
    late_n = max(acc["late_n"], 1)
    early_rate = acc["early_n"]  # raw count (rate computed after span known)
    late_rate = acc["late_n"]
    row["behavioral_volume_change"] = late_rate / max(early_rate, 1)
    early_mean = acc["early_sum_abs"] / early_n
    late_mean = acc["late_sum_abs"] / late_n
    row["behavioral_amount_change"] = late_mean / max(early_mean, 1e-6)
    early_cr_ratio = acc["early_n_credit"] / early_n
    late_cr_ratio = acc["late_n_credit"] / late_n
    row["behavioral_credit_shift"] = late_cr_ratio - early_cr_ratio
    row["pct_activity_late"] = acc["late_n"] / total_txn

    return row


def build_txn_features():
    if TXN_FEATURES_PATH.exists():
        log.info("Loading cached transaction features from %s", TXN_FEATURES_PATH)
        return pd.read_parquet(TXN_FEATURES_PATH)

    parts = sorted(glob(str(TXN_DIR / "batch-*" / "part_*.parquet")))
    log.info("Processing %d transaction parts", len(parts))

    random.seed(SEED)
    _global_mcc_stats.clear()
    accounts = {}
    map_chunks = []
    MAP_CHUNK_SIZE = 50

    for i, df in enumerate(tqdm(
        _prefetch_iterator(parts, parse_ts=True),
        total=len(parts), desc="Pass 1: Transactions",
    )):
        map_chunks.append(df[["transaction_id", "account_id"]].copy())
        if len(map_chunks) >= MAP_CHUNK_SIZE:
            chunk_df = pd.concat(map_chunks, ignore_index=True)
            chunk_df.to_parquet(OUTPUT_DIR / f"txn_map_chunk_{i}.parquet", index=False)
            del chunk_df
            map_chunks = []
            gc.collect()

        for acct_id, grp in df.groupby("account_id"):
            if acct_id not in accounts:
                accounts[acct_id] = _make_account_accum()
            _update_account(
                accounts[acct_id],
                grp["amount"].values, grp["txn_type"].values,
                grp["channel"].values, grp["mcc_code"].values,
                grp["counterparty_id"].values, grp["transaction_timestamp"],
            )
        del df
        if (i + 1) % 100 == 0:
            gc.collect()
            log.info("  Processed %d/%d parts, %d accounts", i + 1, len(parts), len(accounts))

    # Save remaining mapping chunks
    if map_chunks:
        chunk_df = pd.concat(map_chunks, ignore_index=True)
        chunk_df.to_parquet(OUTPUT_DIR / "txn_map_chunk_final.parquet", index=False)
        del chunk_df, map_chunks
        gc.collect()

    # Combine mapping chunks
    if not TXN_ID_MAP_PATH.exists():
        log.info("Combining transaction ID mapping chunks")
        map_files = sorted(glob(str(OUTPUT_DIR / "txn_map_chunk_*.parquet")))
        txn_map = pd.concat([pd.read_parquet(f) for f in map_files], ignore_index=True)
        txn_map.to_parquet(TXN_ID_MAP_PATH, index=False)
        log.info("Saved mapping: %d transaction IDs", len(txn_map))
        del txn_map
        import os
        for f in map_files:
            os.remove(f)
        gc.collect()

    log.info("Global MCC stats: %d unique MCC codes tracked", len(_global_mcc_stats))

    # Finalize features
    log.info("Finalizing transaction features for %d accounts", len(accounts))
    items = list(accounts.items())
    del accounts

    def _finalize_chunk(chunk):
        return [_finalize_account(aid, acc) for aid, acc in chunk]

    chunk_size = max(1, len(items) // N_THREADS)
    chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    del items

    rows = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        for chunk_rows in pool.map(_finalize_chunk, chunks):
            rows.extend(chunk_rows)
    del chunks
    gc.collect()

    features = pd.DataFrame(rows).set_index("account_id")
    features.to_parquet(TXN_FEATURES_PATH)
    log.info("Saved transaction features: %s", features.shape)
    return features


# ═══════════════════════════════════════════════════════════════════════
# PASS 2: Transaction additional features
# ═══════════════════════════════════════════════════════════════════════

def build_txn_additional_features():
    """
    Pass 2: Chunked vectorized — for each chunk of additional parts,
    load only the needed mapping rows via pyarrow filter, then vectorized agg.
    Merges running sums across chunks. No full mapping in memory.
    """
    if TXN_ADD_FEATURES_PATH.exists():
        log.info("Loading cached additional features from %s", TXN_ADD_FEATURES_PATH)
        return pd.read_parquet(TXN_ADD_FEATURES_PATH)

    import pyarrow.parquet as pq

    parts = sorted(glob(str(TXN_ADDITIONAL_DIR / "batch-*" / "part_*.parquet")))
    log.info("Pass 2: Processing %d transactions_additional parts (chunked vectorized)", len(parts))

    all_aggs = []
    all_pt_counts = []
    all_st_counts = []
    all_bal_diffs = []

    CHUNK_SIZE = 60

    for chunk_start in range(0, len(parts), CHUNK_SIZE):
        chunk_parts = parts[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_num = chunk_start // CHUNK_SIZE + 1
        total_chunks = (len(parts) + CHUNK_SIZE - 1) // CHUNK_SIZE
        log.info("  Chunk %d/%d: loading %d parts...", chunk_num, total_chunks, len(chunk_parts))

        dfs = [pd.read_parquet(p) for p in chunk_parts]
        txn = pd.concat(dfs, ignore_index=True)
        del dfs
        gc.collect()
        n_rows = len(txn)
        log.info("    Loaded %d rows. Resolving account_ids via mapping chunks...", n_rows)

        # Get unique txn_ids in this chunk
        chunk_txn_ids = set(txn["transaction_id"].values)

        # Scan mapping file in chunks to find account_ids for these txn_ids
        partial_lookup = {}
        map_file = pq.ParquetFile(TXN_ID_MAP_PATH)
        MAP_BATCH = 50_000_000
        for batch in map_file.iter_batches(batch_size=MAP_BATCH, columns=["transaction_id", "account_id"]):
            chunk_df = batch.to_pandas()
            mask = chunk_df["transaction_id"].isin(chunk_txn_ids)
            matched = chunk_df[mask]
            for tid, aid in zip(matched["transaction_id"].values, matched["account_id"].values):
                partial_lookup[tid] = aid
            del chunk_df, matched, batch
            if len(partial_lookup) >= len(chunk_txn_ids):
                break
        del map_file, chunk_txn_ids

        log.info("    Matched %d/%d txn_ids. Running vectorized agg...", len(partial_lookup), n_rows)

        txn["account_id"] = txn["transaction_id"].map(partial_lookup)
        del partial_lookup
        txn = txn.dropna(subset=["account_id"])

        if len(txn) == 0:
            gc.collect()
            continue

        # Helper columns (vectorized)
        txn["has_geo"] = txn["latitude"].notna() & txn["longitude"].notna()
        txn["near_zero_bal"] = txn["balance_after_transaction"].abs() < 100
        txn["has_atm_deposit"] = txn["atm_deposit_channel_code"].notna()

        g = txn.groupby("account_id")

        # Running-sum-compatible aggregations
        agg = g.agg(
            n_total=("transaction_id", "count"),
            n_geo_txns=("has_geo", "sum"),
            lat_sum=("latitude", "sum"),
            lat_sq_sum=("latitude", lambda x: (x ** 2).sum()),
            lat_count=("latitude", "count"),
            lat_min=("latitude", "min"),
            lat_max=("latitude", "max"),
            lon_sum=("longitude", "sum"),
            lon_sq_sum=("longitude", lambda x: (x ** 2).sum()),
            lon_min=("longitude", "min"),
            lon_max=("longitude", "max"),
            bal_sum=("balance_after_transaction", "sum"),
            bal_sq_sum=("balance_after_transaction", lambda x: (x ** 2).sum()),
            bal_count=("balance_after_transaction", "count"),
            bal_min=("balance_after_transaction", "min"),
            bal_max=("balance_after_transaction", "max"),
            n_near_zero_balance=("near_zero_bal", "sum"),
            n_atm_deposits=("has_atm_deposit", "sum"),
            n_unique_ips=("ip_address", "nunique"),
            n_unique_subnets=("mnemonic_code", "nunique"),
        )
        all_aggs.append(agg)

        pt = txn.groupby(["account_id", "part_transaction_type"]).size().unstack(fill_value=0)
        all_pt_counts.append(pt)

        st = txn.groupby(["account_id", "transaction_sub_type"]).size().unstack(fill_value=0)
        all_st_counts.append(st)

        bal = txn[["account_id", "balance_after_transaction"]].dropna(subset=["balance_after_transaction"]).copy()
        bal["bal_diff"] = bal.groupby("account_id")["balance_after_transaction"].diff()
        bv = bal.groupby("account_id")["bal_diff"].agg(
            bal_diff_abs_sum=lambda x: x.abs().sum(),
            bal_diff_count="count",
            bal_diff_min="min",
            neg_diff_sum=lambda x: x[x < 0].sum(),
            n_neg_diff=lambda x: (x < 0).sum(),
        )
        all_bal_diffs.append(bv)

        del txn, g, agg, pt, st, bal, bv
        gc.collect()
        log.info("    Chunk %d/%d done", chunk_num, total_chunks)

    # ── Merge partial results across chunks ──
    log.info("  Merging %d chunk results...", len(all_aggs))

    def _merge_sum(frames):
        result = frames[0]
        for f in frames[1:]:
            result = result.add(f, fill_value=0)
        return result

    def _merge_minmax(frames, col, op):
        result = frames[0][[col]].copy()
        for f in frames[1:]:
            if col in f.columns:
                other = f[[col]]
                result = result.join(other, rsuffix="_r", how="outer")
                if op == "min":
                    result[col] = result.min(axis=1)
                else:
                    result[col] = result.max(axis=1)
                result = result[[col]]
        return result

    merged = _merge_sum(all_aggs)
    # Fix min/max — sum doesn't work for these, need actual min/max across chunks
    for col, op in [("lat_min", "min"), ("lat_max", "max"), ("lon_min", "min"),
                     ("lon_max", "max"), ("bal_min", "min"), ("bal_max", "max")]:
        col_vals = _merge_minmax(all_aggs, col, op)
        merged[col] = col_vals[col]
    del all_aggs

    # Build features from running sums
    features = pd.DataFrame(index=merged.index)
    n_total = merged["n_total"].clip(lower=1)
    features["n_geo_txns"] = merged["n_geo_txns"]
    features["geo_coverage_ratio"] = merged["n_geo_txns"] / n_total

    lat_n = merged["lat_count"].clip(lower=1)
    lat_mean = merged["lat_sum"] / lat_n
    lat_var = (merged["lat_sq_sum"] / lat_n - lat_mean ** 2).clip(lower=0)
    lon_mean = merged["lon_sum"] / lat_n
    lon_var = (merged["lon_sq_sum"] / lat_n - lon_mean ** 2).clip(lower=0)
    features["lat_std"] = lat_var ** 0.5
    features["lon_std"] = lon_var ** 0.5
    features["geo_spread"] = (lat_var + lon_var) ** 0.5
    lat_range = (merged["lat_max"] - merged["lat_min"]).fillna(0)
    lon_range = (merged["lon_max"] - merged["lon_min"]).fillna(0)
    features["geo_range_km"] = ((lat_range * 111) ** 2 + (lon_range * 85) ** 2) ** 0.5
    features["n_geo_clusters"] = merged["n_geo_txns"].clip(upper=500)

    features["n_unique_ips"] = merged["n_unique_ips"]
    features["n_unique_subnets"] = merged["n_unique_subnets"]
    features["ip_per_txn"] = merged["n_unique_ips"] / n_total

    bal_n = merged["bal_count"].clip(lower=1)
    bal_mean_s = merged["bal_sum"] / bal_n
    bal_var = (merged["bal_sq_sum"] / bal_n - bal_mean_s ** 2).clip(lower=0)
    features["bal_mean"] = bal_mean_s
    features["bal_std"] = bal_var ** 0.5
    features["bal_min"] = merged["bal_min"]
    features["bal_max"] = merged["bal_max"]
    features["bal_cv"] = (bal_var ** 0.5) / bal_mean_s.abs().clip(lower=1e-6)
    features["bal_range"] = (merged["bal_max"] - merged["bal_min"]).fillna(0)
    features["n_near_zero_balance"] = merged["n_near_zero_balance"]
    features["pct_near_zero_balance"] = merged["n_near_zero_balance"] / bal_n
    features["n_atm_deposits"] = merged["n_atm_deposits"]
    del merged

    # Balance volatility
    bal_diff = _merge_sum(all_bal_diffs)
    # Fix min across chunks for drawdown
    dd_vals = _merge_minmax(all_bal_diffs, "bal_diff_min", "min")
    bal_diff["bal_diff_min"] = dd_vals["bal_diff_min"]
    del all_bal_diffs, dd_vals

    features["max_drawdown"] = bal_diff["bal_diff_min"]
    features["mean_drawdown"] = bal_diff["neg_diff_sum"] / bal_diff["n_neg_diff"].clip(lower=1)
    features["bal_volatility"] = (
        bal_diff["bal_diff_abs_sum"] / bal_diff["bal_diff_count"].clip(lower=1)
    ) / bal_mean_s.abs().clip(lower=1e-6)
    del bal_diff

    # Transaction type percentages
    log.info("  Merging transaction type distributions...")
    pt_merged = _merge_sum(all_pt_counts)
    del all_pt_counts
    pt_total = pt_merged.sum(axis=1).clip(lower=1)
    for pt in ["CI", "BI", "IP", "IC"]:
        features[f"pct_pt_{pt}"] = pt_merged[pt] / pt_total if pt in pt_merged.columns else 0.0

    st_merged = _merge_sum(all_st_counts)
    del all_st_counts
    st_total = st_merged.sum(axis=1).clip(lower=1)
    for st in ["NORMAL", "CLT_CASH", "LOAN"]:
        features[f"pct_st_{st}"] = st_merged[st] / st_total if st in st_merged.columns else 0.0
    features["n_cash_txns"] = st_merged["CLT_CASH"] if "CLT_CASH" in st_merged.columns else 0
    del pt_merged, st_merged

    features = features.fillna(0)
    features.to_parquet(TXN_ADD_FEATURES_PATH)
    log.info("Saved additional features: %s", features.shape)
    return features


# ═══════════════════════════════════════════════════════════════════════
# PASS 3: Static features (NO LABEL LEAKAGE)
# ═══════════════════════════════════════════════════════════════════════

def build_static_features():
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
        "branch_pin", "product_family", "nomination_flag",
        "cheque_allowed", "cheque_availed", "kyc_compliant", "rural_branch",
        "last_mobile_update_date", "last_kyc_date", "freeze_date", "unfreeze_date",
    ]
    # KEEP branch_code for target encoding in models.py
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

    # Branch features (structural only — NO mule rate, NO label info)
    # branch_code already in acct_features from accounts table (line ~824)
    branch_feat = branch.set_index("branch_code").copy()
    branch_type_dummies = pd.get_dummies(branch_feat["branch_type"], prefix="branch_type")
    branch_feat = branch_feat.drop(
        columns=["branch_address", "branch_pin_code", "branch_city", "branch_state", "branch_type"],
        errors="ignore"
    )
    branch_feat = branch_feat.join(branch_type_dummies)

    # Branch structural stats (label-free): accounts per branch, employee ratios
    branch_account_counts = accounts.groupby("branch_code").size().rename("branch_n_accounts")
    branch_feat = branch_feat.join(branch_account_counts)
    branch_feat["accounts_per_employee"] = branch_feat["branch_n_accounts"] / branch_feat["branch_employee_count"].clip(lower=1)

    acct_features = acct_features.join(branch_feat, on="branch_code")

    # Drop customer_id (used only for joins) but KEEP branch_code for target encoding
    acct_features = acct_features.drop(columns=["customer_id"], errors="ignore")
    acct_features = acct_features.fillna(0)

    acct_features.to_parquet(STATIC_FEATURES_PATH)
    log.info("Saved static features: %s", acct_features.shape)
    return acct_features


# ═══════════════════════════════════════════════════════════════════════
# PASS 4: Graph features (network centrality, community detection)
# ═══════════════════════════════════════════════════════════════════════

def build_graph_features():
    if GRAPH_FEATURES_PATH.exists():
        log.info("Loading cached graph features from %s", GRAPH_FEATURES_PATH)
        return pd.read_parquet(GRAPH_FEATURES_PATH)

    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    log.info("Building account-counterparty transaction graph")

    # Aggregate edges from transaction data — vectorized
    parts = sorted(glob(str(TXN_DIR / "batch-*" / "part_*.parquet")))
    log.info("Pass 4: Loading edges from %d transaction parts", len(parts))

    edge_dfs = []
    for i, part_path in enumerate(parts):
        df = pd.read_parquet(part_path, columns=["account_id", "counterparty_id", "amount"])
        df = df.dropna(subset=["counterparty_id"])
        agg = df.groupby(["account_id", "counterparty_id"]).agg(
            n_txn=("amount", "count"), total_amount=("amount", "sum")
        ).reset_index()
        edge_dfs.append(agg)
        del df, agg
        if (i + 1) % 50 == 0:
            log.info("  Loaded edges from %d/%d parts", i + 1, len(parts))

    log.info("  Concatenating all edge aggregates...")
    all_edges = pd.concat(edge_dfs, ignore_index=True)
    del edge_dfs
    gc.collect()

    log.info("  Final groupby on %d edge rows...", len(all_edges))
    all_edges = all_edges.groupby(["account_id", "counterparty_id"]).agg(
        n_txn=("n_txn", "sum"), total_amount=("total_amount", "sum")
    ).reset_index()
    log.info("Graph has %d unique edges", len(all_edges))

    G = nx.DiGraph()
    log.info("  Building NetworkX graph from DataFrame...")
    for src, dst, n, amt in zip(all_edges["account_id"], all_edges["counterparty_id"],
                                 all_edges["n_txn"], all_edges["total_amount"]):
        G.add_edge(src, dst, weight=int(n), amount=float(amt))
    del all_edges
    gc.collect()

    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # Centrality metrics
    log.info("Computing PageRank")
    pagerank = nx.pagerank(G, weight="weight", max_iter=100, tol=1e-6)

    log.info("Computing HITS")
    try:
        hubs, authorities = nx.hits(G, max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        hubs = {n: 0.0 for n in G.nodes()}
        authorities = {n: 0.0 for n in G.nodes()}

    log.info("Computing degree centrality")
    in_degree_w = dict(G.in_degree(weight="weight"))
    out_degree_w = dict(G.out_degree(weight="weight"))
    in_degree_c = dict(G.in_degree())
    out_degree_c = dict(G.out_degree())
    in_amount = dict(G.in_degree(weight="amount"))
    out_amount = dict(G.out_degree(weight="amount"))

    log.info("Computing betweenness centrality (sampled k=500)")
    betweenness = nx.betweenness_centrality(
        G, k=min(500, G.number_of_nodes()), weight="weight"
    )

    log.info("Running Louvain community detection")
    G_undirected = G.to_undirected()
    try:
        communities = louvain_communities(G_undirected, weight="weight", seed=SEED)
        node_community = {}
        community_sizes = {}
        for i, comm in enumerate(communities):
            community_sizes[i] = len(comm)
            for node in comm:
                node_community[node] = i
    except Exception as e:
        log.warning("Community detection failed: %s", e)
        node_community = {n: 0 for n in G.nodes()}
        community_sizes = {0: G.number_of_nodes()}

    log.info("Computing clustering coefficients")
    clustering = nx.clustering(G_undirected, weight="weight")

    # Assemble features for ACCT_ nodes only
    account_nodes = [n for n in G.nodes() if str(n).startswith("ACCT_")]
    log.info("Assembling graph features for %d account nodes", len(account_nodes))

    rows = []
    for node in account_nodes:
        comm_id = node_community.get(node, -1)
        in_a = in_amount.get(node, 0)
        out_a = out_amount.get(node, 0)
        rows.append({
            "account_id": node,
            "graph_pagerank": pagerank.get(node, 0),
            "graph_hub_score": hubs.get(node, 0),
            "graph_authority_score": authorities.get(node, 0),
            "graph_in_degree_weighted": in_degree_w.get(node, 0),
            "graph_out_degree_weighted": out_degree_w.get(node, 0),
            "graph_in_degree_count": in_degree_c.get(node, 0),
            "graph_out_degree_count": out_degree_c.get(node, 0),
            "graph_degree_ratio": out_degree_c.get(node, 0) / max(in_degree_c.get(node, 0), 1),
            "graph_in_amount": in_a,
            "graph_out_amount": out_a,
            "graph_amount_ratio": out_a / max(in_a, 1),  # pass-through indicator
            "graph_betweenness": betweenness.get(node, 0),
            "graph_clustering_coeff": clustering.get(node, 0),
            "graph_community_id": comm_id,
            "graph_community_size": community_sizes.get(comm_id, 0),
        })

    df = pd.DataFrame(rows).set_index("account_id")
    df.to_parquet(GRAPH_FEATURES_PATH)
    log.info("Saved graph features: %s", df.shape)
    return df


# ═══════════════════════════════════════════════════════════════════════
# COMBINE ALL FEATURES
# ═══════════════════════════════════════════════════════════════════════

def build_all_features():
    if FULL_FEATURES_PATH.exists():
        log.info("Loading cached full features from %s", FULL_FEATURES_PATH)
        return pd.read_parquet(FULL_FEATURES_PATH)

    txn = build_txn_features()
    txn_add = build_txn_additional_features()
    static = build_static_features()
    graph = build_graph_features()

    full = static.join(txn, how="left")
    full = full.join(txn_add, how="left")
    full = full.join(graph, how="left")
    full = full.fillna(0)

    full.to_parquet(FULL_FEATURES_PATH)
    log.info("Saved full features: %s", full.shape)
    return full


if __name__ == "__main__":
    build_all_features()
