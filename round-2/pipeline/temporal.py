"""
NFPC Phase 2 — Temporal Window Prediction (V3)

For accounts predicted as mules, identify suspicious_start/suspicious_end.

V3: Fully vectorized numpy — no Python loops over windows.
Uses searchsorted + cumsum tricks for O(n) sliding window stats.
"""
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from config import (
    TXN_DIR, TRAIN_LABELS_PATH, TEMPORAL_FEATURES_PATH, OUTPUT_DIR,
    RAPID_PASSTHROUGH_HOURS, DORMANCY_DAYS,
    log, N_THREADS, SEED,
)

ROUND_AMOUNTS = np.array([1000, 2000, 5000, 10000, 25000, 50000, 100000])


def _learn_window_stats():
    """Learn typical suspicious activity window from mule_flag_date in training."""
    labels = pd.read_parquet(TRAIN_LABELS_PATH)
    mules = labels[labels["is_mule"] == 1].copy()
    mules["mule_flag_date"] = pd.to_datetime(mules["mule_flag_date"], errors="coerce")
    valid = mules.dropna(subset=["mule_flag_date"])

    if len(valid) == 0:
        return {"median_lookback_days": 60}

    flag_dates = valid["mule_flag_date"]
    ref = pd.Timestamp("2025-06-30")
    days_before_ref = (ref - flag_dates).dt.days
    return {"median_lookback_days": max(int(days_before_ref.median()), 30)}


def _compute_window_scores(ts_sec, abs_amounts, is_credit, is_debit,
                           starts, ends, cum_abs, cum_credit, cum_debit,
                           cum_round, cum_near_50k, cum_credit_amt, cum_debit_amt,
                           baseline_rate, baseline_amt, n, window_days):
    """Vectorized scoring for a batch of windows. Returns (scores, left_idx, right_idx, counts)."""
    left_idx = np.searchsorted(ts_sec, starts, side="left")
    right_idx = np.searchsorted(ts_sec, ends, side="left")
    counts = right_idx - left_idx

    valid_mask = counts >= 2
    if not valid_mask.any():
        return None, valid_mask, None, None, None

    w_counts = counts[valid_mask].astype(np.float64)
    l = left_idx[valid_mask]
    r = right_idx[valid_mask]

    w_abs_sum = cum_abs[r] - cum_abs[l]
    w_abs_mean = w_abs_sum / w_counts
    w_n_credit = cum_credit[r] - cum_credit[l]
    w_n_debit = cum_debit[r] - cum_debit[l]
    w_n_round = cum_round[r] - cum_round[l]
    w_n_near_50k = cum_near_50k[r] - cum_near_50k[l]
    w_credit_sum = cum_credit_amt[r] - cum_credit_amt[l]
    w_debit_sum = cum_debit_amt[r] - cum_debit_amt[l]

    velocity = (w_counts / window_days) / max(baseline_rate, 0.01)
    amt_score = w_abs_mean / max(baseline_amt, 1.0)
    imbalance = np.abs(w_n_credit - w_n_debit) / w_counts
    round_ratio = w_n_round / w_counts
    near_50k_ratio = w_n_near_50k / w_counts
    max_cd = np.maximum(w_credit_sum, w_debit_sum)
    max_cd = np.where(max_cd < 1, 1, max_cd)
    passthrough = np.minimum(w_credit_sum, w_debit_sum) / max_cd
    concentration = w_counts / n

    # Recency bias: windows ending closer to the last transaction score higher
    # Mule activity is typically recent relative to detection
    span_sec = max(ts_sec[-1] - ts_sec[0], 1)
    recency = (ends[valid_mask] - ts_sec[0]).astype(np.float64) / span_sec

    scores = (
        0.22 * np.minimum(velocity, 10) / 10
        + 0.18 * np.minimum(amt_score, 10) / 10
        + 0.14 * imbalance
        + 0.14 * passthrough
        + 0.09 * round_ratio
        + 0.09 * near_50k_ratio
        + 0.04 * concentration
        + 0.10 * recency  # bias toward recent windows
    )

    return scores, valid_mask, starts[valid_mask], ends[valid_mask], w_counts


def _sliding_window_anomaly_fast(ts_sec, abs_amounts, is_credit, is_debit,
                                  window_days=30, stride_days=7):
    """
    Vectorized sliding window anomaly scoring using cumsum + searchsorted.
    V4: Adds recency bias, window refinement, and transaction-boundary clipping.
    ts_sec: sorted int64 array of unix seconds
    """
    n = len(ts_sec)
    if n < 3:
        return None, None, 0.0

    window_sec = window_days * 86400
    stride_sec = stride_days * 86400
    span_sec = max(ts_sec[-1] - ts_sec[0], 1)
    baseline_rate = n / (span_sec / 86400)
    baseline_amt = np.mean(abs_amounts) if n > 0 else 1.0

    # Build window start positions
    starts = np.arange(ts_sec[0], ts_sec[-1] + stride_sec, stride_sec, dtype=np.int64)
    ends = starts + window_sec
    n_windows = len(starts)

    if n_windows == 0:
        return None, None, 0.0

    # Precompute cumulative sums (shared across coarse + refinement passes)
    cum_abs = np.concatenate([[0], np.cumsum(abs_amounts)])
    cum_credit = np.concatenate([[0], np.cumsum(is_credit)])
    cum_debit = np.concatenate([[0], np.cumsum(is_debit)])

    is_round = np.isin(abs_amounts.astype(np.int64), ROUND_AMOUNTS)
    cum_round = np.concatenate([[0], np.cumsum(is_round)])

    is_near_50k = (abs_amounts >= 42500) & (abs_amounts < 50000)
    cum_near_50k = np.concatenate([[0], np.cumsum(is_near_50k)])

    credit_amts = abs_amounts * is_credit
    debit_amts = abs_amounts * is_debit
    cum_credit_amt = np.concatenate([[0], np.cumsum(credit_amts)])
    cum_debit_amt = np.concatenate([[0], np.cumsum(debit_amts)])

    # Coarse pass
    result = _compute_window_scores(
        ts_sec, abs_amounts, is_credit, is_debit,
        starts, ends, cum_abs, cum_credit, cum_debit,
        cum_round, cum_near_50k, cum_credit_amt, cum_debit_amt,
        baseline_rate, baseline_amt, n, window_days,
    )
    if result[0] is None:
        return None, None, 0.0

    scores, valid_mask, valid_starts, valid_ends, w_counts = result
    best_idx = np.argmax(scores)
    best_start_sec = valid_starts[best_idx]
    best_end_sec = valid_ends[best_idx]
    best_score = float(scores[best_idx])

    # Refinement pass: finer stride around the best coarse window
    if stride_days > 2:
        refine_stride = max(stride_days // 3, 1) * 86400
        refine_margin = window_sec  # search one window-width on each side
        ref_starts = np.arange(
            max(best_start_sec - refine_margin, ts_sec[0]),
            min(best_start_sec + refine_margin, ts_sec[-1]) + refine_stride,
            refine_stride, dtype=np.int64,
        )
        ref_ends = ref_starts + window_sec
        if len(ref_starts) > 0:
            ref_result = _compute_window_scores(
                ts_sec, abs_amounts, is_credit, is_debit,
                ref_starts, ref_ends, cum_abs, cum_credit, cum_debit,
                cum_round, cum_near_50k, cum_credit_amt, cum_debit_amt,
                baseline_rate, baseline_amt, n, window_days,
            )
            if ref_result[0] is not None:
                ref_scores = ref_result[0]
                ref_best = np.argmax(ref_scores)
                if float(ref_scores[ref_best]) > best_score:
                    best_start_sec = ref_result[2][ref_best]
                    best_end_sec = ref_result[3][ref_best]
                    best_score = float(ref_scores[ref_best])

    # Clip window to actual transaction boundaries
    # Find transactions within the window and tighten to first/last actual txn
    w_left = np.searchsorted(ts_sec, best_start_sec, side="left")
    w_right = np.searchsorted(ts_sec, best_end_sec, side="left")
    if w_right > w_left and w_left < n:
        # Tighten start to first transaction, end to last transaction in window
        actual_start = ts_sec[w_left]
        actual_end = ts_sec[min(w_right - 1, n - 1)]
        # Add small padding (1 hour) so the window covers the edge transactions
        best_start_sec = actual_start - 3600
        best_end_sec = actual_end + 3600

    best_start = pd.Timestamp(best_start_sec, unit="s")
    best_end = pd.Timestamp(best_end_sec, unit="s")

    return best_start, best_end, best_score


def _process_account(acct_id, ts_sec, abs_amounts, is_credit, is_debit,
                     amounts, txn_types, window_scales, is_window_account):
    """Process one account: temporal features + window prediction."""
    n = len(ts_sec)
    row = {"account_id": acct_id}

    if n < 2:
        return row, None

    # Rapid pass-through: vectorized
    if is_credit.any() and is_debit.any():
        threshold_sec = RAPID_PASSTHROUGH_HOURS * 3600
        credit_ts = ts_sec[is_credit]
        debit_ts = ts_sec[is_debit]
        # For each credit, find first debit after it using searchsorted
        insert_pos = np.searchsorted(debit_ts, credit_ts, side="right")
        valid = insert_pos < len(debit_ts)
        if valid.any():
            diffs = debit_ts[insert_pos[valid]] - credit_ts[valid]
            rapid_count = int((diffs < threshold_sec).sum())
        else:
            rapid_count = 0
        row["rapid_passthrough_count"] = rapid_count
        row["rapid_passthrough_ratio"] = rapid_count / max(int(is_credit.sum()), 1)
    else:
        row["rapid_passthrough_count"] = 0
        row["rapid_passthrough_ratio"] = 0.0

    # Dormancy detection
    time_diffs = np.diff(ts_sec)
    dormancy_threshold = DORMANCY_DAYS * 86400
    dormant_gaps = np.where(time_diffs > dormancy_threshold)[0]
    row["n_dormancy_bursts"] = len(dormant_gaps)

    if len(dormant_gaps) > 0:
        longest_gap_idx = dormant_gaps[np.argmax(time_diffs[dormant_gaps])]
        post_gap_idx = longest_gap_idx + 1
        post_gap_window = min(post_gap_idx + 30, n)
        post_gap_amounts = abs_amounts[post_gap_idx:post_gap_window]
        row["post_dormancy_txn_count"] = len(post_gap_amounts)
        row["post_dormancy_amt_mean"] = float(np.mean(post_gap_amounts)) if len(post_gap_amounts) > 0 else 0
    else:
        row["post_dormancy_txn_count"] = 0
        row["post_dormancy_amt_mean"] = 0.0

    # Salary cycle
    if is_credit.sum() > 5:
        # Extract day-of-month from unix seconds
        credit_days = pd.to_datetime(ts_sec[is_credit], unit="s").day
        salary_window = int(((credit_days <= 5) | (credit_days >= 28)).sum())
        row["salary_cycle_ratio"] = salary_window / int(is_credit.sum())
    else:
        row["salary_cycle_ratio"] = 0.0

    # Recent activity spike
    ref_sec = int(pd.Timestamp("2025-06-30").timestamp())
    recent_mask = ts_sec >= (ref_sec - 30 * 86400)
    recent_count = recent_mask.sum()
    historical_rate = n / max((ts_sec[-1] - ts_sec[0]) / 86400, 1)
    row["recent_30d_velocity_ratio"] = (recent_count / 30) / max(historical_rate, 0.01)

    # Window prediction
    window_row = None
    if is_window_account:
        candidates = []
        for wd, sd in window_scales:
            s, e, sc = _sliding_window_anomaly_fast(
                ts_sec, abs_amounts, is_credit, is_debit, wd, sd
            )
            if s is not None:
                candidates.append((s, e, sc))

        if candidates:
            best = max(candidates, key=lambda x: x[2])
            window_row = {
                "account_id": acct_id,
                "suspicious_start": best[0].isoformat(),
                "suspicious_end": best[1].isoformat(),
                "window_score": best[2],
            }

    return row, window_row


def build_temporal_features_and_windows(mule_predictions=None, mule_threshold=0.3):
    """
    Build temporal features and suspicious windows for predicted mules.
    V3: fully vectorized, processes ~1000 accounts in minutes not hours.
    """
    log.info("Building temporal features (V3 vectorized)")

    if mule_predictions is not None:
        window_accounts = set(
            mule_predictions[mule_predictions > mule_threshold].index
        )
        log.info("Will predict windows for %d accounts (threshold=%.2f)",
                 len(window_accounts), mule_threshold)
    else:
        window_accounts = set()

    if not window_accounts:
        empty = pd.DataFrame(columns=["account_id"]).set_index("account_id")
        empty.to_parquet(TEMPORAL_FEATURES_PATH)
        return empty, pd.DataFrame()

    # Learn typical window patterns
    window_stats = _learn_window_stats()
    median_lb = window_stats["median_lookback_days"]
    window_scales = [(14, 3), (30, 7), (60, 14), (90, 14), (median_lb, 7)]

    # Load transaction data for target accounts
    acct_data = defaultdict(lambda: {"ts": [], "amt": [], "txn_type": []})

    parts = sorted(glob(str(TXN_DIR / "batch-*" / "part_*.parquet")))
    PREFETCH = min(8, N_THREADS)

    def _read_part(path):
        df = pd.read_parquet(
            path, columns=["account_id", "transaction_timestamp", "amount", "txn_type"],
        )
        df = df[df["account_id"].isin(window_accounts)]
        if len(df) == 0:
            return None
        df["ts_sec"] = pd.to_datetime(df["transaction_timestamp"], format="mixed").astype(np.int64) // 10**9
        return df[["account_id", "ts_sec", "amount", "txn_type"]]

    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        for df in tqdm(pool.map(_read_part, parts),
                       total=len(parts), desc="Temporal: loading txns"):
            if df is None or len(df) == 0:
                continue
            for acct_id, grp in df.groupby("account_id"):
                acc = acct_data[acct_id]
                acc["ts"].append(grp["ts_sec"].values)
                acc["amt"].append(grp["amount"].values)
                acc["txn_type"].append(grp["txn_type"].values)
            del df

    log.info("Loaded temporal data for %d accounts", len(acct_data))

    feature_rows = []
    window_rows = []

    for i, (acct_id, acc) in enumerate(tqdm(acct_data.items(), desc="Computing windows")):
        # Concatenate chunks and sort
        ts_sec = np.concatenate(acc["ts"]) if len(acc["ts"]) > 1 else acc["ts"][0]
        amounts = np.concatenate(acc["amt"]) if len(acc["amt"]) > 1 else acc["amt"][0]
        txn_types = np.concatenate(acc["txn_type"]) if len(acc["txn_type"]) > 1 else acc["txn_type"][0]

        sort_idx = np.argsort(ts_sec)
        ts_sec = ts_sec[sort_idx]
        amounts = amounts[sort_idx]
        txn_types = txn_types[sort_idx]

        abs_amounts = np.abs(amounts).astype(np.float64)
        is_credit = (txn_types == "C")
        is_debit = (txn_types == "D")

        row, window_row = _process_account(
            acct_id, ts_sec, abs_amounts, is_credit, is_debit,
            amounts, txn_types, window_scales, acct_id in window_accounts
        )
        feature_rows.append(row)
        if window_row:
            window_rows.append(window_row)

        if (i + 1) % 200 == 0:
            log.info("  Processed %d/%d accounts, %d windows found",
                     i + 1, len(acct_data), len(window_rows))

    temporal_features = pd.DataFrame(feature_rows)
    if "account_id" in temporal_features.columns:
        temporal_features = temporal_features.set_index("account_id")
    temporal_features = temporal_features.fillna(0)
    temporal_features.to_parquet(TEMPORAL_FEATURES_PATH)
    log.info("Saved temporal features: %s", temporal_features.shape)

    windows = pd.DataFrame(window_rows)
    if len(windows) > 0:
        windows.to_parquet(OUTPUT_DIR / "suspicious_windows.parquet", index=False)
        log.info("Saved suspicious windows for %d accounts", len(windows))

    return temporal_features, windows
