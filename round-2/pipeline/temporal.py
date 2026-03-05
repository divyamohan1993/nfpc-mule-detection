"""
NFPC Phase 2 — Temporal Window Prediction

For accounts predicted as mules, identify the suspicious activity window
(suspicious_start, suspicious_end) for temporal IoU scoring.

Research-grade approaches:
  1. Sliding window anomaly scoring — score each time window by
     transaction behavior deviation from account baseline
  2. Change-point detection (PELT via ruptures) — detect regime changes
     in transaction patterns
  3. Combined scoring — merge sliding window + change-point signals
  4. Rapid pass-through detection — credit→debit time deltas within window

The temporal features are also extracted per-account for use in the
main classification model.
"""
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from tqdm import tqdm

from config import (
    TXN_DIR, TEMPORAL_FEATURES_PATH, OUTPUT_DIR,
    RAPID_PASSTHROUGH_HOURS, DORMANCY_DAYS, STRUCTURING_THRESHOLDS,
    log,
)


def _sliding_window_anomaly(
    timestamps: np.ndarray,
    amounts: np.ndarray,
    txn_types: np.ndarray,
    window_days: int = 30,
    stride_days: int = 7,
) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """
    Find the time window with highest anomaly concentration.

    Anomaly score per window = weighted combination of:
      - Transaction velocity (count / window_days)
      - Amount intensity (sum / baseline_mean)
      - Credit-debit imbalance
      - Round amount ratio
      - Structuring ratio (near-threshold)

    Returns (start, end, peak_score).
    """
    if len(timestamps) < 3:
        return None, None, 0.0

    ts = pd.to_datetime(timestamps)
    sort_idx = ts.argsort()
    ts = ts[sort_idx]
    amounts = amounts[sort_idx]
    txn_types = txn_types[sort_idx]

    # Baseline stats (full history)
    span_days = max((ts.max() - ts.min()).days, 1)
    baseline_rate = len(ts) / span_days  # txn/day
    baseline_amt = np.mean(np.abs(amounts)) if len(amounts) > 0 else 1.0

    best_score = 0.0
    best_start = None
    best_end = None

    window = pd.Timedelta(days=window_days)
    stride = pd.Timedelta(days=stride_days)

    t = ts.min()
    while t + window <= ts.max() + stride:
        mask = (ts >= t) & (ts < t + window)
        if mask.sum() < 2:
            t += stride
            continue

        w_amounts = amounts[mask]
        w_types = txn_types[mask]
        w_count = mask.sum()

        # Velocity anomaly
        velocity_score = (w_count / window_days) / max(baseline_rate, 0.01)

        # Amount intensity
        amt_score = np.mean(np.abs(w_amounts)) / max(baseline_amt, 1.0)

        # Credit-debit imbalance
        n_credit = (w_types == "C").sum()
        n_debit = (w_types == "D").sum()
        imbalance = abs(n_credit - n_debit) / max(w_count, 1)

        # Round amount ratio
        abs_w = np.abs(w_amounts)
        round_ratio = np.sum(np.isin(abs_w.astype(int), [1000, 2000, 5000, 10000, 25000, 50000, 100000])) / max(w_count, 1)

        # Near-threshold ratio
        near_50k = np.sum((abs_w >= 42500) & (abs_w < 50000)) / max(w_count, 1)

        # Combined score
        score = (
            0.35 * min(velocity_score, 10) / 10
            + 0.25 * min(amt_score, 10) / 10
            + 0.15 * imbalance
            + 0.15 * round_ratio
            + 0.10 * near_50k
        )

        if score > best_score:
            best_score = score
            best_start = t
            best_end = t + window

        t += stride

    return best_start, best_end, best_score


def _detect_change_points(
    timestamps: np.ndarray,
    amounts: np.ndarray,
    min_size: int = 10,
) -> list[int]:
    """
    Detect change points in transaction amount time series using PELT.
    Falls back to simple variance-based detection if ruptures not available.
    """
    if len(amounts) < min_size * 2:
        return []

    try:
        import ruptures as rpt
        signal = np.abs(amounts).reshape(-1, 1)
        algo = rpt.Pelt(model="rbf", min_size=min_size).fit(signal)
        change_points = algo.predict(pen=10)
        return change_points[:-1]  # last element is len(signal)
    except ImportError:
        # Fallback: simple rolling variance change detection
        abs_amounts = np.abs(amounts)
        if len(abs_amounts) < 20:
            return []
        rolling_var = pd.Series(abs_amounts).rolling(10).var().values
        rolling_var = rolling_var[~np.isnan(rolling_var)]
        if len(rolling_var) < 2:
            return []
        threshold = np.mean(rolling_var) + 2 * np.std(rolling_var)
        change_points = np.where(rolling_var > threshold)[0].tolist()
        return change_points


def build_temporal_features_and_windows(
    mule_predictions: pd.Series = None,
    mule_threshold: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build temporal features for all accounts and suspicious windows for predicted mules.

    Args:
        mule_predictions: Series indexed by account_id with mule probability
        mule_threshold: predict windows for accounts above this threshold

    Returns:
        temporal_features: DataFrame of temporal features per account
        windows: DataFrame with suspicious_start, suspicious_end per predicted mule
    """
    log.info("Building temporal features")

    # Determine which accounts need window prediction
    if mule_predictions is not None:
        window_accounts = set(
            mule_predictions[mule_predictions > mule_threshold].index
        )
        log.info("Will predict windows for %d accounts (threshold=%.2f)",
                 len(window_accounts), mule_threshold)
    else:
        window_accounts = set()

    # If no accounts need window prediction, skip the expensive full scan
    if not window_accounts:
        log.info("No accounts need window prediction — skipping temporal scan")
        empty_features = pd.DataFrame(columns=["account_id"]).set_index("account_id")
        empty_features.to_parquet(TEMPORAL_FEATURES_PATH)
        return empty_features, pd.DataFrame()

    # Only accumulate data for accounts that need window prediction (memory-safe)
    acct_data = defaultdict(lambda: {
        "timestamps": [], "amounts": [], "txn_types": [], "channels": [],
    })

    parts = sorted(glob(str(TXN_DIR / "batch-*" / "part_*.parquet")))
    for part_path in tqdm(parts, desc="Temporal: scanning transactions"):
        df = pd.read_parquet(
            part_path,
            columns=["account_id", "transaction_timestamp", "amount", "txn_type", "channel"],
        )
        df["transaction_timestamp"] = pd.to_datetime(df["transaction_timestamp"], format="mixed")

        # Only keep rows for target accounts
        df = df[df["account_id"].isin(window_accounts)]
        if len(df) == 0:
            continue

        for acct_id, grp in df.groupby("account_id"):
            acc = acct_data[acct_id]
            acc["timestamps"].extend(grp["transaction_timestamp"].values.tolist())
            acc["amounts"].extend(grp["amount"].values.tolist())
            acc["txn_types"].extend(grp["txn_type"].values.tolist())
            acc["channels"].extend(grp["channel"].values.tolist())
        del df

    log.info("Loaded temporal data for %d accounts (of %d requested)",
             len(acct_data), len(window_accounts))

    # ── Compute temporal features ─────────────────────────
    feature_rows = []
    window_rows = []

    for acct_id, acc in tqdm(acct_data.items(), desc="Computing temporal features"):
        ts = np.array(acc["timestamps"])
        amounts = np.array(acc["amounts"])
        txn_types = np.array(acc["txn_types"])
        channels = np.array(acc["channels"])

        if len(ts) < 2:
            feature_rows.append({"account_id": acct_id})
            continue

        sort_idx = np.argsort(ts)
        ts = ts[sort_idx]
        amounts = amounts[sort_idx]
        txn_types = txn_types[sort_idx]
        channels = channels[sort_idx]

        row = {"account_id": acct_id}

        # ── Rapid pass-through detection ──────────────
        credit_mask = txn_types == "C"
        debit_mask = txn_types == "D"
        credit_ts = pd.to_datetime(ts[credit_mask])
        debit_ts = pd.to_datetime(ts[debit_mask])

        if len(credit_ts) > 0 and len(debit_ts) > 0:
            # For each credit, find time to next debit
            rapid_count = 0
            rapid_amounts = []
            threshold_sec = RAPID_PASSTHROUGH_HOURS * 3600

            credit_arr = credit_ts.values.astype("datetime64[s]").astype(np.int64)
            debit_arr = debit_ts.values.astype("datetime64[s]").astype(np.int64)

            for ct in credit_arr:
                next_debits = debit_arr[debit_arr > ct]
                if len(next_debits) > 0 and (next_debits[0] - ct) < threshold_sec:
                    rapid_count += 1

            row["rapid_passthrough_count"] = rapid_count
            row["rapid_passthrough_ratio"] = rapid_count / max(len(credit_ts), 1)
        else:
            row["rapid_passthrough_count"] = 0
            row["rapid_passthrough_ratio"] = 0.0

        # ── Dormancy burst detection ──────────────────
        ts_pd = pd.to_datetime(ts)
        time_diffs = np.diff(ts_pd).astype("timedelta64[s]").astype(float)
        dormancy_threshold = DORMANCY_DAYS * 86400

        dormant_gaps = np.where(time_diffs > dormancy_threshold)[0]
        row["n_dormancy_bursts"] = len(dormant_gaps)

        if len(dormant_gaps) > 0:
            # Activity burst after longest dormancy
            longest_gap_idx = dormant_gaps[np.argmax(time_diffs[dormant_gaps])]
            post_gap_idx = longest_gap_idx + 1
            post_gap_window = min(post_gap_idx + 30, len(ts))
            post_gap_amounts = np.abs(amounts[post_gap_idx:post_gap_window])
            row["post_dormancy_txn_count"] = len(post_gap_amounts)
            row["post_dormancy_amt_mean"] = float(np.mean(post_gap_amounts)) if len(post_gap_amounts) > 0 else 0
            row["post_dormancy_intensity"] = row["post_dormancy_txn_count"] * row["post_dormancy_amt_mean"]
        else:
            row["post_dormancy_txn_count"] = 0
            row["post_dormancy_amt_mean"] = 0.0
            row["post_dormancy_intensity"] = 0.0

        # ── Change point features ─────────────────────
        change_points = _detect_change_points(ts, amounts)
        row["n_change_points"] = len(change_points)

        # ── Salary cycle detection ────────────────────
        # Check if credits cluster around month boundaries (days 1-5, 28-31)
        ts_pd_series = pd.to_datetime(ts)
        credit_days = ts_pd_series[credit_mask].day
        if len(credit_days) > 5:
            salary_window = ((credit_days <= 5) | (credit_days >= 28)).sum()
            row["salary_cycle_ratio"] = salary_window / len(credit_days)
        else:
            row["salary_cycle_ratio"] = 0.0

        # ── Recent activity spike ─────────────────────
        # Compare last 30 days to historical average
        ref_date = pd.Timestamp("2025-06-30")
        recent_mask = ts_pd_series >= (ref_date - pd.Timedelta(days=30))
        recent_count = recent_mask.sum()
        historical_rate = len(ts) / max((ts_pd_series.max() - ts_pd_series.min()).days, 1)
        row["recent_30d_velocity_ratio"] = (recent_count / 30) / max(historical_rate, 0.01)

        feature_rows.append(row)

        # ── Window prediction for predicted mules ─────
        if acct_id in window_accounts:
            start, end, score = _sliding_window_anomaly(
                ts, amounts, txn_types,
                window_days=30, stride_days=7,
            )
            # Also try wider windows
            start_60, end_60, score_60 = _sliding_window_anomaly(
                ts, amounts, txn_types,
                window_days=60, stride_days=14,
            )
            start_90, end_90, score_90 = _sliding_window_anomaly(
                ts, amounts, txn_types,
                window_days=90, stride_days=14,
            )

            # Pick best window
            candidates = [
                (start, end, score),
                (start_60, end_60, score_60),
                (start_90, end_90, score_90),
            ]
            best = max(candidates, key=lambda x: x[2])

            if best[0] is not None:
                window_rows.append({
                    "account_id": acct_id,
                    "suspicious_start": best[0].isoformat(),
                    "suspicious_end": best[1].isoformat(),
                    "window_score": best[2],
                })

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


if __name__ == "__main__":
    build_temporal_features_and_windows()
