"""
NFPC Phase 2 — Pipeline Orchestrator (V5)

Uses models_v5: V3 base + feature interactions for mule detection.
Re-uses existing features (cached) and label cleaning from V2.
"""
import sys
import time
import argparse
import numpy as np
import pandas as pd
import gc

from config import (
    OUTPUT_DIR, FEATURES_DIR, MODELS_DIR, FULL_FEATURES_PATH,
    TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH, log,
)


def _stage_done(name):
    marker = OUTPUT_DIR / f".stage_{name}_done"
    return marker.exists()


def _mark_done(name):
    marker = OUTPUT_DIR / f".stage_{name}_done"
    marker.touch()


def _clear_stage(name):
    marker = OUTPUT_DIR / f".stage_{name}_done"
    if marker.exists():
        marker.unlink()


def run_pipeline(skip_optuna=False, force_stage=None):
    """Run V3 pipeline — re-uses features, re-trains models."""
    t0 = time.time()

    # ── Stage 1: Features (re-use from V2) ──
    if force_stage == "features":
        _clear_stage("features")

    if not _stage_done("features"):
        log.info("══════ STAGE 1: Feature Engineering ══════")
        from features import build_all_features
        features = build_all_features()
        _mark_done("features")
        log.info("Stage 1 complete: %s features", features.shape)
    else:
        log.info("Stage 1 (features) already done — loading from cache")
        features = pd.read_parquet(FULL_FEATURES_PATH)
    gc.collect()

    # ── Stage 2: Label Cleaning (re-use from V2) ──
    if force_stage == "labels":
        _clear_stage("labels")

    sample_weights = None
    weights_path = OUTPUT_DIR / "sample_weights.npy"
    if not _stage_done("labels"):
        log.info("══════ STAGE 2: Label Cleaning ══════")
        from label_cleaning import compute_sample_weights

        labels = pd.read_parquet(TRAIN_LABELS_PATH)
        train_ids = labels["account_id"].values
        X_train = features.loc[features.index.isin(train_ids)].copy()
        labels_aligned = labels.set_index("account_id").loc[X_train.index].reset_index()
        y_train = labels_aligned.set_index("account_id")["is_mule"]

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        sample_weights = compute_sample_weights(labels_aligned, X_train[numeric_cols], y_train)
        np.save(weights_path, sample_weights)
        _mark_done("labels")
        log.info("Stage 2 complete: %d sample weights computed", len(sample_weights))
    else:
        log.info("Stage 2 (labels) already done — loading from cache")
        if weights_path.exists():
            sample_weights = np.load(weights_path)
    gc.collect()

    # ── Stage 3: Model Training (V3 — always re-run) ──
    # Always clear and re-run models with V3
    _clear_stage("models_v5")

    log.info("══════ STAGE 3: Model Training (V3) ══════")
    from models_v5 import train_and_predict
    results = train_and_predict(
        features,
        sample_weights=sample_weights,
        skip_optuna=skip_optuna,
    )
    _mark_done("models_v5")
    log.info("Stage 3 complete: ensemble AUC=%.5f (%s)",
             results["oof_auc"], results["ensemble_method"])
    gc.collect()

    # ── Stage 4: Temporal Windows ──
    if force_stage == "temporal":
        _clear_stage("temporal")

    # Always re-run temporal with new predictions
    _clear_stage("temporal")

    log.info("══════ STAGE 4: Temporal Window Prediction ══════")
    from temporal import build_temporal_features_and_windows

    mule_preds = results.get("mule_predictions")
    _, windows = build_temporal_features_and_windows(
        mule_predictions=mule_preds,
        mule_threshold=results.get("best_threshold", 0.3),
    )
    _mark_done("temporal")
    log.info("Stage 4 complete: %d windows predicted", len(windows))
    gc.collect()

    # ── Stage 5: Submission ──
    log.info("══════ STAGE 5: Submission Generation ══════")
    test_accounts = pd.read_parquet(TEST_ACCOUNTS_PATH)
    test_ids = test_accounts["account_id"].values

    submission = pd.DataFrame({"account_id": test_ids})

    # Merge predictions
    if "mule_predictions" in results:
        pred_series = results["mule_predictions"]
        submission["is_mule"] = submission["account_id"].map(pred_series).fillna(0.0)
    else:
        preds = pd.Series(results["test_predictions"], index=results["test_accounts"])
        submission["is_mule"] = submission["account_id"].map(preds).fillna(0.0)

    # Merge temporal windows
    submission["suspicious_start"] = ""
    submission["suspicious_end"] = ""

    if len(windows) > 0:
        window_dict = windows.set_index("account_id").to_dict("index")
        for idx, row in submission.iterrows():
            acct = row["account_id"]
            if acct in window_dict:
                submission.at[idx, "suspicious_start"] = window_dict[acct].get("suspicious_start", "")
                submission.at[idx, "suspicious_end"] = window_dict[acct].get("suspicious_end", "")

    # Save raw predictions (for cache)
    submission.to_parquet(OUTPUT_DIR / "submission_raw.parquet", index=False)

    # Save final CSV
    submission.to_csv(OUTPUT_DIR / "submission_v5.csv", index=False)

    # Also save as submission.csv for backward compat
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    # Stats
    n_mules_30 = (submission["is_mule"] >= 0.3).sum()
    n_mules_50 = (submission["is_mule"] >= 0.5).sum()
    n_windows = (submission["suspicious_start"] != "").sum()
    log.info("Submission saved: %d rows", len(submission))
    log.info("  Predicted mules (p>=0.3): %d (%.1f%%)", n_mules_30, 100 * n_mules_30 / len(submission))
    log.info("  Predicted mules (p>=0.5): %d (%.1f%%)", n_mules_50, 100 * n_mules_50 / len(submission))
    log.info("  Accounts with temporal windows: %d", n_windows)

    elapsed = time.time() - t0
    log.info("V5 Pipeline complete in %.1f minutes", elapsed / 60)
    log.info("Output: %s", OUTPUT_DIR / "submission_v5.csv")

    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFPC Phase 2 Pipeline V5")
    parser.add_argument("--skip-optuna", action="store_true",
                        help="Skip Optuna HPO (use saved/default params)")
    parser.add_argument("--force-stage", type=str, default=None,
                        choices=["features", "labels", "models", "temporal"],
                        help="Force re-run a specific stage")
    args = parser.parse_args()
    run_pipeline(skip_optuna=args.skip_optuna, force_stage=args.force_stage)
