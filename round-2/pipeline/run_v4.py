"""
NFPC Phase 2 — Pipeline V4
Uses models_v4: freq+target encoding, feature interactions, no-weights test.
Re-uses existing features and label cleaning.
"""
import sys, time, argparse
import numpy as np, pandas as pd, gc

from config import (
    OUTPUT_DIR, FEATURES_DIR, MODELS_DIR, FULL_FEATURES_PATH,
    TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH, log,
)


def _stage_done(name):
    return (OUTPUT_DIR / f".stage_{name}_done").exists()

def _mark_done(name):
    (OUTPUT_DIR / f".stage_{name}_done").touch()

def _clear_stage(name):
    m = OUTPUT_DIR / f".stage_{name}_done"
    if m.exists(): m.unlink()


def run_pipeline(skip_optuna=False, force_stage=None):
    t0 = time.time()

    # Stage 1: Features (cached)
    if not _stage_done("features"):
        log.info("══════ STAGE 1: Feature Engineering ══════")
        from features import build_all_features
        features = build_all_features()
        _mark_done("features")
    else:
        log.info("Stage 1 (features) — cached")
        features = pd.read_parquet(FULL_FEATURES_PATH)
    gc.collect()

    # Stage 2: Label Cleaning (cached)
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
    else:
        log.info("Stage 2 (labels) — cached")
        if weights_path.exists():
            sample_weights = np.load(weights_path)
    gc.collect()

    # Stage 3: Model Training (V4 — always re-run)
    log.info("══════ STAGE 3: Model Training (V4) ══════")
    from models_v4 import train_and_predict
    results = train_and_predict(features, sample_weights=sample_weights, skip_optuna=skip_optuna)
    log.info("Stage 3: AUC=%.5f (%s)", results["oof_auc"], results["ensemble_method"])
    gc.collect()

    # Stage 4: Temporal Windows
    _clear_stage("temporal")
    log.info("══════ STAGE 4: Temporal Windows ══════")
    from temporal import build_temporal_features_and_windows
    mule_preds = results.get("mule_predictions")
    _, windows = build_temporal_features_and_windows(
        mule_predictions=mule_preds,
        mule_threshold=results.get("best_threshold", 0.3),
    )
    _mark_done("temporal")
    log.info("Stage 4: %d windows", len(windows))
    gc.collect()

    # Stage 5: Submission
    log.info("══════ STAGE 5: Submission ══════")
    test_accounts = pd.read_parquet(TEST_ACCOUNTS_PATH)
    test_ids = test_accounts["account_id"].values
    submission = pd.DataFrame({"account_id": test_ids})

    pred_series = results["mule_predictions"]
    submission["is_mule"] = submission["account_id"].map(pred_series).fillna(0.0)

    submission["suspicious_start"] = ""
    submission["suspicious_end"] = ""
    if len(windows) > 0:
        window_dict = windows.set_index("account_id").to_dict("index")
        for idx, row in submission.iterrows():
            acct = row["account_id"]
            if acct in window_dict:
                submission.at[idx, "suspicious_start"] = window_dict[acct].get("suspicious_start", "")
                submission.at[idx, "suspicious_end"] = window_dict[acct].get("suspicious_end", "")

    submission.to_parquet(OUTPUT_DIR / "submission_raw.parquet", index=False)
    submission.to_csv(OUTPUT_DIR / "submission_v4.csv", index=False)
    submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

    n30 = (submission["is_mule"] >= 0.3).sum()
    n50 = (submission["is_mule"] >= 0.5).sum()
    nw = (submission["suspicious_start"] != "").sum()
    log.info("Submission: %d rows, p>=0.3: %d, p>=0.5: %d, windows: %d", len(submission), n30, n50, nw)

    elapsed = time.time() - t0
    log.info("V4 Pipeline complete in %.1f minutes", elapsed / 60)
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--force-stage", type=str, default=None,
                        choices=["features", "labels", "models", "temporal"])
    args = parser.parse_args()
    run_pipeline(skip_optuna=args.skip_optuna, force_stage=args.force_stage)
