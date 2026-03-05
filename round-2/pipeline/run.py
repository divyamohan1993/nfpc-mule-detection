"""
NFPC Phase 2 — End-to-End Orchestrator

Runs the complete mule detection pipeline:
  1. Feature engineering (3 passes + graph + temporal)
  2. Label cleaning (confident learning + heuristic)
  3. Model training (LGB + XGB + CatBoost ensemble)
  4. Probability calibration + ensemble
  5. Temporal window prediction for mules
  6. Submission file generation

Usage:
  python run.py                  # Full pipeline
  python run.py --skip-features  # Skip feature engineering (use cached)
  python run.py --skip-graph     # Skip graph features (slow)
  python run.py --stage features # Run only feature engineering
  python run.py --stage train    # Run only training (features must exist)
  python run.py --stage predict  # Run only prediction (models must exist)
"""
import argparse
import time
import numpy as np
import pandas as pd

from config import (
    TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH, FULL_FEATURES_PATH,
    GRAPH_FEATURES_PATH, OUTPUT_DIR, MODELS_DIR, log,
)


def stage_features(skip_graph: bool = False):
    """Run all feature engineering."""
    from features import build_txn_features, build_txn_additional_features, build_static_features, build_all_features

    log.info("=" * 60)
    log.info("STAGE 1: FEATURE ENGINEERING")
    log.info("=" * 60)

    t0 = time.time()

    # Pass 1: Transaction features
    log.info("── Pass 1: Transaction features ──")
    build_txn_features()

    # Pass 2: Transaction additional features
    log.info("── Pass 2: Transaction additional features ──")
    build_txn_additional_features()

    # Pass 3: Static features
    log.info("── Pass 3: Static features ──")
    build_static_features()

    # Graph features (optional — slow but valuable)
    if not skip_graph:
        log.info("── Graph features ──")
        from graph_features import build_graph_features
        build_graph_features()
    else:
        log.info("Skipping graph features (--skip-graph)")

    # Temporal features skipped here — computed during prediction stage
    # for predicted mule accounts only (avoids OOM on 400M transactions)
    log.info("Temporal features deferred to prediction stage")

    # Combine all
    log.info("── Combining all features ──")
    features = build_all_features()

    elapsed = time.time() - t0
    log.info("Feature engineering complete: %s (%d features, %.1f min)",
             features.shape, features.shape[1], elapsed / 60)

    return features


def stage_train(features: pd.DataFrame = None):
    """Run label cleaning and model training."""
    from label_cleaning import compute_sample_weights
    from models import (
        adversarial_validation, train_ensemble, calibrate_and_ensemble,
        save_models, run_shap_analysis,
    )

    log.info("=" * 60)
    log.info("STAGE 2: MODEL TRAINING")
    log.info("=" * 60)

    t0 = time.time()

    # Load data
    if features is None:
        features = pd.read_parquet(FULL_FEATURES_PATH)
    train_labels = pd.read_parquet(TRAIN_LABELS_PATH)
    test_accounts = pd.read_parquet(TEST_ACCOUNTS_PATH)

    # Split train/test features
    train_ids = train_labels["account_id"].values
    test_ids = test_accounts["account_id"].values

    X_train = features.loc[features.index.isin(train_ids)].copy()
    X_test = features.loc[features.index.isin(test_ids)].copy()

    # Align labels
    labels_indexed = train_labels.set_index("account_id")
    y_train = labels_indexed.loc[X_train.index, "is_mule"]

    log.info("Train: %s, Test: %s", X_train.shape, X_test.shape)
    log.info("Mule rate: %.4f (%d mules / %d total)",
             y_train.mean(), y_train.sum(), len(y_train))

    # Adversarial validation
    adversarial_validation(X_train, X_test)

    # Label cleaning → sample weights
    log.info("── Label cleaning ──")
    sample_weights = compute_sample_weights(
        labels_indexed.loc[X_train.index].reset_index(),
        X_train,
        y_train,
    )

    # Train ensemble
    log.info("── Training ensemble ──")
    results = train_ensemble(X_train, y_train, sample_weights=sample_weights)

    # Calibrate + ensemble
    ensemble_oof = calibrate_and_ensemble(results, X_train, y_train)

    # SHAP analysis
    run_shap_analysis(results, X_train)

    # Save everything
    save_models(results)

    elapsed = time.time() - t0
    log.info("Training complete (%.1f min)", elapsed / 60)

    return results, X_test


def stage_predict(results: dict = None, X_test: pd.DataFrame = None):
    """Generate predictions and submission file."""
    from models import predict_test
    from temporal import build_temporal_features_and_windows

    log.info("=" * 60)
    log.info("STAGE 3: PREDICTION & SUBMISSION")
    log.info("=" * 60)

    t0 = time.time()

    # Load if not provided
    if results is None or X_test is None:
        import joblib
        features = pd.read_parquet(FULL_FEATURES_PATH)
        test_accounts = pd.read_parquet(TEST_ACCOUNTS_PATH)
        test_ids = test_accounts["account_id"].values
        X_test = features.loc[features.index.isin(test_ids)].copy()

        # Reconstruct results
        results = {"models": {}, "calibrators": None, "ensemble_weights": None}
        results["calibrators"] = joblib.load(MODELS_DIR / "calibrators.joblib")
        results["ensemble_weights"] = joblib.load(MODELS_DIR / "ensemble_weights.joblib")

        for model_type in results["ensemble_weights"].keys():
            fold_models = []
            for i in range(5):
                path = MODELS_DIR / f"{model_type}_fold{i}.joblib"
                fold_models.append(joblib.load(path))
            results["models"][model_type] = fold_models

    # Generate test predictions
    test_preds = predict_test(results, X_test)

    # Create predictions series for temporal window detection
    pred_series = pd.Series(test_preds, index=X_test.index, name="is_mule")

    # Temporal window prediction for high-probability mules
    log.info("── Temporal window prediction ──")
    _, windows = build_temporal_features_and_windows(
        mule_predictions=pred_series,
        mule_threshold=0.3,
    )

    # Build submission
    log.info("── Building submission ──")
    submission = pd.DataFrame({
        "account_id": X_test.index,
        "is_mule": test_preds,
    })

    # Merge windows
    if len(windows) > 0:
        submission = submission.merge(
            windows[["account_id", "suspicious_start", "suspicious_end"]],
            on="account_id",
            how="left",
        )
    else:
        submission["suspicious_start"] = ""
        submission["suspicious_end"] = ""

    submission["suspicious_start"] = submission["suspicious_start"].fillna("")
    submission["suspicious_end"] = submission["suspicious_end"].fillna("")

    # Save
    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    log.info("Saved submission to %s", submission_path)
    log.info("Submission shape: %s", submission.shape)
    log.info("Predicted mules (>0.5): %d", (test_preds > 0.5).sum())
    log.info("Predicted mules (>0.3): %d", (test_preds > 0.3).sum())
    log.info("Mean prediction: %.4f", test_preds.mean())

    elapsed = time.time() - t0
    log.info("Prediction complete (%.1f min)", elapsed / 60)

    # Summary
    log.info("=" * 60)
    log.info("PIPELINE COMPLETE")
    log.info("=" * 60)
    log.info("Submission: %s", submission_path)
    log.info("Models: %s", MODELS_DIR)
    log.info("Features: %s", FULL_FEATURES_PATH)

    return submission


def main():
    parser = argparse.ArgumentParser(description="NFPC Phase 2 Pipeline")
    parser.add_argument("--stage", choices=["features", "train", "predict", "all"],
                        default="all", help="Pipeline stage to run")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Skip graph feature extraction (slow)")
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature engineering (use cached)")
    args = parser.parse_args()

    log.info("NFPC Phase 2 Pipeline — Starting")
    log.info("Stage: %s", args.stage)
    pipeline_start = time.time()

    if args.stage == "features":
        stage_features(skip_graph=args.skip_graph)

    elif args.stage == "train":
        features = None
        if not args.skip_features:
            features = stage_features(skip_graph=args.skip_graph)
        stage_train(features)

    elif args.stage == "predict":
        stage_predict()

    elif args.stage == "all":
        features = None
        if not args.skip_features:
            features = stage_features(skip_graph=args.skip_graph)
        results, X_test = stage_train(features)
        stage_predict(results, X_test)

    total_elapsed = time.time() - pipeline_start
    log.info("Total pipeline time: %.1f min", total_elapsed / 60)


if __name__ == "__main__":
    main()
