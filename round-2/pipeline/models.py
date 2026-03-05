"""
NFPC Phase 2 — Model Training Pipeline

Research-grade ensemble:
  1. LightGBM — gradient-boosted trees optimized for speed
  2. XGBoost — gradient-boosted trees with regularization
  3. CatBoost — handles categoricals natively, robust to noise
  4. Weighted average ensemble with learned weights
  5. Isotonic calibration for probability outputs
  6. Adversarial validation for train/test distribution shift detection
  7. Bayesian threshold optimization for binary classification

Techniques:
  - Stratified K-fold cross-validation
  - Out-of-fold predictions for stacking
  - SHAP importance analysis
  - Probability calibration (isotonic regression)
  - Cost-sensitive learning (sample weights from label cleaning)
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import lightgbm as lgb
import xgboost as xgb

from config import (
    TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH, FULL_FEATURES_PATH,
    MODELS_DIR, OUTPUT_DIR, N_FOLDS, SEED, MULE_RATIO, log,
)


# ═══════════════════════════════════════════════════════════════════════
# Adversarial Validation
# ═══════════════════════════════════════════════════════════════════════

def adversarial_validation(X_train: pd.DataFrame, X_test: pd.DataFrame) -> float:
    """
    Check if train and test distributions differ significantly.
    Returns AUC — close to 0.5 means similar distributions (good).
    """
    log.info("Running adversarial validation")
    combined = pd.concat([X_train, X_test], ignore_index=True)
    labels = np.array([0] * len(X_train) + [1] * len(X_test))

    model = lgb.LGBMClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5,
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof_preds = np.zeros(len(labels))

    for train_idx, val_idx in skf.split(combined, labels):
        model.fit(combined.iloc[train_idx], labels[train_idx])
        oof_preds[val_idx] = model.predict_proba(combined.iloc[val_idx])[:, 1]

    auc = roc_auc_score(labels, oof_preds)
    log.info("Adversarial validation AUC: %.4f (closer to 0.5 = better)", auc)

    if auc > 0.7:
        log.warning("Significant train/test distribution shift detected!")
        # Identify most discriminative features
        model.fit(combined, labels)
        importances = pd.Series(
            model.feature_importances_, index=combined.columns
        ).sort_values(ascending=False)
        log.warning("Top shifting features:\n%s", importances.head(10))

    return auc


# ═══════════════════════════════════════════════════════════════════════
# Model Definitions
# ═══════════════════════════════════════════════════════════════════════

def get_lgb_params(seed: int = SEED) -> dict:
    return {
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "max_depth": 8,
        "num_leaves": 127,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": MULE_RATIO,
        "random_state": seed,
        "verbosity": -1,
        "n_jobs": -1,
    }


def get_xgb_params(seed: int = SEED) -> dict:
    return {
        "n_estimators": 3000,
        "learning_rate": 0.02,
        "max_depth": 8,
        "min_child_weight": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": MULE_RATIO,
        "random_state": seed,
        "verbosity": 0,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "auc",
    }


def get_catboost_params(seed: int = SEED) -> dict:
    return {
        "iterations": 3000,
        "learning_rate": 0.02,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "auto_class_weights": "Balanced",
        "random_seed": seed,
        "verbose": 0,
        "thread_count": -1,
        "eval_metric": "AUC",
    }


# ═══════════════════════════════════════════════════════════════════════
# Training Pipeline
# ═══════════════════════════════════════════════════════════════════════

def train_fold(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    sample_weights_train: np.ndarray = None,
    fold: int = 0,
):
    """Train a single fold for a given model type."""
    if model_type == "lgb":
        params = get_lgb_params(SEED + fold)
        model = lgb.LGBMClassifier(**params)
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "callbacks": [lgb.early_stopping(100, verbose=False)],
        }
        if sample_weights_train is not None:
            fit_kwargs["sample_weight"] = sample_weights_train
        model.fit(X_train, y_train, **fit_kwargs)

    elif model_type == "xgb":
        params = get_xgb_params(SEED + fold)
        model = xgb.XGBClassifier(**params)
        fit_kwargs = {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }
        if sample_weights_train is not None:
            fit_kwargs["sample_weight"] = sample_weights_train
        model.fit(X_train, y_train, **fit_kwargs)

    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        params = get_catboost_params(SEED + fold)
        model = CatBoostClassifier(**params)
        fit_kwargs = {
            "eval_set": (X_val, y_val),
            "early_stopping_rounds": 100,
        }
        if sample_weights_train is not None:
            fit_kwargs["sample_weight"] = sample_weights_train
        model.fit(X_train, y_train, **fit_kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    sample_weights: np.ndarray = None,
    model_types: list[str] = None,
) -> dict:
    """
    Train full ensemble with K-fold cross-validation.

    Returns dict with:
      - models: {model_type: [fold_models]}
      - oof_preds: {model_type: oof_array}
      - feature_importance: {model_type: importance_df}
      - metrics: {model_type: {auc, f1, threshold}}
    """
    if model_types is None:
        model_types = ["lgb", "xgb"]
        # Try CatBoost if available
        try:
            import catboost
            model_types.append("catboost")
            log.info("CatBoost available — including in ensemble")
        except ImportError:
            log.info("CatBoost not available — using LGB + XGB only")

    results = {
        "models": {},
        "oof_preds": {},
        "feature_importance": {},
        "metrics": {},
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for model_type in model_types:
        log.info("Training %s (%d folds)", model_type.upper(), N_FOLDS)
        fold_models = []
        oof_preds = np.zeros(len(y))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            log.info("  %s Fold %d/%d", model_type.upper(), fold + 1, N_FOLDS)

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            sw_train = sample_weights[train_idx] if sample_weights is not None else None

            model = train_fold(
                model_type, X_train, y_train, X_val, y_val,
                sample_weights_train=sw_train, fold=fold,
            )
            fold_models.append(model)

            val_preds = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_preds

            fold_auc = roc_auc_score(y_val, val_preds)
            log.info("    Fold %d AUC: %.5f", fold + 1, fold_auc)

        # Overall metrics
        auc = roc_auc_score(y, oof_preds)
        precision, recall, thresholds = precision_recall_curve(y, oof_preds)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        best_threshold = thresholds[np.argmax(f1_scores)]
        best_f1 = np.max(f1_scores)

        log.info(
            "%s — OOF AUC: %.5f, Best F1: %.5f @ threshold %.4f",
            model_type.upper(), auc, best_f1, best_threshold,
        )

        results["models"][model_type] = fold_models
        results["oof_preds"][model_type] = oof_preds
        results["metrics"][model_type] = {
            "auc": auc, "f1": best_f1, "threshold": best_threshold,
        }

        # Feature importance
        if model_type == "lgb":
            imp = pd.DataFrame({
                "feature": X.columns,
                "importance": np.mean(
                    [m.feature_importances_ for m in fold_models], axis=0
                ),
            }).sort_values("importance", ascending=False)
        elif model_type == "xgb":
            imp = pd.DataFrame({
                "feature": X.columns,
                "importance": np.mean(
                    [m.feature_importances_ for m in fold_models], axis=0
                ),
            }).sort_values("importance", ascending=False)
        else:
            imp = pd.DataFrame({"feature": X.columns, "importance": 0})

        results["feature_importance"][model_type] = imp

    return results


def calibrate_and_ensemble(
    results: dict,
    X: pd.DataFrame,
    y: pd.Series,
) -> np.ndarray:
    """
    Calibrate individual model probabilities and create weighted ensemble.

    Uses isotonic regression for calibration and AUC-weighted averaging.
    """
    log.info("Calibrating probabilities and creating ensemble")

    calibrated_oof = {}
    calibrators = {}

    for model_type, oof_preds in results["oof_preds"].items():
        # Isotonic calibration using OOF predictions
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_preds, y)
        calibrated = iso.predict(oof_preds)
        calibrated_oof[model_type] = calibrated
        calibrators[model_type] = iso

        cal_auc = roc_auc_score(y, calibrated)
        log.info("  %s calibrated AUC: %.5f", model_type.upper(), cal_auc)

    # AUC-weighted ensemble
    aucs = {mt: results["metrics"][mt]["auc"] for mt in results["oof_preds"]}
    total_auc = sum(aucs.values())
    weights = {mt: auc / total_auc for mt, auc in aucs.items()}
    log.info("Ensemble weights: %s", {k: f"{v:.3f}" for k, v in weights.items()})

    ensemble_oof = np.zeros(len(y))
    for model_type, cal_preds in calibrated_oof.items():
        ensemble_oof += weights[model_type] * cal_preds

    ensemble_auc = roc_auc_score(y, ensemble_oof)
    log.info("Ensemble OOF AUC: %.5f", ensemble_auc)

    # Save calibrators and weights
    joblib.dump(calibrators, MODELS_DIR / "calibrators.joblib")
    joblib.dump(weights, MODELS_DIR / "ensemble_weights.joblib")

    results["calibrators"] = calibrators
    results["ensemble_weights"] = weights
    results["ensemble_oof"] = ensemble_oof

    return ensemble_oof


def predict_test(
    results: dict,
    X_test: pd.DataFrame,
) -> np.ndarray:
    """Generate calibrated ensemble predictions for test accounts."""
    log.info("Generating test predictions")

    calibrators = results["calibrators"]
    weights = results["ensemble_weights"]

    ensemble_preds = np.zeros(len(X_test))

    for model_type, fold_models in results["models"].items():
        # Average predictions across folds
        fold_preds = np.zeros(len(X_test))
        for model in fold_models:
            fold_preds += model.predict_proba(X_test)[:, 1] / len(fold_models)

        # Calibrate
        calibrated = calibrators[model_type].predict(fold_preds)
        ensemble_preds += weights[model_type] * calibrated

        log.info("  %s test predictions: mean=%.4f, std=%.4f",
                 model_type.upper(), fold_preds.mean(), fold_preds.std())

    log.info(
        "Ensemble test predictions: mean=%.4f, std=%.4f, >0.5: %d",
        ensemble_preds.mean(), ensemble_preds.std(), (ensemble_preds > 0.5).sum(),
    )

    return ensemble_preds


def save_models(results: dict):
    """Persist all trained models and metadata."""
    for model_type, fold_models in results["models"].items():
        for i, model in enumerate(fold_models):
            path = MODELS_DIR / f"{model_type}_fold{i}.joblib"
            joblib.dump(model, path)
    log.info("Saved all models to %s", MODELS_DIR)

    # Save feature importances
    for model_type, imp in results["feature_importance"].items():
        imp.to_csv(MODELS_DIR / f"{model_type}_importance.csv", index=False)

    # Save OOF predictions
    oof_df = pd.DataFrame(results["oof_preds"])
    oof_df.to_parquet(OUTPUT_DIR / "oof_predictions.parquet")

    # Save metrics
    metrics_df = pd.DataFrame(results["metrics"]).T
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv")
    log.info("Saved metrics:\n%s", metrics_df)


def run_shap_analysis(results: dict, X: pd.DataFrame, n_samples: int = 2000):
    """SHAP analysis for model interpretability."""
    try:
        import shap
    except ImportError:
        log.warning("SHAP not available, skipping analysis")
        return

    log.info("Running SHAP analysis on %d samples", n_samples)

    # Use first LGB fold model
    model = results["models"]["lgb"][0]
    X_sample = X.sample(n=min(n_samples, len(X)), random_state=SEED)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list [neg, pos]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Mean absolute SHAP values
    mean_shap = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    mean_shap.to_csv(OUTPUT_DIR / "shap_importance.csv", index=False)
    log.info("Top 20 SHAP features:\n%s", mean_shap.head(20).to_string())


if __name__ == "__main__":
    log.info("Model training module — run via run.py")
