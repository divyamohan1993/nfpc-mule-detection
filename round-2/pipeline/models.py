"""
NFPC Phase 2 — Model Training & Inference (V2)

Pipeline:
  1. Adversarial validation → feature debiasing
  2. LOO target encoding for branch_code (inside CV folds only)
  3. Label cleaning → sample weights
  4. LightGBM + XGBoost + CatBoost with Optuna HPO
  5. Stacking: L0 OOF → L1 meta-learner (LogReg)
  6. Isotonic calibration
  7. SHAP analysis
  8. Submission generation with temporal windows
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
import joblib
import gc

from config import (
    TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH, FULL_FEATURES_PATH,
    OUTPUT_DIR, MODELS_DIR, N_FOLDS, SEED, MULE_RATIO, log,
)


# ═══════════════════════════════════════════════════════════════════════
# ADVERSARIAL VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def adversarial_validation(X_train, X_test, threshold=0.7):
    """
    Train a classifier to distinguish train vs test distributions.
    Returns features with AV importance above threshold (likely distribution-shifted).
    """
    log.info("Running adversarial validation")

    # Combine train/test with domain labels
    av_X = pd.concat([X_train, X_test], axis=0)
    av_y = np.array([0] * len(X_train) + [1] * len(X_test))

    # Drop non-numeric columns for AV
    numeric_cols = av_X.select_dtypes(include=[np.number]).columns.tolist()
    av_X = av_X[numeric_cols]

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    importances = np.zeros(len(numeric_cols))
    oof_preds = np.zeros(len(av_y))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(av_X, av_y)):
        model = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            num_leaves=31, subsample=0.8, colsample_bytree=0.7,
            random_state=SEED, verbosity=-1, n_jobs=-1,
        )
        model.fit(
            av_X.iloc[tr_idx], av_y[tr_idx],
            eval_set=[(av_X.iloc[val_idx], av_y[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )
        oof_preds[val_idx] = model.predict_proba(av_X.iloc[val_idx])[:, 1]
        importances += model.feature_importances_

    av_auc = roc_auc_score(av_y, oof_preds)
    log.info("Adversarial validation AUC: %.4f", av_auc)

    importances /= 3
    feat_imp = pd.Series(importances, index=numeric_cols).sort_values(ascending=False)

    # Identify features that strongly distinguish train from test
    total_imp = feat_imp.sum()
    cumulative = feat_imp.cumsum() / total_imp
    shifted_features = cumulative[cumulative < threshold].index.tolist()

    log.info("Top 10 AV features:\n%s", feat_imp.head(10).to_string())
    log.info("Features contributing to %.0f%% of AV signal: %d",
             threshold * 100, len(shifted_features))

    # Save AV report
    av_report = pd.DataFrame({
        "feature": feat_imp.index,
        "av_importance": feat_imp.values,
        "cumulative_pct": (feat_imp.cumsum() / total_imp).values,
    })
    av_report.to_csv(OUTPUT_DIR / "adversarial_validation.csv", index=False)

    return shifted_features, av_auc


def debias_features(X, shifted_features, strategy="downweight"):
    """
    Debias shifted features instead of removing them entirely.
    Strategy: downweight by multiplying by a shrinkage factor.
    """
    if strategy == "remove":
        return X.drop(columns=shifted_features, errors="ignore")
    elif strategy == "downweight":
        X = X.copy()
        for feat in shifted_features:
            if feat in X.columns:
                X[feat] = X[feat] * 0.5  # Shrink toward zero
        return X
    return X


# ═══════════════════════════════════════════════════════════════════════
# LOO TARGET ENCODING (within each CV fold)
# ═══════════════════════════════════════════════════════════════════════

def _target_encode_fold(X_train, y_train, X_val, X_test, col, global_mean, min_samples=30, smoothing=20):
    """
    Leave-one-out target encoding for a single fold.
    Uses Bayesian smoothing: encoded = (n * mean_target + smoothing * global_mean) / (n + smoothing)
    """
    stats = y_train.groupby(X_train[col]).agg(["mean", "count"])
    stats.columns = ["mean_target", "n"]

    # Bayesian smoothing to regularize rare categories
    stats["encoded"] = (
        (stats["n"] * stats["mean_target"] + smoothing * global_mean)
        / (stats["n"] + smoothing)
    )

    mapping = stats["encoded"].to_dict()

    X_train_encoded = X_train[col].map(mapping).fillna(global_mean)
    X_val_encoded = X_val[col].map(mapping).fillna(global_mean)
    X_test_encoded = X_test[col].map(mapping).fillna(global_mean) if X_test is not None else None

    return X_train_encoded, X_val_encoded, X_test_encoded


# ═══════════════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

def _optimize_lgb(X, y, sample_weights, n_trials=40):
    """Optimize LightGBM hyperparameters with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": MULE_RATIO,
            "random_state": SEED, "verbosity": -1, "n_jobs": -1,
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                sample_weight=sample_weights[tr_idx] if sample_weights is not None else None,
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info("LGB best trial: AUC=%.5f, params=%s", study.best_value, study.best_params)
    return study.best_params


def _optimize_xgb(X, y, sample_weights, n_trials=30):
    """Optimize XGBoost hyperparameters with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "scale_pos_weight": MULE_RATIO,
            "random_state": SEED, "verbosity": 0, "n_jobs": -1,
            "tree_method": "hist", "eval_metric": "auc",
            "early_stopping_rounds": 50,
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            model = xgb.XGBClassifier(**params)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                sample_weight=sample_weights[tr_idx] if sample_weights is not None else None,
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                verbose=False,
            )
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info("XGB best trial: AUC=%.5f, params=%s", study.best_value, study.best_params)
    return study.best_params


def _optimize_catboost(X, y, sample_weights, n_trials=20):
    """Optimize CatBoost hyperparameters with Optuna."""
    import optuna
    from catboost import CatBoostClassifier
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "iterations": 2000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 5),
            "random_strength": trial.suggest_float("random_strength", 0, 5),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "scale_pos_weight": MULE_RATIO,
            "random_seed": SEED, "verbose": 0, "thread_count": -1,
            "eval_metric": "AUC",
        }

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
        scores = []
        for tr_idx, val_idx in skf.split(X, y):
            model = CatBoostClassifier(**params)
            model.fit(
                X.iloc[tr_idx], y.iloc[tr_idx],
                sample_weight=sample_weights[tr_idx] if sample_weights is not None else None,
                eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
                early_stopping_rounds=50,
            )
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            scores.append(roc_auc_score(y.iloc[val_idx], preds))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    log.info("CatBoost best trial: AUC=%.5f, params=%s", study.best_value, study.best_params)
    return study.best_params


# ═══════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def train_and_predict(features, sample_weights=None, skip_optuna=False):
    """
    Full training pipeline:
      1. Split train/test
      2. Adversarial validation + debiasing
      3. N-fold CV with LOO target encoding
      4. LGB + XGB + CatBoost ensemble
      5. Stacking (L1 meta-learner)
      6. Isotonic calibration
      7. SHAP analysis
    """
    from catboost import CatBoostClassifier

    labels = pd.read_parquet(TRAIN_LABELS_PATH)
    test_accounts = pd.read_parquet(TEST_ACCOUNTS_PATH)

    train_ids = labels["account_id"].values
    test_ids = test_accounts["account_id"].values

    X_all = features
    X_train = X_all.loc[X_all.index.isin(train_ids)].copy()
    X_test = X_all.loc[X_all.index.isin(test_ids)].copy()

    # Align labels
    labels_indexed = labels.set_index("account_id")
    y_train = labels_indexed.loc[X_train.index, "is_mule"]

    log.info("Train: %d samples (%d mules, %.2f%%)",
             len(X_train), y_train.sum(), 100 * y_train.mean())
    log.info("Test: %d samples", len(X_test))

    # ── Step 1: Adversarial validation ──
    # Identify branch_code column for target encoding
    te_cols = []
    if "branch_code" in X_train.columns:
        te_cols.append("branch_code")

    # Prep numeric features for AV (exclude string columns)
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    shifted_features, av_auc = adversarial_validation(
        X_train[numeric_cols], X_test[numeric_cols]
    )

    # Only debias if AV AUC is significant (>0.6 means distribution shift exists)
    if av_auc > 0.6:
        debias_strategy = "downweight" if av_auc < 0.75 else "remove"
        # Don't remove too many features — cap at top 15
        features_to_debias = shifted_features[:15]
        log.info("Debiasing %d features (strategy=%s)", len(features_to_debias), debias_strategy)
        X_train = debias_features(X_train, features_to_debias, debias_strategy)
        X_test = debias_features(X_test, features_to_debias, debias_strategy)
    else:
        log.info("AV AUC %.4f < 0.6 — no debiasing needed", av_auc)

    # ── Step 2: Prepare features ──
    # Drop non-numeric columns (keep branch_code for target encoding, then drop)
    drop_cols = [c for c in X_train.columns if X_train[c].dtype == "object" and c not in te_cols]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # Ensure consistent columns
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    log.info("Feature count: %d (including %d for target encoding)", len(common_cols), len(te_cols))

    # ── Step 3: Optuna HPO ──
    # Use numeric features only for HPO (target encoding done inside CV)
    numeric_for_hpo = [c for c in common_cols if c not in te_cols]
    X_hpo = X_train[numeric_for_hpo].copy()

    saved_params_path = MODELS_DIR / "best_params.pkl"
    if not skip_optuna:
        log.info("Starting Optuna hyperparameter optimization")
        lgb_best_params = _optimize_lgb(X_hpo, y_train, sample_weights)
        xgb_best_params = _optimize_xgb(X_hpo, y_train, sample_weights)
        cb_best_params = _optimize_catboost(X_hpo, y_train, sample_weights)

        # Save best params
        joblib.dump({
            "lgb": lgb_best_params,
            "xgb": xgb_best_params,
            "catboost": cb_best_params,
        }, saved_params_path)
    elif saved_params_path.exists():
        # Try v2 params first (tuned with early stopping), fall back to v1
        v2_path = MODELS_DIR / "best_params_v2.pkl"
        if v2_path.exists():
            log.info("Loading Optuna v2 params (tuned with early stopping)")
            saved = joblib.load(v2_path)
        else:
            log.info("Loading Optuna v1 params")
            saved = joblib.load(saved_params_path)
        lgb_best_params = saved["lgb"]
        xgb_best_params = saved["xgb"]
        cb_best_params = saved["catboost"]
        # Override CB l2_leaf_reg if too low (causes cross-fold calibration drift)
        if cb_best_params.get("l2_leaf_reg", 0) < 1.0:
            log.info("CB l2_leaf_reg=%.4f too low, clamping to 1.0", cb_best_params["l2_leaf_reg"])
            cb_best_params["l2_leaf_reg"] = 1.0
        log.info("LGB params: %s", lgb_best_params)
        log.info("XGB params: %s", xgb_best_params)
        log.info("CB params: %s", cb_best_params)
    else:
        log.info("Skipping Optuna — using default hyperparameters")
        lgb_best_params = {
            "learning_rate": 0.02, "max_depth": 7, "num_leaves": 63,
            "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        }
        xgb_best_params = {
            "learning_rate": 0.02, "max_depth": 7, "min_child_weight": 10,
            "subsample": 0.8, "colsample_bytree": 0.7,
            "reg_alpha": 0.1, "reg_lambda": 1.0, "gamma": 0.5,
        }
        cb_best_params = {
            "learning_rate": 0.03, "depth": 7, "l2_leaf_reg": 3.0,
            "bagging_temperature": 1.0, "random_strength": 1.0, "border_count": 128,
        }

    del X_hpo
    gc.collect()

    # ── Step 4: N-fold CV with stacking ──
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_lgb = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_cb = np.zeros(len(X_train))

    test_lgb = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_cb = np.zeros(len(X_test))

    global_mean = y_train.mean()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        log.info("═══ Fold %d/%d ═══", fold + 1, N_FOLDS)

        X_tr = X_train.iloc[tr_idx].copy()
        X_val = X_train.iloc[val_idx].copy()
        X_te = X_test.copy()
        y_tr = y_train.iloc[tr_idx]
        y_val = y_train.iloc[val_idx]

        sw_tr = sample_weights[tr_idx] if sample_weights is not None else None

        # LOO target encoding inside fold
        for col in te_cols:
            tr_enc, val_enc, te_enc = _target_encode_fold(
                X_tr, y_tr, X_val, X_te, col, global_mean
            )
            X_tr[f"{col}_te"] = tr_enc.values
            X_val[f"{col}_te"] = val_enc.values
            X_te[f"{col}_te"] = te_enc.values
            X_tr = X_tr.drop(columns=[col])
            X_val = X_val.drop(columns=[col])
            X_te = X_te.drop(columns=[col])

        # Ensure all numeric
        feature_cols = [c for c in X_tr.columns if X_tr[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.float16, np.uint8]]
        X_tr = X_tr[feature_cols]
        X_val = X_val[feature_cols]
        X_te = X_te[feature_cols]

        # ── LightGBM ──
        lgb_params = {
            "n_estimators": 3000, "scale_pos_weight": MULE_RATIO,
            "random_state": SEED, "verbosity": -1, "n_jobs": -1,
            **lgb_best_params,
        }
        m_lgb = lgb.LGBMClassifier(**lgb_params)
        m_lgb.fit(
            X_tr, y_tr, sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
        test_lgb += m_lgb.predict_proba(X_te)[:, 1] / N_FOLDS

        # ── XGBoost ──
        xgb_params = {
            "n_estimators": 3000, "scale_pos_weight": MULE_RATIO,
            "random_state": SEED, "verbosity": 0, "n_jobs": -1,
            "tree_method": "hist", "eval_metric": "auc",
            "early_stopping_rounds": 100,
            **xgb_best_params,
        }
        m_xgb = xgb.XGBClassifier(**xgb_params)
        m_xgb.fit(
            X_tr, y_tr, sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        test_xgb += m_xgb.predict_proba(X_te)[:, 1] / N_FOLDS

        # ── CatBoost ──
        # Use scale_pos_weight instead of auto_class_weights when sample_weight provided
        # (double-weighting causes instability)
        cb_params_full = {
            "iterations": 3000,
            "scale_pos_weight": MULE_RATIO,
            "random_seed": SEED, "verbose": 100, "thread_count": -1,
            "eval_metric": "AUC", "od_type": "Iter", "od_wait": 100,
            **cb_best_params,
        }
        m_cb = CatBoostClassifier(**cb_params_full)
        m_cb.fit(
            X_tr, y_tr, sample_weight=sw_tr,
            eval_set=(X_val, y_val),
        )
        oof_cb[val_idx] = m_cb.predict_proba(X_val)[:, 1]
        test_cb += m_cb.predict_proba(X_te)[:, 1] / N_FOLDS

        fold_auc_lgb = roc_auc_score(y_val, oof_lgb[val_idx])
        fold_auc_xgb = roc_auc_score(y_val, oof_xgb[val_idx])
        fold_auc_cb = roc_auc_score(y_val, oof_cb[val_idx])
        log.info("  Fold %d AUC — LGB: %.5f, XGB: %.5f, CB: %.5f",
                 fold + 1, fold_auc_lgb, fold_auc_xgb, fold_auc_cb)

        # Save fold models
        joblib.dump(m_lgb, MODELS_DIR / f"lgb_fold{fold}.pkl")
        joblib.dump(m_xgb, MODELS_DIR / f"xgb_fold{fold}.pkl")
        m_cb.save_model(str(MODELS_DIR / f"cb_fold{fold}.cbm"))

        del m_lgb, m_xgb, m_cb, X_tr, X_val, X_te
        gc.collect()

    # ── Overall OOF metrics ──
    auc_lgb = roc_auc_score(y_train, oof_lgb)
    auc_xgb = roc_auc_score(y_train, oof_xgb)
    auc_cb = roc_auc_score(y_train, oof_cb)
    log.info("Overall OOF AUC — LGB: %.5f, XGB: %.5f, CB: %.5f", auc_lgb, auc_xgb, auc_cb)

    # ── Step 5: Stacking (L1 meta-learner) ──
    log.info("Training L1 meta-learner (stacking)")
    oof_stack = np.column_stack([oof_lgb, oof_xgb, oof_cb])
    test_stack = np.column_stack([test_lgb, test_xgb, test_cb])

    # L1: Logistic Regression on OOF predictions
    meta = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED
    )
    meta.fit(oof_stack, y_train)
    oof_meta = meta.predict_proba(oof_stack)[:, 1]
    test_meta = meta.predict_proba(test_stack)[:, 1]

    auc_meta = roc_auc_score(y_train, oof_meta)
    log.info("Stacked meta-learner OOF AUC: %.5f", auc_meta)
    log.info("Meta-learner weights: LGB=%.3f, XGB=%.3f, CB=%.3f",
             meta.coef_[0][0], meta.coef_[0][1], meta.coef_[0][2])

    # Also compute simple AUC-weighted average as fallback
    aucs = np.array([auc_lgb, auc_xgb, auc_cb])
    w = aucs / aucs.sum()
    oof_wavg = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cb
    test_wavg = w[0] * test_lgb + w[1] * test_xgb + w[2] * test_cb
    auc_wavg = roc_auc_score(y_train, oof_wavg)
    log.info("AUC-weighted average OOF AUC: %.5f (weights: %.3f, %.3f, %.3f)",
             auc_wavg, w[0], w[1], w[2])

    # Pick best ensemble method
    if auc_meta >= auc_wavg:
        oof_ensemble = oof_meta
        test_ensemble = test_meta
        ensemble_method = "stacking"
    else:
        oof_ensemble = oof_wavg
        test_ensemble = test_wavg
        ensemble_method = "auc_weighted"

    log.info("Selected ensemble method: %s (AUC=%.5f)", ensemble_method, max(auc_meta, auc_wavg))

    # ── Step 6: Isotonic calibration ──
    log.info("Applying isotonic calibration")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(oof_ensemble, y_train)
    oof_calibrated = iso.predict(oof_ensemble)
    test_calibrated = iso.predict(test_ensemble)

    auc_calibrated = roc_auc_score(y_train, oof_calibrated)
    log.info("Calibrated OOF AUC: %.5f", auc_calibrated)

    # ── Step 7: Threshold optimization ──
    expected_mule_rate = 1.0 / (1 + MULE_RATIO)  # ~2.86%
    precision, recall, thresholds = precision_recall_curve(y_train, oof_calibrated)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_f1_idx]

    log.info("Optimal threshold: %.4f (F1=%.4f)", best_threshold, best_f1)

    # Count predicted mules at various thresholds
    for t in [0.1, 0.2, 0.3, 0.5, best_threshold]:
        n_mules = (test_calibrated >= t).sum()
        log.info("  Threshold %.2f → %d predicted mules (%.1f%%)",
                 t, n_mules, 100 * n_mules / len(test_calibrated))

    # ── Step 8: SHAP analysis ──
    log.info("Computing SHAP values (last fold LGB)")
    try:
        import shap
        last_lgb = joblib.load(MODELS_DIR / f"lgb_fold{N_FOLDS - 1}.pkl")

        # Use last fold's feature set for SHAP
        X_shap = X_train.copy()
        for col in te_cols:
            if col in X_shap.columns:
                # Simple encoding for SHAP display (not used for prediction)
                X_shap[f"{col}_te"] = X_shap[col].map(
                    y_train.groupby(X_train[col]).mean()
                ).fillna(global_mean)
                X_shap = X_shap.drop(columns=[col])

        feature_cols = [c for c in X_shap.columns if X_shap[c].dtype in [np.float64, np.float32, np.int64, np.int32, np.float16, np.uint8]]
        X_shap = X_shap[feature_cols]

        explainer = shap.TreeExplainer(last_lgb)
        shap_values = explainer.shap_values(X_shap[:5000])  # sample for speed

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_imp = pd.Series(mean_abs_shap, index=feature_cols).sort_values(ascending=False)
        shap_imp.to_csv(OUTPUT_DIR / "shap_importance.csv")
        log.info("Top 20 SHAP features:\n%s", shap_imp.head(20).to_string())

        # Check for potential leakage (any single feature dominating by 4x+)
        if len(shap_imp) > 1:
            top_ratio = shap_imp.iloc[0] / max(shap_imp.iloc[1], 1e-10)
            if top_ratio > 4.0:
                log.warning("LEAKAGE WARNING: Top feature '%s' has %.1fx higher SHAP than #2 '%s'",
                           shap_imp.index[0], top_ratio, shap_imp.index[1])
    except Exception as e:
        log.warning("SHAP analysis failed: %s", e)

    # ── Save outputs ──
    joblib.dump(meta, MODELS_DIR / "meta_learner.pkl")
    joblib.dump(iso, MODELS_DIR / "isotonic_calibrator.pkl")

    # Save OOF predictions for analysis
    oof_df = pd.DataFrame({
        "account_id": X_train.index,
        "y_true": y_train.values,
        "oof_lgb": oof_lgb,
        "oof_xgb": oof_xgb,
        "oof_cb": oof_cb,
        "oof_ensemble": oof_ensemble,
        "oof_calibrated": oof_calibrated,
    })
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    # Return mule predictions for temporal window module
    mule_predictions = pd.Series(test_calibrated, index=X_test.index, name="mule_prob")

    results = {
        "test_predictions": test_calibrated,
        "test_accounts": X_test.index,
        "mule_predictions": mule_predictions,
        "oof_auc": max(auc_meta, auc_wavg),
        "calibrated_auc": auc_calibrated,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "ensemble_method": ensemble_method,
        "individual_aucs": {"lgb": auc_lgb, "xgb": auc_xgb, "cb": auc_cb},
    }

    log.info("Training complete. Ensemble AUC: %.5f, Calibrated AUC: %.5f",
             results["oof_auc"], results["calibrated_auc"])

    return results
