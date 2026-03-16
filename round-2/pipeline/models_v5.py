"""
NFPC Phase 2 — Model Training & Inference (V5)

Key changes from V3:
  1. All V3 improvements retained (no AV debiasing, freq encoding, rank avg, multi-seed)
  2. Added 20+ derived feature interactions for mule detection:
     - Pass-through detection (flow ratios, credit-debit matching)
     - Activity concentration (txn per counterparty, per channel)
     - Graph x volume interactions (PageRank x txn count)
     - Dormancy-burst patterns (dormancy x volume change)
     - Temporal regularity features
  3. Uses V2 params (best known public score 0.968)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from scipy.stats import rankdata
import joblib
import gc

from config import (
    TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH, FULL_FEATURES_PATH,
    OUTPUT_DIR, MODELS_DIR, N_FOLDS, SEED, MULE_RATIO, log,
)


# ═══════════════════════════════════════════════════════════════════════
# FREQUENCY ENCODING (safe, no label leakage)
# ═══════════════════════════════════════════════════════════════════════

def _frequency_encode(X_train, X_test, col):
    """
    Encode categorical column by its frequency in training data.
    No label leakage possible since we only count occurrences.
    """
    freq = X_train[col].value_counts(normalize=True)
    X_train_enc = X_train[col].map(freq).fillna(0.0)
    X_test_enc = X_test[col].map(freq).fillna(0.0)

    # Also add count encoding
    counts = X_train[col].value_counts()
    X_train_cnt = X_train[col].map(counts).fillna(0).astype(float)
    X_test_cnt = X_test[col].map(counts).fillna(0).astype(float)

    return X_train_enc, X_test_enc, X_train_cnt, X_test_cnt


# ═══════════════════════════════════════════════════════════════════════
# RANK AVERAGING
# ═══════════════════════════════════════════════════════════════════════

def _rank_average(*arrays):
    """
    Rank-average multiple prediction arrays.
    More robust than probability averaging since it's invariant to
    monotonic transformations of individual model outputs.
    """
    n = len(arrays[0])
    ranked = np.zeros(n)
    for arr in arrays:
        ranked += rankdata(arr) / n
    ranked /= len(arrays)
    return ranked


# ═══════════════════════════════════════════════════════════════════════
# OPTUNA HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

def _optimize_lgb(X, y, sample_weights, n_trials=40):
    """Optimize LightGBM hyperparameters with Optuna."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
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
            "n_estimators": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 0.85),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.8),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 50.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.1, 5),
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
            "iterations": 3000,
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
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
# V5: FEATURE INTERACTIONS
# ═══════════════════════════════════════════════════════════════════════

def _safe_div(a, b, fill=0.0):
    """Safe division avoiding div-by-zero."""
    return np.where(b > 1e-10, a / b, fill)


def _add_feature_interactions(df):
    """
    Add derived feature interactions that capture mule behavior patterns.
    All computed from existing features — no raw data access needed.
    Operates in-place on the DataFrame.
    """
    eps = 1e-10

    # ── Pass-through detection ──
    # Mules receive and forward money: credit ~ debit amounts
    if "sum_credit" in df.columns and "sum_debit" in df.columns:
        abs_debit = df["sum_debit"].abs()
        total_flow = df["sum_credit"] + abs_debit + eps
        min_flow = np.minimum(df["sum_credit"], abs_debit)
        max_flow = np.maximum(df["sum_credit"], abs_debit).clip(lower=eps)

        df["v5_flow_through_ratio"] = min_flow / max_flow  # ~1.0 for mules
        df["v5_net_flow_pct"] = df["net_flow"].abs() / total_flow  # ~0 for mules
        df["v5_credit_minus_debit_abs"] = (df["sum_credit"] - abs_debit).abs()
        df["v5_flow_symmetry"] = 1.0 - df["v5_net_flow_pct"]  # ~1 for mules

    # ── Activity concentration ──
    if "n_txn" in df.columns and "n_unique_counterparties" in df.columns:
        df["v5_txn_per_cp"] = _safe_div(df["n_txn"], df["n_unique_counterparties"].clip(lower=1))
    if "n_txn" in df.columns and "n_channels_used" in df.columns:
        df["v5_txn_per_channel"] = _safe_div(df["n_txn"], df["n_channels_used"].clip(lower=1))
    if "sum_credit" in df.columns and "n_unique_cp_credit" in df.columns:
        df["v5_credit_per_cp"] = _safe_div(df["sum_credit"], df["n_unique_cp_credit"].clip(lower=1))
    if "sum_debit" in df.columns and "n_unique_cp_debit" in df.columns:
        df["v5_debit_per_cp"] = _safe_div(df["sum_debit"].abs(), df["n_unique_cp_debit"].clip(lower=1))

    # ── Graph x Volume interactions ──
    if "graph_pagerank" in df.columns and "n_txn" in df.columns:
        df["v5_pagerank_x_txn"] = df["graph_pagerank"] * np.log1p(df["n_txn"])
    if "graph_betweenness" in df.columns and "n_txn" in df.columns:
        df["v5_betweenness_x_txn"] = df["graph_betweenness"] * np.log1p(df["n_txn"])
    if "graph_in_amount" in df.columns and "graph_out_amount" in df.columns:
        ga_in = df["graph_in_amount"]
        ga_out = df["graph_out_amount"]
        ga_total = ga_in + ga_out + eps
        df["v5_graph_flow_asymmetry"] = (ga_in - ga_out).abs() / ga_total
        df["v5_graph_pass_through"] = np.minimum(ga_in, ga_out) / np.maximum(ga_in, ga_out).clip(lower=eps)

    # ── Dormancy-burst patterns ──
    if "dormancy_events" in df.columns and "behavioral_volume_change" in df.columns:
        df["v5_dormancy_x_burst"] = df["dormancy_events"] * df["behavioral_volume_change"]
    if "dormancy_events" in df.columns and "n_txn" in df.columns:
        df["v5_dormancy_x_volume"] = df["dormancy_events"] * np.log1p(df["n_txn"])

    # ── Temporal regularity ──
    if "inter_txn_cv" in df.columns and "burstiness" in df.columns:
        df["v5_temporal_irregularity"] = df["inter_txn_cv"] * (1 + df["burstiness"].abs())
    if "txn_span_days" in df.columns and "n_active_months" in df.columns:
        df["v5_activity_density"] = _safe_div(
            df["n_active_months"], df["txn_span_days"].clip(lower=1) / 30
        )

    # ── Amount pattern features ──
    if "amt_cv" in df.columns and "round_amount_ratio" in df.columns:
        df["v5_low_cv_x_round"] = (1.0 / (df["amt_cv"] + eps)) * df["round_amount_ratio"]
    if "amt_mean" in df.columns and "n_txn" in df.columns:
        df["v5_total_volume"] = df["amt_mean"] * df["n_txn"]

    # ── Structuring intensity ──
    near_cols = [c for c in df.columns if c.startswith("near_") and c.endswith("_ratio")]
    if near_cols:
        df["v5_total_structuring_ratio"] = df[near_cols].sum(axis=1)

    # ── Fan-in/fan-out asymmetry combined with amounts ──
    if "fan_in_out_ratio" in df.columns and "credit_debit_amount_ratio" in df.columns:
        df["v5_fanio_x_amt_ratio"] = df["fan_in_out_ratio"] * df["credit_debit_amount_ratio"]

    # ── Account age x behavior ──
    if "account_age_days" in df.columns and "n_txn" in df.columns:
        df["v5_txn_per_age_day"] = _safe_div(df["n_txn"], df["account_age_days"].clip(lower=1))
    if "account_age_days" in df.columns and "was_frozen" in df.columns:
        df["v5_frozen_x_young"] = df["was_frozen"] * _safe_div(1.0, df["account_age_days"].clip(lower=1))

    # ── Balance behavior ──
    if "bal_mean" in df.columns and "amt_mean" in df.columns:
        df["v5_bal_to_txn_ratio"] = _safe_div(df["bal_mean"].abs(), df["amt_mean"].clip(lower=eps))
    if "n_near_zero_balance" in df.columns and "n_txn" in df.columns:
        df["v5_zero_bal_intensity"] = _safe_div(df["n_near_zero_balance"], df["n_txn"].clip(lower=1))

    # ── MCC anomaly x graph ──
    if "mcc_amt_zscore_max" in df.columns and "graph_pagerank" in df.columns:
        df["v5_mcc_anomaly_x_pr"] = df["mcc_amt_zscore_max"] * df["graph_pagerank"]

    # ── Community features ──
    if "graph_community_size" in df.columns and "graph_pagerank" in df.columns:
        df["v5_pr_within_community"] = df["graph_pagerank"] * np.log1p(df["graph_community_size"])

    # Replace any NaN/inf
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)


# ═══════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE (V5)
# ═══════════════════════════════════════════════════════════════════════

def train_and_predict(features, sample_weights=None, skip_optuna=False):
    """
    V5 training pipeline:
      1. NO adversarial debiasing (keep all features)
      2. Frequency encoding for categoricals (no target leakage)
      3. V5 feature interactions (pass-through, graph, dormancy, etc.)
      4. N-fold CV: LGB + XGB + CatBoost
      5. Rank averaging ensemble (AUC-optimal)
      6. Raw probabilities for submission (no isotonic calibration)
      7. Multi-seed for stability
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

    # ── Step 1: Frequency encoding for categoricals (NO target encoding) ──
    cat_cols = []
    if "branch_code" in X_train.columns:
        cat_cols.append("branch_code")

    for col in cat_cols:
        freq_tr, freq_te, cnt_tr, cnt_te = _frequency_encode(X_train, X_test, col)
        X_train[f"{col}_freq"] = freq_tr.values
        X_test[f"{col}_freq"] = freq_te.values
        X_train[f"{col}_count"] = cnt_tr.values
        X_test[f"{col}_count"] = cnt_te.values
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

    # Drop non-numeric columns
    drop_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # ── V5: Feature interactions (derived from existing features) ──
    for df in [X_train, X_test]:
        _add_feature_interactions(df)

    # Ensure consistent columns
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    log.info("Feature count: %d (V5 with feature interactions)", len(common_cols))

    # ── Step 2: Load hyperparameters ──
    # V5: Prefer V2 params (gave best public score 0.968), then V3, then defaults
    saved_params_path = MODELS_DIR / "best_params_v5.pkl"
    if not skip_optuna:
        log.info("Starting Optuna hyperparameter optimization (V5)")
        lgb_best_params = _optimize_lgb(X_train, y_train, sample_weights)
        xgb_best_params = _optimize_xgb(X_train, y_train, sample_weights)
        cb_best_params = _optimize_catboost(X_train, y_train, sample_weights)

        joblib.dump({
            "lgb": lgb_best_params,
            "xgb": xgb_best_params,
            "catboost": cb_best_params,
        }, saved_params_path)
    elif saved_params_path.exists():
        log.info("Loading Optuna V5 params")
        saved = joblib.load(saved_params_path)
        lgb_best_params = saved["lgb"]
        xgb_best_params = saved["xgb"]
        cb_best_params = saved["catboost"]
    else:
        # Prefer V2 params (best public), then V3, then V1, then defaults
        for path_name in ["best_params_v2_backup.pkl", "best_params_v2.pkl",
                          "best_params_v3.pkl", "best_params.pkl"]:
            p = MODELS_DIR / path_name
            if p.exists():
                log.info("Loading params from %s", path_name)
                saved = joblib.load(p)
                lgb_best_params = saved["lgb"]
                xgb_best_params = saved["xgb"]
                cb_best_params = saved["catboost"]
                break
        else:
            log.info("Using default hyperparameters (light regularization)")
            lgb_best_params = {
                "learning_rate": 0.01, "max_depth": 6, "num_leaves": 31,
                "min_child_samples": 50, "subsample": 0.7, "colsample_bytree": 0.6,
                "reg_alpha": 0.01, "reg_lambda": 0.1, "min_split_gain": 0.0,
            }
            xgb_best_params = {
                "learning_rate": 0.01, "max_depth": 6, "min_child_weight": 20,
                "subsample": 0.7, "colsample_bytree": 0.6,
                "reg_alpha": 0.01, "reg_lambda": 0.1, "gamma": 0.0,
            }
            cb_best_params = {
                "learning_rate": 0.02, "depth": 6, "l2_leaf_reg": 0.1,
                "bagging_temperature": 1.0, "random_strength": 1.0, "border_count": 128,
            }

    log.info("LGB params: %s", lgb_best_params)
    log.info("XGB params: %s", xgb_best_params)
    log.info("CB params: %s", cb_best_params)

    # ── Step 3: Multi-seed N-fold CV with 3 models ──
    seeds = [SEED, SEED + 1, SEED + 2]  # 3 seeds for stability
    n_seeds = len(seeds)

    oof_lgb_all = np.zeros(len(X_train))
    oof_xgb_all = np.zeros(len(X_train))
    oof_cb_all = np.zeros(len(X_train))

    test_lgb_all = np.zeros(len(X_test))
    test_xgb_all = np.zeros(len(X_test))
    test_cb_all = np.zeros(len(X_test))

    for seed_idx, seed in enumerate(seeds):
        log.info("═══ Seed %d/%d (seed=%d) ═══", seed_idx + 1, n_seeds, seed)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

        oof_lgb = np.zeros(len(X_train))
        oof_xgb = np.zeros(len(X_train))
        oof_cb = np.zeros(len(X_train))

        test_lgb = np.zeros(len(X_test))
        test_xgb = np.zeros(len(X_test))
        test_cb = np.zeros(len(X_test))

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            log.info("  Fold %d/%d (seed=%d)", fold + 1, N_FOLDS, seed)

            X_tr = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            X_te = X_test
            y_tr = y_train.iloc[tr_idx]
            y_val = y_train.iloc[val_idx]

            sw_tr = sample_weights[tr_idx] if sample_weights is not None else None

            # Ensure all numeric
            feature_cols = [c for c in X_tr.columns if X_tr[c].dtype in
                          [np.float64, np.float32, np.int64, np.int32, np.float16, np.uint8]]
            X_tr = X_tr[feature_cols]
            X_val = X_val[feature_cols]
            X_te = X_te[feature_cols]

            # ── LightGBM ──
            lgb_params = {
                "n_estimators": 5000, "scale_pos_weight": MULE_RATIO,
                "random_state": seed, "verbosity": -1, "n_jobs": -1,
                **lgb_best_params,
            }
            m_lgb = lgb.LGBMClassifier(**lgb_params)
            m_lgb.fit(
                X_tr, y_tr, sample_weight=sw_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(150, verbose=False)],
            )
            oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
            test_lgb += m_lgb.predict_proba(X_te)[:, 1] / N_FOLDS

            # ── XGBoost ──
            xgb_params = {
                "n_estimators": 5000, "scale_pos_weight": MULE_RATIO,
                "random_state": seed, "verbosity": 0, "n_jobs": -1,
                "tree_method": "hist", "eval_metric": "auc",
                "early_stopping_rounds": 150,
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
            cb_params_full = {
                "iterations": 5000,
                "scale_pos_weight": MULE_RATIO,
                "random_seed": seed, "verbose": 100, "thread_count": -1,
                "eval_metric": "AUC", "od_type": "Iter", "od_wait": 150,
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

            # Save models for seed 0 only (primary)
            if seed_idx == 0:
                joblib.dump(m_lgb, MODELS_DIR / f"lgb_fold{fold}.pkl")
                joblib.dump(m_xgb, MODELS_DIR / f"xgb_fold{fold}.pkl")
                m_cb.save_model(str(MODELS_DIR / f"cb_fold{fold}.cbm"))

            del m_lgb, m_xgb, m_cb
            gc.collect()

        # Accumulate across seeds
        oof_lgb_all += oof_lgb / n_seeds
        oof_xgb_all += oof_xgb / n_seeds
        oof_cb_all += oof_cb / n_seeds
        test_lgb_all += test_lgb / n_seeds
        test_xgb_all += test_xgb / n_seeds
        test_cb_all += test_cb / n_seeds

        # Per-seed OOF AUC
        auc_lgb_s = roc_auc_score(y_train, oof_lgb)
        auc_xgb_s = roc_auc_score(y_train, oof_xgb)
        auc_cb_s = roc_auc_score(y_train, oof_cb)
        log.info("Seed %d OOF AUC — LGB: %.5f, XGB: %.5f, CB: %.5f",
                 seed, auc_lgb_s, auc_xgb_s, auc_cb_s)

    # ── Overall OOF metrics ──
    auc_lgb = roc_auc_score(y_train, oof_lgb_all)
    auc_xgb = roc_auc_score(y_train, oof_xgb_all)
    auc_cb = roc_auc_score(y_train, oof_cb_all)
    log.info("Overall OOF AUC (multi-seed avg) — LGB: %.5f, XGB: %.5f, CB: %.5f",
             auc_lgb, auc_xgb, auc_cb)

    # ── Step 4: Ensemble via RANK AVERAGING ──
    log.info("Computing rank-averaged ensemble")

    # Rank averaging on OOF
    oof_rank = _rank_average(oof_lgb_all, oof_xgb_all, oof_cb_all)
    auc_rank = roc_auc_score(y_train, oof_rank)
    log.info("Rank-averaged OOF AUC: %.5f", auc_rank)

    # Rank averaging on test
    test_rank = _rank_average(test_lgb_all, test_xgb_all, test_cb_all)

    # Also try probability averaging and stacking as alternatives
    # Simple probability average
    oof_avg = (oof_lgb_all + oof_xgb_all + oof_cb_all) / 3
    test_avg = (test_lgb_all + test_xgb_all + test_cb_all) / 3
    auc_avg = roc_auc_score(y_train, oof_avg)
    log.info("Simple avg OOF AUC: %.5f", auc_avg)

    # AUC-weighted average
    aucs = np.array([auc_lgb, auc_xgb, auc_cb])
    w = aucs / aucs.sum()
    oof_wavg = w[0] * oof_lgb_all + w[1] * oof_xgb_all + w[2] * oof_cb_all
    test_wavg = w[0] * test_lgb_all + w[1] * test_xgb_all + w[2] * test_cb_all
    auc_wavg = roc_auc_score(y_train, oof_wavg)
    log.info("AUC-weighted avg OOF AUC: %.5f (w: %.3f, %.3f, %.3f)",
             auc_wavg, w[0], w[1], w[2])

    # Stacking
    oof_stack = np.column_stack([oof_lgb_all, oof_xgb_all, oof_cb_all])
    test_stack = np.column_stack([test_lgb_all, test_xgb_all, test_cb_all])
    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
    meta.fit(oof_stack, y_train)
    oof_meta = meta.predict_proba(oof_stack)[:, 1]
    test_meta = meta.predict_proba(test_stack)[:, 1]
    auc_meta = roc_auc_score(y_train, oof_meta)
    log.info("Stacked meta OOF AUC: %.5f", auc_meta)

    # Pick best ensemble method
    methods = {
        "rank_avg": (oof_rank, test_rank, auc_rank),
        "simple_avg": (oof_avg, test_avg, auc_avg),
        "auc_weighted": (oof_wavg, test_wavg, auc_wavg),
        "stacking": (oof_meta, test_meta, auc_meta),
    }
    best_method = max(methods, key=lambda k: methods[k][2])
    oof_ensemble, test_ensemble, best_auc = methods[best_method]
    log.info("Selected ensemble: %s (AUC=%.5f)", best_method, best_auc)

    # ── Step 5: Use RAW ensemble output (no isotonic calibration for AUC) ──
    # Isotonic calibration can create ties that hurt AUC ranking.
    # We use raw probabilities for is_mule.
    # Only calibrate for F1 threshold optimization.
    test_final = test_ensemble

    # But we still need proper threshold for F1
    precision, recall, thresholds = precision_recall_curve(y_train, oof_ensemble)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_f1_idx]

    log.info("Optimal threshold: %.4f (F1=%.4f)", best_threshold, best_f1)

    # Count predicted mules at various thresholds
    for t in [0.1, 0.2, 0.3, 0.5, best_threshold]:
        n_mules = (test_final >= t).sum()
        log.info("  Threshold %.2f -> %d predicted mules (%.1f%%)",
                 t, n_mules, 100 * n_mules / len(test_final))

    # ── Step 6: SHAP analysis ──
    log.info("Computing SHAP values (primary seed, last fold LGB)")
    try:
        import shap
        last_lgb = joblib.load(MODELS_DIR / f"lgb_fold{N_FOLDS - 1}.pkl")

        feature_cols_shap = [c for c in X_train.columns if X_train[c].dtype in
                            [np.float64, np.float32, np.int64, np.int32, np.float16, np.uint8]]
        X_shap = X_train[feature_cols_shap]

        explainer = shap.TreeExplainer(last_lgb)
        shap_values = explainer.shap_values(X_shap[:5000])

        if isinstance(shap_values, list):
            shap_vals = shap_values[1]
        else:
            shap_vals = shap_values

        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_imp = pd.Series(mean_abs_shap, index=feature_cols_shap).sort_values(ascending=False)
        shap_imp.to_csv(OUTPUT_DIR / "shap_importance_v5.csv")
        log.info("Top 20 SHAP features:\n%s", shap_imp.head(20).to_string())

        if len(shap_imp) > 1:
            top_ratio = shap_imp.iloc[0] / max(shap_imp.iloc[1], 1e-10)
            if top_ratio > 4.0:
                log.warning("LEAKAGE WARNING: Top feature '%s' has %.1fx higher SHAP than #2 '%s'",
                           shap_imp.index[0], top_ratio, shap_imp.index[1])
    except Exception as e:
        log.warning("SHAP analysis failed: %s", e)

    # ── Save outputs ──
    joblib.dump(meta, MODELS_DIR / "meta_learner_v5.pkl")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "account_id": X_train.index,
        "y_true": y_train.values,
        "oof_lgb": oof_lgb_all,
        "oof_xgb": oof_xgb_all,
        "oof_cb": oof_cb_all,
        "oof_ensemble": oof_ensemble,
    })
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions_v5.csv", index=False)

    # Return predictions
    mule_predictions = pd.Series(test_final, index=X_test.index, name="mule_prob")

    results = {
        "test_predictions": test_final,
        "test_accounts": X_test.index,
        "mule_predictions": mule_predictions,
        "oof_auc": best_auc,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "ensemble_method": best_method,
        "individual_aucs": {"lgb": auc_lgb, "xgb": auc_xgb, "cb": auc_cb},
    }

    log.info("V5 Training complete. Best ensemble AUC: %.5f (%s)", best_auc, best_method)

    return results
