"""
NFPC Phase 2 — Model Training & Inference (V6)

Key changes from V3:
  1. All V3 improvements retained
  2. Pseudo-labeling: use confident predictions to augment training
  3. Two-stage training: first train on original labels, then retrain
     with pseudo-labeled test data added
  4. Uses V2 params (best known public score 0.968)
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
    freq = X_train[col].value_counts(normalize=True)
    X_train_enc = X_train[col].map(freq).fillna(0.0)
    X_test_enc = X_test[col].map(freq).fillna(0.0)
    counts = X_train[col].value_counts()
    X_train_cnt = X_train[col].map(counts).fillna(0).astype(float)
    X_test_cnt = X_test[col].map(counts).fillna(0).astype(float)
    return X_train_enc, X_test_enc, X_train_cnt, X_test_cnt


def _rank_average(*arrays):
    n = len(arrays[0])
    ranked = np.zeros(n)
    for arr in arrays:
        ranked += rankdata(arr) / n
    ranked /= len(arrays)
    return ranked


# ═══════════════════════════════════════════════════════════════════════
# MAIN TRAINING PIPELINE (V6 — with pseudo-labeling)
# ═══════════════════════════════════════════════════════════════════════

def train_and_predict(features, sample_weights=None, skip_optuna=False):
    """
    V6 training pipeline with pseudo-labeling:
      Stage A: Train on original labels, get test predictions
      Stage B: Add confident pseudo-labels to training, retrain
    """
    from catboost import CatBoostClassifier

    labels = pd.read_parquet(TRAIN_LABELS_PATH)
    test_accounts = pd.read_parquet(TEST_ACCOUNTS_PATH)

    train_ids = labels["account_id"].values
    test_ids = test_accounts["account_id"].values

    X_all = features
    X_train_orig = X_all.loc[X_all.index.isin(train_ids)].copy()
    X_test = X_all.loc[X_all.index.isin(test_ids)].copy()

    labels_indexed = labels.set_index("account_id")
    y_train_orig = labels_indexed.loc[X_train_orig.index, "is_mule"]

    log.info("Train: %d samples (%d mules, %.2f%%)",
             len(X_train_orig), y_train_orig.sum(), 100 * y_train_orig.mean())
    log.info("Test: %d samples", len(X_test))

    # Frequency encoding for categoricals
    cat_cols = []
    if "branch_code" in X_train_orig.columns:
        cat_cols.append("branch_code")

    for col in cat_cols:
        freq_tr, freq_te, cnt_tr, cnt_te = _frequency_encode(X_train_orig, X_test, col)
        X_train_orig[f"{col}_freq"] = freq_tr.values
        X_test[f"{col}_freq"] = freq_te.values
        X_train_orig[f"{col}_count"] = cnt_tr.values
        X_test[f"{col}_count"] = cnt_te.values
        X_train_orig = X_train_orig.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

    drop_cols = [c for c in X_train_orig.columns if X_train_orig[c].dtype == "object"]
    X_train_orig = X_train_orig.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    common_cols = sorted(set(X_train_orig.columns) & set(X_test.columns))
    X_train_orig = X_train_orig[common_cols]
    X_test = X_test[common_cols]

    log.info("Feature count: %d", len(common_cols))

    # Load hyperparameters (prefer V2)
    lgb_best_params = None
    xgb_best_params = None
    cb_best_params = None
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

    if lgb_best_params is None:
        lgb_best_params = {"learning_rate": 0.01, "max_depth": 6, "num_leaves": 31,
                           "min_child_samples": 50, "subsample": 0.7, "colsample_bytree": 0.6,
                           "reg_alpha": 0.01, "reg_lambda": 0.1}
        xgb_best_params = {"learning_rate": 0.01, "max_depth": 6, "min_child_weight": 20,
                           "subsample": 0.7, "colsample_bytree": 0.6,
                           "reg_alpha": 0.01, "reg_lambda": 0.1, "gamma": 0.0}
        cb_best_params = {"learning_rate": 0.02, "depth": 6, "l2_leaf_reg": 0.1,
                          "bagging_temperature": 1.0, "random_strength": 1.0, "border_count": 128}

    log.info("LGB params: %s", lgb_best_params)
    log.info("XGB params: %s", xgb_best_params)
    log.info("CB params: %s", cb_best_params)

    # ════════════════════════════════════════════════
    # STAGE A: Initial training (same as V3)
    # ════════════════════════════════════════════════
    log.info("═══ STAGE A: Initial training ═══")

    test_preds_a = _train_stage(
        X_train_orig, y_train_orig, X_test, sample_weights,
        lgb_best_params, xgb_best_params, cb_best_params,
        stage_name="A", save_models=True,
    )

    # ════════════════════════════════════════════════
    # STAGE B: Pseudo-labeling
    # ════════════════════════════════════════════════
    log.info("═══ STAGE B: Pseudo-labeling ═══")

    # Select confident pseudo-labels
    CONF_HIGH = 0.85  # confident mule
    CONF_LOW = 0.10   # confident non-mule

    test_probs = test_preds_a["test_ensemble"]

    high_mask = test_probs >= CONF_HIGH
    low_mask = test_probs <= CONF_LOW
    confident_mask = high_mask | low_mask

    n_pseudo_mule = high_mask.sum()
    n_pseudo_legit = low_mask.sum()
    n_pseudo = confident_mask.sum()

    log.info("Pseudo-labels: %d mules (>=%.2f), %d legit (<=%.2f), %d total (%.1f%% of test)",
             n_pseudo_mule, CONF_HIGH, n_pseudo_legit, CONF_LOW,
             n_pseudo, 100 * n_pseudo / len(X_test))

    if n_pseudo > 100:
        # Create pseudo-labeled dataset
        X_pseudo = X_test[confident_mask].copy()
        y_pseudo = pd.Series(
            (test_probs[confident_mask] >= 0.5).astype(int),
            index=X_pseudo.index,
            name="is_mule"
        )

        # Combine with original training data
        X_train_pl = pd.concat([X_train_orig, X_pseudo])
        y_train_pl = pd.concat([y_train_orig, y_pseudo])

        # Create sample weights for pseudo-labeled data
        # Original data gets weight 1.0, pseudo-labeled gets lower weight
        PSEUDO_WEIGHT = 0.5
        if sample_weights is not None:
            sw_pl = np.concatenate([
                sample_weights,
                np.full(len(X_pseudo), PSEUDO_WEIGHT)
            ])
        else:
            sw_pl = np.concatenate([
                np.ones(len(X_train_orig)),
                np.full(len(X_pseudo), PSEUDO_WEIGHT)
            ])

        log.info("Combined training: %d original + %d pseudo = %d total (%d mules, %.2f%%)",
                 len(X_train_orig), len(X_pseudo), len(X_train_pl),
                 y_train_pl.sum(), 100 * y_train_pl.mean())

        # Retrain with pseudo-labels
        test_preds_b = _train_stage(
            X_train_pl, y_train_pl, X_test, sw_pl,
            lgb_best_params, xgb_best_params, cb_best_params,
            stage_name="B", save_models=False,
        )

        # Use stage B predictions
        test_final = test_preds_b["test_ensemble"]
        oof_auc = test_preds_b["oof_auc"]
        best_method = test_preds_b["ensemble_method"]
        oof_ensemble = test_preds_b["oof_ensemble"]
        y_for_threshold = y_train_pl
    else:
        log.info("Too few pseudo-labels, using Stage A results")
        test_final = test_preds_a["test_ensemble"]
        oof_auc = test_preds_a["oof_auc"]
        best_method = test_preds_a["ensemble_method"]
        oof_ensemble = test_preds_a["oof_ensemble"]
        y_for_threshold = y_train_orig

    # Threshold optimization
    precision, recall, thresholds = precision_recall_curve(y_for_threshold, oof_ensemble)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_f1_idx]

    log.info("Optimal threshold: %.4f (F1=%.4f)", best_threshold, best_f1)

    mule_predictions = pd.Series(test_final, index=X_test.index, name="mule_prob")

    return {
        "test_predictions": test_final,
        "test_accounts": X_test.index,
        "mule_predictions": mule_predictions,
        "oof_auc": oof_auc,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "ensemble_method": best_method,
    }


def _train_stage(X_train, y_train, X_test, sample_weights,
                 lgb_params_base, xgb_params_base, cb_params_base,
                 stage_name="", save_models=False):
    """Run a full training stage: multi-seed CV with 3 models."""
    from catboost import CatBoostClassifier

    seeds = [SEED, SEED + 1, SEED + 2]
    n_seeds = len(seeds)

    oof_lgb_all = np.zeros(len(X_train))
    oof_xgb_all = np.zeros(len(X_train))
    oof_cb_all = np.zeros(len(X_train))
    test_lgb_all = np.zeros(len(X_test))
    test_xgb_all = np.zeros(len(X_test))
    test_cb_all = np.zeros(len(X_test))

    for seed_idx, seed in enumerate(seeds):
        log.info("═══ Stage %s Seed %d/%d (seed=%d) ═══", stage_name, seed_idx + 1, n_seeds, seed)

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

            feature_cols = [c for c in X_tr.columns if X_tr[c].dtype in
                          [np.float64, np.float32, np.int64, np.int32, np.float16, np.uint8]]
            X_tr = X_tr[feature_cols]
            X_val = X_val[feature_cols]
            X_te = X_te[feature_cols]

            # LightGBM
            lgb_p = {"n_estimators": 5000, "scale_pos_weight": MULE_RATIO,
                     "random_state": seed, "verbosity": -1, "n_jobs": -1,
                     **lgb_params_base}
            m_lgb = lgb.LGBMClassifier(**lgb_p)
            m_lgb.fit(X_tr, y_tr, sample_weight=sw_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
            oof_lgb[val_idx] = m_lgb.predict_proba(X_val)[:, 1]
            test_lgb += m_lgb.predict_proba(X_te)[:, 1] / N_FOLDS

            # XGBoost
            xgb_p = {"n_estimators": 5000, "scale_pos_weight": MULE_RATIO,
                     "random_state": seed, "verbosity": 0, "n_jobs": -1,
                     "tree_method": "hist", "eval_metric": "auc",
                     "early_stopping_rounds": 150,
                     **xgb_params_base}
            m_xgb = xgb.XGBClassifier(**xgb_p)
            m_xgb.fit(X_tr, y_tr, sample_weight=sw_tr,
                      eval_set=[(X_val, y_val)], verbose=False)
            oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
            test_xgb += m_xgb.predict_proba(X_te)[:, 1] / N_FOLDS

            # CatBoost
            cb_p = {"iterations": 5000, "scale_pos_weight": MULE_RATIO,
                    "random_seed": seed, "verbose": 100, "thread_count": -1,
                    "eval_metric": "AUC", "od_type": "Iter", "od_wait": 150,
                    **cb_params_base}
            m_cb = CatBoostClassifier(**cb_p)
            m_cb.fit(X_tr, y_tr, sample_weight=sw_tr,
                     eval_set=(X_val, y_val))
            oof_cb[val_idx] = m_cb.predict_proba(X_val)[:, 1]
            test_cb += m_cb.predict_proba(X_te)[:, 1] / N_FOLDS

            fold_auc = roc_auc_score(y_val, oof_lgb[val_idx])
            fold_auc_x = roc_auc_score(y_val, oof_xgb[val_idx])
            fold_auc_c = roc_auc_score(y_val, oof_cb[val_idx])
            log.info("  Fold %d AUC — LGB: %.5f, XGB: %.5f, CB: %.5f",
                     fold + 1, fold_auc, fold_auc_x, fold_auc_c)

            if save_models and seed_idx == 0:
                joblib.dump(m_lgb, MODELS_DIR / f"lgb_fold{fold}.pkl")
                joblib.dump(m_xgb, MODELS_DIR / f"xgb_fold{fold}.pkl")
                m_cb.save_model(str(MODELS_DIR / f"cb_fold{fold}.cbm"))

            del m_lgb, m_xgb, m_cb
            gc.collect()

        oof_lgb_all += oof_lgb / n_seeds
        oof_xgb_all += oof_xgb / n_seeds
        oof_cb_all += oof_cb / n_seeds
        test_lgb_all += test_lgb / n_seeds
        test_xgb_all += test_xgb / n_seeds
        test_cb_all += test_cb / n_seeds

        auc_l = roc_auc_score(y_train, oof_lgb)
        auc_x = roc_auc_score(y_train, oof_xgb)
        auc_c = roc_auc_score(y_train, oof_cb)
        log.info("Seed %d OOF AUC — LGB: %.5f, XGB: %.5f, CB: %.5f", seed, auc_l, auc_x, auc_c)

    # Ensemble
    auc_lgb = roc_auc_score(y_train, oof_lgb_all)
    auc_xgb = roc_auc_score(y_train, oof_xgb_all)
    auc_cb = roc_auc_score(y_train, oof_cb_all)
    log.info("Stage %s Overall OOF — LGB: %.5f, XGB: %.5f, CB: %.5f", stage_name, auc_lgb, auc_xgb, auc_cb)

    oof_rank = _rank_average(oof_lgb_all, oof_xgb_all, oof_cb_all)
    test_rank = _rank_average(test_lgb_all, test_xgb_all, test_cb_all)
    auc_rank = roc_auc_score(y_train, oof_rank)

    oof_avg = (oof_lgb_all + oof_xgb_all + oof_cb_all) / 3
    test_avg = (test_lgb_all + test_xgb_all + test_cb_all) / 3
    auc_avg = roc_auc_score(y_train, oof_avg)

    aucs = np.array([auc_lgb, auc_xgb, auc_cb])
    w = aucs / aucs.sum()
    oof_wavg = w[0] * oof_lgb_all + w[1] * oof_xgb_all + w[2] * oof_cb_all
    test_wavg = w[0] * test_lgb_all + w[1] * test_xgb_all + w[2] * test_cb_all
    auc_wavg = roc_auc_score(y_train, oof_wavg)

    oof_stack = np.column_stack([oof_lgb_all, oof_xgb_all, oof_cb_all])
    test_stack = np.column_stack([test_lgb_all, test_xgb_all, test_cb_all])
    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
    meta.fit(oof_stack, y_train)
    oof_meta = meta.predict_proba(oof_stack)[:, 1]
    test_meta = meta.predict_proba(test_stack)[:, 1]
    auc_meta = roc_auc_score(y_train, oof_meta)

    log.info("Stage %s — rank_avg: %.5f, simple_avg: %.5f, weighted: %.5f, stacking: %.5f",
             stage_name, auc_rank, auc_avg, auc_wavg, auc_meta)

    methods = {
        "rank_avg": (oof_rank, test_rank, auc_rank),
        "simple_avg": (oof_avg, test_avg, auc_avg),
        "auc_weighted": (oof_wavg, test_wavg, auc_wavg),
        "stacking": (oof_meta, test_meta, auc_meta),
    }
    best_method = max(methods, key=lambda k: methods[k][2])
    oof_ensemble, test_ensemble, best_auc = methods[best_method]
    log.info("Stage %s selected: %s (AUC=%.5f)", stage_name, best_method, best_auc)

    return {
        "test_ensemble": test_ensemble,
        "oof_ensemble": oof_ensemble,
        "oof_auc": best_auc,
        "ensemble_method": best_method,
    }
