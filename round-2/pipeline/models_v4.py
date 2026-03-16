"""
NFPC Phase 2 — Model Training & Inference (V4)

Key changes from V3:
  1. Both frequency AND target encoding (combined signals)
  2. Test WITHOUT sample weights (label cleaning may hurt)
  3. Feature interactions between top discriminators
  4. Proper F1 optimization with calibrated probabilities
  5. Keep rank averaging ensemble
  6. Multi-seed stability
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve
from scipy.stats import rankdata
import joblib
import gc

from config import (
    TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH, FULL_FEATURES_PATH,
    OUTPUT_DIR, MODELS_DIR, N_FOLDS, SEED, MULE_RATIO, log,
)


def _frequency_encode(X_train, X_test, col):
    """Frequency + count encoding for categorical."""
    freq = X_train[col].value_counts(normalize=True)
    counts = X_train[col].value_counts()
    return (
        X_train[col].map(freq).fillna(0.0),
        X_test[col].map(freq).fillna(0.0),
        X_train[col].map(counts).fillna(0).astype(float),
        X_test[col].map(counts).fillna(0).astype(float),
    )


def _target_encode_fold(X_train, y_train, X_val, X_test, col, global_mean,
                         min_samples=30, smoothing=20):
    """LOO target encoding within a single fold."""
    stats = y_train.groupby(X_train[col]).agg(["mean", "count"])
    stats.columns = ["mean_target", "n"]
    stats["encoded"] = (
        (stats["n"] * stats["mean_target"] + smoothing * global_mean)
        / (stats["n"] + smoothing)
    )
    mapping = stats["encoded"].to_dict()
    return (
        X_train[col].map(mapping).fillna(global_mean),
        X_val[col].map(mapping).fillna(global_mean),
        X_test[col].map(mapping).fillna(global_mean) if X_test is not None else None,
    )


def _rank_average(*arrays):
    """Rank-average multiple prediction arrays."""
    n = len(arrays[0])
    ranked = np.zeros(n)
    for arr in arrays:
        ranked += rankdata(arr) / n
    ranked /= len(arrays)
    return ranked


def _add_feature_interactions(df, top_features, max_interactions=15):
    """Add interaction features between top discriminators."""
    added = 0
    for i in range(len(top_features)):
        if added >= max_interactions:
            break
        for j in range(i + 1, len(top_features)):
            if added >= max_interactions:
                break
            f1, f2 = top_features[i], top_features[j]
            if f1 in df.columns and f2 in df.columns:
                # Ratio
                denom = df[f2].abs() + 1e-8
                df[f"{f1}_div_{f2}"] = df[f1] / denom
                # Product
                df[f"{f1}_x_{f2}"] = df[f1] * df[f2]
                added += 2
    return df


def train_and_predict(features, sample_weights=None, skip_optuna=False):
    """
    V4 training pipeline:
      1. Frequency encoding for all (safe, no leakage)
      2. Target encoding WITHIN folds (captures branch-level signal)
      3. Feature interactions between top features
      4. N-fold CV: LGB + XGB + CatBoost (multi-seed)
      5. Rank averaging + probability averaging ensemble
      6. Isotonic calibration for F1, raw for AUC
    """
    from catboost import CatBoostClassifier

    labels = pd.read_parquet(TRAIN_LABELS_PATH)
    test_accounts = pd.read_parquet(TEST_ACCOUNTS_PATH)

    train_ids = labels["account_id"].values
    test_ids = test_accounts["account_id"].values

    X_all = features
    X_train = X_all.loc[X_all.index.isin(train_ids)].copy()
    X_test = X_all.loc[X_all.index.isin(test_ids)].copy()

    labels_indexed = labels.set_index("account_id")
    y_train = labels_indexed.loc[X_train.index, "is_mule"]

    log.info("Train: %d samples (%d mules, %.2f%%)",
             len(X_train), y_train.sum(), 100 * y_train.mean())
    log.info("Test: %d samples", len(X_test))

    # ── Step 1: Frequency encoding (safe, applied globally) ──
    te_cols = []
    if "branch_code" in X_train.columns:
        te_cols.append("branch_code")

    for col in te_cols:
        freq_tr, freq_te, cnt_tr, cnt_te = _frequency_encode(X_train, X_test, col)
        X_train[f"{col}_freq"] = freq_tr.values
        X_test[f"{col}_freq"] = freq_te.values
        X_train[f"{col}_count"] = cnt_tr.values
        X_test[f"{col}_count"] = cnt_te.values

    # Drop non-numeric (keep branch_code for target encoding inside folds)
    drop_cols = [c for c in X_train.columns if X_train[c].dtype == "object" and c not in te_cols]
    X_train = X_train.drop(columns=drop_cols, errors="ignore")
    X_test = X_test.drop(columns=drop_cols, errors="ignore")

    # Ensure consistent columns
    common_cols = sorted(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    log.info("Feature count: %d (freq + target encoding within folds)", len(common_cols))

    # ── Step 2: Add feature interactions from V3 SHAP analysis ──
    top_features = [
        "mcc_hhi", "graph_degree_ratio", "graph_out_degree_count",
        "was_frozen", "hour_entropy", "dormancy_events",
        "balance_consistency", "cp_entropy", "account_status_frozen",
        "burstiness",
    ]
    existing_top = [f for f in top_features if f in X_train.columns]
    if len(existing_top) >= 2:
        n_before = len(X_train.columns)
        X_train = _add_feature_interactions(X_train, existing_top, max_interactions=20)
        X_test = _add_feature_interactions(X_test, existing_top, max_interactions=20)
        log.info("Added %d interaction features", len(X_train.columns) - n_before)

    # ── Step 3: Load or use default hyperparameters ──
    # Try to load existing params
    for path_name in ["best_params_v3.pkl", "best_params_v2.pkl", "best_params.pkl"]:
        p = MODELS_DIR / path_name
        if p.exists():
            log.info("Loading params from %s", path_name)
            saved = joblib.load(p)
            lgb_best_params = saved["lgb"]
            xgb_best_params = saved["xgb"]
            cb_best_params = saved["catboost"]
            break
    else:
        log.info("Using default V4 hyperparameters")
        lgb_best_params = {
            "learning_rate": 0.01, "max_depth": 6, "num_leaves": 31,
            "min_child_samples": 50, "subsample": 0.7, "colsample_bytree": 0.6,
            "reg_alpha": 1.0, "reg_lambda": 5.0,
        }
        xgb_best_params = {
            "learning_rate": 0.01, "max_depth": 6, "min_child_weight": 20,
            "subsample": 0.7, "colsample_bytree": 0.6,
            "reg_alpha": 1.0, "reg_lambda": 5.0, "gamma": 1.0,
        }
        cb_best_params = {
            "learning_rate": 0.02, "depth": 6, "l2_leaf_reg": 5.0,
            "bagging_temperature": 1.0, "random_strength": 1.0, "border_count": 128,
        }

    log.info("LGB params: %s", lgb_best_params)
    log.info("XGB params: %s", xgb_best_params)
    log.info("CB params: %s", cb_best_params)

    # ── Step 4: Multi-seed N-fold CV ──
    # Run both WITH and WITHOUT sample weights, pick best
    weight_configs = [
        ("no_weights", None),
    ]
    if sample_weights is not None:
        weight_configs.append(("with_weights", sample_weights))

    best_config_auc = -1
    best_config_results = None

    for config_name, sw in weight_configs:
        log.info("══════ Config: %s ══════", config_name)

        seeds = [SEED, SEED + 1, SEED + 2]
        n_seeds = len(seeds)

        oof_lgb_all = np.zeros(len(X_train))
        oof_xgb_all = np.zeros(len(X_train))
        oof_cb_all = np.zeros(len(X_train))

        test_lgb_all = np.zeros(len(X_test))
        test_xgb_all = np.zeros(len(X_test))
        test_cb_all = np.zeros(len(X_test))

        global_mean = y_train.mean()

        for seed_idx, seed in enumerate(seeds):
            log.info("  Seed %d/%d (seed=%d)", seed_idx + 1, n_seeds, seed)

            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

            oof_lgb = np.zeros(len(X_train))
            oof_xgb = np.zeros(len(X_train))
            oof_cb = np.zeros(len(X_train))

            test_lgb = np.zeros(len(X_test))
            test_xgb = np.zeros(len(X_test))
            test_cb = np.zeros(len(X_test))

            for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                log.info("    Fold %d/%d", fold + 1, N_FOLDS)

                X_tr = X_train.iloc[tr_idx].copy()
                X_val = X_train.iloc[val_idx].copy()
                X_te = X_test.copy()
                y_tr = y_train.iloc[tr_idx]
                y_val = y_train.iloc[val_idx]

                sw_tr = sw[tr_idx] if sw is not None else None

                # Target encoding within fold
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

                # All numeric
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
                log.info("    Fold %d AUC — LGB: %.5f, XGB: %.5f, CB: %.5f",
                         fold + 1, fold_auc_lgb, fold_auc_xgb, fold_auc_cb)

                # Save models for primary seed + no_weights config
                if seed_idx == 0 and config_name == "no_weights":
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

        # Compute ensemble scores
        auc_lgb = roc_auc_score(y_train, oof_lgb_all)
        auc_xgb = roc_auc_score(y_train, oof_xgb_all)
        auc_cb = roc_auc_score(y_train, oof_cb_all)
        log.info("  %s OOF — LGB: %.5f, XGB: %.5f, CB: %.5f", config_name, auc_lgb, auc_xgb, auc_cb)

        # Rank average
        oof_rank = _rank_average(oof_lgb_all, oof_xgb_all, oof_cb_all)
        test_rank = _rank_average(test_lgb_all, test_xgb_all, test_cb_all)
        auc_rank = roc_auc_score(y_train, oof_rank)

        # Simple average
        oof_avg = (oof_lgb_all + oof_xgb_all + oof_cb_all) / 3
        test_avg = (test_lgb_all + test_xgb_all + test_cb_all) / 3
        auc_avg = roc_auc_score(y_train, oof_avg)

        # Stacking
        oof_stack = np.column_stack([oof_lgb_all, oof_xgb_all, oof_cb_all])
        test_stack = np.column_stack([test_lgb_all, test_xgb_all, test_cb_all])
        meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=SEED)
        meta.fit(oof_stack, y_train)
        oof_meta = meta.predict_proba(oof_stack)[:, 1]
        test_meta = meta.predict_proba(test_stack)[:, 1]
        auc_meta = roc_auc_score(y_train, oof_meta)

        log.info("  %s ensemble — rank: %.5f, avg: %.5f, stack: %.5f",
                 config_name, auc_rank, auc_avg, auc_meta)

        methods = {
            "rank_avg": (oof_rank, test_rank, auc_rank),
            "simple_avg": (oof_avg, test_avg, auc_avg),
            "stacking": (oof_meta, test_meta, auc_meta),
        }
        best_method = max(methods, key=lambda k: methods[k][2])
        oof_best, test_best, auc_best = methods[best_method]

        if auc_best > best_config_auc:
            best_config_auc = auc_best
            best_config_results = {
                "oof": oof_best,
                "test": test_best,
                "auc": auc_best,
                "method": f"{config_name}_{best_method}",
                "oof_lgb": oof_lgb_all,
                "oof_xgb": oof_xgb_all,
                "oof_cb": oof_cb_all,
                "test_lgb": test_lgb_all,
                "test_xgb": test_xgb_all,
                "test_cb": test_cb_all,
                "individual_aucs": {"lgb": auc_lgb, "xgb": auc_xgb, "cb": auc_cb},
                "meta": meta,
            }

    log.info("Best config: %s (AUC=%.5f)", best_config_results["method"], best_config_auc)

    oof_ensemble = best_config_results["oof"]
    test_ensemble = best_config_results["test"]

    # ── F1 optimization with isotonic calibration ──
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999)
    iso.fit(oof_ensemble, y_train)
    oof_cal = iso.predict(oof_ensemble)
    test_cal = iso.predict(test_ensemble)

    precision, recall, thresholds = precision_recall_curve(y_train, oof_cal)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_f1_idx]

    log.info("F1 optimal threshold: %.4f (F1=%.4f)", best_threshold, best_f1)

    # For submission, use raw ensemble for AUC (no calibration needed for ranking)
    # But normalize to [0, 1] range
    test_final = test_ensemble
    test_min, test_max = test_final.min(), test_final.max()
    if test_max > test_min:
        test_final_norm = (test_final - test_min) / (test_max - test_min)
    else:
        test_final_norm = test_final

    # For threshold-based decisions, use calibrated probabilities
    # For is_mule column, we want probabilities that:
    # 1. Preserve ranking (for AUC)
    # 2. Are well-calibrated (for F1)
    # Use calibrated if it preserves ranking better
    auc_raw = roc_auc_score(y_train, oof_ensemble)
    auc_cal = roc_auc_score(y_train, oof_cal)

    if auc_cal >= auc_raw - 0.001:
        test_output = test_cal.clip(0, 1)
        log.info("Using calibrated probabilities (AUC preserved: %.5f vs %.5f)", auc_cal, auc_raw)
    else:
        test_output = test_final_norm
        log.info("Using raw normalized probabilities (calibration hurt AUC: %.5f vs %.5f)", auc_cal, auc_raw)

    for t in [0.1, 0.2, 0.3, 0.5, best_threshold]:
        n_mules = (test_output >= t).sum()
        log.info("  Threshold %.2f -> %d predicted mules (%.1f%%)",
                 t, n_mules, 100 * n_mules / len(test_output))

    # ── Save ──
    joblib.dump(best_config_results["meta"], MODELS_DIR / "meta_learner_v4.pkl")
    joblib.dump(iso, MODELS_DIR / "isotonic_calibrator_v4.pkl")

    oof_df = pd.DataFrame({
        "account_id": X_train.index,
        "y_true": y_train.values,
        "oof_lgb": best_config_results["oof_lgb"],
        "oof_xgb": best_config_results["oof_xgb"],
        "oof_cb": best_config_results["oof_cb"],
        "oof_ensemble": oof_ensemble,
    })
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions_v4.csv", index=False)

    mule_predictions = pd.Series(test_output, index=X_test.index, name="mule_prob")

    results = {
        "test_predictions": test_output,
        "test_accounts": X_test.index,
        "mule_predictions": mule_predictions,
        "oof_auc": best_config_auc,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "ensemble_method": best_config_results["method"],
        "individual_aucs": best_config_results["individual_aucs"],
    }

    log.info("V4 Training complete. Best AUC: %.5f (%s)", best_config_auc, best_config_results["method"])
    return results
