"""Run Optuna HPO for XGB and CB only, then retrain with best params."""
import sys
import time
import numpy as np
import pandas as pd
import joblib
import gc

from config import (
    OUTPUT_DIR, FEATURES_DIR, MODELS_DIR, FULL_FEATURES_PATH,
    TRAIN_LABELS_PATH, TEST_ACCOUNTS_PATH, log, MULE_RATIO,
)

t0 = time.time()

# Load cached features
log.info("Loading cached features")
features = pd.read_parquet(FULL_FEATURES_PATH)
labels = pd.read_parquet(TRAIN_LABELS_PATH)

train_ids = labels["account_id"].values
X_train = features.loc[features.index.isin(train_ids)].copy()
X_test = features.loc[features.index.isin(set(pd.read_parquet(TEST_ACCOUNTS_PATH)["account_id"]))].copy()

labels_indexed = labels.set_index("account_id")
y_train = labels_indexed.loc[X_train.index, "is_mule"]

# Load sample weights
weights_path = OUTPUT_DIR / "sample_weights.npy"
sample_weights = np.load(weights_path) if weights_path.exists() else None

# Prep features (same as models.py)
te_cols = ["branch_code"] if "branch_code" in X_train.columns else []
drop_cols = [c for c in X_train.columns if X_train[c].dtype == "object" and c not in te_cols]
X_train = X_train.drop(columns=drop_cols, errors="ignore")
X_test = X_test.drop(columns=drop_cols, errors="ignore")
common_cols = sorted(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
numeric_cols = [c for c in common_cols if c not in te_cols]
X_hpo = X_train[numeric_cols].copy()

log.info("X_hpo shape: %s, y_train: %d mules / %d total", X_hpo.shape, y_train.sum(), len(y_train))

# Load existing best params
saved = joblib.load(MODELS_DIR / "best_params.pkl")
lgb_best = saved["lgb"]
log.info("Keeping LGB params: %s", lgb_best)

# Run Optuna for XGB
from models import _optimize_xgb, _optimize_catboost
log.info("═══ Running Optuna for XGBoost (50 trials) ═══")
xgb_best = _optimize_xgb(X_hpo, y_train, sample_weights, n_trials=50)

log.info("═══ Running Optuna for CatBoost (30 trials) ═══")
cb_best = _optimize_catboost(X_hpo, y_train, sample_weights, n_trials=30)

# Save updated params
new_params = {"lgb": lgb_best, "xgb": xgb_best, "catboost": cb_best}
joblib.dump(new_params, MODELS_DIR / "best_params_v2.pkl")
log.info("Saved updated params to best_params_v2.pkl")
log.info("LGB: %s", lgb_best)
log.info("XGB: %s", xgb_best)
log.info("CB: %s", cb_best)

elapsed = time.time() - t0
log.info("Optuna complete in %.1f minutes", elapsed / 60)
