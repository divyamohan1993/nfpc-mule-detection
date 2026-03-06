"""Quick submission generator using saved LGB fold models."""
import pandas as pd
import numpy as np
import joblib
from config import *

features = pd.read_parquet(FULL_FEATURES_PATH)
test_accounts = pd.read_parquet(TEST_ACCOUNTS_PATH)
labels = pd.read_parquet(TRAIN_LABELS_PATH)

test_ids = test_accounts["account_id"].values
train_ids = labels["account_id"].values
X_test = features.loc[features.index.isin(test_ids)].copy()
X_train = features.loc[features.index.isin(train_ids)].copy()
y_train = labels.set_index("account_id").loc[X_train.index, "is_mule"]

log.info("X_test: %s, X_train: %s", X_test.shape, X_train.shape)

# Target encode branch_code using full training data (same logic as models.py)
if "branch_code" in X_test.columns:
    global_mean = y_train.mean()
    stats = y_train.groupby(X_train["branch_code"]).agg(["mean", "count"])
    stats.columns = ["mean_target", "n"]
    stats["encoded"] = (stats["n"] * stats["mean_target"] + 20 * global_mean) / (stats["n"] + 20)
    mapping = stats["encoded"].to_dict()
    X_test["branch_code_te"] = X_test["branch_code"].map(mapping).fillna(global_mean)
    X_test = X_test.drop(columns=["branch_code"], errors="ignore")

# Get exact feature names from fold 0 model
m0 = joblib.load(MODELS_DIR / "lgb_fold0.pkl")
model_features = list(m0.feature_name_)
log.info("Model trained on %d features", len(model_features))

# Ensure all model features exist in X_test (fill missing with 0)
for col in model_features:
    if col not in X_test.columns:
        log.warning("Missing feature %s — filling with 0", col)
        X_test[col] = 0.0

X_test_np = X_test[model_features]
log.info("X_test aligned: %s", X_test_np.shape)

# LGB-only predictions (best model, no overfit issues)
test_lgb = np.zeros(len(X_test))
for fold in range(5):
    m = joblib.load(MODELS_DIR / f"lgb_fold{fold}.pkl")
    test_lgb += m.predict_proba(X_test_np)[:, 1] / 5
    log.info("LGB fold %d done", fold)

# Calibrate
iso = joblib.load(MODELS_DIR / "isotonic_calibrator.pkl")
test_cal = iso.predict(test_lgb)

# Submission
submission = pd.DataFrame({"account_id": test_ids})
pred_series = pd.Series(test_cal, index=X_test.index)
submission["is_mule"] = submission["account_id"].map(pred_series).fillna(0.0)
submission["suspicious_start"] = ""
submission["suspicious_end"] = ""
submission.to_csv(OUTPUT_DIR / "submission.csv", index=False)

n30 = (submission["is_mule"] >= 0.3).sum()
n50 = (submission["is_mule"] >= 0.5).sum()
log.info("Submission: %d rows, p>=0.3: %d (%.1f%%), p>=0.5: %d (%.1f%%)",
         len(submission), n30, 100*n30/len(submission), n50, 100*n50/len(submission))
log.info("Mean prob: %.4f, Max prob: %.4f", submission["is_mule"].mean(), submission["is_mule"].max())
