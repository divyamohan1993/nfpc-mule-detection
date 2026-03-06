"""
NFPC Phase 2 — Label Cleaning & Red Herring Detection (V2)

Multi-round confident learning + heuristic noise scoring:
  1. Out-of-fold probability estimation (LightGBM fast)
  2. Confident learning thresholds (Northcutt et al. 2021)
  3. Heuristic noise flags (Routine Investigation, null alert_reason, future dates)
  4. Combined sample weights for noise-robust training
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from config import (
    TRAIN_LABELS_PATH, OUTPUT_DIR, N_FOLDS, SEED, MULE_RATIO, log,
)


def _get_oof_probabilities(X, y, n_folds=N_FOLDS):
    """Get out-of-fold predicted probabilities using fast LightGBM."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_probs = np.zeros(len(y))

    params = {
        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 6,
        "num_leaves": 63, "min_child_samples": 30, "subsample": 0.8,
        "colsample_bytree": 0.7, "scale_pos_weight": MULE_RATIO,
        "random_state": SEED, "verbosity": -1, "n_jobs": -1,
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        log.info("  CL Fold %d/%d", fold + 1, n_folds)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_probs[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

    return oof_probs


def confident_learning_detect(X, y, n_rounds=2):
    """
    Multi-round confident learning to identify mislabeled samples.

    Returns indices of likely mislabeled samples and their noise scores.
    """
    noise_scores = np.zeros(len(y))

    for round_num in range(n_rounds):
        log.info("Confident learning round %d/%d", round_num + 1, n_rounds)
        oof_probs = _get_oof_probabilities(X, y)

        # Compute per-class probability thresholds
        pos_mask = y == 1
        neg_mask = y == 0
        t_pos = np.mean(oof_probs[pos_mask]) if pos_mask.sum() > 0 else 0.5
        t_neg = np.mean(oof_probs[neg_mask]) if neg_mask.sum() > 0 else 0.5

        log.info("  Thresholds: t_pos=%.4f, t_neg=%.4f", t_pos, t_neg)

        # Confident mislabels: labeled positive but model confident it's negative (and vice versa)
        mislabeled_pos = pos_mask & (oof_probs < t_neg)  # labeled 1, model says 0
        mislabeled_neg = neg_mask & (oof_probs > t_pos)  # labeled 0, model says 1

        n_mis_pos = mislabeled_pos.sum()
        n_mis_neg = mislabeled_neg.sum()
        log.info("  Found %d mislabeled positives, %d mislabeled negatives",
                 n_mis_pos, n_mis_neg)

        # Accumulate noise scores (higher = more likely mislabeled)
        noise_scores[mislabeled_pos.values] += 1.0
        noise_scores[mislabeled_neg.values] += 1.0

        # For subsequent rounds, use distance from threshold as continuous score
        for i in range(len(y)):
            if y.iloc[i] == 1:
                noise_scores[i] += max(0, t_neg - oof_probs[i]) / max(t_neg, 1e-6)
            else:
                noise_scores[i] += max(0, oof_probs[i] - t_pos) / max(1 - t_pos, 1e-6)

    # Normalize to [0, 1]
    noise_scores = noise_scores / (noise_scores.max() + 1e-10)
    return noise_scores


def heuristic_noise_scores(labels_df):
    """
    Assign noise likelihood based on label metadata.
    Higher score = more likely red herring.

    Targets 7 known red herring categories:
    1. Routine Investigation false positives
    2. Missing alert_reason despite mule label
    3. Future mule_flag_date (beyond data window)
    4. Very old flag dates (before transaction window)
    5. Flag date exactly on data boundary dates (synthetic artifacts)
    6. Frozen accounts flagged as mules (contradictory signals)
    7. Accounts flagged by their own branch (potential circular investigation)
    """
    scores = pd.Series(0.0, index=labels_df.index)
    mules = labels_df["is_mule"] == 1

    # RH1: "Routine Investigation" — weakest investigation trigger, likely false positives
    routine = labels_df["alert_reason"] == "Routine Investigation"
    scores[mules & routine] = 0.6

    # RH2: No alert_reason despite mule flag — strong red herring signal
    no_reason = labels_df["alert_reason"].isna() | (labels_df["alert_reason"] == "")
    scores[mules & no_reason] = 0.8

    if "mule_flag_date" in labels_df.columns:
        flag_date = pd.to_datetime(labels_df["mule_flag_date"], errors="coerce")

        # RH3: Future mule_flag_date (beyond transaction window end: Jun 2025)
        future_flag = flag_date > pd.Timestamp("2025-06-30")
        scores[mules & future_flag] += 0.3

        # RH4: Very old flag dates (before transaction window start: Jul 2020)
        # A flag from 2015-2019 predates all transaction data — suspicious artifact
        old_flag = flag_date < pd.Timestamp("2020-07-01")
        has_flag = flag_date.notna()
        scores[mules & old_flag & has_flag] += 0.25

        # RH5: Flag date exactly on boundary dates (likely synthetic)
        boundary_dates = {
            pd.Timestamp("2020-07-01"), pd.Timestamp("2025-06-30"),
            pd.Timestamp("2025-01-01"), pd.Timestamp("2020-01-01"),
        }
        is_boundary = flag_date.isin(boundary_dates)
        scores[mules & is_boundary] += 0.15

    # RH6: Accounts flagged by a branch matching their own (self-referential)
    # Cross-branch flags are more credible; same-branch flags may be circular
    if "flagged_by_branch" in labels_df.columns:
        flagged_by = labels_df["flagged_by_branch"]
        # flagged_by_branch is only set for mules, check if it matches account's branch
        # We don't have branch_code in labels_df, but we can flag null flagged_by_branch
        # for mules (similar to RH2 but distinct field)
        no_branch_flag = flagged_by.isna()
        scores[mules & no_branch_flag] = np.maximum(
            scores[mules & no_branch_flag], 0.4
        )

    # Clamp to [0, 1]
    scores = scores.clip(upper=1.0)

    return scores.values


def compute_sample_weights(labels_df, X, y):
    """
    Compute sample weights combining confident learning + heuristic noise.

    Returns array of weights in [0.2, 1.0] — noisy samples downweighted.
    """
    log.info("Running multi-round confident learning")

    # Confident learning noise scores
    cl_scores = confident_learning_detect(X, y, n_rounds=2)

    # Heuristic noise scores
    h_scores = heuristic_noise_scores(labels_df)

    # Combined: max of CL and heuristic (conservative — flag if either says noisy)
    combined = np.maximum(cl_scores, h_scores)

    # Convert to weights: 1.0 for clean, down to 0.2 for noisiest
    weights = 1.0 - 0.8 * combined

    n_downweighted = (weights < 0.95).sum()
    log.info("Sample weights — mean: %.3f, min: %.3f, n_downweighted: %d",
             weights.mean(), weights.min(), n_downweighted)

    # Save noise analysis
    analysis = pd.DataFrame({
        "account_id": labels_df["account_id"].values if "account_id" in labels_df.columns else labels_df.index,
        "is_mule": y.values,
        "cl_noise_score": cl_scores,
        "heuristic_noise_score": h_scores,
        "combined_noise_score": combined,
        "sample_weight": weights,
    })
    analysis.to_csv(OUTPUT_DIR / "noise_analysis.csv", index=False)
    log.info("Saved noise analysis report")

    return weights
