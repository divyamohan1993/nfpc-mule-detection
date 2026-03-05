"""
NFPC Phase 2 — Label Cleaning & Red Herring Detection

Research-grade noisy label handling:
  1. Confident Learning (Cleanlab) — identify samples where the label
     is likely wrong based on out-of-fold predicted probabilities
  2. Heuristic noise scoring — flag "Routine Investigation" and null
     alert_reason labels as potentially noisy
  3. Multi-round iterative cleaning — train → clean → retrain
  4. Sample weight generation — downweight noisy samples instead of
     removing them (preserves signal while reducing noise impact)

References:
  - Northcutt et al. (2021) "Confident Learning: Estimating Uncertainty
    in Dataset Labels" — Journal of AI Research
  - Wei et al. (2022) "Learning with Noisy Labels Revisited"
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

from config import (
    TRAIN_LABELS_PATH, OUTPUT_DIR, N_FOLDS, SEED, MULE_RATIO, log,
)


def compute_heuristic_noise_scores(labels_df: pd.DataFrame) -> pd.Series:
    """
    Assign noise likelihood scores based on label metadata.

    Higher score = more likely to be a red herring / noisy label.
    """
    scores = pd.Series(0.0, index=labels_df.index)

    mules = labels_df["is_mule"] == 1

    # "Routine Investigation" is a weak signal — high noise likelihood
    routine = labels_df["alert_reason"] == "Routine Investigation"
    scores[mules & routine] = 0.6

    # No alert_reason despite being flagged as mule — very suspicious
    no_reason = labels_df["alert_reason"].isna() | (labels_df["alert_reason"] == "")
    scores[mules & no_reason] = 0.8

    # Future mule_flag_date (beyond 2025-06-30) — potential data artifact
    if "mule_flag_date" in labels_df.columns:
        flag_date = pd.to_datetime(labels_df["mule_flag_date"], errors="coerce")
        future_flag = flag_date > pd.Timestamp("2025-06-30")
        scores[mules & future_flag] = scores[mules & future_flag].clip(lower=0.5)

    # Very early flag dates (before 2020) — suspicious for 5-year window
    early_flag = flag_date < pd.Timestamp("2020-01-01")
    scores[mules & early_flag] = scores[mules & early_flag].clip(lower=0.5)

    return scores


def confident_learning_clean(
    X: pd.DataFrame,
    y: pd.Series,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use confident learning to identify likely mislabeled samples.

    Returns:
        noise_mask: boolean array, True for likely noisy samples
        oof_probs: out-of-fold predicted probabilities
    """
    log.info("Running confident learning with %d folds", n_folds)

    oof_probs = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        log.info("  Fold %d/%d", fold + 1, n_folds)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=7,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=MULE_RATIO,
            random_state=seed + fold,
            verbosity=-1,
            n_jobs=-1,
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

    # Confident learning threshold
    # For each class, find the self-confidence threshold
    pos_probs = oof_probs[y == 1]
    neg_probs = oof_probs[y == 0]

    # Threshold: expected probability given the class
    t_pos = np.mean(pos_probs)  # avg predicted prob for actual positives
    t_neg = np.mean(neg_probs)  # avg predicted prob for actual negatives

    log.info("Confident learning thresholds: t_pos=%.4f, t_neg=%.4f", t_pos, t_neg)

    # Identify noisy samples:
    # - Labeled positive but predicted prob < t_neg (confident negative → likely mislabeled)
    # - Labeled negative but predicted prob > t_pos (confident positive → likely mislabeled)
    noise_mask = np.zeros(len(y), dtype=bool)
    noise_mask[(y == 1) & (oof_probs < t_neg)] = True
    noise_mask[(y == 0) & (oof_probs > t_pos)] = True

    n_noisy = noise_mask.sum()
    n_noisy_pos = ((y == 1) & noise_mask).sum()
    n_noisy_neg = ((y == 0) & noise_mask).sum()
    log.info(
        "Confident learning found %d noisy samples: %d mislabeled positives, %d mislabeled negatives",
        n_noisy, n_noisy_pos, n_noisy_neg,
    )

    return noise_mask, oof_probs


def compute_sample_weights(
    labels_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
) -> np.ndarray:
    """
    Compute sample weights combining heuristic and confident learning signals.

    Strategy:
      - Start with weight=1.0 for all samples
      - Downweight heuristically suspicious labels
      - Further downweight confident learning noise candidates
      - Never zero out completely (preserve some signal)
    """
    heuristic_scores = compute_heuristic_noise_scores(labels_df)
    noise_mask, oof_probs = confident_learning_clean(X, y)

    # Base weights
    weights = np.ones(len(y))

    # Heuristic downweighting: weight = 1 - 0.5 * noise_score
    weights *= (1.0 - 0.5 * heuristic_scores.values)

    # Confident learning downweighting
    weights[noise_mask] *= 0.3  # heavy downweight for CL-identified noise

    # Floor: minimum weight 0.1 (never fully remove)
    weights = np.clip(weights, 0.1, 1.0)

    # Log summary
    log.info(
        "Sample weights — mean: %.3f, min: %.3f, n_downweighted: %d",
        weights.mean(), weights.min(), (weights < 0.9).sum(),
    )

    # Save noise analysis
    noise_report = labels_df[["account_id", "is_mule", "alert_reason"]].copy()
    noise_report["heuristic_noise_score"] = heuristic_scores.values
    noise_report["cl_noise_flag"] = noise_mask.astype(int)
    noise_report["oof_probability"] = oof_probs
    noise_report["sample_weight"] = weights
    noise_report.to_parquet(OUTPUT_DIR / "noise_analysis.parquet", index=False)
    log.info("Saved noise analysis report")

    return weights


if __name__ == "__main__":
    # Standalone test
    labels = pd.read_parquet(TRAIN_LABELS_PATH)
    scores = compute_heuristic_noise_scores(labels)
    print(f"Heuristic noise scores > 0.5: {(scores > 0.5).sum()}")
    print(f"Breakdown:\n{scores.value_counts().sort_index()}")
