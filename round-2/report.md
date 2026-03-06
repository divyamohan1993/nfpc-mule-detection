# NFPC Phase 2 — Mule Account Detection Report

**Team**: Divya Mohan | **Date**: March 2026 | **Competition**: National Fraud Prevention Challenge, RBI Innovation Hub × IIT Delhi

---

## 1. Executive Summary

We present a multi-stage pipeline for detecting mule accounts used in money laundering from 400 million banking transactions across 160,000 accounts. Our approach achieves **0.940 calibrated AUC-ROC** and **0.756 F1** through:

- **192 engineered features** across 4 data passes (transaction statistics, geolocation/balance trajectories, static account/customer attributes, and network graph centrality)
- **Label noise detection** via 2-round confident learning + heuristic red herring scoring, producing sample weights that downweight likely mislabeled accounts
- **3-model ensemble** (LightGBM + XGBoost + CatBoost) with stacking meta-learner and isotonic calibration
- **Vectorized temporal window prediction** using searchsorted + cumulative sum tricks for O(n) sliding window anomaly scoring across 5 time scales

---

## 2. Data Understanding

### 2.1 Scale and Structure

| Dimension | Value |
|-----------|-------|
| Total accounts | 160,153 |
| Train labels | 96,091 (2,683 mules, 93,408 legitimate) |
| Test accounts | 64,062 |
| Mule ratio | 1:34 (2.79%) |
| Transactions | ~400 million across 396 parquet parts |
| Transaction window | Jul 2020 — Jun 2025 (5 years) |
| Total data size | 16.2 GB |

### 2.2 Key Observations

**Label noise is real and deliberate.** Analysis of the 2,683 mule labels revealed:
- 705 (26.3%) flagged for "Routine Investigation" — a weak signal likely containing false positives
- 245 (9.1%) have `is_mule=1` but no `alert_reason` — highly suspicious red herrings
- Some `mule_flag_date` values extend to March 2026, beyond the transaction window — possible data artifacts

**Class imbalance** at 1:34 is significant but improved from Phase 1's 1:90, enabling more reliable minority class learning.

**Distribution shift** between train and test sets was confirmed via adversarial validation (AUC 0.76), indicating certain features behave differently across splits.

### 2.3 Alert Reason Distribution

| Alert Reason | Count | % of Mules |
|---|---|---|
| Routine Investigation | 705 | 26.3% |
| Dormant Account Reactivation | 188 | 7.0% |
| Rapid Movement of Funds | 177 | 6.6% |
| MCC-Amount Distribution Anomaly | 150 | 5.6% |
| Structuring Below Threshold | 146 | 5.4% |
| Geographic Anomaly | 144 | 5.4% |
| Other 7 categories | 973 | 36.3% |
| No reason given | 245 | 9.1% |

---

## 3. Feature Engineering (192 Features)

Feature engineering was the most compute-intensive stage, processing 400M transactions in 4 passes with memory-efficient batch processing on a 64GB RAM machine with 182GB swap.

### 3.1 Pass 1: Transaction Features (~100 features)

Processed all 396 transaction parts with running accumulators per account.

**Volume and amount statistics:**
- Transaction counts (total, credit, debit), credit/debit ratio
- Amount distribution: mean, std, CV, min, max, median, percentiles (p25/p75/p95/p99), IQR, skewness, kurtosis
- Separate credit and debit amount statistics (mean, std, max)
- Net flow, credit-to-debit amount ratio

**Temporal patterns:**
- Transaction span (days), transactions per day
- Inter-transaction time: mean, std, min, max, CV
- Burstiness index: `(std - mean) / (std + mean)` — high values indicate bursty rather than regular activity
- Maximum gap (days), dormancy events (gaps > 90 days)
- Hour-of-day entropy, day-of-week entropy
- Night transaction percentage (00:00-06:00), business hours percentage, weekend percentage
- Monthly transaction count standard deviation and CV
- Active months count

**Channel diversity (Pattern #4, #12):**
- Number of unique channels, channel entropy, channel HHI (Herfindahl-Hirschman Index)
- Top channel share, per-channel percentages for key channels (UPI Credit/Debit, ATM, Cheque, Cash Deposit, NEFT, IMPS, E-commerce)

**MCC-Amount Anomaly (Pattern #13 — new in Phase 2):**
- Global per-MCC amount population statistics (mean, std) computed across all 400M transactions
- Per-account z-scores: how much each account's average spend per MCC deviates from the population
- Features: mean z-score, max absolute z-score, count of outlier MCCs (|z| > 2), outlier ratio
- Reservoir sampling (500 samples/account) for memory-efficient percentile and Benford's Law computation

**Counterparty analysis (Pattern #4 — Fan-in/Fan-out):**
- Unique credit counterparties, unique debit counterparties, total unique counterparties
- Fan-in/fan-out ratio (credit CPs / debit CPs)
- Counterparty concentration: entropy, HHI, top counterparty share, top-3 share

**Structuring detection (Pattern #2):**
- Counts and ratios of transactions near thresholds: 50K, 100K, 200K, 500K (within 85-100% of threshold)

**Round amount patterns (Pattern #9):**
- Count and ratio of exact round amounts (1K, 2K, 5K, 10K, 25K, 50K, 100K)

**Benford's Law divergence:**
- KL divergence of first-digit distribution from Benford's expected distribution — laundered funds often violate natural digit patterns

**Behavioral change detection:**
- Before/after midpoint (2023-01-01) comparison: volume change ratio, mean amount change, credit ratio shift, late-period activity concentration

### 3.2 Pass 2: Transaction Additional Features (~30 features)

Joined 311 transaction_additional parts with the main transaction ID mapping using chunked vectorized processing.

**Geographic features (Pattern #5):**
- Latitude/longitude standard deviation, geographic spread (combined std), geographic range (km)
- Geographic coverage ratio (% of transactions with location data)

**Balance trajectory (Pattern #7 — Income Mismatch):**
- Running balance: mean, std, min, max, CV, range
- Near-zero balance count and percentage
- Maximum drawdown, mean drawdown
- Balance volatility: `mean(|balance_diff|) / |mean_balance|`

**IP diversity:**
- Unique IP count, IPs per transaction — high IP diversity can indicate VPN usage or account sharing

**Transaction sub-type distribution:**
- Percentages of NORMAL, CLT_CASH, LOAN transaction sub-types
- Cash transaction count (CLT_CASH)

**Part transaction type:**
- Distribution of CI (Customer Induced), BI (Bank Induced), IP (Interest Paid), IC (Interest Collected)
- High BI/IP ratio with low CI suggests automated/institutional activity inconsistent with mule patterns

### 3.3 Pass 3: Static Features (~45 features)

No label leakage — all features derived from non-label metadata.

**Account attributes:**
- Account age (days), balance ratios (monthly/quarterly/daily vs average)
- Balance consistency (std across balance timeframes)
- Freeze/unfreeze history: binary flags, freeze duration
- Mobile update recency, KYC recency
- Product family (Savings/K-family/Overdraft), nomination flag, cheque availability

**Customer attributes:**
- Age, relationship length
- KYC document count (PAN + Aadhaar + Passport)
- Digital score (mobile banking + internet banking + ATM card + demat + credit card + FASTag)
- PIN mismatch (residential vs permanent address PIN — Pattern #5)

**Demographics:**
- Gender, joint account flag, NRI flag
- Address update recency, passbook update recency

**Product holdings:**
- Loan/credit card/overdraft sums and counts
- Total product count, total outstanding

**Scheme codes:**
- One-hot encoding of government scheme participation (PMJDY, PMSBY, PMJJBY, APY, SCSS, SSA, REGULAR)

**Branch features (structural only, no label leakage):**
- `branch_code` preserved for target encoding within CV folds
- Employee count, turnover, asset size
- Branch type (urban/semi-urban/rural)
- Accounts per employee ratio

### 3.4 Pass 4: Graph Features (~16 features)

Built a directed account-counterparty transaction graph from all 396 transaction parts.

**Network centrality:**
- PageRank (weighted by transaction count) — high PageRank indicates important nodes in the money flow network
- HITS hub and authority scores — hubs send money widely, authorities receive from many sources
- Weighted in/out degree (by count and amount)
- Degree ratio (out/in), amount ratio (out_amount/in_amount) — pass-through indicator

**Community structure:**
- Louvain community detection — community ID and community size
- Clustering coefficient — how tightly connected a node's neighbors are
- Betweenness centrality (sampled k=500) — accounts that bridge different communities may act as intermediaries

---

## 4. Red Herring Avoidance

Red herring handling was a critical component, accounting for 15% of the evaluation. Our approach combines model-driven and heuristic detection.

### 4.1 Confident Learning (Model-Driven)

We implemented 2-round confident learning following Northcutt et al. (2021):

1. **Round 1**: Train a fast LightGBM (500 trees) using 5-fold OOF predictions. Compute per-class probability thresholds `t_pos` and `t_neg` as the mean OOF probability within each class.
2. **Flag mislabels**: Accounts labeled as mule but predicted below `t_neg` (likely false positives). Accounts labeled legitimate but predicted above `t_pos` (likely false negatives).
3. **Round 2**: Repeat with accumulated noise scores for convergence.
4. **Continuous scoring**: Distance from threshold provides a continuous noise likelihood, not just binary flags.

### 4.2 Heuristic Noise Scoring

Based on domain knowledge of the label metadata, we target 7 red herring categories:

| # | Signal | Noise Score | Rationale |
|---|--------|-------------|-----------|
| RH1 | `alert_reason = "Routine Investigation"` | 0.6 | Weakest investigation trigger — 26% of mules, likely many false positives |
| RH2 | `alert_reason` is null/empty | 0.8 | Flagged as mule with no investigative basis — strong red herring signal |
| RH3 | `mule_flag_date` beyond June 2025 | +0.3 | Dates beyond the transaction window suggest data artifacts |
| RH4 | `mule_flag_date` before July 2020 | +0.25 | Flag predates all transaction data — no evidence basis |
| RH5 | `mule_flag_date` on exact boundary dates | +0.15 | Dates like 2020-07-01, 2025-06-30 suggest synthetic generation |
| RH6 | `flagged_by_branch` is null for mule | 0.4 | Mule flagged without branch attribution — weak evidence |

### 4.3 Sample Weight Computation

Combined noise scores (max of confident learning and heuristic scores) are converted to sample weights:

```
weight = 1.0 - 0.8 × combined_noise_score
```

This maps clean samples to weight 1.0 and the noisiest samples to weight 0.2. Critically, we never fully remove samples — even noisy labels carry some signal.

### 4.4 Adversarial Validation as Debiasing

We used adversarial validation (LGB classifier distinguishing train vs test distributions) to identify and handle distribution-shifted features:

- **AV AUC = 0.76** confirmed moderate distribution shift
- Top 15 features contributing most to train/test separability were removed (AV AUC > 0.75 → "remove" strategy)
- This prevents the model from learning train-specific patterns that don't generalize

---

## 5. Model Architecture

### 5.1 Overview

```
Features (192) → Adversarial Debiasing → LOO Target Encoding (branch_code)
    → 5-Fold Stratified CV:
        ├─ LightGBM (3000 trees, Optuna-tuned)     → OOF predictions
        ├─ XGBoost  (3000 trees, early stopping)     → OOF predictions
        └─ CatBoost (3000 trees, early stopping)     → OOF predictions
    → Stacking: LogisticRegression(OOF_lgb, OOF_xgb, OOF_cb)
    → Isotonic Calibration
    → Final probabilities
```

### 5.2 Individual Models

**LightGBM** (strongest model, AUC 0.941):
- Optuna-optimized (40 trials, 3-fold inner CV)
- Best hyperparameters found via TPE sampler
- Early stopping at 100 rounds patience

**XGBoost** (AUC 0.935):
- Conservative default hyperparameters (learning_rate=0.02, max_depth=7)
- `early_stopping_rounds=100` in constructor (XGBoost 3.x requirement)
- Histogram-based tree method for speed

**CatBoost** (AUC 0.847):
- `scale_pos_weight=34` instead of `auto_class_weights` (avoids double-weighting with sample weights)
- `od_type="Iter"` with `od_wait=100` for overfitting detection

### 5.3 Key Technical Decisions

**Why Optuna only for LightGBM:**
We discovered that Optuna-tuned hyperparameters for XGBoost and CatBoost performed worse in the main 5-fold CV than conservative defaults. The root cause: 3-fold inner HPO finds parameters with low regularization (e.g., `l2_leaf_reg < 1.0`) that overfit the HPO folds, causing different prediction scales across the 5 main folds. This cross-fold calibration drift degraded the stacking layer. Conservative defaults with high regularization produce more consistent predictions.

| Configuration | LGB AUC | XGB AUC | CB AUC | Calibrated AUC | F1 |
|---|---|---|---|---|---|
| All defaults | 0.901 | 0.935 | 0.847 | 0.940 | 0.650 |
| All Optuna | 0.941 | 0.911 | 0.700 | 0.941 | 0.710 |
| **Hybrid (best)** | **0.941** | **0.935** | **0.847** | **0.940** | **0.756** |
| Optuna v2 | 0.941 | 0.925 | 0.722 | 0.928 | 0.700 |

**LOO Target Encoding for branch_code:**
Branch code was the dominant feature by SHAP importance (4.7x higher than #2). We used leave-one-out target encoding with Bayesian smoothing (smoothing=20, min_samples=30) computed strictly within each CV fold to prevent leakage. Investigation confirmed this is not data leakage — branch code is genuinely predictive of mule clustering (Pattern #12: Branch-Level Collusion).

### 5.4 Stacking and Calibration

- **L1 Meta-learner**: LogisticRegression on the 3 OOF prediction columns, learning optimal weights per model
- **Isotonic Calibration**: Monotonic function mapping ensemble predictions to calibrated probabilities, ensuring well-ordered probability estimates
- **Fallback**: AUC-weighted average computed alongside stacking; the method with higher OOF AUC is selected automatically

---

## 6. Temporal Window Prediction

For each account predicted as a mule (probability >= 0.3), we identify the most suspicious activity window.

### 6.1 Algorithm

We use a vectorized sliding window approach at 5 time scales:

| Window (days) | Stride (days) | Purpose |
|---|---|---|
| 14 | 3 | Catch short, intense bursts |
| 30 | 7 | Standard monthly window |
| 60 | 14 | Medium-term patterns |
| 90 | 14 | Quarterly patterns |
| Median lookback | 7 | Calibrated from training mule_flag_date |

The median lookback is learned from the training data: we compute the median number of days between each mule's `mule_flag_date` and the reference date (June 30, 2025), clamped to a minimum of 30 days.

### 6.2 Window Scoring

Each candidate window is scored on 8 dimensions:

```
score = 0.22 × velocity_score        (txn rate relative to account baseline)
      + 0.18 × amount_score          (mean amount relative to account baseline)
      + 0.14 × imbalance_score       (|credits - debits| / total)
      + 0.14 × passthrough_score     (min(credit_sum, debit_sum) / max — rapid passthrough)
      + 0.09 × round_amount_ratio    (proportion of round amounts)
      + 0.09 × near_50k_ratio        (structuring indicator)
      + 0.04 × concentration_score   (what fraction of account's total activity is in this window)
      + 0.10 × recency_bias          (bias toward recent windows — mules detected after activity)
```

The recency bias reflects the domain insight that mule accounts are typically flagged *after* suspicious activity, so the most suspicious window is likely to be recent relative to the account's transaction history.

### 6.3 Two-Pass Refinement

After the coarse pass identifies the best window, a refinement pass searches with 3x finer stride around the best window (one window-width margin on each side). This improves temporal alignment without increasing overall complexity.

Additionally, the final window is **clipped to actual transaction boundaries** — the start is tightened to the first transaction within the window, and the end to the last transaction, with 1-hour padding. This prevents unnecessarily wide windows that would reduce IoU.

### 6.4 Implementation: 3000x Speedup

The naive approach (Python loops over windows with per-window datetime parsing) processed ~1 account/39 seconds. Our vectorized implementation:

1. Convert timestamps to int64 seconds once
2. Use `np.searchsorted` to find window boundaries in O(log n)
3. Use cumulative sum arrays (`np.cumsum`) for O(1) range sum queries
4. Compute all window scores simultaneously via numpy broadcasting

Result: 1,249 accounts processed in 2 seconds — a 3000x speedup.

### 6.5 Additional Temporal Features

Beyond window prediction, we compute per-account temporal features used as model inputs:

- **Rapid pass-through**: Count and ratio of credit-debit pairs within 24 hours (vectorized via searchsorted)
- **Dormancy bursts**: Number of gaps > 90 days, post-dormancy transaction count and mean amount
- **Salary cycle ratio**: Fraction of credits occurring on days 1-5 or 28-31 (Pattern #11)
- **Recent velocity ratio**: Last 30 days transaction rate vs historical baseline

---

## 7. Experiments and Iterations

### 7.1 Summary of Pipeline Versions

| Version | Changes | Calibrated AUC | F1 | Notes |
|---|---|---|---|---|
| v1 | Initial pipeline, all defaults | 0.901 | 0.550 | Baseline |
| v2 | Added early stopping for XGB/CB, fixed double-weighting | 0.940 | 0.650 | Major improvement |
| v3 | Added Optuna for all 3 models | 0.941 | 0.710 | LGB improved, CB degraded |
| **v4** | **Hybrid: Optuna LGB + default XGB/CB** | **0.940** | **0.756** | **Best overall** |
| v5 | Optuna v2 for XGB/CB with l2_leaf_reg clamping | 0.928 | 0.700 | Regression — abandoned |

### 7.2 What Worked

1. **Confident learning for label noise** — correctly identified and downweighted red herring labels
2. **Branch code target encoding** — dominant predictive feature when encoded within CV folds
3. **Graph features** (PageRank, betweenness, community) — captured network structure of money flows
4. **MCC-amount anomaly z-scores** — population-level anomaly detection per merchant category
5. **Adversarial validation + feature removal** — eliminated distribution-shifted features
6. **Multi-scale temporal windows** — 5 window sizes capture different mule behavior durations

### 7.3 What Didn't Work

1. **Optuna for XGBoost/CatBoost** — 3-fold HPO parameters didn't generalize to 5-fold main CV
2. **CatBoost auto_class_weights + sample_weight** — double-weighting caused extreme instability
3. **Downweighting instead of removing AV features** — when AV AUC > 0.75, full removal was necessary
4. **XGB without early_stopping_rounds in constructor** — XGB 3.x silently ignores the parameter if passed only to fit()

### 7.4 Bugs Found and Fixed

| Bug | Impact | Fix |
|---|---|---|
| XGB early stopping not activating | OOF 0.854 (trained all 3000 trees) | Pass `early_stopping_rounds` in constructor, not fit() |
| CatBoost double class weighting | OOF 0.746 (extreme variance) | Replace `auto_class_weights` with `scale_pos_weight` |
| Cross-fold calibration drift | OOF 0.700-0.720 for CB | Clamp `l2_leaf_reg >= 1.0`, use conservative defaults |
| Feature count mismatch (197 vs 182) | Inference crash | Read model's `feature_name_` and align test features |

---

## 8. Results

### 8.1 Final Model Performance (5-fold CV)

| Metric | Value |
|--------|-------|
| LightGBM OOF AUC | 0.941 |
| XGBoost OOF AUC | 0.935 |
| CatBoost OOF AUC | 0.847 |
| Stacked Ensemble AUC | 0.937 |
| **Calibrated AUC** | **0.940** |
| **Best F1** | **0.756** |

### 8.2 Submission Statistics

| Metric | Value |
|--------|-------|
| Total test accounts | 64,062 |
| Predicted mules (p >= 0.3) | 992 (1.55%) |
| Predicted mules (p >= 0.5) | 886 (1.38%) |
| Temporal windows assigned | 989 |
| Mean prediction probability | ~0.03 |

### 8.3 SHAP Feature Importance (Top 20)

Branch code target encoding dominated, followed by transaction volume, amount statistics, graph centrality metrics, and behavioral change features. The full SHAP importance ranking confirmed that our feature engineering captured meaningful mule signals across all 13 known behavior patterns.

---

## 9. Infrastructure

- **Training**: GCP n2-highmem-8 (8 vCPU, 64GB RAM, 100GB SSD + 182GB swap), SPOT instance
- **Cost**: ~5.5 INR/hour
- **Pipeline runtime**: ~4 hours (features: 2.5h, label cleaning: 0.5h, training: 0.5h, temporal: 0.5h)
- **Libraries**: Python 3.10, LightGBM 4.6, XGBoost 3.2, CatBoost 1.2, scikit-learn, NetworkX, Optuna, SHAP

---

## 10. Conclusion

Our approach demonstrates that mule account detection at scale requires careful attention to three often-overlooked aspects:

1. **Label quality**: The deliberate noise injection in this competition mirrors real-world challenges where investigation labels are noisy. Our confident learning + heuristic approach explicitly handles this.

2. **Feature stability**: Features that discriminate well on training data may not generalize. Adversarial validation provides an automated safeguard against distribution shift.

3. **Model diversity with consistency**: While ensemble diversity helps, the individual models must produce consistent prediction scales across CV folds. Aggressive hyperparameter tuning can actually harm ensemble performance by creating cross-fold calibration drift.

The pipeline processes 16.2GB of data in under 4 hours on a single 8-core machine, producing both mule probability scores and temporal activity windows for a complete AML investigation output.
