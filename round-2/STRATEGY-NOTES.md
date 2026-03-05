# NFPC Phase 2 — Strategy Notes

## Key Observations

### Label Noise / Red Herrings
- 705 mules flagged for "Routine Investigation" — likely many false positives
- 245 mules have is_mule=1 but no alert_reason — suspect red herrings
- Strategy: Use confident learning (cleanlab) to identify and downweight noisy labels
- Consider training with sample weights inversely proportional to noise likelihood

### Scale Challenge
- 400M transactions won't fit in memory at once (even 64GB)
- Must process in batches: aggregate features per account from parquet parts
- Parquet is columnar — can read only needed columns to save memory
- Strategy: Iterate through 396+311 parts, accumulate per-account aggregations

### New Data Signals (vs Phase 1)
1. **Geolocation** (lat/long in transactions_additional) — geographic anomaly detection
2. **IP addresses** — can detect shared IPs across accounts, VPN usage
3. **Balance after transaction** — real-time balance trajectory, detect rapid drawdowns
4. **Demographics** (name, address, phone) — detect shared identities, name patterns
5. **Branch metadata** (employee count, turnover, asset size) — branch collusion signals
6. **Scheme codes** (PMJDY etc.) — government scheme accounts may have different mule patterns
7. **Transaction sub-type** (CLT_CASH, LOAN, NORMAL) — cash transaction patterns
8. **Part transaction type** (CI, BI, IP, IC) — customer vs bank initiated

### Temporal IoU (15% of score)
- Must identify suspicious_start and suspicious_end per mule
- Strategy: For predicted mules, find the time window with highest anomaly density
- Use change-point detection or sliding window anomaly scoring
- mule_flag_date in training data gives the flag date — suspicious activity likely precedes it

### Imbalance
- 1:34 ratio (better than Phase 1's 1:90)
- Still needs: scale_pos_weight, stratified folds, threshold optimization

## Proposed Pipeline Architecture
1. **Feature Engineering** (batch processing)
   - Read parquet parts in batches
   - Aggregate per account_id: txn stats, channel usage, MCC patterns, temporal patterns
   - Join with account/customer/branch static features
   - New: geo features, IP features, balance trajectory features

2. **Label Cleaning**
   - Train initial model → identify likely mislabeled samples
   - Confident learning or loss-based noise detection
   - Create cleaned labels or sample weights

3. **Model Training**
   - LightGBM + XGBoost ensemble (proven in Phase 1)
   - Consider: CatBoost, neural network for sequence patterns
   - Multi-stage: coarse filter → fine classifier

4. **Temporal Window Prediction**
   - For accounts predicted as mules
   - Sliding window anomaly score over transaction timeline
   - Identify window with highest anomaly concentration
   - Output suspicious_start/suspicious_end

5. **Submission**
   - 64,062 rows with probability + time windows

## Feature Ideas (Beyond Phase 1's 125)

### Geographic (NEW)
- Distance between transaction lat/long and customer PIN centroid
- Number of distinct cities/states transacted from
- Max distance between any two transactions
- Transactions from known high-risk locations

### IP-Based (NEW)
- Number of unique IPs per account
- IP entropy (how spread out)
- Shared IPs with other accounts (network feature)
- VPN/datacenter IP detection (IP range analysis)

### Balance Trajectory (NEW)
- Balance volatility (std of balance_after_transaction)
- Min/max balance ratio
- Number of near-zero balance events
- Speed of balance depletion after credits

### Identity/Demographics (NEW)
- Shared phone numbers across customers
- Shared addresses
- Name similarity clusters (potential fake identities)
- Recent address changes near mule activity

### Branch-Level (NEW)
- Mule density at branch (from training data)
- Branch size vs account activity ratio
- Branch turnover anomalies
- Accounts-per-employee ratio

### Scheme-Based (NEW)
- PMJDY accounts with high-value transactions (government scheme misuse)
- Scheme code distribution among mules vs legitimate

## Memory Management Plan
```python
# Process 400M transactions in chunks
from glob import glob
import pandas as pd

parts = sorted(glob("transactions/batch-*/part_*.parquet"))
agg = {}  # account_id -> running aggregations

for part_path in parts:
    chunk = pd.read_parquet(part_path, columns=["account_id", "amount", "channel", ...])
    # Update per-account aggregations
    for acct, group in chunk.groupby("account_id"):
        if acct not in agg:
            agg[acct] = initialize_accumulators()
        update_accumulators(agg[acct], group)
    del chunk  # free memory

features = pd.DataFrame.from_dict(agg, orient="index")
```
