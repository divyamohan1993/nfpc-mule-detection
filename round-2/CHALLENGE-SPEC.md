# NFPC Phase 2 — Challenge Specification

## Objective
Identify mule accounts used for money laundering. Given labelled training data and unlabelled test accounts, predict which test accounts are mules.

## Scale (vs Phase 1)
| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| Accounts | 24,000 | 160,000 |
| Transactions | ~5M | ~400M |
| Train labels | 24,000 | 96,091 |
| Test accounts | 15,848 | 64,062 |
| Mule ratio | 1:90 | 1:34 |
| Mules in train | 263 | 2,683 |
| Total data | ~500MB | 16.2GB |

## Evaluation Criteria
| Weight | Criterion |
|--------|-----------|
| **40%** | Model/Feature Ingenuity — creative and innovative ideas, multiple models/algorithms in series/parallel |
| **20%** | Model Performance — AUC-ROC, F1 scores |
| **15%** | Avoidance of Red Herrings — labels contain noise, not all labels are correct |
| **15%** | Additional Insights — Temporal IoU scores, other insights |
| **10%** | Report Quality — clear, concise, data-driven logic |

## Submission Format
```csv
account_id,is_mule,suspicious_start,suspicious_end
ACCT_000000,0.02,,
ACCT_000003,0.87,2023-11-15T09:30:00,2024-02-20T16:45:00
```
- `is_mule`: Probability 0-1
- `suspicious_start`/`suspicious_end`: ISO timestamps of suspected mule activity window
- Primary scoring on `is_mule`; temporal IoU as bonus metric

## 13 Known Mule Behavior Patterns
1. Dormant Activation
2. Structuring (near 50K threshold)
3. Rapid Pass-Through (credit-to-debit near unity)
4. Fan-In / Fan-Out
5. Geographic Anomaly (lat/long, PIN mismatches)
6. New Account High Value
7. Income Mismatch
8. Post-Mobile-Change Spike
9. Round Amount Patterns
10. Layered/Subtle (multi-signal)
11. Salary Cycle Exploitation
12. Branch-Level Collusion
13. **MCC-Amount Anomaly** (NEW in Phase 2)

## Red Herring Warning
> Labels may contain noise/red-herrings. Not all labels are guaranteed to be correct.

Key implications:
- Need label-noise-robust training (e.g., confident learning, loss reweighting)
- "Routine Investigation" is the top alert reason (705 of 2,683 mules) — likely contains false positives
- 245 mules have alert_reason = null despite is_mule = 1 (2683 mules - 2438 with reasons)

## Alert Reason Distribution (train mules)
| Alert Reason | Count |
|---|---|
| Routine Investigation | 705 |
| Dormant Account Reactivation | 188 |
| Rapid Movement of Funds | 177 |
| MCC-Amount Distribution Anomaly | 150 |
| Structuring Transactions Below Threshold | 146 |
| Geographic Anomaly Detected | 144 |
| Unusual Fund Flow Pattern | 141 |
| Income-Transaction Mismatch | 134 |
| Branch Cluster Investigation | 128 |
| High-Value Activity on New Account | 120 |
| Post-Contact-Update Spike | 109 |
| Round Amount Pattern | 105 |
| Layered Transaction Pattern | 98 |
| Salary Cycle Anomaly | 93 |

## Mule Flag Date Range
- Earliest: 2015-09-05
- Latest: 2026-03-20 (future dates — possible data generation artifact)
