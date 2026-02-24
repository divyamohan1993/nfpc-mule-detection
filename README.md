# Mule Account Detection - National Fraud Prevention Challenge

**Team dmj.one** | RBIH x IIT Delhi TRYST Hackathon 2025

Detecting money mule accounts in Indian banking data using machine learning. Built for the [National Fraud Prevention Challenge (NFPC)](https://rbihub.in) hosted by **Reserve Bank Innovation Hub (RBIH)** in association with **IIT Delhi TRYST**.

> **Challenge repo & dataset**: [IITD-Tryst-Hackathon](https://github.com/AkhilPuppala/IITD-Tryst-Hackathon) | **Challenge docs**: [EDA Guide](https://github.com/AkhilPuppala/IITD-Tryst-Hackathon/blob/main/EDA-Phase-1/EDA_GUIDE.md) | [Submission Guidelines](https://github.com/AkhilPuppala/IITD-Tryst-Hackathon/blob/main/EDA-Phase-1/SUBMISSION_GUIDELINES.md)

## Key Results

| Metric | LightGBM | XGBoost | Ensemble |
|---|---|---|---|
| OOF AUC-ROC | 0.9834 | 0.9789 | **0.9851** |
| Mean Fold AUC | 0.9831 +/- 0.0058 | 0.9785 +/- 0.0067 | - |

- **125 engineered features** across 13 categories
- **12 mule behavior patterns** identified with statistical evidence
- **47 statistical tables** and **25 analytical visualizations**
- Suspicious activity time windows for high-risk accounts
- Extreme class imbalance handled (263 mules vs 23,760 legitimate = **1:90 ratio**)

## Top Discriminative Features

| Rank | Feature | Importance | What It Captures |
|---|---|---|---|
| 1 | `mcc_6051_rate` | 275,234 | Rate of MCC 6051 (quasi-cash/money orders) transactions |
| 2 | `was_frozen` | 105,127 | Account freeze history (19.6x differential) |
| 3 | `ch_UPD_rate` | 33,884 | UPI Debit channel usage rate |
| 4 | `cp_per_txn` | 30,480 | Unique counterparties per transaction (fan-in/fan-out) |
| 5 | `days_since_kyc` | 30,205 | Recency of KYC verification |
| 6 | `mcc_5933_rate` | 28,738 | Rate of MCC 5933 (pawn shops) transactions |
| 7 | `p25_amount` | 24,823 | 25th percentile transaction amount |
| 8 | `ch_CHQ_rate` | 20,250 | Cheque channel usage rate |
| 9 | `rel_years` | 15,593 | Customer relationship tenure in years |
| 10 | `ch_ATW_rate` | 13,464 | ATM withdrawal rate |

## Mule Behavior Patterns

All 12 known patterns from the [challenge specification](https://github.com/AkhilPuppala/IITD-Tryst-Hackathon/blob/main/EDA-Phase-1/README.md#known-mule-behavior-patterns) were identified in the data:

1. **Dormant Activation** - Inactive accounts with sudden high-value bursts
2. **Structuring** - Transactions just below 50K INR reporting threshold
3. **Rapid Pass-Through** - Near-unity credit-to-debit ratio
4. **Fan-In / Fan-Out** - Many-to-one or one-to-many fund flows
5. **Geographic Anomaly** - PIN code mismatches across customer/branch/address
6. **New Account High Value** - Young accounts with disproportionate volume
7. **Income Mismatch** - Transaction values vs account balance
8. **Post-Mobile-Change Spike** - Activity surge after mobile number update
9. **Round Amount Patterns** - Overuse of exact round amounts
10. **Layered/Subtle** - Weak multi-signal combinations
11. **Salary Cycle Exploitation** - Laundering within salary credit cycles
12. **Branch-Level Collusion** - Suspicious account clusters at same branch

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── full_pipeline.py          # Complete: EDA + features + models + predictions
│   ├── eda_phase1.py             # Standalone EDA script
│   ├── md_to_html.py             # Markdown -> HTML/PDF converter
│   └── fix_tables.py             # Table caption post-processor
├── reports/
│   ├── NFPC_Phase1_EDA_Report.md # Full EDA report (markdown)
│   └── plots/                    # 25 analytical visualizations
└── models/
    ├── predictions.csv           # 15,848 test account predictions
    └── feature_importance.csv    # 123 features ranked by importance
```

> HTML and PDF reports are generated from the markdown source. See [Generating the Report](#generating-the-report) below.

## Feature Engineering (125 features, 13 categories)

| Category | Count | Examples |
|---|---|---|
| Transaction Aggregation | 7 | `txn_count`, `mean_amount`, `std_amount` |
| Structuring Detection | 7 | `near_50k_rate`, `round_10k_rate` |
| Velocity & Burstiness | 10 | `min_gap_hrs`, `med_gap_hrs`, `burstiness` |
| Pass-Through & Flow | 4 | `passthrough_ratio`, `net_flow` |
| Graph & Network | 8 | `n_unique_counterparties`, `cp_per_txn` |
| Channel Usage | 12 | `ch_UPD_rate`, `ch_CHQ_rate`, `ch_ATW_rate` |
| Temporal | 4 | `weekend_rate`, `night_rate`, `hour_entropy` |
| MCC-Based | 6 | `mcc_6051_rate`, `mcc_5933_rate` |
| Derived Ratios | 3 | `txn_to_balance_ratio`, `max_txn_to_balance` |
| Unsupervised/Anomaly | 18 | Digital scores, KYC scores |
| Geographic | 1 | `pin_mismatch` |
| Balance | 8 | `balance_std`, `balance_range` |
| Demographics | 37 | Encoded flags, product holdings, account age |

## Model Training

- **LightGBM** (GBDT, 63 leaves, lr=0.05, scale_pos_weight=90.3)
- **XGBoost** (max_depth=6, lr=0.05, scale_pos_weight=90.3)
- **Ensemble**: 50/50 average of both models
- **Validation**: 5-fold Stratified K-Fold cross-validation
- **Early stopping**: 50 rounds patience on validation AUC
- **Explainability**: SHAP TreeExplainer on top-25 features

## Visualizations

25 plots in [`reports/plots/`](reports/plots/):

| # | Plot | # | Plot |
|---|---|---|---|
| 01 | Class Distribution | 14 | Branch Analysis |
| 02 | Alert Reasons | 15 | Velocity |
| 03 | Account Status & Freeze | 16 | Correlations |
| 04 | Balance Distributions | 17 | Feature Importance |
| 05 | Account Opening | 18 | Model Evaluation (ROC/PR) |
| 06 | Customer Demographics | 19 | SHAP Summary |
| 07 | Flags Heatmap | 20 | Geographic Analysis |
| 08 | Channel Analysis | 21 | Unsupervised Features |
| 09 | Temporal Patterns | 22 | Focused Heatmap |
| 10 | Amount Distribution | 23 | Network Topology |
| 11 | Structuring | 24 | False Positive Analysis |
| 12 | Counterparty | 25 | Cost-Sensitive Matrix |
| 13 | MCC Analysis | | |

## Quick Start

```bash
pip install -r requirements.txt

# Clone the dataset (not included in this repo)
git clone https://github.com/AkhilPuppala/IITD-Tryst-Hackathon.git data

# Run the full pipeline
python src/full_pipeline.py
```

### Generating the Report

```bash
python src/md_to_html.py          # Markdown -> HTML
playwright install chromium        # One-time browser install
python -c "
from playwright.sync_api import sync_playwright
import os
html = os.path.abspath('reports/NFPC_Phase1_EDA_Report.html')
with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page()
    page.goto(f'file:///{html}')
    page.wait_for_load_state('networkidle')
    page.pdf(path='reports/pdf/report.pdf', format='A4', print_background=True)
    b.close()
"
```

## Statistical Methods

- **Mann-Whitney U test** - Non-parametric continuous variable comparison
- **Kolmogorov-Smirnov test** - Distribution shape comparison
- **Chi-square with Cramer's V** - Categorical association strength
- **Effect size (rank-biserial)** - Practical significance beyond p-values
- **Bonferroni correction** - Multiple hypothesis testing adjustment

## Tech Stack

Python 3.10+ | LightGBM | XGBoost | SHAP | Pandas | NumPy | Matplotlib | Seaborn | SciPy | scikit-learn | Playwright | markdown2

## License

MIT License. See [LICENSE](LICENSE).

Dataset is synthetic, provided by RBIH for challenge purposes. See [IP & Data Governance](https://github.com/AkhilPuppala/IITD-Tryst-Hackathon/blob/main/EDA-Phase-1/README.md) for details.

## Acknowledgments

- [Reserve Bank Innovation Hub (RBIH)](https://rbihub.in) for hosting the challenge
- [IIT Delhi TRYST](https://tryst-iitd.org) for the hackathon platform
- [Challenge Dataset Repository](https://github.com/AkhilPuppala/IITD-Tryst-Hackathon)
