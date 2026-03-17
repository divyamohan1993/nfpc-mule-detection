# Mule Account Detection - National Fraud Prevention Challenge

**Team dmj.one** | RBIH x IIT Delhi TRYST 2025

Detecting money mule accounts in Indian banking data using machine learning. Built for the [National Fraud Prevention Challenge (NFPC)](https://rbihub.in) hosted by **Reserve Bank Innovation Hub (RBIH)** in association with **IIT Delhi TRYST**.

## Datasets

| Phase | Source | Link | Size |
|-------|--------|------|------|
| Phase 1 | GitHub (IITD-Tryst-Hackathon) | [AkhilPuppala/IITD-Tryst-Hackathon](https://github.com/AkhilPuppala/IITD-Tryst-Hackathon) | ~2 GB |
| Phase 2 | Kaggle | [rbih-nfpc-phase-2](https://www.kaggle.com/datasets/abhyudayrbih/rbih-nfpc-phase-2/) | ~16 GB (Parquet) |

> Datasets are synthetic, provided by RBIH for challenge purposes. Not included in this repo.

## Results

### Phase 2 (Final Submission)

| Metric | Score |
|--------|-------|
| **Public AUC-ROC** | **0.968136** |
| **Private AUC-ROC** | **0.955815** |

- 3-model ensemble: LightGBM + XGBoost + CatBoost
- 208 engineered features, 3-seed x 5-fold CV, rank averaging
- Confident learning for label noise, heuristic red-herring avoidance

### Phase 1

| Metric | LightGBM | XGBoost | Ensemble |
|--------|----------|---------|----------|
| OOF AUC-ROC | 0.9834 | 0.9789 | **0.9851** |

- 125 engineered features across 13 categories
- 12 mule behavior patterns identified with statistical evidence
- 47 statistical tables and 25 analytical visualizations

## Repository Structure

```
.
├── README.md
├── LICENSE
├── requirements.txt               # Phase 1 dependencies
│
├── src/                            # Phase 1 pipeline
│   ├── full_pipeline.py            # Complete: EDA + features + models + predictions
│   ├── eda_phase1.py               # Standalone EDA script
│   ├── md_to_html.py               # Markdown -> HTML/PDF report converter
│   └── fix_tables.py               # Table caption post-processor
│
├── models/                         # Phase 1 outputs
│   ├── predictions.csv             # 15,848 test account predictions
│   └── feature_importance.csv      # 123 features ranked by importance
│
├── reports/                        # Phase 1 EDA
│   ├── NFPC_Phase1_EDA_Report.md   # Full EDA report (markdown source)
│   └── plots/                      # 25 analytical visualizations
│
├── round-2/                        # Phase 2
│   ├── pipeline/                   # All pipeline code (see below)
│   ├── deliverables/               # Final RBIHub submissions (CSV + report)
│   ├── CHALLENGE-SPEC.md           # Challenge rules and scoring
│   ├── DATA-SCHEMAS.md             # Dataset column definitions
│   ├── DATASET-README.md           # Official dataset documentation
│   ├── STRATEGY-NOTES.md           # Approach planning notes
│   ├── VM-INFO.md                  # VM setup details
│   ├── rbihub-info.md              # Challenge overview, Kaggle link, evaluation criteria
│   ├── submission-details.md       # Submission format specs
│   ├── report.md                   # Solution report (markdown source)
│   ├── report.html                 # Solution report (HTML)
│   ├── report.pdf                  # Solution report (PDF, submitted to RBIHub)
│   └── generate_report.py          # Report generation script
│
└── web/                            # Next.js showcase site (deployed to Vercel)
    ├── app/page.tsx                # Main showcase page
    └── app/pitch/                  # Pitch deck
```

### Phase 2 Pipeline (`round-2/pipeline/`)

```
config.py           # Paths, constants, seeds, logging
features.py         # Feature engineering (4 passes, 208 features)
label_cleaning.py   # Confident learning + heuristic noise detection
models_v3.py        # Best model version (LGB + XGB + CatBoost ensemble)
temporal.py         # Suspicious activity window prediction
run_v3.py           # Pipeline orchestrator (best version)
requirements.txt    # Phase 2 dependencies
models*.py          # All model iterations (v1-v8)
run*.py             # All pipeline iterations
```

## Reproducing Results

### Phase 2 (Best Submission)

```bash
# 1. Clone this repo
git clone https://github.com/divyamohan1993/nfpc-mule-detection.git
cd nfpc-mule-detection

# 2. Download the Phase 2 dataset from Kaggle
#    https://www.kaggle.com/datasets/abhyudayrbih/rbih-nfpc-phase-2/
#    Place parquet files in a data directory

# 3. Install dependencies
pip install -r round-2/pipeline/requirements.txt

# 4. Set data path (adjust to your download location)
export NFPC_DATA_DIR=/path/to/phase2/data
export NFPC_OUTPUT_DIR=./round-2/pipeline/output

# 5. Run the pipeline (skip Optuna for exact reproduction)
cd round-2/pipeline
python run_v3.py --skip-optuna
```

This produces `submission_v3.csv` matching the best public score (0.968136 AUC-ROC).

### Phase 1

```bash
pip install -r requirements.txt

# Download Phase 1 data
git clone https://github.com/AkhilPuppala/IITD-Tryst-Hackathon.git data

python src/full_pipeline.py
```

## Feature Engineering Highlights

### Phase 2 (208 features)
- **Transaction**: Volume, amounts, temporal histograms, channel diversity, MCC patterns, counterparty fan-in/out, structuring detection, behavioral change metrics
- **Static**: Account age, freeze history, customer demographics, branch characteristics, scheme codes
- **Graph**: PageRank, HITS scores, betweenness centrality, Louvain community detection, clustering coefficients

### Phase 1 (125 features, 13 categories)

| Category | Count | Examples |
|----------|-------|---------|
| Transaction Aggregation | 7 | `txn_count`, `mean_amount`, `std_amount` |
| Structuring Detection | 7 | `near_50k_rate`, `round_10k_rate` |
| Velocity & Burstiness | 10 | `min_gap_hrs`, `med_gap_hrs`, `burstiness` |
| Graph & Network | 8 | `n_unique_counterparties`, `cp_per_txn` |
| Channel Usage | 12 | `ch_UPD_rate`, `ch_CHQ_rate`, `ch_ATW_rate` |
| Unsupervised/Anomaly | 18 | Digital scores, KYC scores |
| Demographics | 37 | Encoded flags, product holdings, account age |

## Mule Behavior Patterns

All 12 known patterns from the challenge specification were identified:

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

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- [Reserve Bank Innovation Hub (RBIH)](https://rbihub.in) for organizing the challenge
- [IIT Delhi TRYST](https://tryst-iitd.org) for the hackathon platform
- [Phase 1 Dataset](https://github.com/AkhilPuppala/IITD-Tryst-Hackathon) | [Phase 2 Dataset (Kaggle)](https://www.kaggle.com/datasets/abhyudayrbih/rbih-nfpc-phase-2/)
