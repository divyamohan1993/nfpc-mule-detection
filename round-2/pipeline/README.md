# NFPC Phase 2: Mule Account Detection Pipeline

**Team**: DMJ.ONE
**Approach**: Gradient-boosted tree ensemble (LightGBM + XGBoost + CatBoost) with rank averaging

## Environment Setup

- Python 3.10+
- Dependencies: `pip install -r requirements.txt`

Key packages: lightgbm, xgboost, catboost, scikit-learn, pandas, numpy, networkx, optuna, shap

## Reproducing Results

1. Place data files in the path specified by `config.py` (`DATA_DIR`)
2. Run the pipeline:

```bash
python run_v3.py --skip-optuna
```

This will:
- Build 208 features from transaction, static, and graph data (Stage 1)
- Compute sample weights via confident learning + heuristic noise detection (Stage 2)
- Train 3-model ensemble with 3-seed x 5-fold CV (Stage 3)
- Generate temporal suspicious activity windows (Stage 4)
- Produce `submission_v3.csv` (Stage 5)

To run Optuna hyperparameter optimization first:
```bash
python run_v3.py
```

## Approach Description

### Feature Engineering (features.py, 208 features)
- **Transaction features**: Volume, amounts, temporal histograms, channel diversity, MCC patterns, counterparty fan-in/out, structuring detection, behavioral change metrics
- **Additional transaction features**: Geographic spread, IP diversity, balance statistics, transaction type distributions
- **Static features**: Account age, freeze history, customer demographics, branch characteristics, scheme codes
- **Graph features**: NetworkX-based PageRank, HITS scores, betweenness centrality, Louvain community detection, clustering coefficients

### Label Cleaning (label_cleaning.py)
- Multi-round confident learning (Northcutt et al.) to detect mislabeled samples
- Heuristic noise scoring targeting 7 known red herring categories
- Sample weights in [0.2, 1.0] for noise-robust training

### Model Training (models_v3.py)
- **3-model ensemble**: LightGBM, XGBoost, CatBoost
- **Frequency encoding** for categorical features (no label leakage)
- **3-seed averaging** (seeds 42, 43, 44) x 5-fold stratified CV for stability
- **Rank averaging** for ensemble combination
- **Optuna** hyperparameter optimization with V2 params (near-zero regularization)
- **SHAP** feature importance analysis

### Temporal Windows (temporal.py)
- Transaction-level analysis for accounts predicted as mules
- Sliding window approach to identify suspicious activity periods
- Outputs suspicious_start and suspicious_end timestamps

### Key Results
- **Public Phase AUC-ROC**: 0.968136 (V3 with V2 params)
- **Private Phase AUC-ROC**: 0.955815

## File Structure
```
config.py          - Paths, constants, logging
features.py        - Feature engineering (4 passes, 208 features)
label_cleaning.py  - Confident learning + heuristic noise detection
models_v3.py       - Model training & inference (best version)
temporal.py        - Temporal window prediction
run_v3.py          - Pipeline orchestrator
requirements.txt   - Python dependencies
models/            - Trained model weights and hyperparameters
```
