# NFPC Phase 2 — Mule Account Detection Pipeline

## Environment

- **Python**: 3.10+
- **OS**: Ubuntu 22.04 (tested on GCP n2-highmem-8, 64GB RAM)
- **Dependencies**: See `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
```

## Data Preparation

Place the Kaggle dataset files in the data directory (default: `/home/DIVYA/nfpc-phase2/data/`):

```
data/
  customers.parquet
  accounts.parquet
  demographics.parquet
  accounts-additional.parquet
  branch.parquet
  customer_account_linkage.parquet
  product_details.parquet
  train_labels.parquet
  test_accounts.parquet
  transactions/
    batch-1/part_*.parquet
    batch-2/part_*.parquet
    ...
  transactions_additional/
    batch-1/part_*.parquet
    ...
```

Override paths via environment variables:
```bash
export NFPC_DATA_DIR=/path/to/data
export NFPC_OUTPUT_DIR=/path/to/output
```

## Running the Pipeline

### Full pipeline (includes Optuna HPO)
```bash
python run.py
```

### Skip Optuna (use saved or default hyperparameters)
```bash
python run.py --skip-optuna
```

### Re-run a specific stage
```bash
python run.py --skip-optuna --force-stage models
python run.py --skip-optuna --force-stage temporal
```

### Standalone scripts
```bash
# Run Optuna for XGB/CB only (after initial pipeline run)
python run_optuna_xgb_cb.py

# Generate submission from saved fold models
python gen_submission.py
```

## Pipeline Stages

1. **Feature Engineering** (`features.py`) — 4 passes: transaction stats, geo/balance, static, graph → 192 features
2. **Label Cleaning** (`label_cleaning.py`) — 2-round confident learning + heuristic noise scoring → sample weights
3. **Model Training** (`models.py`) — Adversarial validation, LOO target encoding, LGB+XGB+CB ensemble, stacking, calibration
4. **Temporal Windows** (`temporal.py`) — Vectorized sliding window anomaly scoring at 5 time scales
5. **Submission** — CSV with `account_id, is_mule, suspicious_start, suspicious_end`

Each stage saves checkpoint markers. Re-running safely skips completed stages.

## Output

```
output/
  submission.csv                  # Final submission
  features/full_features.parquet  # Combined feature matrix
  models/lgb_fold*.pkl           # Saved fold models
  models/best_params.pkl         # Optuna hyperparameters
  oof_predictions.csv            # Out-of-fold predictions
  noise_analysis.csv             # Label noise scores
  adversarial_validation.csv     # AV feature importance
  shap_importance.csv            # SHAP feature ranking
  suspicious_windows.parquet     # Temporal predictions
```

## File Descriptions

| File | Description |
|------|-------------|
| `config.py` | Paths, constants, hyperparameters |
| `features.py` | 4-pass feature engineering (192 features) |
| `label_cleaning.py` | Confident learning + heuristic red herring detection |
| `models.py` | Training pipeline: AV, target encoding, 3-model ensemble, stacking, calibration, SHAP |
| `temporal.py` | Vectorized temporal window prediction |
| `run.py` | Pipeline orchestrator with checkpointing |
| `gen_submission.py` | Quick submission from saved models |
| `run_optuna_xgb_cb.py` | Targeted Optuna HPO for XGB/CB |

## Approach Summary

- **192 features** across transaction stats, geolocation, balance trajectories, account/customer metadata, and network graph centrality
- **Label noise handling**: 2-round confident learning + heuristic scoring for red herring avoidance
- **3-model ensemble**: LightGBM (Optuna-tuned) + XGBoost + CatBoost with stacking meta-learner and isotonic calibration
- **Temporal windows**: O(n) vectorized sliding window anomaly scoring using searchsorted + cumsum

## Results

- Calibrated AUC-ROC: **0.940**
- Best F1: **0.756**
- Predicted mules: 992 (p >= 0.3), 886 (p >= 0.5)
- Temporal windows: 989 accounts
