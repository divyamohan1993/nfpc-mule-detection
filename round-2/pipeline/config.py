"""
NFPC Phase 2 — Configuration (V2)
"""
import os
from pathlib import Path
import logging

# Paths
DATA_DIR = Path(os.environ.get("NFPC_DATA_DIR", "/home/DIVYA/nfpc-phase2/data"))
OUTPUT_DIR = Path(os.environ.get("NFPC_OUTPUT_DIR", "/home/DIVYA/nfpc-phase2/output"))
FEATURES_DIR = OUTPUT_DIR / "features"
MODELS_DIR = OUTPUT_DIR / "models"
for d in [OUTPUT_DIR, FEATURES_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Static file paths
CUSTOMERS_PATH = DATA_DIR / "customers.parquet"
ACCOUNTS_PATH = DATA_DIR / "accounts.parquet"
DEMOGRAPHICS_PATH = DATA_DIR / "demographics.parquet"
ACCOUNTS_ADDITIONAL_PATH = DATA_DIR / "accounts-additional.parquet"
BRANCH_PATH = DATA_DIR / "branch.parquet"
LINKAGE_PATH = DATA_DIR / "customer_account_linkage.parquet"
PRODUCT_DETAILS_PATH = DATA_DIR / "product_details.parquet"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.parquet"
TEST_ACCOUNTS_PATH = DATA_DIR / "test_accounts.parquet"
TXN_DIR = DATA_DIR / "transactions"
TXN_ADDITIONAL_DIR = DATA_DIR / "transactions_additional"

# Feature output paths
TXN_ID_MAP_PATH = OUTPUT_DIR / "txn_id_to_acct_id.parquet"
TXN_FEATURES_PATH = FEATURES_DIR / "txn_features.parquet"
TXN_ADD_FEATURES_PATH = FEATURES_DIR / "txn_additional_features.parquet"
STATIC_FEATURES_PATH = FEATURES_DIR / "static_features.parquet"
GRAPH_FEATURES_PATH = FEATURES_DIR / "graph_features.parquet"
TEMPORAL_FEATURES_PATH = FEATURES_DIR / "temporal_features.parquet"
FULL_FEATURES_PATH = FEATURES_DIR / "full_features.parquet"

# Feature engineering constants
STRUCTURING_THRESHOLDS = [50_000, 100_000, 200_000, 500_000]
ROUND_AMOUNTS = [1000, 2000, 5000, 10_000, 25_000, 50_000, 100_000]
DORMANCY_DAYS = 90
RAPID_PASSTHROUGH_HOURS = 24
BEHAVIORAL_SPLIT_DATE = "2023-01-01"  # midpoint for before/after comparison
RESERVOIR_SIZE = 500

ALL_CHANNELS = [
    "UPC", "UPD", "END", "IPM", "STD", "P2A", "FTD", "NTD", "MCR", "FTC",
    "MAC", "TPD", "APD", "CHQ", "ATW", "TPC", "STC", "OCD", "RCD", "IFD",
    "ETD", "NWD", "CSD", "IFC", "PCA", "MAD", "CHD", "RTD", "CCL", "OPI",
    "CTC", "SID", "ASD", "IAD", "SCW",
]

# Model parameters
N_FOLDS = 5
SEED = 42
MULE_RATIO = 34
N_THREADS = int(os.environ.get("NFPC_N_THREADS", os.cpu_count() or 4))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("nfpc")
