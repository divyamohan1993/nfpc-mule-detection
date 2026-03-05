# NFPC Phase 2 — GCP VM Reference

## VM Details
- **Name**: `nfpc-training`
- **Zone**: `us-central1-a`
- **IP**: `136.113.147.155`
- **Type**: `n2-highmem-8` (8 vCPU, 64GB RAM)
- **Provisioning**: SPOT (auto-STOP on preemption, disk preserved)
- **Disk**: 100GB SSD, auto-delete DISABLED
- **OS**: Ubuntu 22.04 LTS
- **Cost**: ~$0.066/hr (~₹5.5/hr, ~₹130/day)

## SSH Access
```bash
gcloud compute ssh nfpc-training --zone=us-central1-a
```

## Data Location on VM
```
/home/DIVYA/nfpc-phase2/data/
├── README.md
├── customers.parquet           (159K rows, 2.4MB)
├── accounts.parquet            (160K rows, 6.8MB)
├── demographics.parquet        (159K rows, 4.7MB)
├── accounts-additional.parquet (160K rows, 963KB)
├── branch.parquet              (9K rows, 261KB)
├── customer_account_linkage.parquet (160K rows, 1.8MB)
├── product_details.parquet     (159K rows, 2.9MB)
├── train_labels.parquet        (96K rows, 642KB)
├── test_accounts.parquet       (64K rows, 428KB)
├── transactions/               (~400M rows, 8.2GB, 396 parquet parts in 4 batches)
└── transactions_additional/    (~400M rows, 8.4GB, 311 parquet parts in 4 batches)
```
Total on disk: 27GB (16.2GB data + 10.5GB zip)

## Installed Software
- Python 3.10
- pandas, pyarrow
- LightGBM 4.6, XGBoost 3.2
- scikit-learn, SHAP, matplotlib, seaborn, scipy, numpy
- Kaggle CLI 1.7.4 (credentials at ~/.kaggle/kaggle.json)

## VM Management
```bash
# Start (if stopped/preempted)
gcloud compute instances start nfpc-training --zone=us-central1-a

# Stop (to save cost when not training)
gcloud compute instances stop nfpc-training --zone=us-central1-a

# Check status
gcloud compute instances describe nfpc-training --zone=us-central1-a --format="value(status)"
```
