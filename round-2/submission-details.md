# Submission Guidelines

This challenge has **four submission phases**. Select the appropriate phase from the dropdown on the submit page.

---

## 1. Public Phase — CSV

Submit predictions as a CSV file.

| Column | Type | Description |
|--------|------|-------------|
| `account_id` | string | Account identifier (must match all accounts in `public_test_accounts.parquet`) |
| `is_mule` | float [0, 1] | Probability that the account is a mule (1 = mule) |
| `suspicious_start` | datetime (optional) | Start of suspected mule activity window (format: `YYYY-MM-DDTHH:MM:SS`) |
| `suspicious_end` | datetime (optional) | End of suspected mule activity window |

**Example CSV:**
```csv
account_id,is_mule,suspicious_start,suspicious_end
ACCT_000005,0.02,,
ACCT_000007,0.95,2023-11-15T09:30:00,2024-02-20T16:45:00
ACCT_000009,0.12,,
```

**Rules:**
- Must include predictions for **all accounts** in the test set
- `is_mule` must be between 0 and 1
- `suspicious_start` / `suspicious_end` are optional but contribute to the Temporal IoU metric
- Leave time fields empty for accounts you predict as legitimate

**Limit: 100 submissions per day**

---

## 2. Private Phase — CSV

Same CSV format as the Public Phase, but evaluated against the **private test set**.

- The private test set has more subtle mule patterns and attenuated signals
- Additional robustness metrics are computed privately
- Leaderboard is **hidden** — scores are visible only to you and organizers

**Limit: 10 submissions total. Choose carefully.**

---

## 3. Code Submission — ZIP

Upload your complete solution code as a **single ZIP archive**.

**Requirements:**
- Maximum uncompressed size: **200 MB**
- Must include a `README.md` or `README.txt` in the root of the ZIP
- **Allowed file types:** `.py`, `.ipynb`, `.r`, `.R`, `.sh`, `.json`, `.yaml`, `.yml`, `.toml`, `.cfg`, `.ini`, `.txt`, `.md`, `.pkl`, `.joblib`, `.pt`, `.pth`, `.onnx`, `.h5` (model weights)
- **Prohibited:** Data files (`.parquet`, `.csv`, `.hdf5`, `.npy`, `.npz`, `.feather`) — do not include training data or dataset copies

**What to Include:**
1. All source code for feature engineering, model training, and inference
2. Trained model weights or serialized models
3. A README describing:
   - Environment setup (Python version, dependencies)
   - Steps to reproduce your results
   - Brief description of your approach

**Limit: 1 submission only. Ensure your code is complete before uploading.**

---

## 4. Report Submission — PDF

Upload your solution report as a **PDF document**.

**Requirements:**
- Format: PDF only (max 50 MB)

**What to Include:**
1. **Approach:** Methodology, feature engineering, and model architecture
2. **Key Findings:** Insights about mule account patterns discovered in the data
3. **Experiments:** Summary of what worked and what didn't
4. **Results:** Performance metrics on public/private test sets
5. **Red Herring Analysis:** How you handled potential red herrings and noise

**Limit: 3 submissions maximum. Make sure your report is polished, professional and comprehensive before submitting.**
