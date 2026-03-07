# National Fraud Prevention Challenge

> **Source:** RBI Innovation Hub — National Fraud Prevention Challenge (Phase 2), hosted on EvalAI.

**Organized by:** Reserve Bank Innovation Hub & IIT Delhi

**Starts on:** Mar 3, 2026 5:30:00 AM IST (GMT + 5:30)
**Ends on:** Mar 13, 2026 11:59:00 PM IST (GMT + 5:30)

---

## Challenge Overview

Money laundering through mule accounts is a major challenge for financial institutions. Mule accounts are bank accounts used to move illicit funds, often recruited through social engineering or opened specifically for laundering purposes.

In this challenge, you are given synthetic banking data modeled after real-world transaction patterns. Your task is to **identify which accounts are mules** from a mix of legitimate and suspicious accounts using transaction history, customer demographics, account attributes, and branch metadata.

### Dataset Overview

- **~160,000 accounts** across ~159,000 customers
- **~400 million transactions** spanning a 5-year window (Jul 2020 - Jun 2025)
- **35 transaction channels** (UPI, NEFT, IMPS, ATM, etc.)
- **13 known mule behavior patterns** ranging from dormant activation to branch-level collusion
- **Multiple supplementary files**: demographics, branch metadata, scheme codes, geolocation data

### Key Challenges

- Labels may contain noise (mislabeled accounts in both directions)
- Red herring features that correlate with labels but don't generalize
- Temporal traps: ghost precursor bursts, post-activity tails, festival decoys
- Freeze/unfreeze patterns that can mislead feature engineering
- Private test set has attenuated signals compared to public test

### Data Access

The dataset (~16 GB, Apache Parquet format) is available on **[Kaggle](https://www.kaggle.com/datasets/abhyudayrbih/rbih-nfpc-phase-2/)**. Download it directly from there.

Refer to the `README.md` provided with the data for complete schema documentation and code examples.

---

## Evaluation Criteria

### Scoring Metrics (Public & Private Phases)

| Metric | Description | Phase |
|--------|-------------|-------|
| **AUC-ROC** | Area under the ROC curve on `is_mule` probability scores. Primary ranking metric. | Public & Private |
| **F1 Score** | Best F1 across 100 thresholds (0.00 to 1.00). Measures precision-recall balance. | Public & Private |
| **Temporal IoU** | Average intersection-over-union of predicted vs. actual suspicious activity time windows for mule accounts. Only computed when time windows are provided. | Public & Private |
| **RH Avoidance 1-7** | Robustness scores measuring how well the model avoids being misled by red-herring patterns in the data. Each score corresponds to a different trap category. | Private only |

### How AUC-ROC Works

AUC-ROC measures your model's ability to discriminate between mule and legitimate accounts across all classification thresholds. A score of 1.0 means perfect separation; 0.5 means random guessing. Rankings are determined by this metric.

### How F1 Score Works

We sweep 100 evenly spaced thresholds from 0.00 to 1.00 on your `is_mule` scores and report the maximum F1 score achieved. This captures the best possible balance between precision and recall for your model.

### How Temporal IoU Works

For accounts predicted as mules, you may optionally provide `suspicious_start` and `suspicious_end` timestamps. For each true mule account where both you and the ground truth have time windows:

- Temporal IoU = Intersection(predicted window, true window) / Union(predicted window, true window)
- The final score is the average IoU across all such accounts
- If no time windows are provided, Temporal IoU defaults to 0

### Red-Herring Avoidance (Private Phase Only)

The private test set contains several categories of tricky accounts designed to mislead models. The RH Avoidance scores (1 through 7) measure your model's accuracy on these specific subgroups. These scores are computed on the private phase only and are visible to organizers for final evaluation.

### Leaderboard

- **Public phase:** Leaderboard is visible to all participants during the challenge. Use it to benchmark and iterate.
- **Private phase:** Leaderboard is hidden. Scores are visible only to you and organizers. Used for final ranking.

### Code & Report Submissions

Code (ZIP) and report (PDF) submissions are validated for format compliance only. They do not produce model scores. A `Validation_Status` of 1 indicates the submission passed all checks.

---

## Terms and Conditions

- This challenge uses synthetic data generated for educational purposes. No real customer data is involved.
- Participants may use any tools, libraries, or techniques to build their models.
- Sharing of solution code between teams during the challenge is not permitted.
- The organizers reserve the right to disqualify submissions that attempt to exploit the evaluation system.
- By participating, you agree to these terms.

---

## Phases

### Phase 1: Public Phase

**Starts on:** Mar 3, 2026 5:30:00 AM IST (GMT + 5:30)
**Ends on:** Mar 13, 2026 11:59:00 PM IST (GMT + 5:30)

Submit your predictions for all test accounts. Your submission will be evaluated on AUC-ROC (primary) and Temporal IoU (bonus). Upload a CSV file with columns: `account_id`, `is_mule`, `suspicious_start`, `suspicious_end`.

**Note:** Failed submissions are not counted against your submission limit. Check the error reason, fix your file, and resubmit.

| Limit | Value |
|-------|-------|
| Max submissions/day | 100 |
| Max submissions/month | 3000 |
| Max total submissions | 100000 |
| Max concurrent submissions | 3 |

---

### Phase 2: Private Phase

**Starts on:** Mar 3, 2026 5:30:00 AM IST (GMT + 5:30)
**Ends on:** Mar 13, 2026 11:59:00 PM IST (GMT + 5:30)

Submit your predictions for all test accounts. Your submission will be evaluated on AUC-ROC (primary) and Temporal IoU (bonus). Upload a CSV file with columns: `account_id`, `is_mule`, `suspicious_start`, `suspicious_end`.

You have **10 submissions** for this phase. Use them wisely — review your private phase scores after each attempt, identify weaknesses (especially around red-herring avoidance), and iterate on your model before submitting again. The **last uploaded submission** will be considered for final scoring.

**Note:** Failed submissions are not counted against your submission limit. Check the error reason, fix your file, and resubmit.

| Limit | Value |
|-------|-------|
| Max submissions/day | 10 |
| Max submissions/month | 10 |
| Max total submissions | 10 |
| Max concurrent submissions | 3 |

---

### Phase 3: Code Submission

**Starts on:** Mar 3, 2026 5:30:00 AM IST (GMT + 5:30)
**Ends on:** Mar 13, 2026 11:59:00 PM IST (GMT + 5:30)

Upload your **complete solution code** as a single ZIP archive.

**Requirements:**

- **Maximum file size:** 200 MB (compressed); 200 MB (uncompressed total)
- **Must include:** A `README.md` or `README.txt` in the root directory explaining how to run your code
- **Allowed file types:** `.py`, `.ipynb`, `.r`, `.R`, `.sh`, `.json`, `.yaml`, `.yml`, `.toml`, `.cfg`, `.ini`, `.txt`, `.md`, `.pkl`, `.joblib`, `.pt`, `.pth`, `.onnx`, `.h5` (model weights only)
- **Prohibited:** Data files (`.parquet`, `.csv`, `.hdf5`, `.npy`, `.npz`, `.feather`) — do not include training data or dataset copies

**What to Include:**

1. All source code used for feature engineering, model training, and inference
2. **Trained model files** (weights, serialized models) — these must be included in the ZIP. If your model files cause the archive to exceed 200 MB, host them on GitHub and provide the link in your README.
3. **Code from earlier iterations:** Include source code for previous model versions and experiments alongside your final solution. If including all iterations pushes the ZIP beyond 200 MB, package only your latest code in the archive and link to older versions in your README.
4. A README with:
   - Environment setup instructions (Python version, dependencies)
   - Steps to reproduce your results
   - Brief description of your approach
   - Links to any externally hosted model files or older code versions (if applicable)

**1 submission only.** Make sure your code is complete and well-documented before uploading.

**Note:** Failed submissions are not counted against your submission limit. Check the error reason, fix your file, and resubmit.

| Limit | Value |
|-------|-------|
| Max submissions/day | 1 |
| Max submissions/month | 1 |
| Max total submissions | 1 |
| Max concurrent submissions | 1 |

---

### Phase 4: Report Submission

**Starts on:** Mar 3, 2026 5:30:00 AM IST (GMT + 5:30)
**Ends on:** Mar 13, 2026 11:59:00 PM IST (GMT + 5:30)

Upload your **solution report** as a PDF document.

**Requirements:**

- **Format:** PDF only
- **Maximum file size:** 50 MB

**What to Include:**

1. **Approach:** Description of your methodology, feature engineering, and model architecture
2. **Key findings:** Insights about mule account patterns discovered in the data
3. **Experiments:** Summary of experiments tried, what worked and what didn't
4. **Results:** Performance metrics on public/private test sets
5. **Red herring analysis:** How you handled potential red herrings and noise in the data

**3 submissions allowed.** You may re-upload your report if you need to correct errors or make improvements. Only the **most recent submission** will be evaluated.

**Note:** Failed submissions are not counted against your submission limit. Check the error reason, fix your file, and resubmit.

| Limit | Value |
|-------|-------|
| Max submissions/day | 3 |
| Max submissions/month | 3 |
| Max total submissions | 3 |
| Max concurrent submissions | 3 |
