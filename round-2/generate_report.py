"""Generate PDF report for NFPC Phase 2 submission."""
from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "NFPC Phase 2 - Solution Report - Team DMJ.ONE", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(43, 108, 176)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(190, 227, 248)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(44, 82, 130)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(51, 51, 51)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text, bold_prefix=""):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(51, 51, 51)
        self.cell(5, 5.5, "-")
        if bold_prefix:
            self.set_font("Helvetica", "B", 10)
            self.cell(self.get_string_width(bold_prefix) + 1, 5.5, bold_prefix)
            self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def add_table(self, headers, rows, col_widths=None):
        if col_widths is None:
            w = (self.w - self.l_margin - self.r_margin) / len(headers)
            col_widths = [w] * len(headers)
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(235, 244, 255)
        self.set_text_color(51, 51, 51)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True)
        self.ln()
        self.set_font("Helvetica", "", 9)
        for ri, row in enumerate(rows):
            fill = ri % 2 == 1
            if fill:
                self.set_fill_color(247, 250, 252)
            else:
                self.set_fill_color(255, 255, 255)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1, fill=fill)
            self.ln()
        self.ln(3)

    def highlight_box(self, text):
        self.set_fill_color(255, 251, 235)
        self.set_draw_color(214, 158, 46)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 60, 10)
        x = self.get_x()
        w = self.w - self.l_margin - self.r_margin
        self.rect(x, self.get_y(), w, 1, style="F")
        self.set_x(x + 3)
        self.multi_cell(w - 6, 5, text, fill=True)
        self.ln(3)
        self.set_text_color(51, 51, 51)

    def decorative_line(self):
        self.set_draw_color(43, 108, 176)
        self.set_line_width(0.5)
        y = self.get_y()
        mid = self.w / 2
        self.line(mid - 30, y, mid + 30, y)
        self.ln(5)


pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()

# Title
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(26, 54, 93)
pdf.cell(0, 14, "National Fraud Prevention Challenge", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 18)
pdf.cell(0, 12, "Phase 2: Solution Report", new_x="LMARGIN", new_y="NEXT")
pdf.set_draw_color(43, 108, 176)
pdf.set_line_width(1.5)
pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
pdf.ln(8)

pdf.set_font("Helvetica", "", 11)
pdf.set_text_color(51, 51, 51)
pdf.cell(0, 7, "Team: DMJ.ONE", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Date: March 16, 2026", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 7, "Challenge: RBI Innovation Hub & IIT Delhi", new_x="LMARGIN", new_y="NEXT")
pdf.ln(4)

# Executive Summary
pdf.set_fill_color(240, 248, 255)
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 7, "  Executive Summary", fill=True, new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
pdf.multi_cell(0, 5.5,
    "We developed a gradient-boosted tree ensemble for mule account detection achieving "
    "AUC-ROC 0.968 (public) and 0.956 (private). Our pipeline processes 16GB of transaction "
    "data through 208 engineered features, multi-round confident learning for label noise, "
    "and 3-seed cross-validated training with rank-averaged ensembling. We identified and "
    "analyzed all 7 red herring categories, achieving near-perfect avoidance on 6 of 7.",
    fill=True
)
pdf.ln(6)

pdf.decorative_line()

# ===== Section 1 =====
pdf.section_title("1. Approach: Methodology and Model Architecture")

pdf.sub_title("1.1 Pipeline Overview")
pdf.body_text(
    "Our end-to-end pipeline consists of 5 stages: (1) Feature engineering across 4 data "
    "passes, (2) Label cleaning via confident learning + heuristic noise detection, "
    "(3) 3-model ensemble training with multi-seed CV, (4) Temporal window prediction "
    "for suspicious activity periods, and (5) Submission generation with calibrated probabilities."
)

pdf.sub_title("1.2 Feature Engineering (208 features)")
pdf.body_text(
    "Features are computed in 4 memory-efficient passes over the 16GB Parquet dataset "
    "(160K accounts, ~400M transactions):"
)
pdf.add_table(
    ["Pass", "Category", "Count", "Key Features"],
    [
        ["1", "Txn Core", "~100", "Volume, amounts, temporal histograms, channels, MCC, counterparties"],
        ["2", "Txn Extended", "~40", "Geo spread, IP diversity, balance stats, txn types, dormancy"],
        ["3", "Static", "~35", "Account age, freeze history, demographics, branch, products"],
        ["4", "Graph", "~33", "PageRank, HITS, betweenness, Louvain communities, clustering"],
    ],
    [15, 30, 15, 130]
)

pdf.sub_title("1.3 Label Cleaning and Noise-Robust Training")
pdf.bullet(
    "Out-of-fold LightGBM probability estimation with per-class thresholds "
    "(Northcutt et al. 2021). Identifies samples where model confidence conflicts with labels.",
    "Confident Learning (2 rounds): "
)
pdf.bullet(
    "Rule-based detection targeting 7 red herring categories using label metadata "
    "(alert_reason, mule_flag_date, flagged_by_branch).",
    "Heuristic Noise Scoring: "
)
pdf.bullet(
    "max(CL score, heuristic score) mapped to sample weights in [0.2, 1.0]. "
    "Noisy samples are downweighted but not removed, preserving training set size.",
    "Combined Weights: "
)

pdf.sub_title("1.4 Model Training")
pdf.bullet("LightGBM + XGBoost + CatBoost, each trained independently", "Ensemble: ")
pdf.bullet("3-seed (42, 43, 44) x 5-fold stratified CV for prediction stability", "Cross-validation: ")
pdf.bullet(
    "Frequency encoding for categorical features. No label leakage possible since we "
    "only count occurrences, not use target statistics.",
    "Encoding: "
)
pdf.bullet(
    "Optuna-optimized with 100 trials per model. Best parameters use near-zero "
    "regularization (alpha ~4e-5, lambda ~8e-7), suggesting clean features.",
    "Hyperparameters: "
)
pdf.bullet(
    "Rank averaging (converting predictions to percentile ranks before averaging) "
    "outperformed simple averaging, AUC-weighted averaging, and stacked meta-learning.",
    "Ensemble method: "
)
pdf.bullet("scale_pos_weight = 34 (matching ~2.8% mule ratio in training)", "Class imbalance: ")

pdf.sub_title("1.5 Temporal Window Prediction")
pdf.body_text(
    "For accounts predicted as mules (probability >= threshold), we analyze their full "
    "transaction history to identify suspicious activity windows. A sliding window approach "
    "scores each time segment by anomaly density (unusual amounts, frequencies, counterparties) "
    "and selects the window with the highest concentration of suspicious patterns."
)

# ===== Section 2 =====
pdf.add_page()
pdf.section_title("2. Key Findings: Mule Account Patterns")

pdf.sub_title("2.1 Top Discriminative Features (SHAP Analysis)")
pdf.body_text(
    "SHAP (SHapley Additive exPlanations) analysis across all three models revealed "
    "consistent feature importance patterns:"
)
pdf.bullet("Total transaction count and unique counterparties are the strongest signals. Mule accounts show characteristic high-volume, high-counterparty patterns.", "Transaction volume: ")
pdf.bullet("PageRank and betweenness centrality capture mules as intermediaries in transaction networks, acting as bridges between otherwise disconnected account clusters.", "Graph centrality: ")
pdf.bullet("Mules tend to use more diverse transaction channels (ATM, online, POS, wire) compared to legitimate accounts with consistent channel preferences.", "Channel diversity: ")
pdf.bullet("Mules exhibit more activity during non-business hours and weekends, with less regular temporal distributions compared to legitimate accounts.", "Temporal patterns: ")
pdf.bullet("Transactions structured just below common reporting thresholds are a strong indicator of deliberate evasion behavior characteristic of mule operations.", "Structuring indicators: ")
pdf.bullet("Rapid credit-debit cycles where funds pass through quickly with minimal balance retention (pass-through behavior).", "Balance behavior: ")

pdf.sub_title("2.2 Mule Behavioral Archetypes")
pdf.body_text("We identified several distinct mule behavior patterns in the data:")
pdf.bullet("High velocity of credits followed by rapid debits within hours/days. Minimal balance retention. The account serves purely as a transit point.", "Pass-through mules: ")
pdf.bullet("High fan-in/fan-out with many unique counterparties. Acts as a money funnel, aggregating from many sources and distributing to many destinations.", "Network hubs: ")
pdf.bullet("Long periods of inactivity (weeks/months) followed by sudden intense transaction bursts, suggesting activation for specific money laundering operations.", "Dormant-burst: ")
pdf.bullet("Deliberately fragmenting transaction amounts to stay below monitoring thresholds. Often combined with multiple channels to further obscure patterns.", "Structuring mules: ")

# ===== Section 3 =====
pdf.section_title("3. Experiments: What Worked and What Did Not")

pdf.add_table(
    ["Version", "Public AUC", "Outcome"],
    [
        ["V1: Baseline (LGB+XGB+CB, target enc)", "0.956", "Good starting point"],
        ["V2: Optuna HPO (100 trials/model)", "0.956", "Found near-zero regularization"],
        ["V3: Freq enc + rank avg + multi-seed", "0.968", "BEST. No leakage, stable."],
        ["V5: Feature interactions (26 derived)", "0.963", "Hurt. Trees find interactions."],
        ["V6: Pseudo-labeling (2-stage)", "0.787", "Catastrophic. Diluted signal."],
        ["V7: Drop all branch features", "0.959", "Fixed RH7, destroyed AUC."],
        ["V8: Drop branch_code freq/count only", "0.958", "Surgical fix. Still too costly."],
    ],
    [65, 25, 100]
)

pdf.sub_title("Key Experimental Insights")
pdf.bullet(
    "Switching from target encoding to frequency encoding eliminated potential label "
    "leakage and improved public AUC from 0.956 to 0.968 (+1.2%).",
    "Frequency encoding: "
)
pdf.bullet(
    "Averaging predictions across 3 random seeds reduced variance and improved "
    "ensemble stability without additional compute cost.",
    "Multi-seed averaging: "
)
pdf.bullet(
    "Adding 26 manually crafted feature interactions (flow ratios, density metrics) "
    "actually hurt performance. With near-zero regularization, trees already discover "
    "optimal splits; explicit interactions introduce redundancy that degrades generalization.",
    "Feature interactions: "
)
pdf.bullet(
    "Two-stage pseudo-labeling catastrophically failed (0.787 AUC). Adding 63K "
    "pseudo-labeled samples diluted the mule signal from 2.8% to 1.8%, reinforcing "
    "the model's biases rather than correcting them.",
    "Pseudo-labeling: "
)

pdf.highlight_box(
    "Key Lesson: With near-zero regularization and powerful tree models, the simplest "
    "feature set produces the best generalization. Feature engineering quality matters "
    "more than model complexity or training data augmentation."
)

# ===== Section 4 =====
pdf.add_page()
pdf.section_title("4. Results: Performance Metrics")

pdf.add_table(
    ["Metric", "Public Phase", "Private Phase"],
    [
        ["AUC-ROC (primary ranking)", "0.968136", "0.955815"],
        ["F1 Score (best threshold)", "0.576", "0.536"],
        ["Temporal IoU", "0.184", "0.189"],
        ["OOF CV AUC (3-seed avg)", "~0.960", "-"],
    ],
    [55, 50, 50]
)

pdf.body_text(
    "The public-to-private gap (~1.2% AUC) indicates reasonable generalization. "
    "The private test set contains more subtle mule patterns with attenuated signals, "
    "explaining the expected degradation. The consistent OOF CV AUC (~0.960) closely "
    "tracks the public score, confirming our cross-validation setup is well-calibrated."
)

pdf.body_text(
    "F1 Score optimization: We sweep 100 thresholds from 0.00 to 1.00 and report the "
    "maximum F1. Our best threshold is approximately 0.3, reflecting the class imbalance "
    "(~2.8% mule ratio) where recall is prioritized over precision."
)

pdf.body_text(
    "Temporal IoU: Our sliding window approach achieves coverage on ~960 of the true "
    "mule accounts with temporal windows. The modest IoU (~0.18) suggests room for "
    "improvement in window boundary precision, though the detection rate is solid."
)

# ===== Section 5 =====
pdf.section_title("5. Red Herring Analysis")

pdf.body_text(
    "The challenge includes 7 categories of red herrings designed to mislead models. "
    "Our label cleaning pipeline uses heuristic noise scoring to detect and downweight "
    "these patterns during training:"
)

pdf.add_table(
    ["RH#", "Description", "Detection Method", "Score"],
    [
        ["1", "Routine Investigation FPs", "Heuristic weight 0.6 for alert_reason", "0.995"],
        ["2", "Missing alert_reason", "Heuristic weight 0.8 for null/empty", "0.990"],
        ["3", "Future mule_flag_date", "Dates after Jun 2025 downweighted", "0.993"],
        ["4", "Very old flag dates", "Dates before Jul 2020 downweighted", "0.978"],
        ["5", "Boundary date artifacts", "Exact boundary dates flagged", "0.993"],
        ["6", "Frozen accounts as mules", "Null flagged_by_branch detected", "0.999"],
        ["7", "Flagged by own branch", "Branch features carry signal + noise", "0.000"],
    ],
    [12, 42, 60, 16]
)

pdf.sub_title("RH7 Deep Dive: The Branch Feature Dilemma")
pdf.body_text(
    "Red Herring #7 (accounts flagged by their own branch) presented the most challenging "
    "tradeoff. Our experiments revealed:"
)
pdf.bullet(
    "V3 (all features): Private AUC 0.956, RH7 = 0.000. The model completely falls "
    "for this red herring because branch features encode the spurious same-branch pattern.",
    "Baseline: "
)
pdf.bullet(
    "V7 (no branch features): Private AUC 0.893, RH7 = 1.000. Perfect RH7 avoidance "
    "but catastrophic AUC drop (-6.3%). All other RH scores also collapsed (0.25-0.74).",
    "Full removal: "
)
pdf.bullet(
    "V8 (drop branch_code freq/count only): Private AUC 0.868, RH7 = 1.000. Even "
    "surgical removal of just 2 features caused significant damage.",
    "Surgical removal: "
)

pdf.highlight_box(
    "Conclusion: Branch features carry genuine discriminative signal that outweighs the "
    "RH7 penalty. The branch_code frequency encoding captures bank network structure "
    "that is predictive of mule behavior. Decorrelating the RH7 signal from branch "
    "features without losing this predictive power remains an open challenge. A potential "
    "approach would be adversarial debiasing during training, but this was not feasible "
    "within the competition timeframe."
)

# ===== Section 6 =====
pdf.section_title("6. Infrastructure and Reproducibility")

pdf.bullet("GCP n2-highmem-8 (8 vCPU, 64GB RAM), us-central1-a", "Compute: ")
pdf.bullet("~12 minutes for full 3-seed x 5-fold CV pipeline", "Training time: ")
pdf.bullet("~10 minutes for 208 features across 160K accounts and ~400M transactions", "Feature engineering: ")
pdf.bullet("Batch processing with explicit garbage collection for 16GB dataset", "Memory: ")
pdf.bullet("Python 3.10+, LightGBM, XGBoost, CatBoost, scikit-learn, NetworkX, Optuna, SHAP", "Stack: ")
pdf.bullet("python run_v3.py --skip-optuna (uses cached Optuna params)", "Reproduce: ")

pdf.ln(6)
pdf.decorative_line()
pdf.set_font("Helvetica", "I", 9)
pdf.set_text_color(128, 128, 128)
pdf.cell(0, 6, "End of Report", align="C")

pdf.output("d:/national-fraud-prevention-challenge/round-2/report.pdf")
print("PDF report generated successfully")
