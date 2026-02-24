"""Fix table captions and remove --- rules from the markdown report."""
import re

with open('r:/national-fraud-prevention-challenge/NFPC_Phase1_EDA_Report.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove any table captions from previous bad run
content = re.sub(r'\n\*Table \d+\.\d+:.*?\*\n', '\n', content)

# Remove all standalone --- horizontal rules
content = re.sub(r'\n\n---\n\n', '\n\n', content)
content = re.sub(r'\n---\n', '\n', content)

# Add table captions with code-block awareness
descs = {
    (1, 1): 'Dataset overview with file sizes, row counts, and column descriptions',
    (1, 2): 'Data integrity verification confirming 100% join coverage',
    (1, 3): 'Missing value analysis across all data tables',
    (2, 1): 'Class distribution showing extreme 1:90 mule-to-legitimate imbalance',
    (2, 2): 'Alert reason distribution across 263 mule accounts',
    (3, 1): 'Account status and freeze rate comparison (19.6x differential)',
    (3, 2): 'Balance metrics comparison across four balance types',
    (3, 3): 'Mule rate by account age cohort with relative risk',
    (3, 4): 'Account-level flag prevalence and statistical significance',
    (4, 1): 'Transaction volume comparison (median 1.78x higher for mules)',
    (4, 2): 'Transaction amount statistics across distribution quantiles',
    (4, 3): 'Payment channels overrepresented in mule accounts',
    (4, 4): 'Payment channels underrepresented in mule accounts',
    (4, 5): 'Pass-through ratio comparison showing near-unity mule ratio',
    (4, 6): 'Temporal transaction pattern differences across time dimensions',
    (4, 7): 'Transaction velocity metrics (4.3x faster median gap for mules)',
    (4, 8): 'Burst detection metrics for dormant activation pattern',
    (4, 9): 'Top MCC codes ranked by mule overrepresentation factor',
    (5, 1): 'Customer demographic comparison by age and relationship tenure',
    (5, 2): 'KYC document availability rates showing Aadhaar gap for mules',
    (5, 3): 'Digital banking and service flag comparison with significance testing',
    (5, 4): 'Product holdings comparison showing 39% higher savings for mules',
    (6, 1): 'Evidence summary for all 12 documented mule behavior patterns',
    (6, 2): 'Counterparty diversity metrics demonstrating fan-in/fan-out topology',
    (6, 3): 'Branches with anomalously high mule concentration rates',
    (6, 4): 'Geographic PIN prefix mismatch at state and district level',
    (7, 1): 'Mann-Whitney U test results for continuous balance variables',
    (7, 2): 'Chi-square test results for categorical variables',
    (7, 3): 'Top feature correlations with mule label ranked by strength',
    (8, 1): 'Data leakage risk assessment for candidate model features',
    (8, 2): 'Missing value patterns by class revealing informative missingness',
    (9, 1): 'Transaction aggregation features (7 proposed)',
    (9, 2): 'Structuring detection features (7 proposed)',
    (9, 3): 'Velocity and burstiness features (10 proposed)',
    (9, 4): 'Pass-through and flow features (4 proposed)',
    (9, 5): 'Graph and network features (8 proposed)',
    (9, 6): 'Channel features (12 proposed)',
    (9, 7): 'Temporal features (4 proposed)',
    (9, 8): 'MCC-based features (6 proposed)',
    (9, 9): 'Derived ratio features (3 proposed)',
    (9, 10): 'Unsupervised anomaly features with extreme statistical significance',
    (9, 11): 'Geographic mismatch feature specification',
    (9, 12): 'Feature count summary across 13 categories totalling 125',
    (10, 1): 'Model performance: LightGBM, XGBoost, and ensemble comparison',
    (10, 2): 'Top 15 features ranked by LightGBM split importance',
    (10, 3): 'Cost matrix assumptions for Indian banking deployment',
    (10, 4): 'Operating point comparison: cost-optimal vs F1-optimal thresholds',
}

lines = content.split('\n')
output = []
current_section = 0
table_count = {}
in_table = False
in_code_block = False

for line in lines:
    stripped = line.strip()

    # Track fenced code blocks (``` markers)
    if stripped.startswith('```'):
        in_code_block = not in_code_block
        output.append(line)
        continue

    # Skip table detection inside code blocks
    if in_code_block:
        output.append(line)
        continue

    m = re.match(r'^## (\d+)\.', line)
    if m:
        current_section = int(m.group(1))
        table_count.setdefault(current_section, 0)

    is_table_line = stripped.startswith('|')

    if is_table_line:
        if not in_table:
            in_table = True
        output.append(line)
    else:
        if in_table:
            in_table = False
            if current_section > 0:
                table_count[current_section] = table_count.get(current_section, 0) + 1
                key = (current_section, table_count[current_section])
                desc = descs.get(key, f'Data table {key[0]}.{key[1]}')
                output.append('')
                output.append(f'*Table {key[0]}.{key[1]}: {desc}*')
        output.append(line)

content = '\n'.join(output)

with open('r:/national-fraud-prevention-challenge/NFPC_Phase1_EDA_Report.md', 'w', encoding='utf-8') as f:
    f.write(content)

total_tables = sum(table_count.values())
print(f'Added {total_tables} table captions')
for sec in sorted(table_count):
    print(f'  Section {sec}: {table_count[sec]} tables')
