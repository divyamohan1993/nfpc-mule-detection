import Link from "next/link";
import Image from "next/image";

const FEATURE_DATA = [
  { name: "mcc_6051_rate", importance: 275234, label: "Wire Transfer Rate", pct: 100 },
  { name: "was_frozen", importance: 105127, label: "Account Freeze History", pct: 38.2 },
  { name: "ch_UPD_rate", importance: 33884, label: "UPI Debit Rate", pct: 12.3 },
  { name: "cp_per_txn", importance: 30480, label: "Counterparties/Txn", pct: 11.1 },
  { name: "days_since_kyc", importance: 30205, label: "KYC Recency", pct: 11.0 },
  { name: "mcc_5933_rate", importance: 28738, label: "Pawn Shop Rate", pct: 10.4 },
  { name: "p25_amount", importance: 24823, label: "P25 Amount", pct: 9.0 },
  { name: "ch_CHQ_rate", importance: 20250, label: "Cheque Rate", pct: 7.4 },
  { name: "rel_years", importance: 15593, label: "Relationship Tenure", pct: 5.7 },
  { name: "ch_ATW_rate", importance: 13464, label: "ATM Withdrawal Rate", pct: 4.9 },
];

const PATTERNS = [
  {
    id: 1,
    name: "Dormant Activation",
    desc: "Inactive accounts suddenly process high-value bursts",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M13 10V3L4 14h7v7l9-11h-7z" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    id: 2,
    name: "Structuring",
    desc: "Transactions just below the 50K INR reporting threshold",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M3 3v18h18M7 16l4-4 4 4 6-6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    id: 3,
    name: "Rapid Pass-Through",
    desc: "Near 1:1 credit-to-debit ratio -- money flows through untouched",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M5 12h14M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    id: 4,
    name: "Fan-In / Fan-Out",
    desc: "Many-to-one or one-to-many fund flows reveal network topology",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <circle cx="12" cy="12" r="3" /><path d="M12 2v4m0 12v4m10-10h-4M6 12H2m15.07-7.07l-2.83 2.83M9.76 14.24l-2.83 2.83m0-10.14l2.83 2.83m4.48 4.48l2.83 2.83" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: 5,
    name: "Geographic Anomaly",
    desc: "PIN code mismatches across customer, branch, and address",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" /><circle cx="12" cy="9" r="2.5" />
      </svg>
    ),
  },
  {
    id: 6,
    name: "New Account High Value",
    desc: "Young accounts with disproportionate transaction volume",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M12 2v20M2 12h20" strokeLinecap="round" /><path d="M12 2l4 4M12 2L8 6" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: 7,
    name: "Income Mismatch",
    desc: "Transaction values vastly exceed account balance patterns",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M12 1v22M17 5H9.5a3.5 3.5 0 000 7h5a3.5 3.5 0 010 7H6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    id: 8,
    name: "Post-Mobile-Change Spike",
    desc: "Activity surges 7x after mobile number update",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <rect x="5" y="2" width="14" height="20" rx="2" /><path d="M12 18h.01" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: 9,
    name: "Round Amount Patterns",
    desc: "Overuse of exact round amounts (1K, 5K, 10K, 50K)",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <circle cx="12" cy="12" r="10" /><path d="M8 12h8M12 8v8" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: 10,
    name: "Layered / Subtle",
    desc: "Weak multi-signal combinations that evade single-rule detection",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    id: 11,
    name: "Salary Cycle Exploitation",
    desc: "Laundering timed to coincide with salary credit windows",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <rect x="3" y="4" width="18" height="18" rx="2" /><path d="M16 2v4M8 2v4M3 10h18" strokeLinecap="round" />
      </svg>
    ),
  },
  {
    id: 12,
    name: "Branch-Level Collusion",
    desc: "Suspicious account clusters originating from the same branch",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="w-6 h-6">
        <path d="M3 21h18M5 21V7l7-4 7 4v14M9 21v-6h6v6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ),
  },
];

const KEY_SIGNALS = [
  { signal: "Accounts Frozen", legit: "3.0%", mule: "58.9%", mult: "19.6x" },
  { signal: "MCC 6051 (Wire Transfer)", legit: "0.12%", mule: "2.10%", mult: "18x" },
  { signal: "Post-Mobile Txn Value", legit: "127K", mule: "903K", mult: "7.1x" },
  { signal: "Txn-to-Balance Ratio", legit: "68.5", mule: "473.9", mult: "6.9x" },
  { signal: "Near-50K Structuring", legit: "1.1%", mule: "5.9%", mult: "5.3x" },
  { signal: "Median Txn Velocity", legit: "336.8h", mule: "78.3h", mult: "4.3x faster" },
  { signal: "Unique Counterparties", legit: "13.7", mule: "37.1", mult: "2.7x" },
  { signal: "Pass-Through Ratio", legit: "1.184", mule: "1.015", mult: "~1:1" },
];

const VIZ_SECTIONS = [
  {
    title: "Data & Distribution",
    plots: [
      { file: "01_class_distribution.png", label: "Class Distribution" },
      { file: "02_alert_reasons.png", label: "Alert Reasons" },
      { file: "03_account_status_freeze.png", label: "Account Status & Freeze" },
      { file: "04_balance_boxplots.png", label: "Balance Distributions" },
    ],
  },
  {
    title: "Account & Customer",
    plots: [
      { file: "05_account_opening.png", label: "Account Opening" },
      { file: "06_customer_demographics.png", label: "Demographics" },
      { file: "07_flags_heatmap.png", label: "Flags Heatmap" },
      { file: "08_channel_analysis.png", label: "Channel Analysis" },
    ],
  },
  {
    title: "Transaction Patterns",
    plots: [
      { file: "09_temporal_patterns.png", label: "Temporal Patterns" },
      { file: "10_amount_distribution.png", label: "Amount Distribution" },
      { file: "11_structuring.png", label: "Structuring Detection" },
      { file: "12_counterparty.png", label: "Counterparty Analysis" },
    ],
  },
  {
    title: "Advanced Analysis",
    plots: [
      { file: "13_mcc_analysis.png", label: "MCC Analysis" },
      { file: "14_branch_analysis.png", label: "Branch Analysis" },
      { file: "15_velocity.png", label: "Velocity" },
      { file: "16_correlations.png", label: "Correlations" },
    ],
  },
  {
    title: "Model & Evaluation",
    plots: [
      { file: "17_feature_importance.png", label: "Feature Importance" },
      { file: "18_model_evaluation.png", label: "ROC & PR Curves" },
      { file: "19_shap_summary.png", label: "SHAP Summary" },
      { file: "20_geographic_analysis.png", label: "Geographic" },
    ],
  },
  {
    title: "Deep Dive",
    plots: [
      { file: "21_unsupervised_features.png", label: "Unsupervised Features" },
      { file: "22_focused_heatmap.png", label: "Focused Heatmap" },
      { file: "23_network_topology.png", label: "Network Topology" },
      { file: "24_false_positive_analysis.png", label: "False Positive Analysis" },
    ],
  },
];

function StatCard({
  value,
  label,
  sub,
}: {
  value: string;
  label: string;
  sub?: string;
}) {
  return (
    <div className="group relative rounded-2xl border border-border bg-surface-raised p-6 transition-all hover:border-accent/30 hover:shadow-[0_0_30px_rgba(0,212,170,0.08)]">
      <div className="text-4xl font-black tracking-tight text-accent sm:text-5xl">
        {value}
      </div>
      <div className="mt-2 text-sm font-medium">{label}</div>
      {sub && <div className="mt-1 text-xs text-[#666]">{sub}</div>}
    </div>
  );
}

function FeatureBar({
  name,
  label,
  pct,
  rank,
}: {
  name: string;
  label: string;
  pct: number;
  rank: number;
}) {
  return (
    <div className="group flex items-center gap-4">
      <span className="w-6 shrink-0 text-right text-xs font-mono text-[#666]">
        {rank}
      </span>
      <div className="flex-1">
        <div className="mb-1 flex items-baseline justify-between">
          <span className="text-sm font-medium group-hover:text-accent transition-colors">
            {label}
          </span>
          <code className="text-xs font-mono text-[#666]">{name}</code>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-surface-overlay">
          <div
            className="h-full rounded-full bg-gradient-to-r from-accent/60 to-accent transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    </div>
  );
}

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Nav */}
      <nav className="fixed top-0 z-50 w-full border-b border-border/50 bg-surface/80 backdrop-blur-xl">
        <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent/10">
              <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4 text-accent" stroke="currentColor" strokeWidth="2">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
              </svg>
            </div>
            <span className="text-sm font-semibold tracking-tight">NFPC</span>
            <span className="hidden text-xs text-[#666] sm:inline">
              Mule Detection
            </span>
          </div>
          <div className="flex items-center gap-6 text-xs text-[#a0a0a0]">
            <a href="#results" className="transition-colors hover:text-accent">
              Results
            </a>
            <a href="#features" className="transition-colors hover:text-accent">
              Features
            </a>
            <a href="#patterns" className="transition-colors hover:text-accent">
              Patterns
            </a>
            <a
              href="#visualizations"
              className="transition-colors hover:text-accent"
            >
              Viz
            </a>
            <Link
              href="/pitch"
              className="rounded-full border border-accent/30 px-3 py-1 text-accent transition-colors hover:bg-accent/10"
            >
              Pitch
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-6 pt-14">
        <div
          className="pointer-events-none absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }}
        />
        <div className="pointer-events-none absolute left-1/2 top-1/3 h-[800px] w-[800px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent/5 blur-[120px]" />

        <div className="relative z-10 max-w-4xl text-center">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-surface-raised px-4 py-1.5 text-xs text-[#a0a0a0]">
            <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
            RBIH x IIT Delhi TRYST 2025
          </div>

          <h1 className="text-5xl font-black tracking-tight sm:text-7xl lg:text-8xl">
            Catching{" "}
            <span className="bg-gradient-to-r from-accent to-emerald-400 bg-clip-text text-transparent">
              Money Mules
            </span>
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-lg text-[#a0a0a0] sm:text-xl">
            Machine learning system detecting fraudulent intermediary accounts in
            Indian banking. 125 engineered features. 12 behavioral patterns. One
            ensemble model.
          </p>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <a
              href="#results"
              className="inline-flex h-12 items-center gap-2 rounded-full bg-accent px-8 text-sm font-semibold text-surface transition-all hover:scale-[1.02] hover:shadow-[0_0_30px_rgba(0,212,170,0.3)] active:scale-[0.98]"
            >
              See Results
              <svg
                viewBox="0 0 16 16"
                className="h-3.5 w-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M1 8h14M8 1l7 7-7 7" />
              </svg>
            </a>
            <a
              href="https://github.com/divyamohan1993/nfpc-mule-detection"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-12 items-center gap-2 rounded-full border border-border px-8 text-sm font-medium text-[#a0a0a0] transition-all hover:border-[#444] hover:text-[#f0f0f0]"
            >
              <svg viewBox="0 0 16 16" fill="currentColor" className="h-4 w-4">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
              GitHub
            </a>
          </div>
        </div>

        <div className="absolute bottom-8 left-1/2 flex -translate-x-1/2 flex-col items-center gap-2 text-[#444]">
          <span className="text-[10px] uppercase tracking-[0.2em]">Scroll</span>
          <div className="h-8 w-[1px] bg-gradient-to-b from-[#444] to-transparent" />
        </div>
      </section>

      {/* Results */}
      <section id="results" className="relative px-6 py-32">
        <div className="mx-auto max-w-7xl">
          <div className="mb-4 text-xs font-mono uppercase tracking-[0.2em] text-accent">
            Performance
          </div>
          <h2 className="text-4xl font-black tracking-tight sm:text-5xl">
            Key Results
          </h2>
          <p className="mt-4 max-w-xl text-[#a0a0a0]">
            Ensemble of LightGBM and XGBoost trained on 24,023 accounts with
            extreme class imbalance (1:90 mule-to-legitimate ratio).
          </p>

          <div className="mt-12 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard value="0.985" label="Ensemble AUC-ROC" sub="Out-of-fold validation" />
            <StatCard value="125" label="Engineered Features" sub="13 categories" />
            <StatCard value="12/12" label="Mule Patterns Found" sub="All known patterns validated" />
            <StatCard value="1:90" label="Class Imbalance" sub="263 mules / 23,760 legitimate" />
          </div>

          <div className="mt-16">
            <h3 className="mb-6 text-xl font-bold">Model Comparison</h3>
            <div className="overflow-x-auto rounded-xl border border-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border bg-surface-raised">
                    <th className="px-6 py-4 text-left font-medium text-[#a0a0a0]">Model</th>
                    <th className="px-6 py-4 text-right font-medium text-[#a0a0a0]">OOF AUC-ROC</th>
                    <th className="px-6 py-4 text-right font-medium text-[#a0a0a0]">Mean Fold AUC</th>
                    <th className="px-6 py-4 text-right font-medium text-[#a0a0a0]">Std Dev</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="border-b border-border/50 transition-colors hover:bg-surface-overlay/50">
                    <td className="px-6 py-4 font-medium">LightGBM</td>
                    <td className="px-6 py-4 text-right font-mono">0.9834</td>
                    <td className="px-6 py-4 text-right font-mono">0.9831</td>
                    <td className="px-6 py-4 text-right font-mono text-[#666]">&plusmn;0.0058</td>
                  </tr>
                  <tr className="border-b border-border/50 transition-colors hover:bg-surface-overlay/50">
                    <td className="px-6 py-4 font-medium">XGBoost</td>
                    <td className="px-6 py-4 text-right font-mono">0.9789</td>
                    <td className="px-6 py-4 text-right font-mono">0.9785</td>
                    <td className="px-6 py-4 text-right font-mono text-[#666]">&plusmn;0.0067</td>
                  </tr>
                  <tr className="transition-colors hover:bg-surface-overlay/50">
                    <td className="px-6 py-4 font-bold text-accent">Ensemble</td>
                    <td className="px-6 py-4 text-right font-mono font-bold text-accent">0.9851</td>
                    <td className="px-6 py-4 text-right font-mono text-[#666]">-</td>
                    <td className="px-6 py-4 text-right font-mono text-[#666]">-</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* Key Signals */}
      <section className="border-t border-border/50 px-6 py-24">
        <div className="mx-auto max-w-7xl">
          <div className="mb-4 text-xs font-mono uppercase tracking-[0.2em] text-accent">
            Discrimination
          </div>
          <h2 className="text-4xl font-black tracking-tight sm:text-5xl">
            Red Flags
          </h2>
          <p className="mt-4 max-w-xl text-[#a0a0a0]">
            Behavioral fingerprints that distinguish mule accounts from legitimate
            ones. These signals, in combination, form the basis for detection.
          </p>

          <div className="mt-12 overflow-x-auto rounded-xl border border-border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border bg-surface-raised">
                  <th className="px-6 py-4 text-left font-medium text-[#a0a0a0]">Signal</th>
                  <th className="px-6 py-4 text-right font-medium text-[#a0a0a0]">Legitimate</th>
                  <th className="px-6 py-4 text-right font-medium text-[#a0a0a0]">Mule</th>
                  <th className="px-6 py-4 text-right font-medium text-[#a0a0a0]">Risk Multiplier</th>
                </tr>
              </thead>
              <tbody>
                {KEY_SIGNALS.map((s) => (
                  <tr
                    key={s.signal}
                    className="border-b border-border/50 transition-colors hover:bg-surface-overlay/50"
                  >
                    <td className="px-6 py-4 font-medium">{s.signal}</td>
                    <td className="px-6 py-4 text-right font-mono text-[#a0a0a0]">{s.legit}</td>
                    <td className="px-6 py-4 text-right font-mono text-[#ff4444]">{s.mule}</td>
                    <td className="px-6 py-4 text-right">
                      <span className="inline-flex items-center rounded-full bg-accent/10 px-2.5 py-0.5 text-xs font-bold text-accent">
                        {s.mult}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Feature Importance */}
      <section id="features" className="border-t border-border/50 px-6 py-24">
        <div className="mx-auto max-w-7xl">
          <div className="mb-4 text-xs font-mono uppercase tracking-[0.2em] text-accent">
            Feature Engineering
          </div>
          <h2 className="text-4xl font-black tracking-tight sm:text-5xl">
            Top 10 Features
          </h2>
          <p className="mt-4 max-w-xl text-[#a0a0a0]">
            125 features across 13 categories. These are the top 10 by LightGBM
            split-based importance.
          </p>

          <div className="mt-12 grid gap-6 lg:grid-cols-2">
            <div className="space-y-4">
              {FEATURE_DATA.slice(0, 5).map((f, i) => (
                <FeatureBar key={f.name} name={f.name} label={f.label} pct={f.pct} rank={i + 1} />
              ))}
            </div>
            <div className="space-y-4">
              {FEATURE_DATA.slice(5, 10).map((f, i) => (
                <FeatureBar key={f.name} name={f.name} label={f.label} pct={f.pct} rank={i + 6} />
              ))}
            </div>
          </div>

          <div className="mt-12 grid gap-4 sm:grid-cols-3 lg:grid-cols-5">
            {[
              { cat: "Transaction Aggregation", n: 7 },
              { cat: "Structuring Detection", n: 7 },
              { cat: "Velocity & Burstiness", n: 10 },
              { cat: "Channel Usage", n: 12 },
              { cat: "Graph & Network", n: 8 },
              { cat: "Temporal", n: 4 },
              { cat: "MCC-Based", n: 6 },
              { cat: "Unsupervised/Anomaly", n: 18 },
              { cat: "Balance", n: 8 },
              { cat: "Demographics", n: 37 },
            ].map((c) => (
              <div
                key={c.cat}
                className="rounded-xl border border-border bg-surface-raised p-4 text-center transition-colors hover:border-accent/20"
              >
                <div className="text-2xl font-black text-accent">{c.n}</div>
                <div className="mt-1 text-xs text-[#a0a0a0]">{c.cat}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Patterns */}
      <section id="patterns" className="border-t border-border/50 px-6 py-24">
        <div className="mx-auto max-w-7xl">
          <div className="mb-4 text-xs font-mono uppercase tracking-[0.2em] text-accent">
            Behavioral Analysis
          </div>
          <h2 className="text-4xl font-black tracking-tight sm:text-5xl">
            12 Mule Patterns
          </h2>
          <p className="mt-4 max-w-xl text-[#a0a0a0]">
            All 12 known mule behavior patterns from the RBIH challenge
            specification were identified and validated with statistical evidence.
          </p>

          <div className="mt-12 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {PATTERNS.map((p) => (
              <div
                key={p.id}
                className="group rounded-xl border border-border bg-surface-raised p-6 transition-all hover:border-accent/30 hover:shadow-[0_0_20px_rgba(0,212,170,0.05)]"
              >
                <div className="mb-3 flex items-center gap-3">
                  <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent/10 text-accent transition-colors group-hover:bg-accent/20">
                    {p.icon}
                  </div>
                  <span className="text-xs font-mono text-[#666]">#{p.id}</span>
                </div>
                <h3 className="font-bold">{p.name}</h3>
                <p className="mt-1 text-sm leading-relaxed text-[#a0a0a0]">
                  {p.desc}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Visualizations */}
      <section id="visualizations" className="border-t border-border/50 px-6 py-24">
        <div className="mx-auto max-w-7xl">
          <div className="mb-4 text-xs font-mono uppercase tracking-[0.2em] text-accent">
            Exploratory Data Analysis
          </div>
          <h2 className="text-4xl font-black tracking-tight sm:text-5xl">
            25 Visualizations
          </h2>
          <p className="mt-4 max-w-xl text-[#a0a0a0]">
            Comprehensive EDA covering class distribution, transaction patterns,
            network topology, model evaluation, and more.
          </p>

          {VIZ_SECTIONS.map((section) => (
            <div key={section.title} className="mt-16">
              <h3 className="mb-6 text-lg font-bold text-[#a0a0a0]">
                {section.title}
              </h3>
              <div className="grid gap-4 sm:grid-cols-2">
                {section.plots.map((plot) => (
                  <div
                    key={plot.file}
                    className="group overflow-hidden rounded-xl border border-border bg-surface-raised transition-all hover:border-accent/20"
                  >
                    <div className="relative aspect-[4/3] overflow-hidden">
                      <Image
                        src={`/plots/${plot.file}`}
                        alt={plot.label}
                        fill
                        className="object-contain p-2 transition-transform group-hover:scale-[1.02]"
                        sizes="(max-width: 768px) 100vw, 50vw"
                        loading="lazy"
                      />
                    </div>
                    <div className="border-t border-border/50 px-4 py-3">
                      <span className="text-sm font-medium text-[#a0a0a0]">
                        {plot.label}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}

          <div className="mt-16">
            <h3 className="mb-6 text-lg font-bold text-[#a0a0a0]">
              Cost-Sensitive Deployment
            </h3>
            <div className="max-w-2xl overflow-hidden rounded-xl border border-border bg-surface-raised">
              <div className="relative aspect-[4/3] overflow-hidden">
                <Image
                  src="/plots/25_cost_sensitive_matrix.png"
                  alt="Cost-Sensitive Confusion Matrix"
                  fill
                  className="object-contain p-2"
                  sizes="(max-width: 768px) 100vw, 50vw"
                  loading="lazy"
                />
              </div>
              <div className="border-t border-border/50 px-4 py-3">
                <span className="text-sm font-medium text-[#a0a0a0]">
                  Cost-Sensitive Confusion Matrix
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Technical Approach */}
      <section className="border-t border-border/50 px-6 py-24">
        <div className="mx-auto max-w-7xl">
          <div className="mb-4 text-xs font-mono uppercase tracking-[0.2em] text-accent">
            Methodology
          </div>
          <h2 className="text-4xl font-black tracking-tight sm:text-5xl">
            How It Works
          </h2>

          <div className="mt-12 grid gap-8 lg:grid-cols-3">
            {[
              {
                step: "01",
                title: "Data Ingestion",
                desc: "40K accounts, 40K customers, 7.4M transactions spanning July 2020 - June 2025. Full integrity verification with 100% join coverage.",
              },
              {
                step: "02",
                title: "Feature Engineering",
                desc: "125 features across 13 categories: transaction aggregation, structuring detection, velocity, pass-through, graph/network, channel, temporal, MCC, anomaly scores.",
              },
              {
                step: "03",
                title: "Model Training",
                desc: "LightGBM + XGBoost ensemble with 5-fold stratified CV. Scale_pos_weight=90.3 handles extreme imbalance. Early stopping on validation AUC.",
              },
            ].map((s) => (
              <div key={s.step} className="relative pl-16">
                <div className="absolute left-0 top-0 flex h-12 w-12 items-center justify-center rounded-xl border border-accent/30 bg-accent/5 text-lg font-black text-accent">
                  {s.step}
                </div>
                <h3 className="text-lg font-bold">{s.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-[#a0a0a0]">
                  {s.desc}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-16 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { label: "Statistical Methods", items: ["Mann-Whitney U", "Kolmogorov-Smirnov", "Chi-square / Cramer's V", "Bonferroni correction"] },
              { label: "ML Models", items: ["LightGBM (GBDT)", "XGBoost", "50/50 Ensemble", "SHAP TreeExplainer"] },
              { label: "Unsupervised", items: ["Isolation Forest", "PCA Reconstruction", "K-Means Distance", "PageRank / Community"] },
              { label: "Tech Stack", items: ["Python 3.10+", "Pandas / NumPy", "scikit-learn", "Matplotlib / Seaborn"] },
            ].map((g) => (
              <div key={g.label} className="rounded-xl border border-border bg-surface-raised p-6">
                <h4 className="mb-3 text-sm font-bold text-accent">{g.label}</h4>
                <ul className="space-y-2">
                  {g.items.map((item) => (
                    <li key={item} className="flex items-center gap-2 text-sm text-[#a0a0a0]">
                      <span className="h-1 w-1 rounded-full bg-accent/50" />
                      {item}
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border/50 px-6 py-16">
        <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-6 sm:flex-row">
          <div>
            <div className="text-sm font-semibold">NFPC Mule Account Detection</div>
            <div className="mt-1 text-xs text-[#666]">
              Divya Mohan &amp; Kumkum Thakur &middot; Team dmj.one &middot; RBIH x IIT Delhi TRYST 2025
            </div>
          </div>
          <div className="flex items-center gap-6 text-xs text-[#666]">
            <a
              href="https://github.com/divyamohan1993/nfpc-mule-detection"
              target="_blank"
              rel="noopener noreferrer"
              className="transition-colors hover:text-accent"
            >
              GitHub
            </a>
            <Link href="/pitch" className="transition-colors hover:text-accent">
              Pitch Deck
            </Link>
            <a
              href="https://dmj.one"
              target="_blank"
              rel="noopener noreferrer"
              className="transition-colors hover:text-accent"
            >
              dmj.one
            </a>
            <span>MIT License</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
