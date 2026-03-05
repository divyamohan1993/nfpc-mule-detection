import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Mule Account Detection - Pitch | dmj.one",
  description: "0.985 AUC-ROC detecting money mule accounts in Indian banking.",
  openGraph: {
    title: "Mule Account Detection - Pitch",
    description: "0.985 AUC-ROC detecting money mule accounts in Indian banking.",
    url: "https://nfpc.dmj.one/pitch",
    type: "website",
  },
};

const SLIDES = [
  // 1. Title
  {
    id: "s1",
    label: "Title",
    content: (
      <>
        <div className="mb-8 inline-flex items-center gap-2 rounded-full border border-[#222] bg-[#141414] px-4 py-1.5 text-xs text-[#a0a0a0]">
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#00d4aa]" />
          National Fraud Prevention Challenge
        </div>
        <h1 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          Catching{" "}
          <span className="bg-gradient-to-r from-[#00d4aa] to-[#34d399] bg-clip-text text-transparent">
            Money Mules
          </span>
        </h1>
        <p className="mt-6 max-w-[65ch] text-[clamp(1.1rem,2vw,1.5rem)] leading-[1.6] text-[rgba(255,255,255,0.85)]">
          ML-powered detection of fraudulent intermediary accounts in Indian banking
        </p>
        <div className="mt-8 text-[0.9rem] text-[rgba(255,255,255,0.4)]">
          Divya Mohan &amp; Kumkum Thakur &middot; Team dmj.one &middot;{" "}
          <a href="https://github.com/divyamohan1993/nfpc-mule-detection" target="_blank" rel="noopener noreferrer" className="underline decoration-[rgba(255,255,255,0.3)] underline-offset-2 transition-colors hover:text-[#00d4aa]">
            github.com/divyamohan1993/nfpc-mule-detection
          </a>
        </div>
      </>
    ),
  },
  // 2. Problem
  {
    id: "s2",
    label: "Problem",
    content: (
      <>
        <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          Money Mules Are{" "}
          <span className="text-[#00d4aa]">Invisible</span>
        </h2>
        <div className="my-4 text-[clamp(3rem,8vw,7rem)] font-[900] leading-none text-[#00d4aa]">
          200 Lakh Cr
        </div>
        <p className="max-w-[65ch] text-[clamp(1.1rem,2vw,1.5rem)] leading-[1.6] text-[rgba(255,255,255,0.85)]">
          processed through Indian payment systems in FY24 alone. Every undetected mule
          account is a breach in UPI, NEFT, RTGS integrity.
        </p>
        <p className="mt-2 max-w-[65ch] text-[clamp(1.1rem,2vw,1.5rem)] leading-[1.6] text-[rgba(255,255,255,0.85)]">
          Banks flag accounts <em>after</em> fraud. Rule-based systems catch only the obvious.
        </p>
      </>
    ),
  },
  // 3. Value Proposition
  {
    id: "s3",
    label: "Value",
    content: (
      <>
        <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          Detect Before the <span className="text-[#00d4aa]">Damage</span>
        </h2>
        <div className="my-4 text-[clamp(3rem,8vw,7rem)] font-[900] leading-none text-[#00d4aa]">
          0.985 AUC
        </div>
        <p className="max-w-[65ch] text-[clamp(1.1rem,2vw,1.5rem)] leading-[1.6] text-[rgba(255,255,255,0.85)]">
          Our ensemble model detects mule accounts with 0.985 AUC-ROC —
          even under extreme 1:90 class imbalance. 12 of 12 known behavioral
          patterns validated with statistical evidence.
        </p>
      </>
    ),
  },
  // 4. Underlying Magic
  {
    id: "s4",
    label: "How it works",
    content: (
      <>
        <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          The <span className="text-[#00d4aa]">Pipeline</span>
        </h2>
        <div className="mt-8 flex flex-col gap-4 sm:flex-row sm:items-center sm:gap-6">
          {[
            { label: "7.4M Txns", sub: "Ingested" },
            { label: "125 Features", sub: "Engineered" },
            { label: "Ensemble", sub: "LGB + XGB" },
            { label: "0.985", sub: "AUC-ROC" },
          ].map((s, i) => (
            <div key={s.label} className="flex items-center gap-4">
              <div className="flex h-16 w-16 shrink-0 flex-col items-center justify-center rounded-xl border border-[#00d4aa30] bg-[#00d4aa08]">
                <span className="text-lg font-[900] text-[#00d4aa]">{s.label}</span>
                <span className="text-[10px] text-[rgba(255,255,255,0.5)]">{s.sub}</span>
              </div>
              {i < 3 && (
                <svg viewBox="0 0 24 24" className="hidden h-4 w-4 text-[#333] sm:block" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M5 12h14M12 5l7 7-7 7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              )}
            </div>
          ))}
        </div>
        <p className="mt-8 max-w-[65ch] text-[clamp(1.1rem,2vw,1.5rem)] leading-[1.6] text-[rgba(255,255,255,0.85)]">
          13 feature categories including transaction aggregation, structuring detection,
          velocity, graph/network metrics, and anomaly scores.
        </p>
      </>
    ),
  },
  // 5. Business Model
  {
    id: "s5",
    label: "Business Model",
    content: (
      <>
        <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          Who <span className="text-[#00d4aa]">Benefits</span>
        </h2>
        <div className="mt-8 grid gap-4 sm:grid-cols-2" style={{ maxWidth: "600px" }}>
          {[
            { who: "Banks & NBFCs", how: "Pre-flag suspicious accounts before losses" },
            { who: "Regulators (RBI/FIU-IND)", how: "Strengthen STR framework, PMLA compliance" },
            { who: "Payment Processors", how: "Risk scoring for UPI/NEFT/RTGS flows" },
            { who: "Insurance/Fintech", how: "KYC enhancement, onboarding risk" },
          ].map((b) => (
            <div key={b.who} className="rounded-lg border border-[#222] bg-[#141414] p-4">
              <div className="text-sm font-bold text-[#00d4aa]">{b.who}</div>
              <div className="mt-1 text-xs text-[rgba(255,255,255,0.6)]">{b.how}</div>
            </div>
          ))}
        </div>
        <div className="mt-8 text-[0.9rem] text-[rgba(255,255,255,0.4)]">
          API-based risk scoring — per-account inference, batch or real-time.
        </div>
      </>
    ),
  },
  // 6. Go-to-Market
  {
    id: "s6",
    label: "GTM",
    content: (
      <>
        <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          Reaching <span className="text-[#00d4aa]">Users</span>
        </h2>
        <div className="mt-8 space-y-4" style={{ maxWidth: "500px" }}>
          {[
            "RBIH network — direct access to Indian banking ecosystem",
            "Open-source model + paper — academic validation, community trust",
            "Regulatory sandbox — pilot with 2-3 banks under RBI oversight",
            "Integration SDKs — drop-in for existing AML/KYC platforms",
          ].map((item, i) => (
            <div key={i} className="flex items-start gap-3">
              <span className="mt-1 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-[#00d4aa15] text-[10px] font-bold text-[#00d4aa]">
                {i + 1}
              </span>
              <span className="text-sm text-[rgba(255,255,255,0.8)]">{item}</span>
            </div>
          ))}
        </div>
      </>
    ),
  },
  // 7. Competitive Analysis
  {
    id: "s7",
    label: "Competition",
    content: (
      <>
        <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          <span className="text-[#00d4aa]">Landscape</span>
        </h2>
        <div className="mt-8 overflow-x-auto">
          <table className="w-full text-sm" style={{ maxWidth: "600px" }}>
            <thead>
              <tr className="border-b border-[#222]">
                <th className="py-2 pr-4 text-left text-xs text-[rgba(255,255,255,0.5)]">Approach</th>
                <th className="px-4 py-2 text-center text-xs text-[rgba(255,255,255,0.5)]">Accuracy</th>
                <th className="px-4 py-2 text-center text-xs text-[rgba(255,255,255,0.5)]">Patterns</th>
                <th className="py-2 pl-4 text-center text-xs text-[rgba(255,255,255,0.5)]">Explainable</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-[#222]/50">
                <td className="py-3 pr-4 text-[rgba(255,255,255,0.6)]">Rule-based (threshold)</td>
                <td className="px-4 py-3 text-center text-[rgba(255,255,255,0.4)]">Low</td>
                <td className="px-4 py-3 text-center text-[rgba(255,255,255,0.4)]">2-3</td>
                <td className="py-3 pl-4 text-center text-[#00d4aa]">Yes</td>
              </tr>
              <tr className="border-b border-[#222]/50">
                <td className="py-3 pr-4 text-[rgba(255,255,255,0.6)]">Single ML model</td>
                <td className="px-4 py-3 text-center text-[rgba(255,255,255,0.4)]">~0.95</td>
                <td className="px-4 py-3 text-center text-[rgba(255,255,255,0.4)]">Implicit</td>
                <td className="py-3 pl-4 text-center text-[rgba(255,255,255,0.4)]">Partial</td>
              </tr>
              <tr className="bg-[#00d4aa08]">
                <td className="py-3 pr-4 font-bold text-[#00d4aa]">Ours (ensemble + SHAP)</td>
                <td className="px-4 py-3 text-center font-bold text-[#00d4aa]">0.985</td>
                <td className="px-4 py-3 text-center font-bold text-[#00d4aa]">12/12</td>
                <td className="py-3 pl-4 text-center font-bold text-[#00d4aa]">Full SHAP</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="mt-8 text-[0.9rem] text-[rgba(255,255,255,0.4)]">
          Structural advantage: interpretability + accuracy. Not a black box.
        </div>
      </>
    ),
  },
  // 8. Team
  {
    id: "s8",
    label: "Team",
    content: (
      <>
        <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          Built By
        </h2>
        <div className="mt-8 space-y-6" style={{ maxWidth: "500px" }}>
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-[#00d4aa15] text-lg font-bold text-[#00d4aa]">
              DM
            </div>
            <div>
              <div className="font-bold">Divya Mohan</div>
              <div className="text-sm text-[rgba(255,255,255,0.5)]">
                Vision, Architecture, Feature Engineering, Analysis
              </div>
              <a href="https://github.com/divyamohan1993" target="_blank" rel="noopener noreferrer" className="mt-1 inline-block text-xs text-[#00d4aa] hover:underline">
                github.com/divyamohan1993
              </a>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-[#00d4aa15] text-lg font-bold text-[#00d4aa]">
              KT
            </div>
            <div>
              <div className="font-bold">Kumkum Thakur</div>
              <div className="text-sm text-[rgba(255,255,255,0.5)]">
                Data Science Guidance, Domain Expertise, Analysis
              </div>
            </div>
          </div>
          <div className="flex items-start gap-4">
            <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-full bg-[#00d4aa08] text-lg font-bold text-[#00d4aa]/60">
              CO
            </div>
            <div>
              <div className="font-bold text-[rgba(255,255,255,0.7)]">Claude Opus</div>
              <div className="text-sm text-[rgba(255,255,255,0.5)]">
                AI Co-Architect &amp; Build Partner (Anthropic)
              </div>
            </div>
          </div>
        </div>
        <div className="mt-8 text-[0.9rem] text-[rgba(255,255,255,0.4)]">
          Team dmj.one &middot; RBIH x IIT Delhi TRYST 2025
        </div>
      </>
    ),
  },
  // 9. Metrics
  {
    id: "s9",
    label: "Metrics",
    content: (
      <>
        <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          Real <span className="text-[#00d4aa]">Numbers</span>
        </h2>
        <div className="mt-8 grid gap-4 sm:grid-cols-3" style={{ maxWidth: "600px" }}>
          {[
            { val: "0.985", label: "AUC-ROC" },
            { val: "125", label: "Features" },
            { val: "12/12", label: "Patterns" },
            { val: "7.4M", label: "Transactions" },
            { val: "47", label: "Statistical Tables" },
            { val: "25", label: "Visualizations" },
          ].map((m) => (
            <div key={m.label} className="rounded-lg border border-[#222] bg-[#141414] p-4 text-center">
              <div className="text-2xl font-[900] text-[#00d4aa]">{m.val}</div>
              <div className="mt-1 text-[11px] text-[rgba(255,255,255,0.5)]">{m.label}</div>
            </div>
          ))}
        </div>
        <div className="mt-8 text-[0.9rem] text-[rgba(255,255,255,0.4)]">
          5-fold stratified cross-validation. All metrics out-of-fold. No leakage.
        </div>
      </>
    ),
  },
  // 10. Status & Roadmap
  {
    id: "s10",
    label: "Roadmap",
    content: (
      <>
        <h2 className="text-[clamp(2.5rem,5vw,4.5rem)] font-[800] leading-[1.05] tracking-[-0.02em]">
          What&apos;s <span className="text-[#00d4aa]">Next</span>
        </h2>
        <div className="mt-8 space-y-4" style={{ maxWidth: "500px" }}>
          {[
            { done: true, text: "EDA + 125-feature pipeline + ensemble model" },
            { done: true, text: "25 visualizations + full statistical report" },
            { done: true, text: "Showcase deployed at nfpc.dmj.one" },
            { done: false, text: "Real-time feature computation (<200ms)" },
            { done: false, text: "Graph neural network for network-level detection" },
            { done: false, text: "Regulatory sandbox pilot with partner bank" },
          ].map((item, i) => (
            <div key={i} className="flex items-start gap-3">
              <span
                className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full text-[10px] ${
                  item.done
                    ? "bg-[#00d4aa] font-bold text-[#0a0a0a]"
                    : "border border-[#333] text-[#666]"
                }`}
              >
                {item.done ? "\u2713" : i - 2}
              </span>
              <span className={`text-sm ${item.done ? "text-[rgba(255,255,255,0.8)]" : "text-[rgba(255,255,255,0.5)]"}`}>
                {item.text}
              </span>
            </div>
          ))}
        </div>
        <div className="mt-10 flex flex-wrap gap-4">
          <a
            href="https://nfpc.dmj.one"
            className="inline-flex h-10 items-center gap-2 rounded-full bg-[#00d4aa] px-6 text-sm font-semibold text-[#0a0a0a] transition-all hover:scale-[1.02]"
          >
            Live Site
          </a>
          <a
            href="https://github.com/divyamohan1993/nfpc-mule-detection"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex h-10 items-center gap-2 rounded-full border border-[#333] px-6 text-sm text-[rgba(255,255,255,0.7)] transition-colors hover:border-[#00d4aa30] hover:text-[#00d4aa]"
          >
            GitHub
          </a>
        </div>
      </>
    ),
  },
];

function PitchNav() {
  return (
    <script
      // Static hardcoded navigation logic - no user input, safe from XSS
      dangerouslySetInnerHTML={{
        __html: [
          "(function(){",
          "var s=document.querySelectorAll('.slide'),p=document.querySelector('.progress-bar'),",
          "c=document.querySelector('.counter'),t=s.length,n=0;",
          "function go(i){if(i<0||i>=t||i===n)return;s[n].classList.remove('active');n=i;",
          "s[n].classList.add('active');p.style.width=((n+1)/t*100)+'%';",
          "c.textContent=(n+1)+' / '+t;history.replaceState(null,'','#s'+(n+1));}",
          "document.addEventListener('keydown',function(e){",
          "if(e.key==='ArrowRight'||e.key===' '){e.preventDefault();go(n+1);}",
          "if(e.key==='ArrowLeft'){e.preventDefault();go(n-1);}",
          "if(e.key==='Home'){e.preventDefault();go(0);}",
          "if(e.key==='End'){e.preventDefault();go(t-1);}",
          "if(e.key==='f'||e.key==='F'){",
          "if(!document.fullscreenElement)document.documentElement.requestFullscreen&&document.documentElement.requestFullscreen();",
          "else document.exitFullscreen&&document.exitFullscreen();}});",
          "document.addEventListener('click',function(e){",
          "if(e.target.closest('a,button'))return;",
          "if(e.clientX>window.innerWidth/2)go(n+1);else go(n-1);});",
          "var tx=0;document.addEventListener('touchstart',function(e){tx=e.touches[0].clientX;},{passive:true});",
          "document.addEventListener('touchend',function(e){",
          "var dx=e.changedTouches[0].clientX-tx;if(Math.abs(dx)>50)go(n+(dx<0?1:-1));});",
          "var h=parseInt((location.hash||'').replace('#s',''))-1;",
          "if(h>0&&h<t)go(h);",
          "})();",
        ].join(""),
      }}
    />
  );
}

export default function PitchPage() {
  return (
    <div className="relative min-h-screen bg-[#0a0a0a] text-[#f0f0f0]" style={{ fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif" }}>
      <style
        // Static CSS for pitch deck layout - no user input
        dangerouslySetInnerHTML={{
          __html: [
            "body{overflow:hidden;margin:0}",
            ".deck{width:100vw;height:100vh;position:relative}",
            ".slide{width:100%;height:100%;position:absolute;top:0;left:0;display:flex;flex-direction:column;justify-content:center;padding:8vh 10vw;opacity:0;pointer-events:none;transition:opacity .3s ease}",
            ".slide.active{opacity:1;pointer-events:auto}",
            ".slide.active>*{animation:pfu .25s ease-out both}",
            ".slide.active>*:nth-child(1){animation-delay:0ms}",
            ".slide.active>*:nth-child(2){animation-delay:80ms}",
            ".slide.active>*:nth-child(3){animation-delay:160ms}",
            ".slide.active>*:nth-child(4){animation-delay:240ms}",
            ".slide.active>*:nth-child(5){animation-delay:320ms}",
            "@keyframes pfu{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}",
            "@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}}",
            ".progress-bar{position:fixed;bottom:0;left:0;height:2px;background:#00d4aa;transition:width .3s ease;z-index:100}",
            ".counter{position:fixed;bottom:1rem;right:1.5rem;font-size:.75rem;color:rgba(255,255,255,.35);font-variant-numeric:tabular-nums;z-index:100}",
            "@media(max-width:768px){.slide{padding:6vh 6vw}}",
          ].join(""),
        }}
      />
      <a
        href="/"
        className="fixed left-6 top-5 z-[100] text-xs text-[rgba(255,255,255,0.35)] no-underline transition-colors hover:text-[#00d4aa]"
      >
        &larr; Back
      </a>
      <div className="deck" role="main" aria-label="Pitch presentation">
        {SLIDES.map((slide, i) => (
          <section
            key={slide.id}
            id={slide.id}
            className={`slide${i === 0 ? " active" : ""}`}
            role="region"
            aria-label={`Slide ${i + 1}: ${slide.label}`}
          >
            {slide.content}
          </section>
        ))}
      </div>
      <div className="progress-bar" role="progressbar" aria-valuenow={1} aria-valuemin={1} aria-valuemax={10} style={{ width: "10%" }} />
      <div className="counter" aria-hidden="true">1 / 10</div>
      <PitchNav />
    </div>
  );
}
