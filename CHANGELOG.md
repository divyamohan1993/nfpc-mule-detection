# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-03-05

### Changed
- Full responsive overhaul: mobile (320px) to 8K projector support
- Lightbox: mobile-safe margins, 48px touch targets, swipe gesture support
- Nav links collapse on mobile, show progressively at sm/md breakpoints
- Table cells use tighter padding on mobile, scale up on sm+
- Section padding reduced on mobile (px-4 py-16) for better space usage
- Pitch deck: inline maxWidth replaced with responsive Tailwind classes
- Pipeline boxes use 2-col grid on mobile, flex row on sm+
- Feature category grid starts at 2-col on mobile
- Feature bar code labels hidden on mobile for cleaner layout
- Report HTML: responsive padding, viewport meta, images/tables constrained
- Headings scale: text-3xl → sm:text-4xl → md:text-5xl

## [1.1.0] - 2026-03-05

### Added
- Web showcase built with Next.js + Tailwind CSS
- 10-slide pitch deck at /pitch with keyboard/touch/swipe navigation
- All 25 EDA visualizations displayed in categorized gallery
- Feature importance bars, red flags table, mule pattern cards
- SEO meta tags, OG tags, security headers via Vercel
- Static export for Vercel deployment

## [1.0.0] - 2025-03-15

### Added
- Full ML pipeline: EDA + feature engineering + model training + predictions
- LightGBM + XGBoost ensemble achieving 0.985 AUC-ROC
- 125 engineered features across 13 categories
- 25 analytical visualizations
- Comprehensive EDA report (Markdown + HTML + PDF)
- 12/12 mule behavior patterns validated with statistical evidence
- Predictions for 15,848 test accounts
