# Financial Distress Early Warning System (FDEWS)

> **End-to-end ML pipeline for corporate insolvency prediction using 60GB+ of SEC EDGAR filings.**  
> Medallion Architecture (Bronze → Silver → Gold) | Multimodal: Financial Ratios + NLP Sentiment | XGBoost + SHAP Explainability

[![HuggingFace Demo](https://img.shields.io/badge/🤗%20Live%20Demo-sec--risk--dashboard-orange)](https://huggingface.co/spaces/Jenababu/sec-risk-dashboard)
[![Dataset Scale](https://img.shields.io/badge/Dataset-60GB%20SEC%20EDGAR-blue)]()
[![Model](https://img.shields.io/badge/Model-XGBoost%20%7C%2096.2%25%20Confidence-green)]()
[![Pipeline](https://img.shields.io/badge/Pipeline-4.9M%20Processed%20Rows-yellow)]()

---

## Overview

FDEWS processes raw SEC 10-K/10-Q filings at scale to predict corporate financial distress 12 months ahead. It fuses 17 engineered financial ratios with FinBERT-derived MD&A sentiment signals into a Multimodal Gold Dataset, then runs XGBoost inference with full SHAP explainability — returning a crash probability score and a ranked feature breakdown for every company.

**Production winner: XGBoost** (96.2% confidence, balanced precision/recall) over LSTM (high recall, low precision — unsuitable for real-world credit decisions).

---

## System Architecture

### 3-Tier Application Design

![3-Tier System Architecture](docs/images/3_tier_system_architecture.png)

The system is organized into three tiers:

- **Presentation Tier** — Company selector, Crash Probability Scorecard (gauge), SHAP Waterfall visualization, Sentiment Analysis indicators.
- **Application Tier** — Feature Engineering Engine (17 features), Feature Order Lock Validator, StandardScaler Synchronization, XGBoost Inference Engine, SHAP Explainability Engine, Joblib Model Loader. LSTM is present as an experimental benchmarking component only.
- **Data Tier** — SEC EDGAR raw filings → Quarterly/Monthly batches → Polars Lazy Execution → Parquet Storage → 4.9M-row Gold dataset → Model hosting.

---

## SEC Multimodal Data Refinery Pipeline

![SEC Multimodal Data Refinery Pipeline](docs/images/SEC_MULTIMODAL_DATA_REFINERY_PIPELINE_V2.png)

The pipeline follows the **Medallion Architecture**:

| Layer | What Happens | Key Output |
|---|---|---|
| **Bronze** | Raw SEC EDGAR ingestion (HTML, JSON, TSV) — 60GB, quarterly + monthly batches | `num.tsv`, `sub.tsv`, HTML filings |
| **Silver** | Polars Lazy Engine, schema harmonization, float32 downcasting, Parquet storage, pivoted financial tables | Clean columnar Parquet files |
| **Gold** | 17 engineered financial ratios, FinBERT sentiment signal, persistent distress flags, multimodal feature fusion | `multimodal_gold_dataset.parquet` — 4.9M rows |
| **Master/Deployment** | Latest filing per company, inference-ready subset, HuggingFace Parquet upload | 9,461 rows, `crash_label` target |

**Sentiment Signal Formula:**

```
Sentiment = Probability_Positive − Probability_Negative
```

Range: `-1.0` (extreme distress) → `+1.0` (extreme confidence). Derived from FinBERT Softmax output on MD&A sections.

---

## Feature Order Lock — Silent Prediction Drift Prevention

![Feature Order Lock](docs/images/FEATURE_ORDER_LOCK_V2.png)

Column mismatch between training and inference causes **silent prediction corruption** — the model runs without error but scores the wrong features. The Feature Order Lock enforces:

1. **Exact column order** — 17 features locked at extraction time.
2. **Column name validation** — names must match training schema exactly.
3. **Feature count check** — any deviation raises a hard error before the scaler is applied.
4. **Scaler synchronization** — `scaler.joblib` applies identical `mean_` and `scale_` parameters from training.
5. **DMatrix construction** — validated, scaled features passed to `model.json` for inference.

**The 17 locked features (in order):**

| # | Feature | # | Feature | # | Feature |
|---|---|---|---|---|---|
| 1 | current_ratio | 7 | roe | 13 | revenue_growth_rate |
| 2 | quick_ratio | 8 | debt_to_assets | 14 | sentiment_signal |
| 3 | cash_ratio | 9 | debt_to_equity | 15 | persistent_distress_flag |
| 4 | roa | 10 | asset_turnover | 16 | Assets |
| 5 | profit_margin | 11 | interest_coverage | 17 | Revenues |
| 6 | operating_margin | 12 | retained_earnings_ratio | | |

Metadata (`cik`, `company_name`, `adsh`, `ddate`, `crash_label`) is excluded from the feature vector.

---

## Modeling Showdown: LSTM vs XGBoost

![Model Comparison](docs/images/Model_Comparision_V2.png)

| Dimension | LSTM | XGBoost |
|---|---|---|
| **Type** | Temporal sequence model | Gradient boosting ensemble |
| **Strength** | Sequential distress pattern analysis | Nonlinear financial relationship detection |
| **Weakness** | High recall, low precision — false alarm machine | — |
| **Status** | Experimental benchmarking only | **Production model** |
| **Why LSTM failed** | Flags nearly all healthy companies as distressed; unusable for credit decisions | — |
| **Why XGBoost won** | Balanced precision/recall, captures threshold interactions (e.g., high debt + negative sentiment = distress), superior stability | ✅ |

**XGBoost Production Output Example:**
- Crash Probability: **82.6%** | Risk Level: HIGH | Confidence: 0.87 | Horizon: 12 months
- Top SHAP drivers: `Assets` (−0.51), `roa` (−0.38), `roe` (−0.29), `debt_to_equity` (+0.09)

---

## Live Dashboard

> **[🚀 Open Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/Jenababu/sec-risk-dashboard)**

![FDEWS Dashboard Screenshot](docs/images/dashboard_screenshot.png)

*Dashboard features: Company selector by CIK → Crash Probability Scorecard → SHAP Waterfall (per-feature risk attribution) → 7-day Sentiment Trend → Feature Impact Ranking.*

---

## Technical Stack

| Layer | Technology | Role |
|---|---|---|
| Data Processing | **Polars** (Lazy/Streaming) | Out-of-core 60GB ingestion, low RAM footprint |
| Storage | **Parquet** | Columnar, compressed, fast analytical reads |
| NLP | **FinBERT** (HuggingFace Transformers) | MD&A sentiment extraction + Softmax signal |
| HTML Parsing | **BeautifulSoup + Regex** | MD&A section extraction from SEC HTML filings |
| ML | **XGBoost** | Primary distress prediction model |
| Temporal ML | **LSTM** (PyTorch/Keras) | Experimental benchmarking only |
| Explainability | **SHAP** | SHapley Additive Explanations, waterfall plots |
| Validation | **Walk-Forward Temporal Split** | Prevents data leakage across economic cycles |
| Serialization | **Joblib** | Scaler + model artifact persistence |
| Interface | **Streamlit / HuggingFace Spaces** | Risk monitoring dashboard |

---

## Repository Structure

```
FDEWS/
├── bronze_layer/          # SEC EDGAR scrapers, raw TSV/HTML ingestion
├── silver_layer/          # Schema enforcement, Parquet partitioning, downcasting
├── gold_layer/            # Feature engineering, FinBERT sentiment, distress flags
├── modeling/              # XGBoost training, LSTM benchmarking, SHAP analysis
├── dashboard/             # Streamlit app, model artifacts (model.json, scaler.joblib)
├── data/                  # dataset_link.md → Google Drive access instructions
└── docs/
    └── images/            # Architecture diagrams (place PNGs here)
        ├── 3_tier_system_architecture.png
        ├── SEC_MULTIMODAL_DATA_REFINERY_PIPELINE_V2.png
        ├── FEATURE_ORDER_LOCK_V2.png
        ├── Model_Comparision_V2.png
        └── dashboard_screenshot.png
```

---

## Dataset Access

Raw (60GB) and processed data are hosted externally due to size.  
See [`data/dataset_link.md`](data/dataset_link.md) for Google Drive access instructions.

**Processed Gold dataset** (9,461 rows, inference-ready) is available on HuggingFace:  
`Jenababu/sec-risk-dashboard` → `multimodal_gold_dataset.parquet`

---

## Known Limitations

- LSTM is benchmarking-only; precision is too low for production credit decisions.
- Sentiment signal quality degrades on companies with boilerplate/templated MD&A language.
- Walk-forward split assumes stationarity of distress patterns across economic cycles — this holds for 2020–2024 but may need recalibration post-2025 rate environment.
- `persistent_distress_flag` uses a 2-quarter window; longer windows untested.
