# FDEWS — Functional Specification

**Version:** 1.0
**Status:** Final
**Last Updated:** July 2026
**Author:** Subrat Kumar Jena

---

## 1. Document Overview

This Functional Specification describes how the FDEWS system fulfills the requirements defined in the PRD. It documents actual implemented system behavior — module by module — verified directly against the project repository. Where a component is experimental or not yet part of the operational pipeline, it is explicitly labeled as such rather than presented as delivered functionality.

**Status Legend used throughout this document:**
- ✅ **Implemented** — operational, part of the production pipeline
- 🧪 **Experimental** — built and present in the repository, but not used in production inference
- 🔲 **Future Scope** — not yet implemented

## 2. Document Scope

This document covers **functional behavior only** — how the system operates, module by module, to deliver the capabilities already defined upstream. It does not restate why the product exists or what it must do; those are covered in separate documents within this documentation set:

- **Business Problem** — explains *why* the product exists (the market gap, current challenges, and business need)
- **PRD** — defines *what* the product must do (user-facing requirements and success criteria)
- **Functional Specification (this document)** — defines *how* the system fulfills those requirements at a module level

## 3. Referenced Documents

| Document | Relationship to This Document |
|---|---|
| `01_Business_Problem.md` | Establishes the business rationale this specification technically fulfills |
| `02_PRD.md` | Defines the product requirements traced to specific modules in Section 5 (Traceability Matrix) |

## 4. System Overview

FDEWS follows a three-tier architecture:

- **Data Tier** — SEC EDGAR raw filings → Bronze → Silver → Gold layers, processed via Polars Lazy Evaluation for out-of-core handling of large filing volumes
- **Application Tier** — Feature engineering, feature validation, model inference (XGBoost), explainability (SHAP), model artifact loading
- **Presentation Tier** — Streamlit dashboard for company-level risk review

This structure is consistent with the Medallion Architecture referenced in the Business Problem and PRD.

## 5. Traceability Matrix — PRD to Functional Specification

| PRD Requirement | Description | Fulfilled By |
|---|---|---|
| FR-01 | Product shall provide a distress risk assessment per covered company | Module 5 (Risk Prediction), Module 8 (Dashboard — Predictions page) |
| FR-02 | Product shall present key contributing factors in an understandable format | Module 6 (SHAP Explainability), Module 8 (Dashboard — SHAP Explainer page) |
| FR-03 | Product shall incorporate both quantitative and qualitative disclosure signals | Module 3 (Feature Engineering — 17 ratios + sentiment signal) |
| FR-04 | Product shall allow analysts to search/select a company | Module 8 (Dashboard — Predictions, Raw Data Explorer pages) |
| FR-05 | Product shall allow tracing a risk assessment to source disclosure | Module 1 (Bronze — raw filing retention), Module 8 (Raw Data Explorer page) |
| FR-06 | Product shall present risk assessments through an analyst-facing interface | Module 8 (Analyst Dashboard) |

## 6. Module Descriptions

---

### Module 1 — Data Ingestion (Bronze Layer)
**Status: ✅ Implemented**

| Field | Description |
|---|---|
| **Purpose** | Ingest raw SEC EDGAR filing data as the unmodified source of truth for downstream processing |
| **Inputs** | SEC EDGAR filings — HTML, JSON, and TSV formats, organized in quarterly and monthly batches |
| **Processing** | Raw files are collected and stored without transformation |
| **Outputs** | Raw filing files (`num.tsv`, `sub.tsv`, HTML filing documents) |
| **Business Rules** | None applied at this stage — data is preserved in original form |
| **Dependencies** | SEC EDGAR as external data source |

---

### Module 2 — Data Cleaning and Standardization (Silver Layer)
**Status: ✅ Implemented**

| Field | Description |
|---|---|
| **Purpose** | Transform raw filing data into a clean, standardized, analysis-ready format |
| **Inputs** | Raw Bronze layer files |
| **Processing** | Schema harmonization across filings, float32 downcasting to reduce memory footprint, pivoting of financial tables, conversion to columnar Parquet format using Polars Lazy Engine |
| **Outputs** | Clean, standardized Parquet files |
| **Business Rules** | Schema must conform to a harmonized structure before advancing to the Gold layer |
| **Dependencies** | Bronze layer output; Polars processing engine |

---

### Module 3 — Feature Engineering and Sentiment Extraction (Gold Layer)
**Status: ✅ Implemented**

| Field | Description |
|---|---|
| **Purpose** | Generate the finalized, model-ready feature set combining financial ratios and disclosure sentiment |
| **Inputs** | Clean Silver layer Parquet data; MD&A narrative text extracted from filing HTML |
| **Processing** | Computes 17 engineered financial ratio and structural features per company per filing period. Extracts MD&A text using HTML parsing, then applies a pretrained financial NLP sentiment model to produce a sentiment signal. |
| **Outputs** | A finalized multimodal Gold dataset combining structured and sentiment-derived features |
| **Business Rules** | The 17-feature set is fixed and locked; feature computation must match the schema used during model training |
| **Dependencies** | Silver layer output; financial NLP sentiment model |

**Implemented Feature Set (17 features, fixed order):**

| # | Feature | # | Feature | # | Feature |
|---|---|---|---|---|---|
| 1 | current_ratio | 7 | roe | 13 | revenue_growth_rate |
| 2 | quick_ratio | 8 | debt_to_assets | 14 | sentiment_signal |
| 3 | cash_ratio | 9 | debt_to_equity | 15 | persistent_distress_flag |
| 4 | roa | 10 | asset_turnover | 16 | Assets |
| 5 | profit_margin | 11 | interest_coverage | 17 | Revenues |
| 6 | operating_margin | 12 | retained_earnings_ratio | | |

**Sentiment Signal — Implemented Behavior:**
- Computed as: Positive Probability − Negative Probability, from a Softmax output over MD&A narrative text
- Range: -1.0 (extreme negative disclosure sentiment) to +1.0 (extreme positive disclosure sentiment)

**Known Coverage Limitation (Implemented, disclosed honestly):**
A sentiment signal has been successfully extracted for a subset of companies in the dataset; the remaining companies default to a neutral sentiment value in the absence of an extracted signal. This means sentiment-driven risk detection is currently strongest for the covered subset and comparatively limited for the remainder.

---

### Module 4 — Feature Validation ("Feature Order Lock")
**Status: ✅ Implemented**

| Field | Description |
|---|---|
| **Purpose** | Prevent silent prediction corruption caused by mismatched feature order or schema between training and inference |
| **Inputs** | Constructed feature vector for a company at inference time |
| **Processing** | Validates exact column order against the locked 17-feature training schema; validates column names match exactly; validates feature count; raises a hard error before scaling if any check fails; applies the training-time scaler parameters to the validated feature vector |
| **Outputs** | A validated, scaled feature vector ready for model inference — or a hard validation error if checks fail |
| **Business Rules** | Any deviation from the locked feature schema (order, naming, or count) blocks inference rather than allowing a silently corrupted prediction |
| **Dependencies** | Module 3 output; persisted scaler artifact from training |

**Missing Data Handling (Implemented Business Rule):**
If a company has a gap in filing history (a missed quarter), the affected ratio features are forward-filled from the most recently available filing rather than left missing. A short rolling window is used to track distress persistence across these gaps, allowing the feature set to remain complete even with incomplete filing history.

---

### Module 5 — Risk Prediction (XGBoost — Production Model)
**Status: ✅ Implemented**

| Field | Description |
|---|---|
| **Purpose** | Generate a financial distress probability for a company based on its validated feature vector |
| **Inputs** | Validated, scaled 17-feature vector (Module 4 output) |
| **Processing** | The trained XGBoost model performs inference and returns a distress probability |
| **Outputs** | A distress probability score, along with a risk classification derived from that score |
| **Business Rules** | The decision threshold used to classify risk level is adjustable rather than fixed, allowing the precision/recall trade-off to be tuned |
| **Dependencies** | Persisted model artifact; Module 4 validated feature vector |

---

### Module 6 — Explainability (SHAP)
**Status: ✅ Implemented**

| Field | Description |
|---|---|
| **Purpose** | Provide transparent, per-prediction reasoning behind each distress risk assessment |
| **Inputs** | Trained XGBoost model; a company's validated feature vector |
| **Processing** | Computes SHAP (SHapley Additive exPlanations) values to quantify each feature's contribution to the prediction |
| **Outputs** | Three explainability views: a global feature importance ranking, a beeswarm-style distribution view across companies, and a single-company waterfall breakdown of contributing factors |
| **Business Rules** | Every prediction surfaced to the dashboard is accompanied by its corresponding SHAP explanation — no unexplained score is shown |
| **Dependencies** | Module 5 output; SHAP library |

---

### Module 7 — Model Evaluation (LSTM Comparison)
**Status: 🧪 Experimental — not part of the production inference pipeline**

| Field | Description |
|---|---|
| **Purpose** | Evaluate a temporal sequence-based modeling approach as a comparison point against the production XGBoost model |
| **Inputs** | Historical company filing sequences |
| **Processing** | An LSTM model was trained and evaluated as part of model experimentation |
| **Outputs** | Training performance curves and a comparison dataset, viewable on a dedicated dashboard page for reference |
| **Business Rules** | LSTM outputs are not used to generate the risk assessments shown as the primary result for any company; XGBoost is the sole model used for production risk scoring |
| **Dependencies** | None — independent of the production inference path |

**Note:** This module is included in the dashboard for transparency into the model selection process, not as an operational prediction source.

---

### Module 8 — Analyst Dashboard (Presentation Layer)
**Status: ✅ Implemented**

| Field | Description |
|---|---|
| **Purpose** | Present risk assessments and supporting analysis to the analyst in an accessible interface |
| **Inputs** | Gold dataset; model predictions; SHAP explainability outputs; sentiment signal data |
| **Processing** | Renders company-level and portfolio-level views across dedicated dashboard pages |
| **Outputs** | Interactive visual interface, described by page below |
| **Business Rules** | Risk classification displayed to the user reflects the currently selected decision threshold |
| **Dependencies** | Modules 3–7 outputs |

**Implemented Dashboard Pages:**

| Page | Function |
|---|---|
| Overview | Distress label distribution; per-feature distribution views split by outcome label |
| Predictions | Adjustable risk threshold control; distribution of predicted risk scores; ranked table of highest-risk companies |
| SHAP Explainer | Global feature importance view; beeswarm distribution view; single-company waterfall view |
| LSTM Analysis *(Experimental)* | Training performance curves; experimental model dataset explorer |
| Sentiment Signals | Table and distribution view of extracted disclosure sentiment signals |
| Raw Data Explorer | Direct access to underlying dataset views for transparency |

---

## 7. Data Flow

```
SEC EDGAR (raw filings)
        ↓
[Bronze Layer] — raw ingestion, no transformation
        ↓
[Silver Layer] — schema harmonization, standardization, Parquet conversion
        ↓
[Gold Layer] — 17-feature engineering + sentiment signal extraction
        ↓
[Feature Order Lock] — validation and scaling
        ↓
[XGBoost Inference] — distress probability
        ↓
[SHAP Explainability] — per-prediction reasoning
        ↓
[Analyst Dashboard] — presentation to user
```

*(LSTM evaluation runs as a parallel, independent experimental track and does not feed into this production data flow.)*

## 8. System Performance Requirements

*(Measurable technical requirements — the appropriate home per our earlier agreement, separate from the business-level PRD)*

| Requirement | Description | Status |
|---|---|---|
| Feature Schema Integrity | Inference must halt with a validation error rather than proceed on any feature order, naming, or count mismatch | ✅ Implemented |
| Missing Data Resilience | The system must produce a complete feature vector even when a company has gaps in filing history, via forward-fill and a persistence-tracking window | ✅ Implemented |
| Explainability Coverage | Every risk prediction displayed on the dashboard must be accompanied by a corresponding SHAP explanation | ✅ Implemented |
| Sentiment Signal Coverage | The system should disclose the proportion of companies with an extracted (non-default) sentiment signal, rather than presenting default neutral values as equivalent to analyzed sentiment | ✅ Implemented (disclosed limitation) |
| Model Selection Transparency | The system should make the rationale for selecting XGBoost over LSTM as the production model available for review | ✅ Implemented (dashboard comparison view) |
| Out-of-Core Data Processing | The system must process filing volumes larger than available memory without failure | ✅ Implemented (Polars Lazy Evaluation) |

## 9. Error Handling

*(Reflects only what is actually implemented — no invented fallback logic)*

| Scenario | System Behavior | Status |
|---|---|---|
| Feature schema mismatch at inference | Hard validation error raised; inference blocked | ✅ Implemented |
| Missing filing quarter for a company | Forward-fill from most recent available filing; distress persistence tracked over a short rolling window | ✅ Implemented |
| Company with no extracted sentiment signal | Defaults to a neutral sentiment value | ✅ Implemented (with disclosed limitation) |
| General pipeline failure handling (e.g., malformed source filings, corrupted downloads) | Not formally implemented | 🔲 Future Scope |

## 10. Constraints and Dependencies

- The system depends on the continued availability and structural consistency of SEC EDGAR filing data
- Sentiment analysis quality is dependent on the clarity and structure of a company's MD&A narrative; heavily templated or boilerplate language reduces signal quality
- The current feature set and model are trained on a specific historical period; predictive validity outside that period has not been independently verified
- LSTM remains an evaluation artifact only and introduces no production dependency

## 11. Assumptions

This Functional Specification describes system behavior as implemented and verified directly against the project repository at the time of writing. Any future modification to the pipeline, feature set, or model should be reflected here to keep this document accurate. Components not present in the repository are excluded rather than assumed.

## 12. Future Enhancements

*(Repository-backed only — not new invented capabilities)*

- **Broader sentiment coverage** — extending sentiment extraction beyond the currently covered subset of companies, reducing reliance on the neutral default value
- **Stronger pipeline error handling** — formalized fallback and recovery logic for malformed source filings or ingestion failures, currently unimplemented
- **Additional model experimentation** — further evaluation of temporal/sequence-based approaches (building on the existing LSTM benchmarking work) as a possible complement to the production XGBoost model, contingent on precision improvements
- **Extended validation coverage** — applying schema-integrity checks (in the spirit of the Feature Order Lock) further upstream in the pipeline, closer to raw ingestion

## 13. Glossary

| Term | Definition |
|---|---|
| **EDGAR** | SEC's Electronic Data Gathering, Analysis, and Retrieval system — the public database of company filings used as the source of all input data for this project |
| **MD&A** | Management Discussion & Analysis — the narrative section of a filing where company management discusses financial results, risks, and outlook in plain text |
| **SHAP** | SHapley Additive exPlanations — a method for attributing a model's prediction to the contribution of each individual input feature |
| **Feature Order Lock** | The implemented validation mechanism ensuring the exact column order, naming, and count of input features at inference time matches the schema used during model training |
| **Bronze / Silver / Gold Layers** | The three-stage Medallion Architecture data pipeline pattern: Bronze holds raw, unmodified data; Silver holds cleaned and standardized data; Gold holds the finalized, model-ready feature set |
| **Distress Probability** | The model's output score representing the estimated likelihood that a company will experience financial distress within the defined prediction horizon |
