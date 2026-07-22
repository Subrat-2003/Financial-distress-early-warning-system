# FDEWS — User Guide

**Version:** 1.0
**Status:** Final
**Last Updated:** July 2026
**Author:** Subrat Kumar Jena

## Quick Start

If you simply want to explore FDEWS:

1. Open the hosted FDEWS demo (HuggingFace Spaces).
2. Navigate to the Predictions page.
3. Select a company.
4. Review the predicted distress probability.
5. Open the SHAP Explainer to understand the prediction.
6. Use the Raw Data Explorer for supporting data.

## 1. Purpose

This guide explains how to access, set up, and operate FDEWS. It is written for someone encountering the project for the first time who wants to understand what it does in practice — not why it was built (see `01_Business_Problem.md`) or how it works internally (see `03_Functional_Specification.md`).

## 2. Intended Audience

- Recruiters and technical reviewers evaluating the project
- Analysts or evaluators exploring the dashboard hands-on
- Anyone setting up the project locally from the repository

## 3. Prerequisites

| Category | Requirement |
|---|---|
| Software | Python (version per `requirements.txt`) |
| Environment | Local Python environment, or no environment at all if using the live hosted demo |
| Dependencies | Listed in `requirements.txt` at the repository root |
| Dataset | Raw (60GB) and processed data are hosted externally due to size; access instructions are in `data/dataset_link.md` (Google Drive). The live hosted demo does not require the user to obtain this data separately — it is already bundled with the deployed dashboard. |

## 4. Installation & Setup

**Two verified ways to access FDEWS:**

**Option A — Live Hosted Demo (verified, no setup required)**
The dashboard is deployed and publicly accessible on HuggingFace Spaces. This is the fastest way to evaluate the project with no installation.

**Option B — Local Setup (repository-based)**
1. Clone the repository
2. Install dependencies from `requirements.txt`
3. Obtain the dataset via the instructions in `data/dataset_link.md`, if running the full pipeline or dashboard locally with the complete dataset
4. Launch the dashboard from the `dashboard/` folder, which contains the deployed model artifacts (`model.json`, `scaler.joblib`)

*Note: the exact local launch command is not explicitly documented in the repository README. Refer to the `dashboard/` folder directly for the entry point when setting up locally.*

## 5. Using the Dashboard

The primary, verified execution path is the hosted dashboard (Option A above). The dashboard runs against the Gold Dataset (9,461 companies, 17 features) and serves predictions and explanations for each covered company without requiring the user to trigger the pipeline manually — ingestion, feature engineering, and model training are pre-executed steps reflected in the deployed artifacts.

## 6. Dashboard Walkthrough

*(Identical to Functional Specification §6, Module 8 — repeated here in user-facing terms for a first-time operator)*

| Page | What You Can Do |
|---|---|
| **Overview** | View the distress label distribution and per-feature distribution views split by outcome (distressed vs. not) |
| **Predictions** | Adjust the risk threshold; view the distribution of predicted risk scores; browse a ranked table of the highest-risk companies |
| **SHAP Explainer** | View global feature importance, a beeswarm distribution across companies, and a waterfall breakdown for a single selected company |
| **LSTM Analysis** *(Experimental)* | View LSTM training performance curves and explore the experimental model's dataset — for reference only; not used for production predictions |
| **Sentiment Signals** | View the table and distribution of extracted MD&A sentiment signals |
| **Raw Data Explorer** | Browse the underlying Gold dataset, sentiment data, LSTM data, and file status directly |

## 7. Typical User Workflow

Open Dashboard
      │
      ▼
Select Company
      │
      ▼
Review Prediction
      │
      ▼
Inspect SHAP Explanation
      │
      ▼
Validate Using Raw Data

## 8. Understanding the Outputs

- **Risk / Distress Probability** — the XGBoost model's output score representing the estimated likelihood of financial distress within the prediction horizon
- **Risk Level** — a classification (e.g., high/low risk) derived from the distress probability relative to the currently selected threshold on the Predictions page
- **SHAP Explanation** — the ranked contributing factors behind a specific company's prediction, shown as a waterfall breakdown; global patterns are visible via the feature importance and beeswarm views
- **Sentiment Signal** — a value from -1.0 (extreme negative disclosure sentiment) to +1.0 (extreme positive), computed as Positive Probability − Negative Probability from the MD&A text. A value of exactly 0 may indicate genuine neutral sentiment **or** that no signal was extracted for that company (see Known Limitations)
- **Dashboard Visualizations** — distribution plots (Overview), threshold-adjustable risk tables (Predictions), and SHAP charts (SHAP Explainer) as described in Section 6

## 9. Known Limitations

*(Summarized from `03_Functional_Specification.md` and `04_Future_Scope.md` — not restated in full)*

- Sentiment signal is extracted for only a subset of companies; the remainder default to neutral (0), which may understate risk for those companies
- LSTM is experimental only; its low precision makes it unsuitable for production use and it does not feed into the risk assessments shown as primary results
- The model's validation covers 2020–2024; predictive validity in more recent market conditions is unverified
- `persistent_distress_flag` uses a fixed 2-quarter window; longer windows are untested
- General pipeline error handling (e.g., malformed filings) is not yet implemented

## 10. Troubleshooting

*(Repository-supported issues only)*

| Issue | Explanation |
|---|---|
| Inference stops with a validation error | The Feature Order Lock detected a mismatch in feature order, naming, or count and blocked inference to prevent a silently corrupted prediction. This is expected, correct behavior — not a bug. |
| A company's sentiment signal shows 0 | Either genuinely neutral sentiment, or no signal was extracted for that company (defaults to neutral). See Section 8. |
| Local setup requires a large dataset download | The full raw and processed datasets are hosted externally (Google Drive) due to size; see `data/dataset_link.md`. The hosted demo does not require this. |

## 11. FAQ

**Why are some sentiment values zero?**
Because a sentiment signal has only been extracted for a subset of companies. Companies without an extracted signal default to neutral (0), which is a known, disclosed limitation — not an error.

**Why did the Feature Order Lock stop inference?**
It detected that the feature vector's order, names, or count didn't match the training schema exactly. This is a deliberate safeguard against silent prediction corruption, not a failure.

**Why isn't the LSTM prediction used?**
LSTM is an experimental benchmarking component only. It showed high recall but low precision during evaluation, making it unsuitable for production. XGBoost is the sole model used to generate the risk assessments shown as primary results.

## 12. Referenced Documents

| Document | Purpose |
|---|---|
| `01_Business_Problem.md` | Why FDEWS exists |
| `02_PRD.md` | What FDEWS must do |
| `03_Functional_Specification.md` | How FDEWS works internally |
| `04_Future_Scope.md` | What is planned but not yet implemented |

