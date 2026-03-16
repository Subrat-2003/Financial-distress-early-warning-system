# Financial Distress Early Warning System (FDEWS)

**Scale:** 60GB+ SEC EDGAR Dataset | **Architecture:** Medallion (Bronze/Silver/Gold)

An end-to-end Machine Learning pipeline designed to predict corporate insolvency and financial distress using high-volume SEC filings (10-K/10-Q). This system leverages **Out-of-Core processing** to engineer signals from 5 years of historical data (2020–2025).

## Technical Highlights
- **High-Volume Engineering:** Processed 60GB of raw SEC data using **Polars Streaming** and **Lazy Evaluation** to maintain low RAM footprint.
- **Quant Feature Store:** Engineered 12 critical financial ratios (Liquidity, Leverage, Profitability) with **temporal persistence logic** to reduce false-positive signals.
- **Multimodal Risk:** (In Progress) Integrating **FinBERT** sentiment scores from MD&A text sections with numeric financial features.
- **Robust Validation:** Implemented **Walk-Forward Temporal Splitting** to mitigate data leakage and ensure model generalization across economic cycles.

## Project Architecture
The project follows the **Medallion Architecture** to ensure data lineage and reliability:
- **Bronze Layer:** Raw ingestion of SEC TSV files and HTML scraping of MD&A sections.
- **Silver Layer:** Schema enforcement, float32 downcasting for memory optimization, and partitioned Parquet storage.
- **Gold Layer:** Feature engineering store containing scaled financial ratios and 2-quarter persistence flags.
- **Modeling:** (Ongoing) XGBoost/LSTM training with SHAP-based interpretability.

## Tech Stack
- **Data:** Polars (Primary Engine), Parquet, DuckDB
- **ML/Analytics:** XGBoost, Scikit-Learn, SHAP
- **NLP:** FinBERT (via Transformers)
- **Deployment:** Streamlit, Google Colab

## Repository Structure
- `bronze_layer/`: Scrapers and raw ingestion logic.
- `silver_layer/`: Cleaning, partitioning, and optimization scripts.
- `gold_layer/`: Quant feature engineering and labeling.
- `data/`: Link to the 60GB Google Drive dataset.
- `docs/`: Architecture diagrams and data dictionary.

## Dataset Access
Due to the 60GB scale, the raw and processed data is hosted on Google Drive. 
**See [data/dataset_link.md](data/dataset_link.md) for access instructions.**
