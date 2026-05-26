# Dataset Guide — Financial RAG Research Assistant

## Overview

The platform is designed to work with real publicly available financial datasets.
Synthetic data is included for testing and demonstration purposes.

## Included Synthetic Data

### `data/synthetic/sec_filings_sample.json`
- 3 synthetic SEC 10-K records: AAPL, MSFT, GOOGL (FY2023)
- Includes key financials, risk factors, and filing metadata

### `data/synthetic/earnings_transcripts_sample.json`
- 2 synthetic earnings call records: AAPL Q4 FY2023, MSFT Q1 FY2024
- Includes prepared remarks, forward guidance, and key signals

### `data/synthetic/market_data_sample.json`
- Market snapshot with S&P 500, VIX, 10-year yield
- Equity fundamentals for AAPL, MSFT, GOOGL

## Real Dataset Sources

### SEC EDGAR
- **URL**: https://www.sec.gov/cgi-bin/browse-edgar
- **API**: https://data.sec.gov/submissions/{CIK}.json
- **Tool**: `sec-edgar-downloader` Python package

```python
from sec_edgar_downloader import Downloader
dl = Downloader("Company Name", "contact@example.com")
dl.get("10-K", "AAPL", limit=5)  # Download 5 most recent 10-Ks
```

### Yahoo Finance
```python
import yfinance as yf
aapl = yf.Ticker("AAPL")
info = aapl.info  # Company fundamentals
hist = aapl.history(period="5y")  # 5-year price history
```

### Hugging Face Financial Datasets
- `financial-phrasebank` — Sentiment analysis
- `edgar-corpus` — SEC filing texts
- `financial_qa` — Financial QA pairs

## Ingestion Pipeline

To ingest real SEC filings:

1. Download filings using `sec-edgar-downloader`
2. Place files in `data/raw/sec_filings/`
3. Run ingestion pipeline:

```bash
python -m scripts.ingest_data \
  --source data/raw/sec_filings/ \
  --collection sec_filings \
  --chunk-strategy semantic
```

## Data Privacy

- Never commit raw financial data files to git (covered by .gitignore)
- Use synthetic data for testing and CI/CD
- Ensure compliance with data vendor terms of service
- Apply PII governance checks before indexing
