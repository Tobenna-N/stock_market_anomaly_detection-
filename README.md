# Stock Market Anomaly Detection
## Overview

This project uses machine learning to detect abnormal trading behavior in stock markets using historical price and volume data.

## Approach

- Engineered financial features (returns, volatility, volume)
- Applied Isolation Forest to identify anomalies
- Flagged unusual market events based on deviations from normal patterns

## Key Results
- Detected major market events such as the 2020 COVID-19 crash
- High-volatility stocks (TSLA, AMZN) showed more anomalies
- Stable stocks (JPM, GS) showed fewer anomalies

##Tools

Python, pandas, numpy, scikit-learn, matplotlib, yfinance
