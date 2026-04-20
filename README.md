
---

## 📝 **Versión corta (si prefieres algo más conciso)**

```markdown
# Commodities Price Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Description

Machine Learning system that predicts commodity prices (natural gas, oil) using Bloomberg historical data. Generates forecasts for 1, 2, and 3 months ahead with chronological train/validation/test splitting.

## Features

- Multi-step forecasting (1, 2, 3 months)
- Automatic feature engineering (lags, rolling means, volatility)
- Interactive Streamlit dashboard
- Model persistence (.pkl files)
- Feature importance analysis

## Quick Start

```bash
# Clone repository
git clone https://github.com/your-username/commodities-predictor.git
cd commodities-predictor

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py