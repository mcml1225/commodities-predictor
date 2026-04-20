
# Commodities Price Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Description

Supervised Machine Learning system that predicts commodity prices (natural gas, oil, etc.) using historical Bloomberg data. The model trains with derived features (lags, rolling means, volatility) and generates predictions for the next 1, 2, and 3 months with chronological data splitting (train/validation/test).

## 🎯 Features

- ✅ **Multi-step forecasting**: Predicts prices 1, 2, and 3 months ahead
- ✅ **Feature engineering**: Automatically creates 20+ derived features
- ✅ **Temporal validation**: Chronological split respecting data order
- ✅ **Interactive dashboard**: Visualizations with Streamlit and Plotly
- ✅ **Persistent models**: Saves trained models as .pkl files
- ✅ **Feature importance**: Identifies most influential variables

## 🏗️ Project Structure
commodities-predictor/
│
├── app.py # Streamlit dashboard
├── model.py # ML model training & prediction
├── data_loader.py # Excel data loading & cleaning
├── requirements.txt # Python dependencies
├── Bloomberg_commodities_prices.xlsx # Input data
├── models/ # Saved trained models (.pkl)
│ ├── models.pkl # RandomForest models (1,2,3 months)
│ ├── scalers.pkl # StandardScaler objects
│ ├── imputers.pkl # SimpleImputer objects
│ └── feature_importance.pkl # Feature importance analysis
└── README.md

text

## 📊 Data Source

The model uses Bloomberg commodity data with the following structure:

| Ticker | Description |
|--------|-------------|
| `BGAS2US LX Equity` | Bloomberg Gas Index |
| `BGASX US Equity` | Bloomberg Gas Total Return |
| `BOIL US Equity` | BOIL ETF (Natural Gas) |
| `CPF US Equity` | CPF ETF |
| `GLCO US Equity` | Global Commodities |
| `NGAS US Equity` | Natural Gas |
| `WETRUCP US Equity` | Wet Crude Oil |

**Fields used:** `PX_LAST`, `PX_BID`, `PX_MID`, `CUR_MKT_CAP`

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/commodities-predictor.git
cd commodities-predictor
2. Create virtual environment (optional but recommended)
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Run the application
bash
streamlit run app.py
🚀 Usage
Load data: The app automatically loads Bloomberg_commodities_prices.xlsx

Train model: Click "Train Model & Predict Next 3 Months"

View results:

Model performance metrics (R², MAE, RMSE)

Price predictions for 1, 2, and 3 months ahead

Historical + projected price chart

Feature importance analysis

📈 Model Architecture
Feature Engineering
Feature Type	Examples
Price lags	lag_1m, lag_3m, lag_6m
Rolling statistics	rolling_mean_3m, rolling_std_6m
Price momentum	price_change_1m, price_change_3m
Volatility	volatility_3m, volatility_6m
Moving average ratios	ma_3_6_ratio, ma_6_12_ratio
Cross-commodity ratios	ratio_vs_BGASX, diff_vs_NGAS
Model Configuration
python
RandomForestRegressor(
    n_estimators=200,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42
)
Data Split (Chronological)
Split	Percentage	Purpose
Train	60%	Model training
Validation	20%	Hyperparameter tuning
Test	20%	Final evaluation
📊 Performance Metrics
The model is evaluated using:

MAE (Mean Absolute Error): Average prediction error in USD

RMSE (Root Mean Square Error): Penalizes larger errors

R² (Coefficient of Determination): Variance explained by the model

Typical results on test data:

1-month prediction: R² ~0.95+

2-month prediction: R² ~0.90+

3-month prediction: R² ~0.85+

🖥️ Dashboard Preview
The Streamlit dashboard includes:

Historical price chart - Interactive time series visualization

Performance metrics - R², MAE, RMSE for each horizon

Price predictions - 1, 2, and 3 month forecasts

Projection chart - Historical + predicted prices

Feature importance - Top 15 most influential features

🔄 Making Predictions with Saved Models
Once trained, models are saved as .pkl files. To make predictions without retraining:

python
from model import predict_with_existing_model

predictions, dates, last_date, metrics = predict_with_existing_model(df, target_col, steps=3)
print(predictions)  # {1: 16.50, 2: 16.80, 3: 17.20}
📦 Dependencies
text
pandas>=2.0.0
numpy>=1.24.0
openpyxl>=3.1.0
streamlit>=1.28.0
plotly>=5.17.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
joblib>=1.3.0
🤝 Contributing
Fork the repository

Create your feature branch (git checkout -b feature/amazing-feature)

Commit your changes (git commit -m 'Add amazing feature')

Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE file for more information.

👤 Author
Your Name

GitHub: @your-username

Email: your.email@example.com

🙏 Acknowledgments
Bloomberg for commodity price data

Streamlit for the amazing dashboard framework

Scikit-learn for ML algorithms

⭐ Star History
If you find this project useful, please give it a star! ⭐