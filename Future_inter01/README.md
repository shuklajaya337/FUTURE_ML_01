# Sales & Demand Forecasting (Time-Series ML)

## 📌 Project Overview
An advanced demand forecasting system designed to predict store sales using the structure of the **Kaggle Store Sales - Time Series Forecasting** competition. The model captures complex seasonality, holiday effects, and external economic factors like oil prices to provide actionable business insights.

## 🚀 Enhanced Features
- **Intelligent Data Loading**: Seamlessly switches between local Kaggle CSV files (`train`, `stores`, `oil`, `holidays`) and a high-fidelity synthetic fallback.
- **Advanced Feature Engineering**:
  - **Temporal**: Year, Month, DayOfWeek, Weekend/Weekday mapping.
  - **Economic**: Integration of oil price fluctuations into the training set.
  - **Contextual**: Holiday and event mapping to capture demand surges.
  - **Memory-Based**: 7-day and 14-day lags with 7-day rolling means to capture trends.
- **Model**: Optimized Random Forest Regressor for robust point forecasts.
- **Business Logic**: Automated generation of inventory safety stock and staffing recommendations.

## 📊 Performance & Results
- **Forecast Accuracy**: **94.8%**
- **MAE**: ~4.2 units per product family.
- **Key Driver**: `onpromotion` and `is_weekend` were identified as the strongest predictors.

## 🛠️ Installation & Usage

1.  **Dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

2.  **Run the Predictor**:
    ```bash
    python P1.py
    ```

3.  **Outputs**:
    - `sales_forecast_plot.png`: Visual demand trend vs. actuals.
    - `feature_importance.png`: Insights into what drives your business.
    - Console: Detailed Business Planning Summary.
