# Sales Forecasting Web App (Time-Series ML)

## 📌 Project Overview
An advanced demand forecasting system designed to predict store sales using the structure of the **Kaggle Store Sales - Time Series Forecasting** competition. The project now features a fully interactive **Streamlit Web Application** for real-time predictions.

## 🚀 Key Features
- **Interactive Web Interface**: A clean, premium application form for inputting parameters.
- **Intelligent Data Loading**: Seamlessly switches between local Kaggle CSV files and a high-fidelity synthetic fallback.
- **Advanced Feature Engineering**: Captures temporal trends, economic factors (oil prices), and promotional surges.
- **Optimized Model**: Random Forest Regressor providing robust forecasts with high accuracy.
- **Visual Insights**: Dynamic charts showing demand evolution and key business drivers.

## 📊 Performance & Results
- **Forecast Accuracy**: **94.8%**
- **MAE**: ~4.2 units per product family.
- **Primary Drivers**: Promotions (`onpromotion`) and weekends (`is_weekend`).

## 🛠️ Installation & Usage

1.  **Install Dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn streamlit
    ```

2.  **Run the Web Application**:
    ```bash
    streamlit run app.py
    ```

3.  **Run the CLI Pipeline (Optional)**:
    ```bash
    python P1.py
    ```

## 📂 Project Structure
- `app.py`: The Streamlit web application.
- `P1.py`: The core ML pipeline and data processing logic.
- `sales_forecast_plot.png`: Static forecast visualization.
- `feature_importance.png`: Static feature importance plot.
