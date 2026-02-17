
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for business-friendly plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def generate_synthetic_data(days=730):
    """
    Generates synthetic daily sales data for 2 years (approx 730 days).
    Includes trend, seasonality, and random noise.
    """
    dates = pd.date_range(start='2024-01-01', periods=days, freq='D')
    
    # Base trend (linear growth)
    trend = np.linspace(100, 300, days)
    
    # Seasonality (weekly pattern: higher on weekends)
    # 0=Mon, 6=Sun. Let's make Fri(4), Sat(5), Sun(6) higher.
    weekly_seasonality = np.array([1.0, 1.0, 1.0, 1.1, 1.3, 1.5, 1.4])
    seasonality_factor = [weekly_seasonality[d.weekday()] for d in dates]
    
    # Yearly seasonality (sine wave for seasonal peaks, e.g., summer/holidays)
    yearly_seasonality = 10 * np.sin(2 * np.pi * np.arange(days) / 365)
    
    # Random noise
    noise = np.random.normal(0, 15, days)
    
    # Combine components
    sales = (trend * seasonality_factor) + yearly_seasonality + noise
    # Ensure no negative sales
    sales = np.maximum(sales, 0)
    
    df = pd.DataFrame({'Date': dates, 'Sales': sales})
    
    # Introduce some missing values to demonstrate data cleaning
    indices_to_drop = np.random.choice(df.index, size=10, replace=False)
    df.loc[indices_to_drop, 'Sales'] = np.nan
    
    return df

def clean_data(df):
    """
    Cleans the dataframe by handling missing values.
    """
    print("\n--- Data Cleaning ---")
    print(f"Missing values before cleaning:\n{df.isnull().sum()}")
    
    # Impute missing values with linear interpolation (good for time series)
    df['Sales'] = df['Sales'].interpolate(method='linear')
    
    print(f"Missing values after cleaning:\n{df.isnull().sum()}")
    return df

def feature_engineering(df):
    """
    Creates time-based features and lag features.
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    # Time-based features
    df['DayOfWeek'] = df.index.dayofweek
    df['Quarter'] = df.index.quarter
    df['Month'] = df.index.month
    df['Year'] = df.index.year
    df['DayOfYear'] = df.index.dayofyear
    
    # Lag features (Past interactions are strong predictors)
    # Sales from 1 day ago, 7 days ago (same day last week), 30 days ago
    df['Lag_1'] = df['Sales'].shift(1)
    df['Lag_7'] = df['Sales'].shift(7)
    df['Lag_30'] = df['Sales'].shift(30)
    
    # Rolling mean (Trend capture)
    df['Rolling_Mean_7'] = df['Sales'].shift(1).rolling(window=7).mean()
    
    # Drop rows with NaN created by shifting
    df.dropna(inplace=True)
    
    return df

def train_model(df):
    """
    Trains a Random Forest Regressor.
    """
    # Define features and target
    features = ['DayOfWeek', 'Quarter', 'Month', 'Year', 'DayOfYear', 
                'Lag_1', 'Lag_7', 'Lag_30', 'Rolling_Mean_7']
    target = 'Sales'
    
    X = df[features]
    y = df[target]
    
    # Split data (Time-based split: Train on past, Test on future)
    split_point = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"\nTraining on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    
    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test, features

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints metrics.
    """
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")
    
    return predictions

def visualize_results(df, y_train, y_test, predictions, model, feature_names):
    """
    Generates business-friendly visualizations.
    """
    # 1. Historical Data vs Forecast
    plt.figure(figsize=(14, 7))
    plt.plot(y_train.index, y_train, label='Historical Sales (Train)', alpha=0.6)
    plt.plot(y_test.index, y_test, label='Actual Sales (Test)', color='green', alpha=0.6)
    plt.plot(y_test.index, predictions, label='Forecasted Sales', color='red', linestyle='--')
    
    plt.title('Sales Forecast: Actual vs Predicted', fontsize=18)
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sales_forecast_plot.png')
    print("\nVisualization saved: sales_forecast_plot.png")
    
    # 2. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
    plt.title('What Drives Sales? (Feature Importance)', fontsize=16)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("Visualization saved: feature_importance.png")
    
    # Show plots if interactive environment
    # plt.show()

def business_summary(predictions, y_test):
    """
    Prints a natural language summary for stakeholders.
    """
    total_actual = y_test.sum()
    total_forecast = predictions.sum()
    error_percentage = abs(total_actual - total_forecast) / total_actual * 100
    
    print("\n=== Business Insight Summary ===")
    print(f"Over the test period, our model forecasted total sales of {int(total_forecast):,}.")
    print(f"Actual sales were {int(total_actual):,}.")
    print(f"The variance is {error_percentage:.2f}%.")
    print("\nKey Takeaways:")
    print("1. The model captures the weekly seasonality (weekend peaks) effectively.")
    print("2. The trend line (Rolling Mean) is a significant predictor of future sales.")
    print("3. This forecast can be used to optimize inventory for the upcoming weeks.")

def main():
    print("Starting Sales & Demand Forecasting Pipeline...")
    
    # 1. Acquire Data
    df = generate_synthetic_data()
    print("Synthetic Data Generated. Head:")
    print(df.head())
    
    # 2. Clean Data
    df = clean_data(df)
    
    # 3. Feature Engineering
    df_features = feature_engineering(df)
    print("Features Engineered. Head:")
    print(df_features.head())
    
    # 4. Train Model
    model, X_train, X_test, y_train, y_test, feature_names = train_model(df_features)
    
    # 5. Evaluate
    predictions = evaluate_model(model, X_test, y_test)
    
    # 6. Visualize
    visualize_results(df, y_train, y_test, predictions, model, feature_names)
    
    # 7. Report
    business_summary(predictions, y_test)
    
    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
