# time-series-analysis-of-eth-usdt-market-projection-
# Suppress warnings (optional)
import warnings
warnings.filterwarnings("ignore")

# Data Collection & Preprocessing
import numpy as np
import pandas as pd

# EDA & Visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import seaborn as sns
import plotly.graph_objects as go

# ARIMA Modeling
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Model Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Load data
ethereum_data = pd.read_csv('/content/ETH_1min.csv')

# Convert to datetime & set as index
ethereum_data['Date'] = pd.to_datetime(ethereum_data['Date'])
ethereum_data.set_index('Date', inplace=True)

# Initial checks
print("=== First 5 Rows ===")
print(ethereum_data.head())

print("\n=== Data Info ===")
print(ethereum_data.info())

print("\n=== Statistical Summary ===")
print(ethereum_data.describe())

print("\n=== Missing Values ===")
print(ethereum_data.isnull().sum())

# Resample to daily data (if needed)
daily_data = ethereum_data['Close'].resample('D').last().ffill()

# Plot closing price
plt.figure(figsize=(12, 6))
daily_data.plot(title='ETH/USDT Daily Closing Price')
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.grid()
plt.show()
adf_result = adfuller(daily_data.dropna())
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")                                                                                                                                      plot_acf(daily_data, lags=30)
plot_pacf(daily_data, lags=30)
plt.show()                                                                                                                                                                # Train-Test Split (80-20)
train_size = int(len(daily_data) * 0.8)
train, test = daily_data[:train_size], daily_data[train_size:]

# Fit ARIMA Model (replace p,d,q with your chosen values)
model = ARIMA(train, order=(2, 1, 2))  # Example: ARIMA(2,1,2)
fitted_model = model.fit()
# Forecast on Test Data
forecast = fitted_model.get_forecast(steps=len(test))
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int()  # Confidence intervals

# Calculate Error Metrics
rmse = np.sqrt(mean_squared_error(test, forecast_mean))
mape = mean_absolute_percentage_error(test, forecast_mean)

print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape * 100:.2f}%")
# Plot Forecast vs Actual
plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train', color='blue')
plt.plot(test.index, test, label='Actual', color='green')
plt.plot(test.index, forecast_mean, label='Forecast', color='red')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.3)
plt.title('ETH/USDT Price Forecast vs Actual')
plt.legend()
plt.grid()
plt.show()
# Residual Analysis
residuals = test - forecast_mean

# Plot Residuals
plt.figure(figsize=(12, 4))
plt.plot(residuals, label='Residuals', color='purple')
plt.axhline(0, linestyle='--', color='gray')
plt.title('Model Residuals')
plt.legend()
plt.grid()
plt.show()                                                                                                                                                           # ACF of Residuals (check for leftover patterns)
plot_acf(residuals, lags=30, title='ACF of Residuals')
plt.show()                                                                                                                                                                                # Histogram of Residuals (check normality)
plt.figure(figsize=(8, 4))
sns.histplot(residuals, kde=True, color='orange')
plt.title('Distribution of Residuals')
plt.show()                                                                                                                                                                         # Load and prepare data (example with synthetic data)
dates = pd.date_range(start="2020-01-01", end="2023-12-31")
daily_data = pd.Series(
    np.cumsum(np.random.normal(loc=0, scale=50, size=len(dates))) + 1500,
    index=dates
)

# Fit final ARIMA model
final_model = ARIMA(daily_data, order=(2,1,2)).fit()

# 30-day forecast
forecast_steps = 30
forecast = final_model.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)  # 95% CI

# Create future date index
last_date = daily_data.index[-1]
forecast_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=forecast_steps
)

# Visualization
plt.figure(figsize=(14, 7))
ax = plt.gca()

# Plot historical data (last 100 days)
ax.plot(
    daily_data[-100:],
    label='Historical Price',
    color='blue',
    linewidth=2
)

# Plot forecast
ax.plot(
    forecast_dates,
    forecast_mean,
    label='30-Day Forecast',
    color='red',
    linestyle='--',
    linewidth=2
)
ax.fill_between(
    forecast_dates,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color='orange',
    alpha=0.2,
    label='95% Confidence Interval'
)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
# Corrected date formatting code

plt.title('ETH/USDT 30-Day Price Forecast with ARIMA(2,1,2)', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USDT)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Trend analysis
trend_direction = "upward" if forecast_mean[-1] > forecast_mean[0] else "downward"
trend_strength = abs((forecast_mean[-1] - forecast_mean[0]) / forecast_mean[0] * 100)

# Add trend annotation
ax.annotate(
    f'{trend_direction.capitalize()} trend ({trend_strength:.1f}% change)',
    xy=(forecast_dates[-10], forecast_mean[-5]),
    xytext=(10, 30),
    textcoords='offset points',
    arrowprops=dict(arrowstyle='->'),
    fontsize=12
)

plt.tight_layout()
plt.show()

# Forecast summary
print("\n=== 30-Day ETH Price Forecast Summary ===")
print(f"Current Price: ${daily_data[-1]:.2f}")
print(f"Forecasted Price in 30 Days: ${forecast_mean[-1]:.2f}")
print(f"Expected Change: {trend_strength:.1f}% ({trend_direction})")
print("\nKey Confidence Intervals:")
print(f"Upper Bound (95% CI): ${conf_int.iloc[-1, 1]:.2f}")
print(f"Lower Bound (95% CI): ${conf_int.iloc[-1, 0]:.2f}")

# Volatility analysis
forecast_volatility = (conf_int.iloc[:, 1] - conf_int.iloc[:, 0]).mean()
print(f"\nAverage Daily Volatility Band: ±${forecast_volatility/2:.2f}") 
