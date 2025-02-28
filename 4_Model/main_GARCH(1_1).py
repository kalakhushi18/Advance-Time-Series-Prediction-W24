import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_pacf

url = "https://raw.githubusercontent.com/kalakhushi18/Advance-Time-Series-Prediction-W24/refs/heads/main/dataset.csv"

# Load Bitcoin closing price
bitcoin_data = pd.read_csv(url)
bitcoin_data['date'] = pd.to_datetime(bitcoin_data['date'])
if bitcoin_data.iloc[-1].isna().any():
    bitcoin_data = bitcoin_data.iloc[:-1]

# Plot Bitcoin closing prices
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_data['date'], bitcoin_data['bitcoin_closing_prices'], label='Bitcoin Closing Price', color='blue')
plt.title('Adjusted Closing Prices of Bitcoin')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc='upper left')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

# Calculate daily returns
bitcoin_data['Daily_Return'] = bitcoin_data['bitcoin_closing_prices'].pct_change()

# Plotting the daily returns
plt.figure(figsize=(12, 6))
plt.plot(bitcoin_data['date'], bitcoin_data['Daily_Return'], label='Bitcoin Daily Returns', color='green')
plt.title('Bitcoin Daily Returns (2014-11-01 to 2024-11-02)')
plt.xlabel('Date')
plt.ylabel('Daily Returns')
plt.legend()
plt.grid(True)

# Calculate squared returns
bitcoin_data['Squared_Return'] = bitcoin_data['Daily_Return'] ** 2

# Plotting the squared returns
plt.figure(figsize=(10, 6))
plt.plot(bitcoin_data['date'], bitcoin_data['Squared_Return'], label='Bitcoin Squared Returns', color='red')
plt.title('Bitcoin Squared Returns (2014-11-01 to 2024-11-02)')
plt.xlabel('Date')
plt.ylabel('Squared Returns')
plt.legend()
plt.grid(True)

# Calculate log returns
bitcoin_data['Log_Return'] = 100*np.log(bitcoin_data['bitcoin_closing_prices'] / bitcoin_data['bitcoin_closing_prices'].shift(1))

# Plotting the log returns
plt.figure(figsize=(10, 6))
plt.plot(bitcoin_data['date'], bitcoin_data['Log_Return'], label='Bitcoin Log Returns', color='purple')
plt.title('Bitcoin Log Returns (2014-11-01 to 2024-11-02)')
plt.xlabel('Date')
plt.ylabel('Log Returns')
plt.legend()
plt.grid(True)


return_data = bitcoin_data[['Squared_Return']]
return_data.reset_index(drop=True, inplace=True)  # Reset index to integers
return_data_array = return_data.iloc[:, 0].values
print(type(return_data_array))

# Plot PACF

plot_pacf(return_data_array)
plt.show()

new_df = bitcoin_data[['Log_Return']]
new_df = new_df.dropna()
new_df.reset_index(drop=True, inplace=True)  # Reset index to integers
transformed_data = new_df.iloc[:, 0].values

model = arch_model(transformed_data, vol='Garch', p=1, q=1)
x = model.fit()
print(x.summary())
fitted_variance = x.conditional_volatility

# Plot the actual data and fitted volatility
plt.figure(figsize=(12, 6))
plt.plot(transformed_data, label="Actual Data", color="blue", alpha=0.6)
plt.plot(fitted_variance, label="Fitted Volatility", color="red", alpha=0.8)
plt.title("Actual Data vs Fitted GARCH(1,1) Volatility")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()

h = 60
forecast = x.forecast(horizon=h)

# Extract forecasted volatility
forecasted_variance = forecast.variance.iloc[-1]
forecasted_volatility = np.sqrt(forecasted_variance)

# Plot the actual data and forecasted values
plt.figure(figsize=(12, 6))

# Plot historical volatility
historical_volatility = x.conditional_volatility
plt.plot(historical_volatility, label="Historical Volatility", color="blue")

# Extend x-axis for forecast
future_index = range(len(historical_volatility), len(historical_volatility) + h)
plt.plot(future_index, forecasted_volatility, label="Forecasted Volatility", color="red", marker="o")


# Add labels and legend
plt.title(f"GARCH(1,1) Forecast: Next {h} Steps")
plt.xlabel("Time")
plt.ylabel("Volatility")
plt.legend()
plt.show()


r2 = r2_score(bitcoin_data['Log_Return'][2:], fitted_variance[1:])

# Calculate MSE
mse = mean_squared_error(bitcoin_data['Log_Return'][2:], fitted_variance[1:])

# Print results
print(f"R² (R-squared): {r2:.4f}")
print(f"MSE (Mean Squared Error): {mse:.4f}")

from sklearn.metrics import mean_absolute_percentage_error
print(mean_absolute_percentage_error(bitcoin_data['Log_Return'][2:], fitted_variance[1:]))

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(bitcoin_data['Log_Return'][2:], fitted_variance[1:]))
