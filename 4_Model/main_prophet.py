import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
print(prophet.__file__)

url = "https://raw.githubusercontent.com/kalakhushi18/Advance-Time-Series-Prediction-W24/refs/heads/main/dataset.csv"

df = pd.read_csv(url)
df['date'] = pd.to_datetime(df['date'])
print(df[df['bitcoin_closing_prices'].isnull()])
if df.iloc[-1].isna().any():
    df = df.iloc[:-1]
print(df.head())
print(df.tail())
plt.figure(figsize=(12, 6))
plt.plot(df['date'], df['bitcoin_closing_prices'], label='Bitcoin Closing Price', color='blue')
plt.title('Bitcoin Closing Price Over Time')
plt.xlabel('Date')
plt.ylabel('Closing Price (USD)')
plt.legend()
plt.grid(True)
plt.show()
df = df[['date', 'bitcoin_closing_prices']]
df = df.reset_index(drop=True)

# Preparing the modeling data frame

df.columns = ['ds', 'y']
print(df.head())

test_number = 3595
#Creating a train-test split
train = df.iloc[:test_number, :]
test = df.iloc[test_number:, :]
print(test)
#Creating a basic Prophet model
m = Prophet()
model = m.fit(train)

future = m.make_future_dataframe(periods=len(test))  # Predict the next year

# Make predictions
forecast = m.predict(future)
print(forecast.tail())

#Creating a basic Prophet model

from sklearn.metrics import r2_score
print(r2_score(list(test['y']), list(forecast.loc[test_number:,'yhat'])))

from sklearn.metrics import mean_absolute_percentage_error
print(mean_absolute_percentage_error(list(test['y']), list(forecast.loc[test_number:,'yhat'])))

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(list(test['y']), list(forecast.loc[test_number:,'yhat'])))

# Plot the forecast
fig1 = m.plot(forecast)
plt.title("Bitcoin Price Prediction")


fig2 = m.plot_components(forecast)
plt.show()

plt.plot(list(test['y']))
plt.plot(list(forecast.loc[test_number:,'yhat'] ))
plt.show()
