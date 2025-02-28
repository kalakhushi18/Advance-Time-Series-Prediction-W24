import matplotlib.pyplot as plt
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

url = "https://raw.githubusercontent.com/kalakhushi18/Advance-Time-Series-Prediction-W24/refs/heads/main/dataset.csv"

bitcoin_data = pd.read_csv(url)
if bitcoin_data.iloc[-1].isna().any():
    bitcoin_data = bitcoin_data.iloc[:-1]

# Rename columns to match TimeSeriesDataFrame format
bitcoin_data = bitcoin_data[['date', 'bitcoin_closing_prices']].rename(columns={'date': 'timestamp', 'bitcoin_closing_prices': 'target'})

# Add 'item_id' column (required for TimeSeriesDataFrame)
bitcoin_data['item_id'] = 'BTC-USD'

# Ensure proper types
bitcoin_data['timestamp'] = pd.to_datetime(bitcoin_data['timestamp'])
bitcoin_data['target'] = bitcoin_data['target'].astype(float)

# Convert to TimeSeriesDataFrame
btc_data = TimeSeriesDataFrame(bitcoin_data)
print(btc_data.tail())
# Define prediction length (e.g., predict the next 30 days)
prediction_length = 60

# Split data into training and testing sets
train_data, test_data = btc_data.train_test_split(prediction_length)

# Initialize the TimeSeriesPredictor
predictor = TimeSeriesPredictor(prediction_length=prediction_length).fit(
    train_data, presets="bolt_small",
)

# Train the predictor on the training data
predictions = predictor.predict(train_data)

# Make predictions on the test data

# Print predictions
print(predictions.tail())

predictor.plot(
    data=test_data,
    predictions=predictions,
    item_ids=btc_data.item_ids[:2],
    max_history_length=200,
)
plt.show()

from sklearn.metrics import r2_score
print(r2_score(list(test_data['target'].iloc[3655-prediction_length:,]), list(predictions['mean'])))

from sklearn.metrics import mean_absolute_percentage_error
print(mean_absolute_percentage_error(list(test_data['target'].iloc[3655-prediction_length:,]), list(predictions['mean'])))

from sklearn.metrics import mean_absolute_error
print(mean_absolute_error(list(test_data['target'].iloc[3655-prediction_length:,]), list(predictions['mean'])))