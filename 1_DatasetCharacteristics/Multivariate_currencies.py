import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download multiple cryptocurrencies' data
crypto_symbols = ["BTC-USD", "ETH-USD", "BNB-USD", " XMR-USD", "USDT-USD", "ADA-USD", ",XRP-USD", "DOGE-USD", "LTC-USD",
                  "XLM-USD"]
btc_eth = ["BTC-USD", "ETH-USD"]
others = [a for a in crypto_symbols if a not in btc_eth]

data = yf.download(tickers=" ".join(crypto_symbols), period="max", group_by='ticker')
data = data.dropna()

# Extract 'Adj Close' data for plotting
adj_close_data = data.xs('Adj Close', axis=1, level=1)
adj_close_data.to_csv("Multiple cryptocurrencies.csv")