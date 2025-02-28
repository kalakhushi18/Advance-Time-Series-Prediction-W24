import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Download multiple cryptocurrencies' data
crypto_symbols = ["BTC-USD", "ETH-USD", "BNB-USD", " XMR-USD", "USDT-USD", "ADA-USD", ",XRP-USD", "DOGE-USD", "LTC-USD",
                  "XLM-USD"]
#btc_eth = ["BTC-USD", "ETH-USD"]
#others = [a for a in crypto_symbols if a not in btc_eth]

df = yf.download(tickers=" ".join(crypto_symbols), period="max", group_by='ticker')
# Compute log returns
for col in df.columns.values:
    df[col] = np.log(df[col]) - np.log(df[col].shift(1))

df = df.dropna()
print(df)

# Extract 'Adj Close' data for plotting
df = df.xs('Adj Close', axis=1, level=1)
print(df)
#adj_close_data.to_csv("Multiple cryptocurrencies.csv")

plt.figure(figsize=(15,10))
plt.plot(df["BTC-USD"], label = "Bitcoin log-return")
plt.plot(df["ETH-USD"], label = "Ethereum log-return")
plt.title("Bitcoin vs Ethereum log-return")
plt.xlabel("Time")
plt.legend()
plt.show()

from copulas.multivariate import GaussianMultivariate
copula = GaussianMultivariate()
copula.fit(df)

# Generate synthetic data from the copula
synthetic_data = copula.sample(len(df))

# Display the synthetic data
import seaborn as sns

# Assuming synthetic_data is your DataFrame containing synthetic data generated from the copula
# Pairwise Scatter Plots
sns.pairplot(synthetic_data, diag_kind='kde')
corr_diff= synthetic_data.corr()-df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr_diff, annot=True, cmap='Greens', fmt=".2f")
plt.show()

# Step 1: Compute the log-likelihood of the fitted copula model
copula_density = copula.pdf(df)

# Step 2: Compute the log-likelihood
log_likelihood = np.sum(np.log(copula_density))

print("Log-Likelihood:", log_likelihood)

# Step 2: Calculate the number of parameters in the model
# For a Gaussian copula, the number of parameters is equal to the number of unique pairwise correlations
num_variables = len(df.columns)
num_params = num_variables * (num_variables - 1) / 2

# Step 3: Compute AIC and BIC
n = len(df)
aic = -2 * log_likelihood + 2 * num_params
bic = -2 * log_likelihood + num_params * np.log(n)

print("AIC:", aic)
print("BIC:", bic)