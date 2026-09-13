import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
def gbm(n_years, n_scenarios, mu, sigma, steps_per_year=12, s_0=1):
    dt = 1 / steps_per_year
    n_steps = int(n_years * steps_per_year)
    L = np.random.normal(size=(n_steps, n_scenarios))
    rets = mu * dt + sigma * np.sqrt(dt) * L
    prices = s_0 * (1 + rets).cumprod(axis=0)
    return pd.DataFrame(prices, columns=[f"path_{i+1}" for i in range(n_scenarios)])
stonks = input("Pick a ticker symbol: ")
scenarios = int(input("How many scenarios do you want to simulate: "))
years = int(input("How many years into the future do you want to simulate: "))
data = yf.download(stonks, start="2023-01-01", end="2026-03-05", auto_adjust=False)["Adj Close"].squeeze()
log_returns = np.log(data / data.shift(1)).dropna()
s0 = float(data.iloc[-1])
simulated_prices = gbm(
    n_years=years, n_scenarios=scenarios, s_0=s0,
    sigma=float(log_returns.std()), mu=float(log_returns.mean())
)
time_axis = np.linspace(0, years, simulated_prices.shape[0])
plt.figure(figsize=(12, 5))
plt.plot(time_axis, simulated_prices.values)
plt.title(f"GBM Simulated Price Paths for {stonks}")
plt.xlabel("Years")
plt.ylabel("Price")
plt.show()