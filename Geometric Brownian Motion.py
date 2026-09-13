import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
def gbm(n_steps, n_scenarios, mu, sigma, s0):
    dt = 1 / 252
    z = np.random.normal(size=(n_steps, n_scenarios))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    prices = s0 * np.exp(np.cumsum(log_returns, axis=0))
    return prices
ticker = input("Ticker: ")
n_scenarios = int(input("Scenarios: "))
n_months = int(input("Months: "))
data = yf.download(ticker, start="2023-01-01", auto_adjust=True)["Close"].dropna()
data = np.array(data).flatten()
log_returns = np.log(data[1:] / data[:-1])
s0 = float(data[-1])
mu = log_returns.mean() * 252
sigma = log_returns.std() * np.sqrt(252)
sim = gbm(n_months, n_scenarios, mu, sigma, s0)
time_months = np.linspace(0, n_months / 252 * 12, n_months)
plt.figure(figsize=(12, 5))
plt.plot(time_months, sim, alpha=0.3)
plt.xlabel("Months")
plt.ylabel("Price")
plt.title(f"GBM Simulation: {ticker}")
plt.show()
final_prices = sim[-1]
avg_growth = (np.mean(final_prices) - s0) / s0 * 100
print(f"Average % growth across simulations: {avg_growth:.2f}%")
horizon = n_months
predicted_end_prices = s0 * np.exp(mu * (horizon / 252))
print(f"Predicted end price (GBM expected value): {predicted_end_prices:.2f}")