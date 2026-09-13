import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from matplotlib.animation import FuncAnimation

def gbm_step(S_prev, mu, sigma, dt):
    Z = np.random.normal()
    return S_prev * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

stonks = input("Pick a ticker symbol: ")
scenarios = int(input("How many scenarios do you want to simulate: "))

data = yf.download(stonks, start="2023-01-01", end="2026-03-05", auto_adjust=False)["Adj Close"].squeeze()

log_returns = np.log(data / data.shift(1)).dropna()

mu = float(log_returns.mean())
sigma = float(log_returns.std())

s0 = float(data.iloc[-1])

steps_per_year = 252
dt = 1 / steps_per_year

prices = [[s0] for _ in range(scenarios)]
time_axis = [0]

fig, ax = plt.subplots(figsize=(12,5))
lines = [ax.plot([], [])[0] for _ in range(scenarios)]

ax.set_title(f"Live GBM Simulated Price Paths for {stonks}")
ax.set_xlabel("Time Step")
ax.set_ylabel("Price")

def update(frame):

    time_axis.append(len(time_axis))

    for i in range(scenarios):
        new_price = gbm_step(prices[i][-1], mu, sigma, dt)
        prices[i].append(new_price)

        lines[i].set_data(time_axis, prices[i])

    ax.set_xlim(0, len(time_axis))
    ax.set_ylim(
        min(min(p) for p in prices),
        max(max(p) for p in prices)
    )

    return lines

ani = FuncAnimation(fig, update, interval=100)

plt.show()