"""Interactive Geometric Brownian Motion stock-price simulator.

Launch with: streamlit run "GBM Inequalities.py"
"""
from __future__ import annotations

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

STEPS_PER_YEAR = 252


@dataclass(frozen=True)
class MarketInputs:
    ticker: str
    start_date: str
    end_date: str
    spot_price: float
    historical_drift: float
    annual_volatility: float
    observations: int


def gbm(n_years, n_scenarios, mu, sigma, steps_per_year=STEPS_PER_YEAR, s_0=1.0, seed=None):
    """Simulate GBM paths using annualized inputs; row zero is the initial price."""
    n_steps = round(n_years * steps_per_year)
    if n_steps <= 0 or n_scenarios <= 0 or steps_per_year <= 0:
        raise ValueError("Horizon, simulations, and steps per year must all be positive.")
    if s_0 <= 0 or sigma < 0 or not np.isfinite([mu, sigma, s_0]).all():
        raise ValueError("Inputs must be finite; price positive; volatility non-negative.")
    rng = np.random.default_rng(seed)
    dt = 1 / steps_per_year
    shocks = rng.standard_normal((n_steps, n_scenarios))
    log_increment = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    prices = s_0 * np.exp(np.cumsum(log_increment, axis=0))
    return np.vstack((np.full((1, n_scenarios), s_0), prices))


def download_market_inputs(ticker, start_date, end_date):
    """Fetch adjusted prices and estimate historical annual GBM inputs."""
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("A ticker symbol is required.")
    if start_date >= end_date:
        raise ValueError("The start date must be before the end date.")
    try:
        history = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False, progress=False)
    except Exception as error:
        raise RuntimeError(f"Could not download price data for {symbol}: {error}") from error
    if history.empty:
        raise RuntimeError("No price data returned. Check the ticker, dates, and internet connection.")
    field = "Adj Close" if "Adj Close" in history.columns else "Close"
    price = history[field]
    if getattr(price, "ndim", 1) == 2:
        if price.shape[1] != 1:
            raise RuntimeError(f"Expected one price series; received {price.shape[1]}.")
        price = price.iloc[:, 0]
    price = price.dropna()
    log_returns = np.log(price / price.shift(1)).dropna()
    if len(log_returns) < 20:
        raise RuntimeError("At least 20 daily returns are needed to estimate volatility.")
    return MarketInputs(symbol, str(start_date), str(end_date), float(price.iloc[-1]),
                        float(log_returns.mean() * STEPS_PER_YEAR),
                        float(log_returns.std() * np.sqrt(STEPS_PER_YEAR)), len(log_returns))


def probability_with_ci(event):
    """Monte Carlo probability and normal-approximation 95% confidence interval."""
    probability = float(np.mean(event))
    margin = 1.96 * np.sqrt(probability * (1 - probability) / len(event))
    return probability, max(0.0, probability - margin), min(1.0, probability + margin)


def summarize_paths(paths, target_return):
    initial, terminal = float(paths[0, 0]), paths[-1]
    target_price = initial * (1 + target_return)
    output = {"target_price": target_price, "mean_terminal": float(terminal.mean()),
              "median_terminal": float(np.median(terminal)),
              "median_return": float(np.median(terminal / initial - 1))}
    for name, event in {"target": terminal >= target_price, "loss": terminal < initial,
                        "double": terminal >= 2 * initial}.items():
        probability, low, high = probability_with_ci(event)
        output.update({f"p_{name}": probability, f"p_{name}_low": low, f"p_{name}_high": high})
    output.update(dict(zip(("p05", "p25", "p50", "p75", "p95"), np.percentile(terminal, [5, 25, 50, 75, 95]))))
    return output


def model_signal(result):
    """Return a transparent scenario label; this is not personalized investment advice."""
    if result["p_target"] >= 0.60 and result["p_loss"] <= 0.40:
        return "BUY", "The model has at least a 60% chance of reaching your target and no more than a 40% chance of a loss."
    if result["p_loss"] >= 0.60 or (result["p_target"] < 0.40 and result["median_return"] < 0):
        return "SELL", "The model assigns a high chance of loss or a low chance of reaching your chosen target with a negative median return."
    return "HOLD", "The simulation is mixed: it does not meet this tool's deliberately simple BUY or SELL thresholds."


def fan_chart(paths, ticker, years):
    quantiles = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    time = np.linspace(0, years, len(paths))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.fill_between(time, quantiles[0], quantiles[4], color="#4C78A8", alpha=.15, label="5th–95th percentile")
    ax.fill_between(time, quantiles[1], quantiles[3], color="#4C78A8", alpha=.30, label="25th–75th percentile")
    ax.plot(time, quantiles[2], color="#153E75", linewidth=2.5, label="Median")
    ax.set(title=f"{ticker}: GBM forecast fan chart", xlabel="Years", ylabel="Price")
    ax.legend(frameon=False); fig.tight_layout()
    return fig


def terminal_chart(paths, target_price, ticker):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.hist(paths[-1], bins=60, density=True, color="#4C78A8", alpha=.75, edgecolor="white")
    ax.axvline(target_price, color="#E45756", linewidth=2.5, label=f"Target: ${target_price:,.2f}")
    ax.axvline(np.median(paths[-1]), color="#153E75", linestyle="--", label="Median terminal price")
    ax.set(title=f"{ticker}: terminal-price distribution", xlabel="Price at forecast horizon", ylabel="Density")
    ax.legend(frameon=False); fig.tight_layout()
    return fig


def run_dashboard():
    import streamlit as st
    from datetime import date
    st.set_page_config(page_title="GBM Inequalities", page_icon="📈", layout="wide")
    st.title("GBM Inequalities")
    st.caption("Configurable Monte Carlo stock-price scenarios — not investment advice.")
    with st.sidebar:
        st.header("Assumptions")
        ticker = st.text_input("Ticker", "AAPL").upper().strip()
        start_date = st.date_input("Historical start", date(2023, 1, 1))
        end_date = st.date_input("Historical end", date.today())
        years = st.slider("Forecast horizon (years)", .25, 10.0, 3.0, .25)
        trials = st.select_slider("Simulations", [1_000, 5_000, 10_000, 25_000, 50_000], value=10_000)
        mode = st.radio("Drift method", ["Historical", "Risk-neutral"])
        risk_free_rate = st.number_input("Risk-free rate (%)", 0.0, 20.0, 4.0, .1) / 100
        target_return = st.number_input("Target return (%)", -90.0, 500.0, 20.0, 1.0) / 100
        seed = st.number_input("Random seed", 0, value=42, step=1)
        run = st.button("Run simulation", type="primary")
    if not run:
        st.info("Set assumptions in the sidebar, then select **Run simulation**.")
        return
    try:
        market = download_market_inputs(ticker, start_date, end_date)
        drift = market.historical_drift if mode == "Historical" else risk_free_rate
        paths = gbm(years, trials, drift, market.annual_volatility, s_0=market.spot_price, seed=seed)
        result = summarize_paths(paths, target_return)
    except (ValueError, RuntimeError) as error:
        st.error(str(error)); return
    st.subheader("Data and model")
    st.write(f"Using **{market.observations:,}** daily returns from {market.start_date} to {market.end_date}. "
             f"Historical drift: **{market.historical_drift:.2%}**; volatility: **{market.annual_volatility:.2%}**. "
             f"Simulation drift: **{drift:.2%}** ({mode.lower()}).")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"P(return ≥ {target_return:.0%})", f"{result['p_target']:.1%}", f"95% CI {result['p_target_low']:.1%}–{result['p_target_high']:.1%}")
    c2.metric("P(loss)", f"{result['p_loss']:.1%}", f"95% CI {result['p_loss_low']:.1%}–{result['p_loss_high']:.1%}")
    c3.metric("P(double)", f"{result['p_double']:.1%}", f"95% CI {result['p_double_low']:.1%}–{result['p_double_high']:.1%}")
    c4.metric("Median terminal price", f"${result['median_terminal']:,.2f}", f"Target ${result['target_price']:,.2f}")
    signal, rationale = model_signal(result)
    st.subheader("Model signal")
    if signal == "BUY":
        st.success(f"**{signal} — simulation signal only.** {rationale}")
    elif signal == "SELL":
        st.error(f"**{signal} — simulation signal only.** {rationale}")
    else:
        st.warning(f"**{signal} — simulation signal only.** {rationale}")
    st.caption("This is an educational model label, not personalized investment advice or a trade instruction. It does not consider your finances, taxes, risk tolerance, other holdings, valuation, news, or market conditions.")
    left, right = st.columns(2)
    left.pyplot(fan_chart(paths, market.ticker, years), clear_figure=True)
    right.pyplot(terminal_chart(paths, result["target_price"], market.ticker), clear_figure=True)
    st.subheader("Terminal-price percentiles")
    st.dataframe({"Percentile": ["5th", "25th", "50th", "75th", "95th"],
                  "Price": [result[key] for key in ("p05", "p25", "p50", "p75", "p95")]}, hide_index=True)
    st.caption("GBM assumes constant drift and volatility, independent normally distributed log returns, and no jumps or regime changes.")


if __name__ == "__main__":
    run_dashboard()
