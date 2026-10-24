"""
INVESTMENT RISK MODULE
Computes a risk score per stock ticker from historical price volatility
and max drawdown. Used by the Wealth Optimization Engine for investment suggestions.
Imported by app.py.
"""
import pandas as pd
import numpy as np

_df = None


def _load(path="data/investment_risk_data/World-Stock-Prices-Dataset.csv"):
    global _df
    if _df is None:
        _df = pd.read_csv(path, parse_dates=["Date"])
        _df.sort_values(["Ticker", "Date"], inplace=True)
    return _df


def get_risk_score(ticker: str) -> dict:
    df = _load()
    stock = df[df["Ticker"].str.upper() == ticker.upper()]
    if stock.empty:
        return {"error": f"No data for ticker '{ticker}'"}

    stock = stock.copy()
    stock["return"] = stock["Close"].pct_change()
    volatility = stock["return"].std() * np.sqrt(252)  # annualized volatility

    cum_max = stock["Close"].cummax()
    drawdown = (stock["Close"] - cum_max) / cum_max
    max_drawdown = drawdown.min()

    # scale volatility (typically 0.1-1.0) to a 0-100 risk score
    risk_score = float(min(100, volatility * 100))
    category = "Low" if risk_score < 30 else "Medium" if risk_score < 60 else "High"

    return {
        "ticker": ticker.upper(),
        "annualized_volatility": round(float(volatility), 3),
        "max_drawdown_pct": round(float(max_drawdown) * 100, 1),
        "risk_score": round(risk_score, 1),
        "risk_category": category,
    }


def rank_by_risk(tickers: list) -> list:
    return sorted([get_risk_score(t) for t in tickers], key=lambda r: r.get("risk_score", 999))
