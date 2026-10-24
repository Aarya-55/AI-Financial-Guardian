"""
CASH FLOW FORECASTING
Simple linear-trend forecast of next month's income/expense from monthly history,
flags a projected shortfall (financial emergency) in advance.
Imported by app.py.
"""
import numpy as np
import pandas as pd


def forecast_next_month(df: pd.DataFrame) -> dict:
    monthly = df.groupby("Month").apply(
        lambda g: pd.Series({
            "income": g.loc[g["Transaction Type"] == "credit", "Amount"].sum(),
            "expense": g.loc[g["Transaction Type"] == "debit", "Amount"].sum(),
        })
    ).reset_index()
    monthly = monthly.sort_values("Month").reset_index(drop=True)
    x = np.arange(len(monthly))

    def trend_forecast(y):
        if len(y) < 2:
            return float(y.mean())
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope * len(y) + intercept)

    next_income = max(0, trend_forecast(monthly["income"].values))
    next_expense = max(0, trend_forecast(monthly["expense"].values))
    projected_balance = next_income - next_expense

    return {
        "history_months": monthly["Month"].tolist(),
        "forecast_next_month_income": round(next_income, 2),
        "forecast_next_month_expense": round(next_expense, 2),
        "projected_net_cash_flow": round(projected_balance, 2),
        "shortfall_risk": bool(projected_balance < 0),
        "message": (
            "Warning: projected expenses exceed income next month."
            if projected_balance < 0 else "Cash flow looks healthy for next month."
        ),
    }
