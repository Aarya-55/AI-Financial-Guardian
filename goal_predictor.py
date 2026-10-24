"""
GOAL-BASED FINANCIAL PREDICTION
Given a goal amount and the user's real average monthly savings,
estimates months-to-goal and how much faster it'd be with behaviour changes.
Imported by app.py.
"""
import pandas as pd


def avg_monthly_savings(df: pd.DataFrame) -> float:
    monthly = df.groupby("Month").apply(
        lambda g: g.loc[g["Transaction Type"] == "credit", "Amount"].sum()
        - g.loc[g["Transaction Type"] == "debit", "Amount"].sum()
    )
    return float(monthly.mean())


def predict_goal(df: pd.DataFrame, goal_amount: float, current_saved: float = 0.0,
                  extra_monthly_saving: float = 0.0) -> dict:
    base_saving = avg_monthly_savings(df)
    remaining = max(0, goal_amount - current_saved)

    months_current = remaining / base_saving if base_saving > 0 else float("inf")
    boosted_saving = base_saving + extra_monthly_saving
    months_boosted = remaining / boosted_saving if boosted_saving > 0 else float("inf")

    return {
        "goal_amount": goal_amount,
        "avg_monthly_savings": round(base_saving, 2),
        "estimated_months_at_current_rate": round(months_current, 1) if months_current != float("inf") else None,
        "estimated_months_with_extra_saving": round(months_boosted, 1) if months_boosted != float("inf") else None,
        "months_saved_by_optimizing": (
            round(months_current - months_boosted, 1)
            if months_current != float("inf") and months_boosted != float("inf") else None
        ),
    }
