"""
WEALTH OPTIMIZATION ENGINE
Looks at category-wise spend and suggests where to cut back + how to split savings.
Imported by app.py.
"""
import pandas as pd
from financial_health import DISCRETIONARY

TOP_N = 3


def recommend(df: pd.DataFrame) -> dict:
    debit = df[df["Transaction Type"] == "debit"]
    by_category = debit.groupby("Category")["Amount"].sum().sort_values(ascending=False)

    discretionary_spend = by_category[by_category.index.isin(DISCRETIONARY)]
    top_cuts = discretionary_spend.head(TOP_N)

    monthly_income = df.loc[df["Transaction Type"] == "credit", "Amount"].sum() / max(1, df["Month"].nunique())
    potential_monthly_saving = top_cuts.sum() / max(1, df["Month"].nunique()) * 0.3  # suggest cutting 30%

    allocation = {
        "emergency_fund_pct": 40,
        "short_term_goals_pct": 30,
        "long_term_investing_pct": 30,
    }

    suggestions = [
        f"Cut '{cat}' spending by ~30% to free up ${amt/max(1, df['Month'].nunique())*0.3:,.0f}/month"
        for cat, amt in top_cuts.items()
    ]

    return {
        "top_discretionary_categories": top_cuts.round(2).to_dict(),
        "suggested_monthly_saving_increase": round(potential_monthly_saving, 2),
        "suggested_allocation_of_new_savings": allocation,
        "suggestions": suggestions,
        "avg_monthly_income": round(monthly_income, 2),
    }
