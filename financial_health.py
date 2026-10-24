"""
FINANCIAL HEALTH SCORE
Computes a 0-100 score from a user's transaction history:
 savings rate, spending discipline (essential vs discretionary),
 debt management, investment/spend consistency, emergency-fund readiness.
Imported by app.py — not run directly.
"""
import pandas as pd

ESSENTIAL = {"Mortgage & Rent", "Groceries", "Utilities", "Gas & Fuel", "Mobile Phone",
             "Internet", "Auto Insurance", "Haircut"}
DISCRETIONARY = {"Shopping", "Restaurants", "Movies & Dvds", "Music", "Fast Food",
                  "Coffee Shops", "Alcohol & Bars", "Entertainment", "Television",
                  "Electronics & Software", "Food & Dining", "Home Improvement"}
DEBT = {"Credit Card Payment"}
INCOME = {"Paycheck"}


def load_transactions(path):
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def compute_health_score(df: pd.DataFrame) -> dict:
    monthly = df.groupby("Month").apply(
        lambda g: pd.Series({
            "income": g.loc[g["Transaction Type"] == "credit", "Amount"].sum(),
            "expense": g.loc[g["Transaction Type"] == "debit", "Amount"].sum(),
            "essential": g.loc[g["Category"].isin(ESSENTIAL), "Amount"].sum(),
            "discretionary": g.loc[g["Category"].isin(DISCRETIONARY), "Amount"].sum(),
            "debt_payment": g.loc[g["Category"].isin(DEBT), "Amount"].sum(),
        })
    ).reset_index()

    monthly["savings"] = monthly["income"] - monthly["expense"]
    monthly["savings_rate"] = (monthly["savings"] / monthly["income"].replace(0, 1)).clip(-1, 1)

    avg_savings_rate = monthly["savings_rate"].mean()
    savings_score = max(0, min(100, (avg_savings_rate + 0.2) / 0.5 * 100))

    disc_ratio = (monthly["discretionary"].sum() /
                  max(1, (monthly["essential"].sum() + monthly["discretionary"].sum())))
    discipline_score = max(0, 100 - disc_ratio * 150)

    debt_to_income = monthly["debt_payment"].sum() / max(1, monthly["income"].sum())
    debt_score = max(0, 100 - debt_to_income * 300)

    consistency_score = max(0, 100 - monthly["savings_rate"].std() * 200) if len(monthly) > 1 else 70

    emergency_months = monthly["savings"].clip(lower=0).sum() / max(1, monthly["expense"].mean())
    emergency_score = min(100, emergency_months / 6 * 100)

    weights = {"savings": 0.30, "discipline": 0.20, "debt": 0.20, "consistency": 0.15, "emergency": 0.15}
    final_score = (
        savings_score * weights["savings"] + discipline_score * weights["discipline"] +
        debt_score * weights["debt"] + consistency_score * weights["consistency"] +
        emergency_score * weights["emergency"]
    )

    return {
        "financial_health_score": round(final_score, 1),
        "breakdown": {
            "savings_score": round(savings_score, 1),
            "spending_discipline_score": round(discipline_score, 1),
            "debt_management_score": round(debt_score, 1),
            "consistency_score": round(consistency_score, 1),
            "emergency_fund_score": round(emergency_score, 1),
        },
        "avg_monthly_savings_rate": round(avg_savings_rate * 100, 1),
    }
