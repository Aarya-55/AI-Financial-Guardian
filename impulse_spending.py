"""
IMPULSE SPENDING PREDICTION
Flags days/categories with clustered discretionary spending
(a proxy for impulsive buying behaviour) and scores overall impulse risk.
Imported by app.py.
"""
import pandas as pd
from financial_health import DISCRETIONARY


def detect_impulse_patterns(df: pd.DataFrame) -> dict:
    disc = df[(df["Transaction Type"] == "debit") & (df["Category"].isin(DISCRETIONARY))].copy()

    per_day = disc.groupby("Date").agg(count=("Amount", "size"), total=("Amount", "sum"))
    flagged_days = per_day[per_day["count"] >= 2]  # 2+ discretionary buys same day = impulse cluster

    total_disc_txns = len(disc)
    impulse_txns = flagged_days["count"].sum()
    impulse_ratio = impulse_txns / max(1, total_disc_txns)
    impulse_score = round(min(100, impulse_ratio * 150), 1)  # 0-100, higher = more impulsive

    top_trigger_categories = (
        disc[disc["Date"].isin(flagged_days.index)]
        .groupby("Category")["Amount"].sum().sort_values(ascending=False).head(3)
    )

    return {
        "impulse_score": impulse_score,
        "flagged_impulse_days": len(flagged_days),
        "top_trigger_categories": top_trigger_categories.round(2).to_dict(),
        "recommendation": (
            "High impulse-buying pattern detected — consider a 24h cooldown rule for discretionary purchases."
            if impulse_score > 40 else "Spending pattern looks controlled."
        ),
    }
