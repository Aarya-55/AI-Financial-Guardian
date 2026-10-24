"""
FRAUD ENGINE
Loads the trained model and gives:
  1. predict_risk(transaction)   -> probability + risk level (low/medium/high)
  2. explain_counterfactual(txn) -> minimal change that would have lowered the risk
This is imported by app.py — you do not run this file directly.
"""
import joblib
import pandas as pd

_bundle = joblib.load("models/fraud_model.pkl")
MODEL = _bundle["model"]
ENCODERS = _bundle["encoders"]
FEATURE_ORDER = _bundle["feature_order"]

BOOL_COLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]


def _vectorize(txn: dict) -> pd.DataFrame:
    row = dict(txn)
    for col, le in ENCODERS.items():
        val = str(row.get(col, le.classes_[0]))
        row[col] = int(le.transform([val])[0]) if val in le.classes_ else 0
    for col in BOOL_COLS:
        row[col] = int(bool(row.get(col, False)))
    return pd.DataFrame([[row.get(f, 0) for f in FEATURE_ORDER]], columns=FEATURE_ORDER)


def risk_level(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.25:
        return "medium"
    return "low"


def predict_risk(txn: dict) -> dict:
    X = _vectorize(txn)
    prob = float(MODEL.predict_proba(X)[0][1])
    return {"probability": round(prob, 4), "risk_level": risk_level(prob)}


def explain_counterfactual(txn: dict, max_tries: int = 20) -> dict:
    """
    Lightweight DiCE-style counterfactual search:
    tries small, realistic edits (lower amount / known merchant / home distance)
    and returns the first one that flips the risk down a tier.
    """
    base = predict_risk(txn)
    if base["risk_level"] == "low":
        return {"message": "Already low risk, no change needed."}

    candidates = []
    if txn.get("amount_usd", 0) > 0:
        for factor in [0.75, 0.5, 0.25]:
            edit = dict(txn, amount_usd=round(txn["amount_usd"] * factor, 2))
            candidates.append((f"lowering the amount to ${edit['amount_usd']}", edit))
    if txn.get("is_new_merchant"):
        edit = dict(txn, is_new_merchant=False)
        candidates.append(("using a previously used merchant", edit))
    if txn.get("distance_from_home_km", 0) > 5:
        edit = dict(txn, distance_from_home_km=2.0)
        candidates.append(("transacting closer to home", edit))
    if txn.get("used_vpn"):
        edit = dict(txn, used_vpn=False)
        candidates.append(("disabling VPN", edit))
    if txn.get("is_foreign_transaction"):
        edit = dict(txn, is_foreign_transaction=False)
        candidates.append(("making it a domestic transaction", edit))

    for reason, edit in candidates[:max_tries]:
        new_risk = predict_risk(edit)
        if new_risk["risk_level"] != base["risk_level"] and new_risk["probability"] < base["probability"]:
            return {
                "original_risk": base,
                "would_have_been": new_risk,
                "explanation": f"This transaction would likely have been approved by {reason}."
            }
    return {"original_risk": base, "explanation": "No small single-field change found; multiple risk factors combined."}
