"""
AI FINANCIAL DECISION ENGINE (Flask API)
This is the ONE file you actually run. It loads all the modules below and
exposes HTTP endpoints that the ESP32 (and any dashboard) can call.

RUN:  python3 app.py
Then it listens on http://<your-computer-ip>:5000
"""
from flask import Flask, request, jsonify
import financial_health as fh
import wealth_optimizer as wo
import goal_predictor as gp
import cash_flow_forecast as cf
import impulse_spending as imp
import investment_risk as ir
import fraud_engine as fe

app = Flask(__name__)

TRANSACTIONS_PATH = "data/finance_tips_data/personal_transactions_dashboard_ready (2).xlsx"
_user_df = fh.load_transactions(TRANSACTIONS_PATH)


# ---------- 1. FRAUD RISK + ADAPTIVE AUTHENTICATION (called by ESP32 on each payment) ----------
@app.route("/api/transaction/check", methods=["POST"])
def check_transaction():
    txn = request.get_json(force=True)
    risk = fe.predict_risk(txn)

    # Adaptive multi-level authentication decision
    if risk["risk_level"] == "low":
        auth_required = "single_factor"       # RFID or fingerprint only
    elif risk["risk_level"] == "medium":
        auth_required = "two_factor"          # RFID + fingerprint
    else:
        auth_required = "blocked_pending_review"

    counterfactual = fe.explain_counterfactual(txn)
    return jsonify({**risk, "auth_required": auth_required, "counterfactual": counterfactual})


# ---------- 2. FINANCIAL HEALTH SCORE ----------
@app.route("/api/health_score", methods=["GET"])
def health_score():
    return jsonify(fh.compute_health_score(_user_df))


# ---------- 3. WEALTH OPTIMIZATION ----------
@app.route("/api/wealth_plan", methods=["GET"])
def wealth_plan():
    return jsonify(wo.recommend(_user_df))


# ---------- 4. GOAL PREDICTION ----------
@app.route("/api/goal", methods=["POST"])
def goal():
    body = request.get_json(force=True)
    result = gp.predict_goal(
        _user_df,
        goal_amount=float(body.get("goal_amount", 0)),
        current_saved=float(body.get("current_saved", 0)),
        extra_monthly_saving=float(body.get("extra_monthly_saving", 0)),
    )
    return jsonify(result)


# ---------- 5. CASH FLOW FORECAST ----------
@app.route("/api/cash_flow_forecast", methods=["GET"])
def cash_flow():
    return jsonify(cf.forecast_next_month(_user_df))


# ---------- 6. IMPULSE SPENDING ----------
@app.route("/api/impulse_spending", methods=["GET"])
def impulse():
    return jsonify(imp.detect_impulse_patterns(_user_df))


# ---------- 7. INVESTMENT RISK ----------
@app.route("/api/investment_risk", methods=["GET"])
def investment_risk():
    ticker = request.args.get("ticker", "AAPL")
    return jsonify(ir.get_risk_score(ticker))


# ---------- 8. EMERGENCY BUTTON (from ESP32 hardware button) ----------
@app.route("/api/emergency", methods=["POST"])
def emergency():
    # In a real deployment: freeze card, alert user's phone, log incident.
    return jsonify({"status": "emergency_received", "action": "account_frozen_and_alert_sent"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
