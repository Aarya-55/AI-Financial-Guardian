"""
TRAIN FRAUD MODEL
Trains a RandomForest to predict is_fraud from credit_card_fraud_2026.csv
Run this ONCE to create models/fraud_model.pkl (used later by fraud_engine.py)
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_PATH = "data/fraud_data/credit_card_fraud_2026.csv"
MODEL_OUT = "models/fraud_model.pkl"

CATEGORICAL_COLS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
BOOL_COLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn",
             "ip_country_mismatch", "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
DROP_COLS = ["transaction_id", "is_fraud"]

def load_and_prepare(path):
    df = pd.read_csv(path)
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    for col in BOOL_COLS:
        df[col] = df[col].astype(bool).astype(int)
    X = df.drop(columns=DROP_COLS)
    y = df["is_fraud"]
    return X, y, encoders

if __name__ == "__main__":
    X, y, encoders = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))

    joblib.dump(
        {"model": model, "encoders": encoders, "feature_order": list(X.columns)},
        MODEL_OUT
    )
    print(f"Saved model -> {MODEL_OUT}")
