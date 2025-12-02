import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="NBA 5-Year Career Predictor", layout="wide")
st.title("🏀 NBA 5-Year Career Predictor")

# ---------------------------------------------------------
# LOAD MODEL PACKAGE
# ---------------------------------------------------------
try:
    pkg = joblib.load("model.joblib")
except:
    st.error("❌ Could not load model.joblib. Make sure it is in the repo root.")
    st.stop()

# Expecting saved package: { "model", "scaler", "features" }
if isinstance(pkg, dict) and "model" in pkg:
    model = pkg["model"]
    scaler = pkg.get("scaler", None)
    features = pkg.get("features", None)
else:
    st.error("❌ model.joblib does not include scaler/features. Re-save model joblib correctly.")
    st.stop()

# ---------------------------------------------------------
# VALIDATE FEATURES
# ---------------------------------------------------------
if not isinstance(features, (list, tuple)):
    st.error("❌ 'features' inside model.joblib must be a list.")
    st.stop()

# Ensure all features are strings
features = [str(f) for f in features]

# ---------------------------------------------------------
# BUILD INPUT FORM
# ---------------------------------------------------------
st.write("Enter player stats below:")

cols = st.columns(2)
user_input = {}

with st.form("predict"):
    for i, col in enumerate(features):
        default = 0.0  # if you want nicer defaults, change this later
        user_input[col] = cols[i % 2].number_input(col, value=float(default), format="%.4f")
    submitted = st.form_submit_button("Predict")

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
if submitted:
    try:
        X = pd.DataFrame([[user_input[f] for f in features]], columns=features)

        if scaler is not None:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values

        pred = model.predict(X_scaled)[0]
        prob = model.predict_proba(X_scaled)[0][1] if hasattr(model, "predict_proba") else None

        label = "✅ Likely 5-Year Career" if int(pred) == 1 else "❌ Not Likely"
        st.success(label)

        if prob is not None:
            st.info(f"Probability: {prob:.3f}")

    except Exception as e:
        st.error(f"Prediction error: {e}")
