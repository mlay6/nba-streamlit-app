# app.py (updated - loads model+scaler+features if available)
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("NBA 5-Year Career Prediction (Logistic Regression)")

# Load packaged model if it has model+scaler+features
model = None
scaler = None
features = None
load_error = None

try:
    pkg = joblib.load("model.joblib")
    if isinstance(pkg, dict) and "model" in pkg:
        model = pkg["model"]
        scaler = pkg.get("scaler", None)
        features = pkg.get("features", None)
        st.info("Loaded model package (model + scaler + features).")
    else:
        # legacy: single model saved directly
        model = pkg
        st.info("Loaded plain model.joblib (no scaler/features).")
except Exception as e:
    load_error = str(e)
    st.error("Could not load model.joblib. Make sure it's in the same folder as app.py.")
    st.stop()

# Optional: load dataset for preview and to infer feature names if needed
example = None
try:
    example = pd.read_csv("nba_logreg.csv")
    st.subheader("Dataset preview")
    st.write(example.head())
    example = example.drop(columns=["Name"], errors="ignore")
    example = example.fillna(example.mean(numeric_only=True))
except Exception:
    example = None

# If features not provided by the model package, try to infer from the dataset
if features is None:
    if example is not None:
        features = [c for c in example.columns if c != "TARGET_5Yrs"]
        st.warning("No features saved with model — using columns from nba_logreg.csv as inputs (may mismatch trained features).")
    else:
        st.error("No feature list found in model and no nba_logreg.csv available. Please upload dataset or re-save model with features.")
        st.stop()

# Show which features the app will ask for
st.subheader("Model input features")
st.write(features)

# Build input form using the model's feature list
st.subheader("Enter player stats (these must match the features above)")
user_input = {}
with st.form("input_form"):
    for col in features:
        default = 0.0
        if example is not None and col in example.columns:
            try:
                default = float(example[col].mean())
            except Exception:
                default = 0.0
        user_input[col] = st.number_input(col, value=default, format="%.4f")
    submitted = st.form_submit_button("Predict")

if submitted:
    X_new = pd.DataFrame([user_input], columns=features)  # preserve order
    # If scaler available, apply it
    if scaler is not None:
        try:
            # scaler might be a fitted sklearn scaler with transform method
            X_scaled = scaler.transform(X_new)
        except Exception as e:
            st.error(f"Error scaling inputs: {e}")
            st.stop()
        X_for_model = X_scaled
    else:
        # no scaler — attempt to use raw inputs (not ideal)
        X_for_model = X_new.values

    try:
        pred = model.predict(X_for_model)
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X_for_model)[:, 1]
            except Exception:
                prob = None
        res_text = "Likely 5-Year Career" if int(pred[0]) == 1 else "Not Likely"
        st.success(f"Prediction: **{res_text}**")
        if prob is not None:
            st.write(f"Probability: {prob[0]:.3f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
