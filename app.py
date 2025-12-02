# app.py - Minimal, stable Streamlit app
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="NBA 5-Year Career Predictor", layout="wide")
st.title("🏀 NBA 5-Year Career Predictor")

# Load model (plain model or package)
try:
    pkg = joblib.load("model.joblib")
except Exception as e:
    st.error("Could not load 'model.joblib'. Make sure it's in the repo root and named exactly 'model.joblib'.")
    st.stop()

if isinstance(pkg, dict) and "model" in pkg:
    model = pkg["model"]
    scaler = pkg.get("scaler", None)
    features = pkg.get("features", None)
else:
    model = pkg
    scaler = None
    features = None

# Load dataset for defaults if available
example_defaults = pd.Series(dtype=float)
try:
    df_example = pd.read_csv("nba_logreg.csv").drop(columns=["Name"], errors="ignore")
    example_defaults = df_example.select_dtypes(include=[np.number]).mean(numeric_only=True)
except Exception:
    pass

# If no features in package, infer numeric columns from dataset
if not features:
    if not example_defaults.empty:
        features = [c for c in example_defaults.index if c != "TARGET_5Yrs"]
    else:
        st.error("No features found in model and no dataset to infer them from. Upload 'nba_logreg.csv' or re-save the model with features.")
        st.stop()

# Normalize feature list
features = [str(f) for f in (features if isinstance(features, (list,tuple,np.ndarray,pd.Index)) else [features])]

# Remove duplicates while preserving order
seen = set(); ordered = []
for f in features:
    if f not in seen:
        ordered.append(f); seen.add(f)
features = ordered

st.write("Enter player stats below (use defaults or your own values).")

# Two-column layout for inputs
user_input = {}
cols = st.columns(2)
with st.form("predict_form"):
    for i, colname in enumerate(features):
        default = 0.0
        if colname in example_defaults.index:
            try:
                default = float(example_defaults[colname])
            except Exception:
                default = 0.0
        user_input[colname] = cols[i % 2].number_input(colname, value=default, format="%.4f")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        row = [[user_input[f] for f in features]]
        X_new = pd.DataFrame(row, columns=features)
        if scaler is not None:
            X_for_model = scaler.transform(X_new)
        else:
            X_for_model = X_new.values
        pred = model.predict(X_for_model)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_for_model)[:,1]
        label = "Likely 5-Year Career" if int(pred[0]) == 1 else "Not Likely"
        st.success(label)
        if prob is not None:
            st.info(f"Probability: {prob[0]:.3f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
