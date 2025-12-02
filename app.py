# app.py - corrected (robust check for features)
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="NBA 5-Year Career Predictor", layout="wide")
st.title("🏀 NBA 5-Year Career Predictor")

# -------------------------
# Load model package
# -------------------------
try:
    pkg = joblib.load("model.joblib")
except Exception:
    st.error("Could not load 'model.joblib'. Ensure it is in the repo root and named exactly 'model.joblib'.")
    st.stop()

# unpack
if isinstance(pkg, dict) and "model" in pkg:
    model = pkg["model"]
    scaler = pkg.get("scaler", None)
    features = pkg.get("features", None)
else:
    model = pkg
    scaler = None
    features = None

# -------------------------
# Optional: load dataset for defaults
# -------------------------
example_defaults = pd.Series(dtype=float)
try:
    df_example = pd.read_csv("nba_logreg.csv").drop(columns=["Name"], errors="ignore")
    example_defaults = df_example.select_dtypes(include=[np.number]).mean(numeric_only=True)
except Exception:
    pass

# -------------------------
# Normalize/validate features safely
# -------------------------
def flatten_features(obj):
    if obj is None:
        return None
    if isinstance(obj, (pd.Index, pd.Series, np.ndarray)):
        return list(np.array(obj).ravel().astype(str))
    if isinstance(obj, (list, tuple)):
        flat = []
        for item in obj:
            if isinstance(item, (list, tuple, np.ndarray, pd.Index, pd.Series)):
                flat.extend(list(np.array(item).ravel().astype(str)))
            else:
                flat.append(str(item))
        return flat
    # single value fallback
    return [str(obj)]

features_list = flatten_features(features)

# If features_list is None or empty, try to infer from dataset
if features_list is None or len(features_list) == 0:
    if not example_defaults.empty:
        # infer numeric columns except target
        inferred = [c for c in example_defaults.index if c != "TARGET_5Yrs"]
        features_list = inferred
    else:
        st.error("No feature list found in model and no 'nba_logreg.csv' to infer features from. Please re-save model with features or upload the dataset.")
        st.stop()

# Deduplicate preserving order
seen = set()
final_features = []
for f in features_list:
    fname = str(f)
    if fname not in seen:
        final_features.append(fname)
        seen.add(fname)
features = final_features

if len(features) == 0:
    st.error("Final feature list is empty. Please provide valid features.")
    st.stop()

# -------------------------
# Build input form (two columns for readability)
# -------------------------
st.markdown("Enter player stats below. Leave defaults if unsure.")
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

# -------------------------
# Predict
# -------------------------
if submitted:
    try:
        # build DataFrame exactly in feature order
        row = [[ user_input[f] for f in features ]]
        X_new = pd.DataFrame(row, columns=features)

        if scaler is not None:
            try:
                X_for_model = scaler.transform(X_new)
            except Exception as e:
                st.error(f"Scaler transform failed: {e}")
                st.stop()
        else:
            X_for_model = X_new.values

        pred = model.predict(X_for_model)
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X_for_model)[:, 1]
            except Exception:
                prob = None

        label = "✅ Likely 5-Year Career" if int(pred[0]) == 1 else "❌ Not Likely"
        st.success(label)
        if prob is not None:
            st.info(f"Predicted probability: {prob[0]:.3f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
