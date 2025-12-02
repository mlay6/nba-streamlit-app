# app.py - Clean, simple Streamlit app for your logistic model
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="NBA 5-Year Career Predictor", layout="centered")
st.title("🏀 NBA 5-Year Career Predictor (Logistic Regression)")

# -------------------------
# Load model.joblib (support both plain model and package)
# -------------------------
try:
    pkg = joblib.load("model.joblib")
except Exception as e:
    st.error("Could not load 'model.joblib'. Make sure it's in the same folder as app.py.")
    st.stop()

# Unpack package if it's a dict, otherwise treat as plain model
if isinstance(pkg, dict) and "model" in pkg:
    model = pkg["model"]
    scaler = pkg.get("scaler", None)
    features = pkg.get("features", None)
else:
    model = pkg
    scaler = None
    features = None

# -------------------------
# Optional: load dataset (only to get sensible defaults)
# -------------------------
example_df = None
try:
    example_df = pd.read_csv("nba_logreg.csv")
    example_df = example_df.drop(columns=["Name"], errors="ignore")
    # convert to numeric means only for defaults
    example_defaults = example_df.select_dtypes(include=[np.number]).mean(numeric_only=True)
except Exception:
    example_defaults = pd.Series(dtype=float)

# -------------------------
# Normalize features into a flat list (if present)
# -------------------------
def flatten_features(obj):
    # Accept many shapes: list, np.ndarray, pandas Index, nested lists
    if obj is None:
        return None
    if isinstance(obj, (pd.Index, np.ndarray)):
        return list(np.array(obj).ravel())
    if isinstance(obj, (list, tuple)):
        flat = []
        for item in obj:
            if isinstance(item, (list, tuple, np.ndarray, pd.Index)):
                flat.extend(list(np.array(item).ravel()))
            else:
                flat.append(item)
        return [str(x) for x in flat]
    # fallback
    return [str(obj)]

if features is not None:
    features = flatten_features(features)

# If no features came with model, try to infer from dataset
if features is None:
    if not example_defaults.empty:
        # Use all numeric columns except TARGET_5Yrs
        features = [c for c in example_defaults.index if c != "TARGET_5Yrs"]
    else:
        st.error("No feature list found in the saved model and no `nba_logreg.csv` available to infer features.")
        st.stop()

# Ensure features is a list of unique strings in order
seen = set()
ordered_features = []
for f in features:
    fname = str(f)
    if fname not in seen:
        ordered_features.append(fname)
        seen.add(fname)
features = ordered_features

# -------------------------
# Input form (no feature table shown)
# -------------------------
st.write("Enter player stats (values should match the model's training scale).")
user_input = {}
with st.form("prediction_form"):
    for col in features:
        default = 0.0
        if col in example_defaults.index:
            try:
                default = float(example_defaults[col])
            except Exception:
                default = 0.0
        user_input[col] = st.number_input(col, value=default, format="%.4f")
    submit = st.form_submit_button("Predict")

# -------------------------
# Build input, scale (if needed), predict
# -------------------------
if submit:
    try:
        # Keep ordering exactly as 'features'
        row = [[ user_input[f] for f in features ]]
        X_new = pd.DataFrame(row, columns=features)

        if scaler is not None:
            try:
                X_for_model = scaler.transform(X_new)
            except Exception as e:
                st.error(f"Error applying saved scaler to inputs: {e}")
                st.stop()
        else:
            X_for_model = X_new.values

        # Predict
        pred = model.predict(X_for_model)
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X_for_model)[:, 1]
            except Exception:
                prob = None

        label = "Likely 5-Year Career" if int(pred[0]) == 1 else "Not Likely"
        st.success(f"Prediction: **{label}**")
        if prob is not None:
            st.write(f"Predicted probability: {prob[0]:.3f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
