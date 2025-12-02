# app.py — Clean, compact, user-friendly Streamlit UI
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
    st.error("Could not load 'model.joblib'. Ensure the file is in the repo root and named exactly 'model.joblib'.")
    st.stop()

if isinstance(pkg, dict) and "model" in pkg:
    model = pkg["model"]
    scaler = pkg.get("scaler", None)
    features = pkg.get("features", None)
else:
    model = pkg
    scaler = None
    features = None

# -------------------------
# Optional: load dataset just for defaults
# -------------------------
example_defaults = pd.Series(dtype=float)
try:
    df_example = pd.read_csv("nba_logreg.csv").drop(columns=["Name"], errors="ignore")
    example_defaults = df_example.select_dtypes(include=[np.number]).mean(numeric_only=True)
except Exception:
    # no dataset available — defaults will be 0.0
    pass

# -------------------------
# Normalize features to simple list
# -------------------------
def flatten_features(obj):
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
    return [str(obj)]

if features is not None:
    features = flatten_features(features)

if features is None:
    if not example_defaults.empty:
        features = [c for c in example_defaults.index if c != "TARGET_5Yrs"]
    else:
        st.error("No feature list available in model and no dataset to infer features. Re-save model with a 'features' list or upload 'nba_logreg.csv'.")
        st.stop()

# Deduplicate preserving order
seen = set()
ordered = []
for f in features:
    fname = str(f)
    if fname not in seen:
        ordered.append(fname)
        seen.add(fname)
features = ordered

if len(features) == 0:
    st.error("Model feature list is empty. Re-save model with feature names.")
    st.stop()

# -------------------------
# Build form: two-column layout for readability
# -------------------------
st.markdown("Enter player stats below. Leave fields at defaults if unsure.")
user_input = {}
cols = st.columns(2)  # two columns layout

with st.form("predict_form"):
    for i, colname in enumerate(features):
        default = 0.0
        if colname in example_defaults.index:
            try:
                default = float(example_defaults[colname])
            except Exception:
                default = 0.0
        # choose left or right column
        col = cols[i % 2]
        user_input[colname] = col.number_input(colname, value=default, format="%.4f")
    submit = st.form_submit_button("Predict")

# -------------------------
# Predict and display results
# -------------------------
if submit:
    try:
        # build DataFrame in exact feature order
        row = [[ user_input[f] for f in features ]]
        X_new = pd.DataFrame(row, columns=features)

        # scale if possible
        if scaler is not None:
            try:
                X_for_model = scaler.transform(X_new)
            except Exception as e:
                st.error(f"Scaler transform failed: {e}")
                st.stop()
        else:
            X_for_model = X_new.values

        # run prediction
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
