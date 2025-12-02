import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("NBA 5-Year Career Prediction (Logistic Regression)")

# -----------------------
# Load model (plain or package)
# -----------------------
try:
    pkg = joblib.load("model.joblib")
except Exception as e:
    st.error("Could not load model.joblib. Make sure it's in the same folder as app.py.")
    st.stop()

# If pkg is a dict with keys, use them; otherwise treat as plain model
if isinstance(pkg, dict) and "model" in pkg:
    model = pkg["model"]
    scaler = pkg.get("scaler", None)
    saved_features = pkg.get("features", None)
else:
    model = pkg
    scaler = None
    saved_features = None

# -----------------------
# Load dataset (optional) for defaults and to infer features
# -----------------------
example = None
example_defaults = pd.Series(dtype=float)
try:
    example = pd.read_csv("nba_logreg.csv")
    st.subheader("Dataset preview")
    st.write(example.head())  # small preview only
    example = example.drop(columns=["Name"], errors="ignore")
    example = example.fillna(example.mean(numeric_only=True))
    example_defaults = example.select_dtypes(include=[np.number]).mean(numeric_only=True)
except Exception:
    example = None

# -----------------------
# Decide which features to ask for
# Priority: saved_features (from model package) -> infer from dataset -> error
# -----------------------
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
    return [str(obj)]

features = flatten_features(saved_features)
if (not features or len(features) == 0) and example is not None:
    # infer numeric columns except target
    features = [c for c in example.select_dtypes(include=[np.number]).columns if c != "TARGET_5Yrs"]

if not features or len(features) == 0:
    st.error("No feature list available. Upload nba_logreg.csv or re-save model.joblib including a 'features' list.")
    st.stop()

# dedupe while keeping order
seen = set(); cleaned = []
for f in features:
    if f not in seen:
        cleaned.append(f); seen.add(f)
features = cleaned

# -----------------------
# Show inputs (simple)
# -----------------------
st.subheader("Enter player stats")
user_input = {}
for col in features:
    default = float(example_defaults[col]) if (not example_defaults.empty and col in example_defaults.index) else 0.0
    user_input[col] = st.number_input(col, value=default, format="%.4f")

# -----------------------
# Predict button & logic
# -----------------------
if st.button("Predict"):
    # Build DataFrame in exact feature order
    try:
        X_new = pd.DataFrame([[user_input[f] for f in features]], columns=features)
    except Exception as e:
        st.error(f"Could not build input DataFrame: {e}")
        st.stop()

    # If scaler exists, align columns to scaler.feature_names_in_ (if available)
    if scaler is not None:
        scaler_cols = None
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = list(getattr(scaler, "feature_names_in_"))
        elif hasattr(scaler, "get_feature_names_out"):
            try:
                scaler_cols = list(scaler.get_feature_names_out())
            except Exception:
                scaler_cols = None

        if scaler_cols:
            missing = [c for c in scaler_cols if c not in X_new.columns]
            unseen = [c for c in X_new.columns if c not in scaler_cols]
            if missing:
                st.warning(f"Filling {len(missing)} missing features with 0 (first few): {missing[:5]}")
            if unseen:
                st.warning(f"Ignoring {len(unseen)} input features not seen by scaler (first few): {unseen[:5]}")

            # reindex to scaler order and fill missing with zeros
            X_aligned = X_new.reindex(columns=scaler_cols, fill_value=0.0)
            try:
                X_for_model = scaler.transform(X_aligned)
            except Exception as e:
                st.error(f"Scaler transform failed after alignment: {e}")
                st.stop()
        else:
            # scaler exists but we don't know expected cols: try transform (may fail)
            try:
                X_for_model = scaler.transform(X_new)
            except Exception as e:
                st.error(f"Scaler transform failed: {e}")
                st.stop()
    else:
        X_for_model = X_new.values

    # Predict
    try:
        pred = model.predict(X_for_model)[0]
        prob = None
        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X_for_model)[:, 1][0]
            except Exception:
                prob = None

        result = "Likely 5-Year Career" if int(pred) == 1 else "Not Likely"
        st.success(f"Prediction: **{result}**")
        if prob is not None:
            st.info(f"Predicted probability: {prob:.3f}")
    except Exception as e:
        st.error(f"Prediction error: {e}")


