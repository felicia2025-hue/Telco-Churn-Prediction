import os, re, io, json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt  # for the chart
import os
from pathlib import Path

st.set_page_config(page_title="Churn Prediction", layout="wide")
st.title("📉 Telco - Churn Prediction")

# (Optional) page-level intro – boleh dibiarkan
st.markdown(
    """
This app lets you **predict churn** in a Telco Company using logistic regression model (Top-10 features)""")

# ====================
# Config
# ====================
# Show env (to verify runtime takes effect)
# --- MUST be at the very top (before joblib.load) ---

import os, sys
from pathlib import Path
import streamlit as st

# joblib (fallback ke pickle kalau joblib tidak ada)
try:
    import joblib
except Exception:
    import pickle as joblib

# PATH model: RELATIF (repo flat)
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "logreg_top10_tuned.pkl"))
DEFAULT_INFO_PATH  = os.getenv("INFO_PATH",  str(BASE_DIR / "logreg_top10_tuned_info.json"))


THRESHOLD = 0.50  # fixed (no UI)

# Ranges for manual inputs
MONTHLY_MIN, MONTHLY_MAX = 0.00, 150.00       # monthly_charges
REF_MIN, REF_MAX         = 0, 15              # number_of_referrals

# ====================
# Helpers
# ====================
def norm_name(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"__+", "_", s)
    s = s.replace("monthly__charges", "monthly_charges")
    return s

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(DEFAULT_MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {DEFAULT_MODEL_PATH}")
    return joblib.load(DEFAULT_MODEL_PATH)

@st.cache_data(show_spinner=False)
def load_info():
    if os.path.exists(DEFAULT_INFO_PATH):
        with open(DEFAULT_INFO_PATH) as f:
            return json.load(f)
    return {}

def infer_schema(pipeline, info: dict):
    pre: ColumnTransformer = pipeline.named_steps.get("preprocess")
    if pre is None:
        raise ValueError("Pipeline must have a 'preprocess' step (ColumnTransformer).")

    num_sel = info.get("num_sel")
    cat_sel = info.get("cat_sel")
    selected_vars = info.get("selected_vars")

    if num_sel is None:
        try:
            num_sel = pre.named_transformers_["num"].feature_names_in_.tolist()
        except Exception:
            num_sel = pre.transformers_[0][2] if len(pre.transformers_) > 0 else []
    if cat_sel is None:
        try:
            cat_sel = pre.named_transformers_["cat"].feature_names_in_.tolist()
        except Exception:
            cat_sel = pre.transformers_[1][2] if len(pre.transformers_) > 1 else []

    if selected_vars is None:
        selected_vars = list(num_sel) + list(cat_sel)

    categories = {}
    try:
        try:
            ohe: OneHotEncoder = pre.named_transformers_["cat"]
        except Exception:
            ohe = pre.transformers_[1][1]
        for feat, cats in zip(cat_sel, getattr(ohe, "categories_", [])):
            categories[feat] = [str(c) for c in cats]
    except Exception:
        for feat in cat_sel:
            categories[feat] = []

    return {
        "selected_vars": list(selected_vars),
        "num_sel": list(num_sel),
        "cat_sel": list(cat_sel),
        "categories": categories,
    }

# ====================
# Load model & schema
# ====================
try:
    model = load_model()
    info  = load_info()
    schema = infer_schema(model, info)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

selected_vars = schema["selected_vars"]
num_sel = schema["num_sel"]
cat_sel = schema["cat_sel"]
categories = schema["categories"]

EXPECTED_NORM_MAP = {norm_name(c): c for c in selected_vars}
ALIASES = {
    "number_of_refferals": "number_of_referrals",
    "monthly__charges": "monthly_charges",
}

# Keep state so left chart persists
if "has_pred" not in st.session_state:
    st.session_state.has_pred = False
if "last_row" not in st.session_state:
    st.session_state.last_row = None

# ====================
# Tabs: Manual | Batch
# ====================
tab_manual, tab_batch = st.tabs(["🧮 Manual Prediction", "📥 Batch CSV Prediction"])

# ---------- Manual ----------
with tab_manual:
    st.subheader("Manual Prediction — Set the values for required features")
    st.caption("Output variable (target): **churn_value**. Fill every field below, then click **Predict**.")

    # Permanent two-column layout
    left_col, right_col = st.columns([1.25, 1])

    # >>> LEFT: requested text (selalu tampil di kiri)
    with left_col:
        st.markdown(
            """
            Set the inputs on the right and click **Predict**. After predicting, the left panel shows a *what-if* curve: how churn probability changes when you vary one feature (others held constant).
            """
        )

    # RIGHT: the form
    with right_col:
        with st.form("manual_form"):
            inputs = {}

            for col in num_sel:
                key = col.lower()
                if key == "satisfaction_score":
                    inputs[col] = st.select_slider(col, options=[1, 2, 3, 4, 5], value=3)
                elif key in {"number_of_referrals", "number_of_refferals"}:
                    inputs[col] = st.slider(col, min_value=REF_MIN, max_value=REF_MAX, value=2, step=1)
                elif key in {"monthly_charges", "monthly_ charges"}:
                    inputs[col] = st.number_input(
                        col,
                        min_value=float(MONTHLY_MIN),
                        max_value=float(MONTHLY_MAX),
                        value=65.0,
                        step=0.05,
                        help="Type a number between 0 and 150",
                    )
                else:
                    inputs[col] = st.number_input(col, value=0.0)

            for col in cat_sel:
                opts = categories.get(col, [])
                if not opts:
                    val = st.text_input(col, value="Unknown")
                else:
                    val = st.selectbox(col, options=[str(o) for o in opts], index=0)
                inputs[col] = val

            submit = st.form_submit_button("Predict")

    # handle submit
    if submit:
        row = {c: inputs.get(c, np.nan) for c in selected_vars}
        X_one = pd.DataFrame([row])
        missing = [c for c in selected_vars if c not in X_one.columns]
        if missing:
            st.error(f"Missing columns in form payload: {missing}")
        else:
            st.session_state.last_row = row
            st.session_state.has_pred = True

    # metrics + chart after first prediction
    if st.session_state.has_pred and st.session_state.last_row is not None:
        row = st.session_state.last_row
        X_one = pd.DataFrame([row])
        prob = float(model.predict_proba(X_one[selected_vars])[:, 1][0])
        pred = int(prob >= THRESHOLD)

        with right_col:
            st.metric("Churn probability", f"{prob:.3f}")
            st.write(f"Predicted label (>= {THRESHOLD:.2f}): **{pred}**")
            st.progress(min(max(prob, 0.0), 1.0))

        with left_col:
            # What-if curve controls & plot (appended below the text)
            st.markdown("**What-if sensitivity — Predicted vs chosen input**")
            st.caption("Pick one numeric feature; others are held constant at the values from your form.")

            preferred = ["monthly_charges", "number_of_referrals", "satisfaction_score", "monthly_ charges"]
            numeric_candidates = [c for c in preferred if c in num_sel]
            if not numeric_candidates and num_sel:
                numeric_candidates = list(num_sel)

            if not numeric_candidates:
                st.info("No numeric feature available for sensitivity chart.")
            else:
                chosen = st.selectbox("Feature to vary", options=numeric_candidates, index=0)

                if chosen in {"monthly_charges", "monthly_ charges"}:
                    grid = np.linspace(float(MONTHLY_MIN), float(MONTHLY_MAX), 200)
                elif chosen == "number_of_referrals":
                    grid = np.arange(REF_MIN, REF_MAX + 1)
                elif chosen == "satisfaction_score":
                    grid = np.arange(1, 6)
                else:
                    curr = pd.to_numeric(X_one[chosen].iloc[0], errors="coerce")
                    curr = float(curr if pd.notna(curr) else 0.0)
                    grid = np.linspace(curr - 3, curr + 3, 100)

                X_grid = pd.DataFrame([row] * len(grid))
                X_grid[chosen] = grid
                probs_grid = model.predict_proba(X_grid[selected_vars])[:, 1]

                fig, ax = plt.subplots()
                ax.plot(grid, probs_grid, linewidth=2)
                try:
                    curr_val = float(pd.to_numeric(X_one[chosen].iloc[0], errors="coerce"))
                except Exception:
                    curr_val = grid[len(grid)//2]
                ax.axvline(curr_val, linestyle="--")
                ax.scatter([curr_val], [prob], zorder=3)
                ax.set_xlabel(chosen)
                ax.set_ylabel("Predicted churn probability")
                ax.set_title("What-if curve (others held constant)")
                st.pyplot(fig)
    else:
        with left_col:
            st.info("Submit the form on the right to see the what-if chart here.")

# ---------- Batch ----------
with tab_batch:
    st.subheader("Batch CSV Prediction — upload your file")
    # >>> italic line right under the subheader (as requested)
    st.markdown("*Upload a CSV to score many customers at once.*")

    st.write("Required columns :")
    st.code(", ".join(selected_vars))

    templ = pd.DataFrame(columns=selected_vars)
    st.download_button(
        "⬇️ Download CSV template",
        data=templ.to_csv(index=False).encode("utf-8"),
        file_name="template_required_columns.csv",
        mime="text/csv",
    )

    csv_pred = st.file_uploader("Upload CSV to score", type=["csv"], key="predcsv")

    if st.button("Run batch prediction"):
        if csv_pred is None:
            st.warning("Please upload a CSV first.")
        else:
            try:
                raw = csv_pred.read()
                text = raw.decode("utf-8", errors="ignore")
                dfp = pd.read_csv(io.StringIO(text), sep=None, engine="python")
                if len(dfp.columns) == 1 or dfp.columns[0].strip().lower() in {
                    "template_required_columns", "sheet1", "sheet 1"
                }:
                    dfp = pd.read_csv(io.StringIO(text), sep=None, engine="python", header=1)

                rename = {}
                for col in dfp.columns:
                    n = norm_name(col)
                    n = ALIASES.get(n, n)
                    if n in EXPECTED_NORM_MAP:
                        rename[col] = EXPECTED_NORM_MAP[n]
                dfp = dfp.rename(columns=rename)

                missing = [c for c in selected_vars if c not in dfp.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                    st.stop()

                req = dfp[selected_vars].copy()
                req_stripped = req.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)
                all_empty = req_stripped.isna() | (req_stripped == "")
                dfp = dfp[~all_empty.all(axis=1)]

                before = len(dfp)
                for c in num_sel:
                    if c in dfp.columns:
                        dfp[c] = pd.to_numeric(dfp[c], errors="coerce")
                if num_sel:
                    dfp = dfp.dropna(subset=[c for c in num_sel if c in dfp.columns])
                dropped = before - len(dfp)

                if len(dfp) == 0:
                    st.error(
                        "No valid rows found after cleaning. "
                        "Ensure at least one row has all required fields filled and numeric values are valid."
                    )
                    st.stop()

                probs = model.predict_proba(dfp[selected_vars])[:, 1]
                labels = (probs >= THRESHOLD).astype(int)
                out = dfp.copy()
                out["churn_prob"] = probs
                out["churn_pred"] = labels

                if dropped > 0:
                    st.info(f"Dropped {dropped} row(s) with empty/invalid numeric values.")

                st.success("Prediction completed. Preview:")
                st.dataframe(out.head(50), use_container_width=True)
                st.download_button(
                    "⬇️ Download predictions CSV",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name="churn_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
