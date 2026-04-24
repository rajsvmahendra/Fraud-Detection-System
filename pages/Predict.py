import streamlit as st
import pandas as pd
import joblib
import utils

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Prediction - Fraud Dashboard", layout="wide")

with open("styles/theme.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------- NAV BAR ----------------
def render_nav(active: str):
    st.markdown("<div class='nav-row'>", unsafe_allow_html=True)
    cols = st.columns(5)
    labels = ["Home", "Overview", "Prediction", "Analytics", "Performance"]
    targets = [
        "app.py",
        "pages/Overview.py",
        "pages/Predict.py",
        "pages/Analytics.py",
        "pages/Model_Performance.py",
    ]

    for col, label, target in zip(cols, labels, targets):
        with col:
            if label == active:
                st.markdown(f"<div class='nav-pill-active'>{label}</div>", unsafe_allow_html=True)
            else:
                if st.button(label, key=f"nav_{label}"):
                    st.switch_page(target if label != "Home" else "app.py")

render_nav("Prediction")

# ---------------- PAGE TITLE ----------------
st.markdown("<div class='page-title'>Predict Transaction Fraud</div>", unsafe_allow_html=True)

# ---------------- LOAD DATA & MODEL ----------------
df = utils.load_data()
model = utils.load_model()

features = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# =====================================================
# PAGE EXPLANATION
# =====================================================
st.write(
    "This page allows you to select a real transaction from the dataset "
    "and run fraud prediction without manually entering feature values."
)

# =====================================================
# SELECT TRANSACTION
# =====================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔴 Fraud Transactions (Class = 1)")
    fraud_indices = df[df["Class"] == 1].index.tolist()
    selected_fraud = st.selectbox(
        "Select a Fraud Transaction (Row Number)",
        fraud_indices
    )

with col2:
    st.markdown("### 🟢 Legitimate Transactions (Class = 0)")
    legit_indices = df[df["Class"] == 0].index.tolist()
    selected_legit = st.selectbox(
        "Select a Legitimate Transaction (Row Number)",
        legit_indices[:5000]  # limit for UI performance
    )

# =====================================================
# CHOOSE WHICH TRANSACTION TO PREDICT
# =====================================================
st.divider()

st.markdown("### 📌 Selected Transaction Details")

select_type = st.radio(
    "Choose Transaction Type to Predict",
    ["Fraud", "Legitimate"],
    horizontal=True
)

if select_type == "Fraud":
    row_idx = selected_fraud
else:
    row_idx = selected_legit

selected_row = df.loc[row_idx, features]

st.write(f"**Selected Dataset Row Number:** {row_idx}")

st.dataframe(
    selected_row.to_frame(name="Value"),
    use_container_width=True
)

# =====================================================
# RUN PREDICTION (✅ FIX IS HERE)
# =====================================================
st.divider()

if st.button("Run Prediction"):

    # Prepare input for model
    input_df = pd.DataFrame([selected_row.values], columns=features)

    # 🔑 IMPORTANT FIX: use probability, not model.predict()
    fraud_probability = model.predict_proba(input_df)[0][1]

    # Custom threshold for fraud detection
    THRESHOLD = 0.25  # realistic for imbalanced fraud data

    if fraud_probability >= THRESHOLD:
        st.error(f"🚨 Fraud Detected — Fraud Probability: {fraud_probability:.2%}")
    else:
        st.success(f"✅ Legitimate Transaction — Fraud Probability: {fraud_probability:.2%}")

    # Show actual class for verification
    actual_class = df.loc[row_idx, "Class"]
    st.write(
        f"**Actual Class in Dataset:** {'Fraud' if actual_class == 1 else 'Legitimate'}"
    )

    # Explain decision (viva-friendly)
    st.write(
        f"""
        **Model Decision Explanation:**
        - Fraud Probability: {fraud_probability:.2%}
        - Decision Threshold: {THRESHOLD:.0%}
        """
    )
