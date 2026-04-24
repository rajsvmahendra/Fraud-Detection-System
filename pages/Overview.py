import streamlit as st
import pandas as pd
import plotly.express as px
import utils

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Overview - Fraud Dashboard", layout="wide")

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

    st.markdown("</div>", unsafe_allow_html=True)

render_nav("Overview")

# ---------------- PAGE TITLE ----------------
st.markdown("<div class='page-title'>Dataset Overview & Monitoring</div>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = utils.load_data()

# =====================================================
# PAGE EXPLANATION
# =====================================================
st.write(
    "This page provides a high-level overview of the credit card fraud dataset. "
    "All values and visualizations shown here are calculated directly from the dataset."
)

# =====================================================
# KEY METRICS (REAL VALUES)
# =====================================================
total_txns = len(df)
fraud_txns = int(df["Class"].sum())
legit_txns = total_txns - fraud_txns
fraud_rate = (fraud_txns / total_txns) * 100
avg_amount = df["Amount"].mean()

c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("Total Transactions", f"{total_txns:,}"),
    ("Fraud Transactions", f"{fraud_txns:,}"),
    ("Fraud Rate (%)", f"{fraud_rate:.3f}%"),
    ("Avg Transaction ($)", f"{avg_amount:,.2f}"),
]

for col, (label, value) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(
            f"""
            <div class='glass-card'>
                <div class='chart-title'>{label}</div>
                <h2 style='color:#D4AF37;margin:0;'>{value}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# SAMPLE TRANSACTIONS TABLE (REAL DATA)
# =====================================================
st.markdown("<div class='chart-title'>Sample Transactions from Dataset</div>", unsafe_allow_html=True)

st.write(
    "This table displays a random sample of real transactions from the dataset, "
    "including both legitimate and fraudulent cases."
)

st.dataframe(df.sample(200, random_state=42), use_container_width=True)

# =====================================================
# FRAUD VS LEGITIMATE DISTRIBUTION (FIXED)
# =====================================================
st.markdown("<div class='chart-title'>Fraud vs Legitimate Distribution</div>", unsafe_allow_html=True)

class_counts = df["Class"].value_counts().reset_index()
class_counts.columns = ["Class", "Count"]

class_counts["Class"] = class_counts["Class"].map({
    0: "Legitimate",
    1: "Fraud"
})

fig = px.pie(
    class_counts,
    names="Class",
    values="Count",
    color="Class",
    color_discrete_map={
        "Legitimate": "#00C985",
        "Fraud": "#D62839"
    }
)

fig.update_traces(
    textinfo="percent+label",
    pull=[0, 0.15]
)

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f8f8f8")
)

st.plotly_chart(fig, use_container_width=True)

st.write(
    "This chart correctly shows that fraud transactions form a very small portion "
    "of the dataset, confirming the extreme class imbalance problem."
)

# =====================================================
# TRANSACTION AMOUNT OVER TIME (REAL SAMPLE)
# =====================================================
st.markdown("<div class='chart-title'>Transaction Amount Over Time (Sample)</div>", unsafe_allow_html=True)

sample_df = df.sample(3000, random_state=42).sort_values("Time")

fig = px.line(
    sample_df,
    x="Time",
    y="Amount",
    color="Class",
    color_discrete_map={
        0: "#00C985",
        1: "#D62839"
    }
)

fig.update_traces(name="Legitimate", selector=dict(name="0"))
fig.update_traces(name="Fraud", selector=dict(name="1"))

fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#f8f8f8"),
    legend_title_text="Transaction Type"
)

st.plotly_chart(fig, use_container_width=True)

st.write(
    "This line chart shows how transaction amounts vary over time. "
    "Fraud transactions appear sparsely and irregularly compared to legitimate transactions."
)
