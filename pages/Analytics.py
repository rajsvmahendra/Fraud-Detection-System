import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import utils

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Analytics - Fraud Dashboard", layout="wide")

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

render_nav("Analytics")

# ---------------- PAGE TITLE ----------------
st.markdown("<div class='page-title'>Analytics & Exploratory Data Analysis</div>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = utils.load_data()

# ---------------- RAW DATA ----------------
with st.expander("Raw Dataset (sample data set)"):
    st.write(
        "This table shows a sample of the raw dataset. "
        "Each row represents one credit card transaction. "
        "`Class = 0` means legitimate transaction and `Class = 1` means fraud."
    )
    st.dataframe(df.sample(300), use_container_width=True)

# =====================================================
# OBJECTIVE 1 — HOW FREQUENT IS FRAUD?
# =====================================================
st.markdown("## 1️⃣ How frequent is fraud?")

st.write(
    "**Definition:** This objective helps us understand how many fraud transactions "
    "exist compared to legitimate transactions."
)

st.write(
    "**What this chart shows:** The bar chart counts the number of transactions "
    "in each class (Legitimate vs Fraud)."
)

fig, ax = plt.subplots()
sns.countplot(x="Class", data=df, ax=ax)
ax.set_xticklabels(["Legitimate", "Fraud"])
ax.set_title("Number of Legitimate vs Fraud Transactions")
st.pyplot(fig)

st.write(
    "**What we observe:** Fraud transactions are extremely rare compared to legitimate ones. "
    "This means the dataset is highly imbalanced."
)

st.divider()

# =====================================================
# OBJECTIVE 2 — TRANSACTION AMOUNT COMPARISON
# =====================================================
st.markdown("## 2️⃣ Transaction Amount Comparison")

st.write(
    "**Definition:** This objective analyzes whether fraud transactions involve "
    "larger amounts than legitimate transactions."
)

st.write(
    "**What this chart shows:** The histogram compares transaction amounts for "
    "legitimate and fraud transactions."
)

fig, ax = plt.subplots()
ax.hist(df[df["Class"] == 0]["Amount"], bins=50, alpha=0.6, label="Legitimate")
ax.hist(df[df["Class"] == 1]["Amount"], bins=50, alpha=0.6, label="Fraud")
ax.set_title("Transaction Amount Distribution")
ax.legend()
st.pyplot(fig)

st.write(
    "**What we observe:** Fraud transactions are not always high-value. "
    "Many frauds occur at smaller amounts to avoid detection."
)

st.divider()

# =====================================================
# OBJECTIVE 3 — FRAUD OCCURRENCE OVER TIME
# =====================================================
st.markdown("## 3️⃣ Fraud Occurrence Over Time")

st.write(
    "**Definition:** This objective studies how fraud transactions are distributed "
    "over time."
)

st.write(
    "**What this chart shows:** The line chart displays the number of fraud "
    "transactions occurring at different time points."
)

fraud_time = df[df["Class"] == 1].groupby("Time").size()

fig, ax = plt.subplots()
ax.plot(fraud_time.index, fraud_time.values)
ax.set_title("Fraud Transactions Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Fraud Count")
st.pyplot(fig)

st.write(
    "**What we observe:** Fraud does not occur randomly. "
    "It appears in clusters, which suggests planned or automated fraudulent activity."
)

st.divider()

# =====================================================
# OBJECTIVE 4 — FEATURE BEHAVIOR COMPARISON
# =====================================================
st.markdown("## 4️⃣ Feature Behavior Comparison")

st.write(
    "**Definition:** This objective compares how a numerical feature behaves "
    "for legitimate and fraud transactions."
)

st.write(
    "**What this chart shows:** The boxplot compares the distribution of feature V1 "
    "for both transaction classes."
)

fig, ax = plt.subplots()
sns.boxplot(x="Class", y="V1", data=df, ax=ax)
ax.set_xticklabels(["Legitimate", "Fraud"])
ax.set_title("Feature V1 Distribution by Class")
st.pyplot(fig)

st.write(
    "**What we observe:** Although the distributions are different, they overlap. "
    "This means simple rules cannot separate fraud clearly."
)

st.divider()

# =====================================================
# OBJECTIVE 5 — AVERAGE FEATURE COMPARISON
# =====================================================
st.markdown("## 5️⃣ Average Feature Comparison")

st.write(
    "**Definition:** This objective compares the average value of a feature "
    "for legitimate and fraud transactions."
)

st.write(
    "**What this chart shows:** The bar chart displays the mean (average) value "
    "of feature V1 for both classes."
)

mean_values = df.groupby("Class")["V1"].mean().reset_index()

fig, ax = plt.subplots()
sns.barplot(x="Class", y="V1", data=mean_values, ax=ax)
ax.set_xticklabels(["Legitimate", "Fraud"])
ax.set_title("Average Value of Feature V1 by Class")
st.pyplot(fig)

st.write(
    "**What we observe:** The average feature values are different for fraud "
    "and legitimate transactions, but averages alone are not enough for detection."
)

st.divider()

# =====================================================
# FINAL SUMMARY
# =====================================================
st.markdown("## ✅ What We Learned from Analytics")

st.markdown("""
- Fraud is **rare**, creating class imbalance  
- Fraud transactions are **not always high-value**  
- Fraud shows **patterns over time**  
- Feature values **overlap between classes**  
- Average values differ, but **machine learning is required**
""")
