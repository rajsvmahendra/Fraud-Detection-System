import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import utils

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Performance - Fraud Dashboard", layout="wide")

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

render_nav("Performance")

# ---------------- PAGE TITLE ----------------
st.markdown("<div class='page-title'>Model Performance Metrics</div>", unsafe_allow_html=True)

# ---------------- LOAD DATA & MODEL ----------------
df = utils.load_data()
model = utils.load_model()

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

y_pred = model.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)

accuracy = report["accuracy"]
precision = report["1"]["precision"]
recall = report["1"]["recall"]
f1 = report["1"]["f1-score"]

# =====================================================
# MODEL PERFORMANCE METRICS
# =====================================================
st.markdown("## 📊 Model Performance Metrics")

st.write(
    "**Definition:** Model performance metrics help us evaluate how well the trained "
    "model detects fraud and legitimate transactions."
)

st.write(
    "**Important Note:** Because fraud is rare, accuracy alone is not sufficient. "
    "We focus more on Precision, Recall, and F1-score for fraud detection."
)

cols = st.columns(4)
values = [
    ("Accuracy", accuracy),
    ("Precision (Fraud)", precision),
    ("Recall (Fraud)", recall),
    ("F1 Score (Fraud)", f1),
]

for col, (label, value) in zip(cols, values):
    with col:
        st.markdown(
            f"""
            <div class='glass-card'>
                <div class='chart-title'>{label}</div>
                <h2 style='color:#D4AF37;margin:0;'>{value:.2%}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- METRIC EXPLANATIONS ----------------
st.divider()

st.markdown("## 🧠 Understanding the Metrics")

st.write(
    "**Accuracy:** This tells us the overall percentage of correct predictions. "
    "The model performs well overall, but accuracy alone can be misleading due to class imbalance."
)

st.write(
    "**Precision (Fraud):** Precision tells us, out of all transactions predicted as fraud, "
    "how many were actually fraud. High precision means fewer false fraud alerts."
)

st.write(
    "**Recall (Fraud):** Recall tells us how many actual fraud transactions were correctly detected. "
    "This is very important because missing fraud is costly."
)

st.write(
    "**F1 Score (Fraud):** The F1 score balances precision and recall. "
    "It gives a single measure of fraud detection performance."
)

# =====================================================
# CONFUSION MATRIX
# =====================================================
st.divider()

st.markdown("## 🔍 Confusion Matrix")

st.write(
    "**Definition:** The confusion matrix shows how many predictions were correct "
    "and where the model made mistakes."
)

st.write(
    "**What this chart shows:** It compares actual transaction classes "
    "with the model’s predicted classes."
)

fig, ax = plt.subplots()
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_pred),
    display_labels=["Legitimate", "Fraud"]
).plot(ax=ax, colorbar=False)

st.pyplot(fig)

st.write(
    "**What we observe:** Most legitimate transactions are correctly classified. "
    "The model also successfully detects most fraud cases, with a small number of false positives and false negatives."
)

# =====================================================
# FINAL CONCLUSION
# =====================================================
st.divider()

st.markdown("## ✅ Model Performance Summary")

st.markdown("""
- The model performs **very well overall**
- High **precision** means fewer false fraud alerts
- High **recall** means most frauds are detected
- F1 score confirms a **good balance** between precision and recall
- The confusion matrix supports these results visually
""")
