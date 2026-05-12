
<!-- ========================================= -->
<!--           FRAUD DETECTION SYSTEM          -->
<!-- ========================================= -->

<p align="center">
  <img src="assets/landing.png" alt="Fraud Detection Dashboard" width="900"/>
</p>

<h1 align="center">💳 Fraud Detection System — Real-Time ML Dashboard</h1>

<p align="center">
  An intelligent machine learning-powered dashboard that detects fraudulent credit card transactions in real time using predictive analytics and interactive visualizations.
</p>

<p align="center">
  <a href="https://fraud-detectionsystem.streamlit.app/" target="_blank"><b>Live Demo</b></a> •
  <a href="https://github.com/rajsvmahendra/fraud-detection-system" target="_blank"><b>GitHub Repository</b></a> •
  <a href="https://www.linkedin.com/in/rajsvmahendra/" target="_blank"><b>LinkedIn</b></a>
</p>

---

# 📌 Project Overview

Financial fraud is rare — but financially devastating.

This project demonstrates how machine learning models can identify suspicious credit card transactions through real-time prediction, exploratory data analysis, and interactive monitoring dashboards.

The application simulates how modern financial institutions monitor transactions and flag potentially fraudulent activities instantly.

---

# 🎯 Objectives

- Detect fraudulent transactions using Machine Learning
- Handle highly imbalanced financial datasets
- Visualize fraud behavior through interactive analytics
- Simulate real-time transaction monitoring
- Evaluate model reliability using industry-standard metrics
- Build a complete end-to-end deployable ML application

---

# 🧠 Machine Learning Model

## 🔹 Algorithm Used
- Logistic Regression

## 🔹 Problem Type
- Binary Classification

## 🔹 Target Classes

| Class | Meaning |
|------|---------|
| 0 | Legitimate Transaction |
| 1 | Fraudulent Transaction |

---

# 📊 Dataset Information

### Kaggle Credit Card Fraud Detection Dataset

| Feature | Value |
|---|---|
| Total Transactions | 284,807 |
| Fraud Cases | 492 |
| Features | 30 Numerical Features |
| Dataset Type | Highly Imbalanced |

### Features Include
- PCA-transformed variables (`V1` → `V28`)
- Transaction Time
- Transaction Amount

---

# ⚙️ Dashboard Features

## 🏠 Landing Page
- Cinematic animated background
- Professional modern UI
- Interactive navigation

---

## 🔍 Real-Time Overview
- Simulated live transaction monitoring
- Fraud vs Legitimate transaction statistics
- Dynamic transaction stream
- Live analytics table

---

## 🧪 Fraud Prediction Module
Users can manually enter transaction values to:
- Predict fraudulent transactions
- View fraud probability score
- Analyze model confidence

---

## 📈 Analytics Dashboard (EDA)

Interactive Exploratory Data Analysis including:

- Class distribution visualization
- Transaction amount analysis
- Time vs Amount trends
- Correlation heatmaps
- Violin plots for key features
- Raw dataset exploration

---

## 📉 Model Performance Dashboard

Performance metrics include:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

> ⚠️ Accuracy alone is misleading in fraud detection because of severe class imbalance.  
> Precision and Recall are prioritized for better fraud identification.

---

# 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core Programming |
| Streamlit | Interactive Web Dashboard |
| Scikit-learn | Machine Learning |
| Pandas & NumPy | Data Processing |
| Plotly | Interactive Charts |
| Seaborn & Matplotlib | Data Visualization |
| VS Code | Development Environment |

---

# 🌍 Real-World Applications

This system reflects practical use cases in:

- 🏦 Banking Systems
- 💳 Payment Gateways
- 🛒 E-Commerce Platforms
- 🔐 Financial Security Monitoring
- 📡 Fraud Detection Infrastructure

---

# 📸 Dashboard Screenshots

## 🔹 Overview Dashboard

<p align="center">
  <img src="assets/overview.png" width="800"/>
</p>

---

## 🔹 Analytics Dashboard

<p align="center">
  <img src="assets/analytics.png" width="800"/>
</p>

---

## 🔹 Prediction Module

<p align="center">
  <img src="assets/prediction.png" width="800"/>
</p>

---

## 🔹 Model Performance

<p align="center">
  <img src="assets/performance.png" width="800"/>
</p>

---

# 🎥 Demo Video

📹 Full Dashboard Walkthrough:

assets/frauddetextionDashboard.mp4

---

# 🚀 Live Application

## 🔗 Streamlit Deployment

👉 [https://fraud-detectionsystem.streamlit.app/](https://fraud-detectionsystem.streamlit.app/)

---

# ⚡ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/rajsvmahendra/fraud-detection-system.git
```

## 2️⃣ Navigate to Project Folder

```bash
cd fraud-detection-system
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```bash
fraud-detection-system/
│
├── assets/
│   ├── landing.png
│   ├── overview.png
│   ├── analytics.png
│   ├── prediction.png
│   ├── performance.png
│   └── frauddetextionDashboard.mp4
│
├── dataset/
│
├── models/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 📈 Model Evaluation Metrics

| Metric           | Importance                  |
| ---------------- | --------------------------- |
| Accuracy         | Overall correctness         |
| Precision        | Controls false positives    |
| Recall           | Detects actual fraud cases  |
| F1 Score         | Balances Precision & Recall |
| Confusion Matrix | Visual performance analysis |

---

# 👨‍💻 Author

## Rajsv Mahendra

Student • Data Science & Machine Learning Enthusiast

🔗 LinkedIn:
[https://www.linkedin.com/in/rajsvmahendra/](https://www.linkedin.com/in/rajsvmahendra/)

📦 GitHub:
[https://github.com/rajsvmahendra](https://github.com/rajsvmahendra)

---

# ⭐ Support

If you found this project useful:

* ⭐ Star the repository
* 🍴 Fork the project
* 📢 Share it with others

---

<p align="center">
  Built with Python, Machine Learning, and a suspicious amount of caffeine ☕
</p>
