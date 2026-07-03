# 💳 Credit Card Fraud Detection System

> An end-to-end Machine Learning system for detecting fraudulent credit
> card transactions using advanced classification models, imbalance
> handling techniques, explainable AI, a FastAPI backend, and an
> interactive Streamlit dashboard.

------------------------------------------------------------------------

## 🚀 Project Overview

Credit card fraud causes billions of dollars in financial losses every
year. Detecting fraudulent transactions is challenging because fraud
cases represent only a tiny fraction of all transactions.

This project builds a complete fraud detection pipeline that: -
Identifies fraudulent transactions in real time - Handles highly
imbalanced data using SMOTE - Compares Logistic Regression, Random
Forest, XGBoost and LightGBM - Optimizes prediction thresholds -
Explains predictions using SHAP - Serves predictions through FastAPI -
Provides an interactive Streamlit dashboard

------------------------------------------------------------------------

## ✨ Key Features

### Data Analysis

-   Exploratory Data Analysis
-   Correlation Analysis
-   Fraud Distribution Analysis
-   Transaction Amount Analysis

### Data Preprocessing

-   RobustScaler
-   Stratified Train/Test Split
-   SMOTE for Imbalanced Data

### Machine Learning

-   Logistic Regression
-   Random Forest
-   XGBoost
-   LightGBM

### Explainable AI

-   SHAP Feature Importance
-   SHAP Summary Plots

### Deployment

-   FastAPI REST API
-   Streamlit Dashboard
-   Real-time Prediction

------------------------------------------------------------------------

## 🏗️ Workflow

``` text
Dataset
 ↓
EDA
 ↓
Preprocessing
(RobustScaler + SMOTE)
 ↓
Model Training
(Logistic Regression, Random Forest, XGBoost, LightGBM)
 ↓
Best Model Selection
 ↓
Threshold Optimization
 ↓
SHAP Explainability
 ↓
FastAPI
 ↓
Streamlit Dashboard
```

------------------------------------------------------------------------

## 📂 Project Structure

``` text
fraud-detection/
│
├── app/
│   ├── main.py
│   └── dashboard.py
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_SHAP_EXP.ipynb
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── threshold.pkl
├── data/
│   └── creditcard.csv
└── README.md
```

------------------------------------------------------------------------

## 📊 Dataset

-   284,807 Transactions
-   492 Fraudulent Transactions
-   30 Features
-   Highly Imbalanced Dataset

Target: - 0 → Legitimate - 1 → Fraud

------------------------------------------------------------------------

## 🔍 Exploratory Data Analysis

Performed: - Missing value analysis - Fraud distribution visualization -
Correlation heatmap - Transaction amount analysis

------------------------------------------------------------------------

## ⚙️ Preprocessing

-   RobustScaler
-   Stratified Split
-   SMOTE Oversampling

------------------------------------------------------------------------

## 🤖 Models

-   Logistic Regression
-   Random Forest
-   XGBoost
-   LightGBM

------------------------------------------------------------------------

## 📈 Evaluation Metrics

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   ROC-AUC
-   Precision-Recall Curve

Threshold optimization was performed to maximize fraud detection
performance.

------------------------------------------------------------------------

## 🧠 Explainable AI

SHAP was used to explain: - Feature Importance - Global
Interpretability - Local Prediction Explanations

------------------------------------------------------------------------

## 🌐 FastAPI

Endpoints

-   GET /
-   POST /predict
-   GET /docs

------------------------------------------------------------------------

## 🖥️ Streamlit Dashboard

Features: - Load Legitimate Sample - Load Fraud Sample - Real-Time
Prediction - Fraud Probability - Transaction Details

------------------------------------------------------------------------

## 🛠 Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-Learn, XGBoost,
LightGBM, SHAP, FastAPI, Streamlit, Joblib

------------------------------------------------------------------------

## 🚀 Installation

``` bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
```

Run API

``` bash
uvicorn app.main:app --reload
```

Run Dashboard

``` bash
streamlit run app/dashboard.py
```

------------------------------------------------------------------------

## 🎯 Skills Demonstrated

-   Machine Learning
-   Fraud Detection
-   Classification
-   Explainable AI
-   Model Deployment
-   REST API Development
-   Dashboard Development

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Docker
-   MLflow
-   Cloud Deployment
-   CI/CD
-   Real-Time Streaming

------------------------------------------------------------------------

## 👨‍💻 Author

**Yanna Phani Manjunath Reddy**

Aspiring Machine Learning Engineer \| AI & Data Science Enthusiast
