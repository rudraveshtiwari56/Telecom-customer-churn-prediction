# 📊 Telecom Customer Churn Prediction

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Project-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📌 Project Overview
This project predicts whether a telecom customer will churn (leave the service) based on their demographics, service usage, and billing details.  
The main objective is to help telecom companies identify customers at risk and take preventive actions to retain them.

---

## 📂 Dataset
- **Source:** Kaggle - Telecom Customer Churn Dataset  
- **Rows:** 7,043  
- **Columns:** 21  
- **Target Variable:** `Churn` (Yes/No)  

---

## 🔍 Key Features
1. **Data Preprocessing** — Missing value handling, label encoding, numeric conversion, SMOTE balancing.
2. **Exploratory Data Analysis** — Distribution plots, correlation heatmaps, count plots.
3. **Model Training** — Decision Tree, Random Forest, XGBoost, KNN, and SVM with 5-fold cross-validation.
4. **Deployment** — Streamlit web app for real-time churn prediction.

---

## 🛠 Tools & Frameworks
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, XGBoost, imbalanced-learn, pickle  
- **Deployment:** Streamlit

---

## 📊 Model Performance

| Model           | Accuracy |
|----------------|----------|
| Decision Tree  | 78%      |
| Random Forest  | 84%      |
| XGBoost        | 83%      |
| KNN            | 77%      |
| SVM            | 64%      |

---

## 🚀 How to Run Locally
```bash
# Clone this repository
git clone https://github.com/yourusername/Telecom-customer-churn-prediction.git

# Navigate to the folder
cd Telecom-customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
