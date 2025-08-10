import streamlit as st
import pickle
import numpy as np
import time
import plotly.graph_objects as go  # For animated chart

# Load model and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)
    model = model_data["model"]
    feature_names = model_data["features_names"]

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Page config
st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    body {
        background-image: url('https://images.unsplash.com/photo-1504384308090-c894fdcc538d?auto=format&fit=crop&w=1950&q=80');
        background-size: cover;
        background-attachment: fixed;
    }

    .title-style {
        background: linear-gradient(to right, #6a11cb, #2575fc);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 30px;
        font-weight: bold;
        animation: fadeInDown 1s ease-out;
    }

    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        font-weight: bold;
        margin-top: 10px;
        transition: 0.3s;
    }

    .stButton button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }

    .result-box {
        background-color: #ffffffdd;
        padding: 15px;
        border-radius: 10px;
        font-size: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        animation: fadeInUp 0.8s ease-in-out;
    }

    @keyframes fadeInDown {
        0% {opacity: 0; transform: translateY(-20px);}
        100% {opacity: 1; transform: translateY(0);}
    }

    @keyframes fadeInUp {
        from { transform: translateY(40px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    .loader {
        border: 8px solid #f3f3f3;
        border-top: 8px solid #3498db;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: 0 auto;
    }

    @keyframes spin {
        0% { transform: rotate(0deg);}
        100% { transform: rotate(360deg);}
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.info("""
    Predict whether a telecom customer will churn or stay using a machine learning model.
    
    Built by **Rudravesh  and Shivam🔥**
    """)

# Title
st.markdown('<div class="title-style">📉 Customer Churn Prediction App</div>', unsafe_allow_html=True)

# Input Fields
gender = st.selectbox("👤 Gender", ["Male", "Female"])
senior = st.selectbox("🧓 Senior Citizen", ["No", "Yes"])
partner = st.selectbox("💑 Partner", ["No", "Yes"])
dependents = st.selectbox("👶 Dependents", ["No", "Yes"])
tenure = st.slider("⏱️ Tenure (in months)", 0, 72)
phone = st.selectbox("📞 Phone Service", ["No", "Yes"])
multiple = st.selectbox("🔁 Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
security = st.selectbox("🛡️ Online Security", ["No", "Yes", "No internet service"])
backup = st.selectbox("💾 Online Backup", ["No", "Yes", "No internet service"])
device = st.selectbox("📱 Device Protection", ["No", "Yes", "No internet service"])
support = st.selectbox("🛠️ Tech Support", ["No", "Yes", "No internet service"])
tv = st.selectbox("📺 Streaming TV", ["No", "Yes", "No internet service"])
movies = st.selectbox("🎬 Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.selectbox("📃 Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("📄 Paperless Billing", ["No", "Yes"])
payment = st.selectbox("💳 Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly = st.number_input("💰 Monthly Charges", 0.0)
total = st.number_input("💵 Total Charges", 0.0)

# Prepare input
user_input = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

# Encode input
for col in user_input:
    if col in label_encoders:
        user_input[col] = label_encoders[col].transform([user_input[col]])[0]

input_data = np.array([list(user_input.values())])

# Prediction
if st.button("🔍 Predict"):
    with st.spinner("Analyzing... Please wait ⏳"):
        time.sleep(2)
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

    # Result Box
    if prediction == 1:
        st.markdown(
            f'<div class="result-box">⚠️ <b>Customer will Churn</b><br>Probability: {prob*100:.2f}%</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box">✅ <b>Customer will Stay</b><br>Probability: {(1 - prob)*100:.2f}%</div>',
            unsafe_allow_html=True
        )

    # Animated Gauge Chart
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prob * 100,
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "red" if prediction == 1 else "green"},
            'steps' : [
                {'range': [0, 50], 'color': "#d4edda"},
                {'range': [50, 100], 'color': "#f8d7da"}
            ],
            'threshold' : {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        },
        title = {'text': "Churn Probability (%)"}
    ))

    st.plotly_chart(fig)
