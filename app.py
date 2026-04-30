import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")

st.markdown("### 📊 Model Performance")
st.write("Accuracy: ~99%")
st.write("ROC-AUC Score: ~0.98")

st.markdown("### Enter Transaction Details")

features = ['V1','V3','V4','V7','V10','V11','V12','V14','V16','V17','V18']

input_values = []

for feature in features:
    val = st.slider(f"{feature}", -10.0, 10.0, 0.0)
    input_values.append(val)

st.markdown("---")

if st.button("🔍 Predict"):
    data = pd.DataFrame([input_values], columns=features)

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Normal Transaction")

st.markdown("---")

if st.button("🧪 Test Fraud Example"):
    sample = [-2.312227, -1.609851, 3.997906, -2.537387, -2.772272, 3.202033, -2.899907, -4.289254, -1.140747, -2.830056, -0.016822]
    data = pd.DataFrame([sample], columns=features)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("🚨 Fraud Detected (Test Case)")
    else:
        st.success("Normal Transaction")