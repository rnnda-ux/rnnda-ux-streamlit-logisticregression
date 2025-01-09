import streamlit as st
import joblib
import numpy as np

model = joblib.load('logistic_regression_model.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')

st.title("Prediksi Spesies Ikan")

# Input data oleh pengguna
length = st.number_input("Panjang Ikan (length)", min_value=0.0)
weight = st.number_input("Berat Ikan (weight)", min_value=0.0)
w_l_ratio = st.number_input("Rasio Berat/Panjang (w_l_ratio)", min_value=0.0)

# Tombol untuk melakukan prediksi
if st.button("Prediksi"):
    # Proses data input pengguna
    input_data = np.array([[length, weight, w_l_ratio]])

    # Standarisasi data input
    scaled_data = scaler.transform(input_data)

    # Prediksi spesies ikan
    prediction = model.predict(scaled_data)

    # Convert hasil prediksi ke nama spesies
    predicted_species = le.inverse_transform(prediction)

    # Tampilkan hasil prediksi
    st.write(f"Prediksi Spesies Ikan: {predicted_species[0]}")
