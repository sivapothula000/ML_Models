import streamlit as st
import pickle
import numpy as np

with open('models/house_price_predict_model.pkl','rb') as f:
    model = pickle.load(f)

st.title("House Price Prediction")

Longitude = st.text_input("Longitude")
Latitude = st.text_input("Latitude")
HouseAge = st.text_input("House Age")
TotalRooms = st.text_input("Total Rooms")
TotalBedRooms = st.text_input("Total Bedrooms")
Population = st.text_input("Population")
AveOccup = st.text_input("Occupancy")
MedInc = st.text_input("Median Income")

if st.button("Predict"):
    data = np.array([
        float(Longitude), float(Latitude), float(HouseAge),
        float(TotalRooms), float(TotalBedRooms),
        float(Population), float(AveOccup), float(MedInc)
    ]).reshape(1,-1)

    pred = model.predict(data)
    st.success(f"Predicted House Price: {pred[0]}")
