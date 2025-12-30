import streamlit as st
import pickle
import numpy as np



with open('house_price_predict_model.pkl','rb') as filename:
  loaded_model=pickle.load(filename)
  
#Creating a function for prediction
def house_price_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.array(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction[0]

def main():
    st.title("Siva's House Price Prediction App")
    Longitude=st.text_input("Longitude Location",placeholder="e.g. 34.05")
    Latitude=st.text_input("Latitude Location",placeholder="e.g. 37.28")
    HouseAge=st.text_input("Median House Age",placeholder="e.g. 25")
    TotalRooms=st.text_input("Total number of rooms",placeholder="e.g. 52")
    TotalBedRooms=st.text_input("Total number of Bed Rooms",placeholder="e.g. 12")
    Population=st.text_input("Population of the District",placeholder="e.g. 3500")
    AveOccup=st.text_input("Total house occupancy",placeholder="e.g. 200")
    MedInc=st.text_input("Median Income of the District",placeholder="e.g. 1475.05")

    prediction =''
    #Code for Prediction using input features
    if st.button("Predict"):
      prediction=house_price_prediction([
        float(Longitude),
        float(Latitude),
        float(HouseAge),
        float(TotalRooms),
        float(TotalBedRooms),
        float(Population),
        float(AveOccup),
        float(MedInc)
      
        ])
    st.success(f"Predicted House Price: {prediction}")


if __name__=='__main__':

  main()


