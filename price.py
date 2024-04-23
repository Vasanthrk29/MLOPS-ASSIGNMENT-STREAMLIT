import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('House_Price.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
def predict_price(squareMeters,numberOfRooms,cityPartRange,numPrevOwners):
    features = np.array([squareMeters,numberOfRooms,cityPartRange,numPrevOwners])
    features = features.reshape(1,-1)
    emission = model.predict(features)
    return emission[0]

# Streamlit UI
st.title('HOUSE PRICE PREDICTION')
st.write("""
## Input Features
ENTER THE VALUES FOR THE INPUT FEATURES TO PREDICT PRICE.
""")

# Input fields for user
squareMeters = st.number_input('SQUARE METERS')
numberOfRooms = st.number_input('NUMBER OF ROOMS')
cityPartRange = st.number_input('CITY PART RANGE')
numPrevOwners = st.number_input('NUMBER OF PREVIOUS OWNERS')

# Prediction button
if st.button('Predict'):
    # Predict EMISSION
    price_prediction = predict_price(squareMeters,numberOfRooms,cityPartRange,numPrevOwners)
    st.write(f"PREDICTED PRICE: {price_prediction}")