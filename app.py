%%writefile app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
loaded_lr_model = joblib.load('linear_regression_model.sav')

st.title('Sales Prediction App')
st.write('Predict sales based on advertising spending for TV, Radio, and Newspaper.')

# Input features from the user
tv = st.slider('TV Advertising Spend', 0.0, 300.0, 150.0)
radio = st.slider('Radio Advertising Spend', 0.0, 50.0, 25.0)
newspaper = st.slider('Newspaper Advertising Spend', 0.0, 100.0, 50.0)

# Create a DataFrame for the input
input_data = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])

# Make prediction
prediction = loaded_lr_model.predict(input_data)[0]

st.subheader('Predicted Sales')
st.write(f'Based on the input advertising spend, the predicted sales are: {prediction:.2f}')
