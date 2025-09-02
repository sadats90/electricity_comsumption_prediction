import streamlit as st
import joblib
import pandas as pd
from datetime import datetime, timedelta

# Load model
model = joblib.load('power_forecast_model.pkl')

st.title("⚡ Electricity Usage Predictor")
st.markdown("Enter the date details and previous usage to forecast the next day's electricity consumption.")

# Default date to predict is tomorrow
today = datetime.today().date()
default_predict_date = today + timedelta(days=1)

# User selects the date to predict usage for
date_to_predict = st.date_input("Select the date you want to predict usage for", default_predict_date)

# Calculate lag dates based on selected prediction date
lag_7_date = date_to_predict - timedelta(days=7)
lag_1_date = date_to_predict - timedelta(days=1)

# Extract dayofweek and month from date_to_predict
dayofweek = date_to_predict.weekday()  # Monday=0, Sunday=6
month = date_to_predict.month

# Inputs for lag usages with dynamic labels
lag_7 = st.number_input(f'Electricity Usage on {lag_7_date.strftime("%Y-%m-%d")} (kW)', min_value=0.0, value=1.0, step=0.1)
lag_1 = st.number_input(f'Electricity Usage on {lag_1_date.strftime("%Y-%m-%d")} (kW)', min_value=0.0, value=1.0, step=0.1)

# Create input DataFrame for model
input_df = pd.DataFrame({
    'dayofweek': [dayofweek],
    'month': [month],
    'lag_1': [lag_1],
    'lag_7': [lag_7]
})

# Predict and display result
if st.button('Predict Usage'):
    prediction = model.predict(input_df)[0]
    st.success(f'🔋 Predicted Electricity Usage for {date_to_predict.strftime("%Y-%m-%d")}: **{prediction:.2f} kW**')
