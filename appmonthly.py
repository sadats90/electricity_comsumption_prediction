import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import calendar

# Load model (we'll use the existing model for now, but adapt it for monthly prediction)
model = joblib.load('power_forecast_model.pkl')

st.title("📅 Monthly Electricity Usage Predictor")
st.markdown("Predict monthly electricity consumption based on historical patterns and seasonal trends.")

# === Section 1: User Input Form ===
st.subheader("🔮 Select Month for Prediction")

# Clear month and year selection with better labels
st.markdown("**Which month would you like to predict electricity usage for?**")

col1, col2 = st.columns(2)
with col1:
    year = st.selectbox('Select Year', list(range(2024, 2030)), index=0, help="Choose the year for your prediction")
with col2:
    month = st.selectbox('Select Month', list(range(1, 13)), index=0, help="Choose the month for your prediction")

# Get month name for display
month_name = calendar.month_name[month]

# Display selected month clearly
st.info(f"🎯 **You are predicting electricity usage for: {month_name} {year}**")

# Calculate previous month and year for lag features
if month == 1:
    prev_month = 12
    prev_year = year - 1
else:
    prev_month = month - 1
    prev_year = year

# Calculate same month previous year
prev_year_month = year - 1

# Input for previous month usage
st.markdown("### 📊 Historical Usage Input")
st.markdown("**Provide your recent electricity usage data:**")

st.markdown("**1️⃣ Previous Month's Usage:**")
lag_1_month = st.number_input(
    f'📅 Total electricity used in {calendar.month_name[prev_month]} {prev_year} (kWh)', 
    min_value=0.0, 
    value=1000.0, 
    step=10.0,
    help=f"This is your electricity consumption for {calendar.month_name[prev_month]} {prev_year} - the month before your target month"
)

st.markdown("**2️⃣ Average Monthly Usage:**")
avg_monthly_usage = st.number_input(
    f'📅 Your typical monthly electricity usage (kWh)', 
    min_value=0.0, 
    value=1200.0, 
    step=10.0,
    help="Enter your average monthly electricity consumption based on your past bills"
)

# Additional factors
st.markdown("### 🏠 Household Information")
st.markdown("**Help us make a more accurate prediction by providing household details:**")

avg_temp = st.slider('Expected Average Temperature (°C)', -10, 40, 20, help="Expected average temperature for the month")
household_size = st.selectbox('Number of People in Household', [1, 2, 3, 4, 5, 6, 7, 8], index=2)
home_size = st.selectbox('Home Size (sq ft)', ['Under 1000', '1000-1500', '1500-2000', '2000-2500', '2500-3000', 'Over 3000'], index=2)
heating_type = st.selectbox('Primary Heating Type', ['Electric', 'Gas', 'Oil', 'Heat Pump', 'Other'], index=0)

# Summary section
st.markdown("---")
st.markdown("### 📋 Prediction Summary")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Target Month", f"{month_name} {year}")
with col2:
    st.metric("Previous Month Usage", f"{lag_1_month:.0f} kWh")
with col3:
    st.metric("Average Monthly Usage", f"{avg_monthly_usage:.0f} kWh")
with col4:
    st.metric("Expected Temperature", f"{avg_temp}°C")

# Create input DataFrame for model prediction
# Note: We'll adapt the existing model features for monthly prediction
input_df = pd.DataFrame({
    'month': [month],
    'dayofweek': [0],  # Default to Monday for monthly aggregation
    'lag_1': [lag_1_month / 30],  # Convert monthly to daily average for model compatibility
    'hour': [12]  # Default to noon for monthly aggregation
})

# === Section 2: Prediction ===
if st.button('🔮 Predict Monthly Usage', type="primary"):
    # Get base prediction from model (this will be daily, so we'll scale it)
    daily_prediction = model.predict(input_df)[0]
    
    # Scale to monthly prediction
    days_in_month = calendar.monthrange(year, month)[1]
    monthly_prediction = daily_prediction * days_in_month
    
    # Apply adjustments based on household factors
    adjustment_factor = 1.0
    
    # Temperature adjustment (heating/cooling)
    if avg_temp < 10:  # Winter heating
        adjustment_factor *= 1.2
    elif avg_temp > 25:  # Summer cooling
        adjustment_factor *= 1.15
    
    # Household size adjustment
    if household_size >= 4:
        adjustment_factor *= 1.1
    elif household_size <= 2:
        adjustment_factor *= 0.9
    
    # Home size adjustment
    size_multipliers = {
        'Under 1000': 0.8,
        '1000-1500': 0.9,
        '1500-2000': 1.0,
        '2000-2500': 1.1,
        '2500-3000': 1.2,
        'Over 3000': 1.3
    }
    adjustment_factor *= size_multipliers[home_size]
    
    # Heating type adjustment
    if heating_type == 'Electric':
        adjustment_factor *= 1.1  # Electric heating uses more electricity
    elif heating_type == 'Heat Pump':
        adjustment_factor *= 1.05  # Heat pumps are efficient but still use electricity
    
    # Apply adjustment factor
    adjusted_prediction = monthly_prediction * adjustment_factor
    
    # Display results
    st.success(f'🔋 **Predicted Monthly Usage for {month_name} {year}:**')
    st.metric(
        label="Total Consumption", 
        value=f"{adjusted_prediction:.0f} kWh",
        delta=f"{adjusted_prediction - monthly_prediction:.0f} kWh (household adjustment)"
    )
    
    # Additional metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Daily Average", f"{adjusted_prediction/days_in_month:.1f} kWh/day")
    with col2:
        st.metric("Weekly Average", f"{adjusted_prediction/4.33:.1f} kWh/week")

# === Section 3: Simple Information ===
st.markdown("---")
st.subheader("💡 Energy Saving Tips")

tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    **Temperature Control**
    - Keep windows open for natural ventilation
    - Use curtains to block direct sunlight
    - Set AC temperature to 26°C or higher
    """)

with tips_col2:
    st.markdown("""
    **Smart Usage**
    - Unplug devices when not in use
    - Use LED bulbs and energy-efficient appliances
    - Run heavy appliances during off-peak hours (night)
    - Use solar panels if available
    """)


