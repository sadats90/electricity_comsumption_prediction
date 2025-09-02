import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load('power_forecast_model.pkl')

st.title("⚡ Electricity Usage Predictor")

st.markdown("Enter the time details and previous hour's usage to forecast the next hour.")

# === Section 1: User Input Form ===
hour = st.slider('Hour of Day (0-23)', 0, 23, 12)
dayofweek = st.selectbox('Day of the Week (0=Mon, 6=Sun)', list(range(7)))
month = st.selectbox('Month (1-12)', list(range(1, 13)))
lag_1 = st.number_input('Previous Hour Power Usage (kW)', min_value=0.0, value=1.0, step=0.1)

input_df = pd.DataFrame({
    'hour': [hour],
    'dayofweek': [dayofweek],
    'month': [month],
    'lag_1': [lag_1]
})

if st.button('Predict Usage'):
    prediction = model.predict(input_df)[0]
    st.success(f'🔋 Predicted Electricity Usage: **{prediction:.2f} kW**')

# === Section 2: Power Usage Graph ===
st.markdown("---")
st.subheader("📊 Historical Daily Power Usage (UCI Dataset)")

# Load the cleaned dataset
try:
    df = pd.read_csv("cleaned_power_data.csv", parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)

    # Resample to daily consumption
    daily_df = df['Global_active_power'].resample('D').sum()

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_df.index, daily_df.values, color='green')
    ax.set_title("Daily Electricity Usage")
    ax.set_ylabel("kW")
    ax.set_xlabel("Date")
    st.pyplot(fig)

except Exception as e:
    st.error(f"Failed to load or plot the historical data: {e}")
