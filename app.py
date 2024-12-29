!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Title and Description
st.title("Time Series Analysis and Forecasting")
st.write("Upload your dataset to analyze and forecast time series data.")

# File Upload
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file, parse_dates=['date_column'], index_col='date_column')
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Data Visualization
    st.write("Time Series Plot:")
    plt.figure(figsize=(10, 6))
    plt.plot(data['value_column'], label="Time Series")
    plt.title("Time Series Data")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    st.pyplot(plt)

    # Model Selection
    model_type = st.selectbox("Select Model", options=["ARIMA", "LSTM"])

    if model_type == "ARIMA":
        st.write("ARIMA Model Selected")
        
        # ARIMA Forecasting
        model = ARIMA(data['value_column'], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)
        
        st.write("Forecasted Values:")
        st.write(forecast)

        # Plot forecast
        plt.figure(figsize=(10, 6))
        plt.plot(data['value_column'], label="Historical Data")
        plt.plot(range(len(data), len(data) + 10), forecast, label="Forecast", color="red")
        plt.title("ARIMA Forecast")
        plt.legend()
        st.pyplot(plt)

    elif model_type == "LSTM":
        st.write("LSTM Model Selected")

        # Prepare data for LSTM
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data['value_column'].values.reshape(-1, 1))

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Build and train LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32)

        # Make predictions
        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)

        st.write("LSTM Forecasted Values:")
        st.write(predictions)

        # Plot LSTM Forecast
        plt.figure(figsize=(10, 6))
        plt.plot(data['value_column'], label="Historical Data")
        plt.plot(range(len(data), len(data) + len(predictions)), predictions, label="Forecast", color="green")
        plt.title("LSTM Forecast")
        plt.legend()
        st.pyplot(plt)

    # Download Predictions
    forecast_df = pd.DataFrame(forecast, columns=["Forecast"]) if model_type == "ARIMA" else pd.DataFrame(predictions, columns=["Forecast"])
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Forecast",
        data=csv,
        file_name='forecast.csv',
        mime='text/csv',
    )