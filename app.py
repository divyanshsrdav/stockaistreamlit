import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Global variables
model = None
scaler = None

def load_trained_model(model_path):
    global model
    model = load_model(model_path)

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

st.title("Stock Price Prediction App")

# File upload for model and CSV
model_file = st.file_uploader("Upload your .h5 model file", type='h5')
csv_file = st.file_uploader("Upload your CSV data file", type='csv')

if model_file is not None:
    load_trained_model(model_file)

if csv_file is not None:
    # Read the CSV file
    data = pd.read_csv(csv_file, date_parser=True)
    data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
    data.set_index('Date', inplace=True)
    data = data[['Close']]

    # Scale data
    data_values = data.values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_values)

    # Create dataset for predictions
    time_step = 60
    X, y = create_dataset(scaled_data, time_step)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Make predictions
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)

    # Plotting
    plt.figure(figsize=(14, 5))
    plt.plot(data.index, data_values, label='True Price', color='blue')
    predicted_dates = data.index[-len(predictions):]
    plt.plot(predicted_dates, predictions, label='Predicted Price', color='red')
    plt.xlabel('Date')
    plt.ylabel('Stock Price (USD)')
    plt.title('Stock Price Prediction')
    plt.legend()
    plt.grid(True)

    # Show plot in Streamlit
    st.pyplot(plt)

