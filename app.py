import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import io

st.title("Sales Forecasting using Multiple Models")

uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel file)", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # Detect date column
    date_col = None
    for col in data.columns:
        try:
            temp = pd.to_datetime(data[col], errors='raise', dayfirst=True, infer_datetime_format=True)
            date_col = col
            break
        except:
            continue

    if date_col is None:
        st.error("No date column detected. Please ensure your dataset contains a recognizable date format.")
        st.stop()

    data[date_col] = pd.to_datetime(data[date_col], errors='coerce', dayfirst=True, infer_datetime_format=True)
    data = data.dropna(subset=[date_col])
    data = data.sort_values(by=date_col)
    data = data.set_index(date_col)

    # Detect sales column
    value_col = None
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64] and data[col].nunique() > 10:
            value_col = col
            break

    if value_col is None:
        st.error("No numeric sales column detected. Please ensure your dataset contains a sales column.")
        st.stop()

    data = data[[value_col]].rename(columns={value_col: 'Sales'})
    st.write("### Raw Data")
    st.write(data.tail())

    # Train-test split
    split_index = int(len(data) * 0.8)
    train, test = data.iloc[:split_index], data.iloc[split_index:]

    forecasts = {}
    rmse_list, r2_list, mape_list, acc_list = [], [], [], []

    def calculate_metrics(true, pred):
        rmse = np.sqrt(mean_squared_error(true, pred))
        r2 = r2_score(true, pred)
        mape = mean_absolute_percentage_error(true, pred)
        accuracy = 100 - (mape * 100)
        return rmse, r2, mape, accuracy

    # ARIMA
    arima_model = ARIMA(train['Sales'], order=(1, 1, 1))
    arima_fit = arima_model.fit()
    arima_pred = arima_fit.forecast(steps=len(test))
    arima_rmse, arima_r2, arima_mape, arima_acc = calculate_metrics(test['Sales'], arima_pred)
    forecasts['ARIMA'] = arima_pred

    # Linear Regression
    X_lr = np.arange(len(train)).reshape(-1, 1)
    y_lr = train['Sales'].values
    model_lr = LinearRegression()
    model_lr.fit(X_lr, y_lr)
    X_test_lr = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    lr_pred = model_lr.predict(X_test_lr)
    lr_rmse, lr_r2, lr_mape, lr_acc = calculate_metrics(test['Sales'], lr_pred)
    forecasts['Linear Regression'] = lr_pred

    # Random Forest
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_lr, y_lr)
    rf_pred = model_rf.predict(X_test_lr)
    rf_rmse, rf_r2, rf_mape, rf_acc = calculate_metrics(test['Sales'], rf_pred)
    forecasts['Random Forest'] = rf_pred

    # LSTM
    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(data['Sales'].values.reshape(-1, 1))
    window = 12
    X_lstm, y_lstm = [], []
    for i in range(window, len(sales_scaled)):
        X_lstm.append(sales_scaled[i-window:i, 0])
        y_lstm.append(sales_scaled[i, 0])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, verbose=0)

    last_window = sales_scaled[-window:].reshape(1, window, 1)
    lstm_pred_scaled = []
    for _ in range(len(test)):
        pred = lstm_model.predict(last_window, verbose=0)[0][0]
        lstm_pred_scaled.append(pred)
        last_window = np.append(last_window[:, 1:, :], [[[pred]]], axis=1)

    lstm_pred = scaler.inverse_transform(np.array(lstm_pred_scaled).reshape(-1, 1)).flatten()
    lstm_rmse, lstm_r2, lstm_mape, lstm_acc = calculate_metrics(test['Sales'], lstm_pred)
    forecasts['LSTM'] = lstm_pred

    # Plot
    st.write("### Forecast Plots")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Sales'], label='Historical Sales')
    for model_name, prediction in forecasts.items():
        ax.plot(test.index, prediction[:len(test)], label=f'{model_name} Forecast', linestyle='--')
    ax.set_title("Sales Forecasting Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    st.pyplot(fig)

    # Metrics Summary
    metrics_df = pd.DataFrame({
        "Model": ["ARIMA", "Linear Regression", "Random Forest", "LSTM"],
        "RMSE": [arima_rmse, lr_rmse, rf_rmse, lstm_rmse],
        "R²": [arima_r2, lr_r2, rf_r2, lstm_r2],
        "MAPE": [arima_mape, lr_mape, rf_mape, lstm_mape],
        "Accuracy (%)": [arima_acc, lr_acc, rf_acc, lstm_acc]
    })

    st.write("### Model Performance")
    st.dataframe(metrics_df.round(3))
