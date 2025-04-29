import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import io

st.title("📈 Predictive Sales Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your sales dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        if 'Month' not in data.columns or 'Sales' not in data.columns:
            st.error("Dataset must contain 'Month' and 'Sales' columns.")
        else:
            data['Month'] = pd.to_datetime(data['Month'])
            data.set_index('Month', inplace=True)
            st.subheader("Dataset Preview")
            st.write(data.head())

            # Split into training and testing sets
            train = data.iloc[:int(0.75 * len(data))]
            test = data.iloc[int(0.75 * len(data)):]

            # ARIMA Forecasting
            arima_model = ARIMA(data['Sales'], order=(1, 1, 1))
            arima_fit = arima_model.fit()
            forecast_arima = arima_fit.forecast(steps=len(test))

            st.subheader("ARIMA Forecast")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data['Sales'], label='Historical Sales')
            ax.plot(test.index, forecast_arima, label='ARIMA Forecast', linestyle='--', marker='o')
            ax.set_xlabel('Month')
            ax.set_ylabel('Sales')
            ax.legend()
            st.pyplot(fig)

            # Evaluation Metrics for ARIMA
            arima_mse = mean_squared_error(test['Sales'], forecast_arima)
            arima_rmse = np.sqrt(arima_mse)
            arima_r2 = r2_score(test['Sales'], forecast_arima)

            st.write("🔍 ARIMA Evaluation Metrics")
            st.write(f"Mean Squared Error (MSE): {arima_mse:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {arima_rmse:.2f}")
            st.write(f"R-squared (R²): {arima_r2:.2f}")

            # Prepare data for regression
            X_train = np.arange(len(train)).reshape(-1, 1)
            X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
            y_train = train['Sales']
            y_test = test['Sales']

            # Linear Regression
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            lr_pred = lr.predict(X_test)

            lr_mse = mean_squared_error(y_test, lr_pred)
            lr_rmse = np.sqrt(lr_mse)
            lr_r2 = r2_score(y_test, lr_pred)

            # Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            rf_mse = mean_squared_error(y_test, rf_pred)
            rf_rmse = np.sqrt(rf_mse)
            rf_r2 = r2_score(y_test, rf_pred)

            # Plot comparisons
            st.subheader("Model Comparison: Linear Regression vs Random Forest")
            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(test.index, y_test.values, label="Actual", marker='x')
            ax2.plot(test.index, lr_pred, label="Linear Regression", linestyle='--')
            ax2.plot(test.index, rf_pred, label="Random Forest", linestyle='-.')
            ax2.legend()
            st.pyplot(fig2)

            st.write("📊 Linear Regression Metrics")
            st.write(f"RMSE: {lr_rmse:.2f}, R²: {lr_r2:.2f}")

            st.write("🌲 Random Forest Metrics")
            st.write(f"RMSE: {rf_rmse:.2f}, R²: {rf_r2:.2f}")

    except Exception as e:
        st.error(f"Error reading file: {e}")
