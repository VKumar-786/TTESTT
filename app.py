import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from io import StringIO

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="centered")
st.title("Predictive Sales Analysis")

st.write("""
Upload your time series sales data and compare forecasting models:
- ARIMA
- Linear Regression
- Random Forest Regressor
""")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'Month' not in data.columns or 'Sales' not in data.columns:
        st.error("CSV must contain 'Month' and 'Sales' columns.")
    else:
        data['Month'] = pd.to_datetime(data['Month'])
        data.set_index('Month', inplace=True)

        st.subheader("Raw Data")
        st.dataframe(data.tail())

        train = data.iloc[:int(0.75 * len(data))]
        test = data.iloc[int(0.75 * len(data)):]        

        # ARIMA Model
        with st.spinner('Fitting ARIMA model...'):
            arima_model = ARIMA(data['Sales'], order=(1,1,1))
            arima_fit = arima_model.fit()
            forecast_arima = arima_fit.forecast(steps=len(test))

        st.subheader("ARIMA Forecast")
        fig, ax = plt.subplots()
        ax.plot(data['Sales'], label='Historical Sales')
        ax.plot(test.index, forecast_arima, label='ARIMA Forecast', linestyle='--')
        ax.set_xlabel('Month')
        ax.set_ylabel('Sales')
        ax.legend()
        st.pyplot(fig)

        arima_rmse = np.sqrt(mean_squared_error(test['Sales'], forecast_arima))
        arima_r2 = r2_score(test['Sales'], forecast_arima)
        st.write(f"ARIMA RMSE: {arima_rmse:.2f}")
        st.write(f"ARIMA R²: {arima_r2:.2f}")

        # Prepare Data for Regression
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
        y_train = train['Sales'].values
        y_test = test['Sales'].values

        # Linear Regression Model
        with st.spinner('Fitting Linear Regression model...'):
            lr = LinearRegression().fit(X_train, y_train)
            lr_pred = lr.predict(X_test)

        st.subheader("Linear Regression Forecast")
        fig_lr, ax_lr = plt.subplots()
        ax_lr.plot(test.index, y_test, label='Actual Sales', marker='o')
        ax_lr.plot(test.index, lr_pred, label='Linear Regression Forecast', linestyle='--')
        ax_lr.set_xlabel('Month')
        ax_lr.set_ylabel('Sales')
        ax_lr.legend()
        st.pyplot(fig_lr)

        st.write("### Linear Regression Metrics")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.2f}")
        st.write(f"R²: {r2_score(y_test, lr_pred):.2f}")

        # Random Forest Model
        with st.spinner('Fitting Random Forest model...'):
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

        st.subheader("Random Forest Forecast")
        fig_rf, ax_rf = plt.subplots()
        ax_rf.plot(test.index, y_test, label='Actual Sales', marker='o')
        ax_rf.plot(test.index, rf_pred, label='Random Forest Forecast', linestyle='-.')
        ax_rf.set_xlabel('Month')
        ax_rf.set_ylabel('Sales')
        ax_rf.legend()
        st.pyplot(fig_rf)

        st.write("### Random Forest Metrics")
        st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}")
        st.write(f"R²: {r2_score(y_test, rf_pred):.2f}")
