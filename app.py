import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
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
        arima_mape = mean_absolute_percentage_error(test['Sales'], forecast_arima)
        arima_accuracy = 100 - arima_mape * 100
        st.write("### ARIMA Metrics")
        st.write(f"RMSE: {arima_rmse:.2f}")
        st.write(f"R²: {arima_r2:.2f}")
        st.write(f"MAPE: {arima_mape:.2f}")
        st.write(f"Accuracy: {arima_accuracy:.2f}%")

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

        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        lr_r2 = r2_score(y_test, lr_pred)
        lr_mape = mean_absolute_percentage_error(y_test, lr_pred)
        lr_accuracy = 100 - lr_mape * 100
        st.write("### Linear Regression Metrics")
        st.write(f"RMSE: {lr_rmse:.2f}")
        st.write(f"R²: {lr_r2:.2f}")
        st.write(f"MAPE: {lr_mape:.2f}")
        st.write(f"Accuracy: {lr_accuracy:.2f}%")

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

        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
        rf_accuracy = 100 - rf_mape * 100
        st.write("### Random Forest Metrics")
        st.write(f"RMSE: {rf_rmse:.2f}")
        st.write(f"R²: {rf_r2:.2f}")
        st.write(f"MAPE: {rf_mape:.2f}")
        st.write(f"Accuracy: {rf_accuracy:.2f}%")

        # Comparison Chart
        st.subheader("Model Comparison: Accuracy")
        models = ['ARIMA', 'Linear Regression', 'Random Forest']
        accuracies = [arima_accuracy, lr_accuracy, rf_accuracy]

        fig_comp, ax_comp = plt.subplots()
        bars = ax_comp.bar(models, accuracies, color=['skyblue', 'orange', 'green'])
        ax_comp.set_ylim([0, 100])
        ax_comp.set_ylabel("Accuracy (%)")
        ax_comp.set_title("Forecasting Model Accuracy Comparison")
        for bar in bars:
            height = bar.get_height()
            ax_comp.text(bar.get_x() + bar.get_width()/2, height - 5, f'{height:.1f}%', ha='center', va='bottom', color='white')
        st.pyplot(fig_comp)
