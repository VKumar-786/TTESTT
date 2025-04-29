import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from io import StringIO, BytesIO

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
        # Data Cleaning
        data.drop_duplicates(inplace=True)
        data.dropna(inplace=True)

        # Handling both Month names and Full date formats
        if data['Month'].str.contains('-').any():  # This checks if there's a full date in your data
            data['Month'] = pd.to_datetime(data['Month'], format='%Y-%m-%d')  # Full date format
        else:
            # For month names like "January", "February", append a year and convert to datetime
            data['Month'] = pd.to_datetime(data['Month'] + ' 2023', format='%B %Y')  # Assuming year is 2023
        
        data.set_index('Month', inplace=True)

        st.subheader("Cleaned Data")
        st.dataframe(data.tail())

        train = data.iloc[:int(0.75 * len(data))]
        test = data.iloc[int(0.75 * len(data)):]        

        # ARIMA Model
        with st.spinner('Fitting ARIMA model...'):
            arima_model = ARIMA(data['Sales'], order=(1,1,1))
            arima_fit = arima_model.fit()
            forecast_arima = arima_fit.forecast(steps=len(test))

        # Linear Regression Model
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
        y_train = train['Sales'].values
        y_test = test['Sales'].values

        with st.spinner('Fitting Linear Regression model...'):
            lr = LinearRegression().fit(X_train, y_train)
            lr_pred = lr.predict(X_test)

        # Random Forest Model
        with st.spinner('Fitting Random Forest model...'):
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

        st.subheader("Forecast Visualization")
        
        # ARIMA plot
        fig_arima, ax_arima = plt.subplots(figsize=(12,6))
        ax_arima.plot(data.index, data['Sales'], label='Historical Sales', linewidth=2)
        ax_arima.plot(test.index, forecast_arima, label='ARIMA Forecast', linestyle='--', color='red')
        ax_arima.set_xlabel('Month')
        ax_arima.set_ylabel('Sales')
        ax_arima.set_title('ARIMA Sales Forecast')
        ax_arima.legend()
        st.pyplot(fig_arima)

        # Linear Regression plot
        fig_lr, ax_lr = plt.subplots(figsize=(12,6))
        ax_lr.plot(data.index, data['Sales'], label='Historical Sales', linewidth=2)
        ax_lr.plot(test.index, lr_pred, label='Linear Regression Forecast', linestyle='-.', color='blue')
        ax_lr.set_xlabel('Month')
        ax_lr.set_ylabel('Sales')
        ax_lr.set_title('Linear Regression Sales Forecast')
        ax_lr.legend()
        st.pyplot(fig_lr)

        # Random Forest plot
        fig_rf, ax_rf = plt.subplots(figsize=(12,6))
        ax_rf.plot(data.index, data['Sales'], label='Historical Sales', linewidth=2)
        ax_rf.plot(test.index, rf_pred, label='Random Forest Forecast', linestyle=':', color='green')
        ax_rf.set_xlabel('Month')
        ax_rf.set_ylabel('Sales')
        ax_rf.set_title('Random Forest Sales Forecast')
        ax_rf.legend()
        st.pyplot(fig_rf)

        # Metrics Calculation
        arima_rmse = np.sqrt(mean_squared_error(y_test, forecast_arima))
        arima_r2 = r2_score(y_test, forecast_arima)
        arima_mape = mean_absolute_percentage_error(y_test, forecast_arima)
        arima_accuracy = 100 - arima_mape * 100

        lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
        lr_r2 = r2_score(y_test, lr_pred)
        lr_mape = mean_absolute_percentage_error(y_test, lr_pred)
        lr_accuracy = 100 - lr_mape * 100

        rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
        rf_r2 = r2_score(y_test, rf_pred)
        rf_mape = mean_absolute_percentage_error(y_test, rf_pred)
        rf_accuracy = 100 - rf_mape * 100

        st.subheader("Model Metrics")
        metrics_df = pd.DataFrame({
            "Model": ["ARIMA", "Linear Regression", "Random Forest"],
            "RMSE": [arima_rmse, lr_rmse, rf_rmse],
            "R²": [arima_r2, lr_r2, rf_r2],
            "MAPE": [arima_mape, lr_mape, rf_mape],
            "Accuracy (%)": [arima_accuracy, lr_accuracy, rf_accuracy]
        })
        st.dataframe(metrics_df.set_index("Model"))

        # Comparison Chart
        st.subheader("Model Comparison: Accuracy")
        models = ["ARIMA", "Linear Regression", "Random Forest"]
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

        # Save and download comparison chart
        buf = BytesIO()
        fig_comp.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Download Comparison Chart as PNG",
            data=buf,
            file_name="model_comparison.png",
            mime="image/png"
        )

        # Forecasted Table
        st.subheader("Forecasted Results")
        forecast_table = pd.DataFrame({
            "Date": test.index,
            "Actual Sales": y_test,
            "ARIMA Forecast": forecast_arima.values,
            "Linear Regression Forecast": lr_pred,
            "Random Forest Forecast": rf_pred
        })
        st.dataframe(forecast_table.set_index("Date"))
