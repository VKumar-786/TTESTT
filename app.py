import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="centered")
st.title("Predictive Sales Analysis")

st.write("""
Upload your time series sales data and compare forecasting models:
- ARIMA
- Linear Regression
- Random Forest Regressor
- LSTM

The app will clean the data by removing missing values and duplicates.
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

        # Flexible date parsing for various formats
        try:
            data['Month'] = pd.to_datetime(data['Month'], errors='raise', dayfirst=True)
        except Exception:
            try:
                data['Month'] = pd.to_datetime(data['Month'].astype(str) + ' 2023', format='%B %Y')
            except Exception:
                try:
                    data['Month'] = pd.to_datetime(data['Month'].astype(str) + ' 2023', format='%b %Y')
                except Exception:
                    try:
                        data['Month'] = pd.to_datetime(data['Month'].astype(str), format='%m', errors='coerce')
                        data['Month'] = data['Month'].fillna(pd.to_datetime('2023-' + data['Month'].astype(str).str.zfill(2) + '-01', errors='coerce'))
                    except Exception as e:
                        st.error(f"Error converting Month data: {e}")
                        st.stop()

        if data['Month'].isnull().any():
            st.error("Some dates could not be parsed. Please check your data.")
            st.stop()

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

        # LSTM Model
        with st.spinner('Fitting LSTM model...'):
            # Scaling data for LSTM
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data['Sales'].values.reshape(-1, 1))
            
            # Prepare LSTM dataset
            X_lstm = []
            y_lstm = []
            for i in range(60, len(scaled_data)):
                X_lstm.append(scaled_data[i-60:i, 0])
                y_lstm.append(scaled_data[i, 0])
            X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

            # Reshape X_lstm for LSTM input [samples, time steps, features]
            X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

            # Build LSTM model
            lstm_model = Sequential()
            lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
            lstm_model.add(LSTM(units=50, return_sequences=False))
            lstm_model.add(Dense(units=1))
            lstm_model.compile(optimizer='adam', loss='mean_squared_error')
            lstm_model.fit(X_lstm, y_lstm, epochs=20, batch_size=32)

            # Predicting the Test set results
            test_data = scaled_data[len(scaled_data) - len(test) - 60:]
            X_test_lstm = []
            for i in range(60, len(test_data)):
                X_test_lstm.append(test_data[i-60:i, 0])
            X_test_lstm = np.array(X_test_lstm)
            X_test_lstm = np.reshape(X_test_lstm, (X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
            lstm_pred = lstm_model.predict(X_test_lstm)
            lstm_pred = scaler.inverse_transform(lstm_pred)

        # Forecast Visualizations (Separate for each model)
        st.subheader("Forecast Visualizations")

        for model_name, prediction in zip([
            "ARIMA", "Linear Regression", "Random Forest", "LSTM"],
            [forecast_arima, lr_pred, rf_pred, lstm_pred]
        ):
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data.index, data['Sales'], label='Historical Sales', linewidth=2)
            ax.plot(test.index, prediction, label=f'{model_name} Forecast', linestyle='--')
            ax.set_title(f"{model_name} Sales Forecast")
            ax.set_xlabel('Month')
            ax.set_ylabel('Sales')
            ax.legend()
            st.pyplot(fig)

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

        lstm_rmse = np.sqrt(mean_squared_error(y_test, lstm_pred))
        lstm_r2 = r2_score(y_test, lstm_pred)
        lstm_mape = mean_absolute_percentage_error(y_test, lstm_pred)
        lstm_accuracy = 100 - lstm_mape * 100

        st.subheader("Model Metrics")
        metrics_df = pd.DataFrame({
            "Model": ["ARIMA", "Linear Regression", "Random Forest", "LSTM"],
            "RMSE": [arima_rmse, lr_rmse, rf_rmse, lstm_rmse],
            "R²": [arima_r2, lr_r2, rf_r2, lstm_r2],
            "MAPE": [arima_mape, lr_mape, rf_mape, lstm_mape],
            "Accuracy (%)": [arima_accuracy, lr_accuracy, rf_accuracy, lstm_accuracy]
        })
        st.dataframe(metrics_df.set_index("Model"))

        # Comparison Chart
        st.subheader("Model Comparison: Accuracy")
        models = ["ARIMA", "Linear Regression", "Random Forest", "LSTM"]
        accuracies = [arima_accuracy, lr_accuracy, rf_accuracy, lstm_accuracy]

        fig_comp, ax_comp = plt.subplots()
        bars = ax_comp.bar(models, accuracies, color=['skyblue', 'orange', 'green', 'red'])
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
            file_name="sales_comparison_chart.png",
            mime="image/png"
        )

        # Forecasted Table
        st.subheader("Forecasted Results")
        forecast_table = pd.DataFrame({
            "Date": test.index,
            "Actual Sales": y_test,
            "ARIMA Forecast": forecast_arima.values,
            "Linear Regression Forecast": lr_pred,
            "Random Forest Forecast": rf_pred,
            "LSTM Forecast": lstm_pred.flatten()
        })
        st.dataframe(forecast_table.set_index("Date"))
