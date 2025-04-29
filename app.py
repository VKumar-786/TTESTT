import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="centered")
st.title("Predictive Sales Analysis")

st.write("""
Upload your time series sales data and compare forecasting models:
- ARIMA
- Linear Regression
- Random Forest Regressor
- LSTM Neural Network

The app will clean the data by removing missing values and duplicates.
""")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if 'Month' not in data.columns or 'Sales' not in data.columns:
        st.error("CSV must contain 'Month' and 'Sales' columns.")
    else:
        data.drop_duplicates(inplace=True)
        data.dropna(inplace=True)

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

        # Linear Regression
        X_train = np.arange(len(train)).reshape(-1, 1)
        X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
        y_train = train['Sales'].values
        y_test = test['Sales'].values

        with st.spinner('Fitting Linear Regression model...'):
            lr = LinearRegression().fit(X_train, y_train)
            lr_pred = lr.predict(X_test)

        # Random Forest
        with st.spinner('Fitting Random Forest model...'):
            rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

        # LSTM Model
        with st.spinner('Fitting LSTM model...'):
            scaler = MinMaxScaler()
            sales_scaled = scaler.fit_transform(data['Sales'].values.reshape(-1,1))

            def create_sequences(data, seq_length):
                x, y = [], []
                for i in range(len(data)-seq_length):
                    x.append(data[i:i+seq_length])
                    y.append(data[i+seq_length])
                return np.array(x), np.array(y)

            SEQ_LEN = 5
            X_lstm, y_lstm = create_sequences(sales_scaled, SEQ_LEN)
            split = int(0.75 * len(X_lstm))
            X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
            y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

            model = Sequential([
                LSTM(50, activation='relu', input_shape=(SEQ_LEN, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_train_lstm, y_train_lstm, epochs=50, verbose=0)

            lstm_pred = model.predict(X_test_lstm)
            lstm_pred = scaler.inverse_transform(lstm_pred).flatten()
            lstm_dates = data.index[SEQ_LEN+split:SEQ_LEN+split+len(lstm_pred)]

        # Forecast Visualizations
        st.subheader("Forecast Visualizations")

        model_preds = [
            ("ARIMA", forecast_arima, test.index),
            ("Linear Regression", lr_pred, test.index),
            ("Random Forest", rf_pred, test.index),
            ("LSTM", lstm_pred, lstm_dates)
        ]

        for name, pred, idx in model_preds:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(data.index, data['Sales'], label='Historical Sales', linewidth=2)
            ax.plot(idx, pred, label=f'{name} Forecast', linestyle='--')
            ax.set_title(f"{name} Sales Forecast")
            ax.set_xlabel('Month')
            ax.set_ylabel('Sales')
            ax.legend()
            st.pyplot(fig)

        # Metrics
        def safe_metrics(y_true, y_pred):
            return (
                np.sqrt(mean_squared_error(y_true, y_pred)),
                r2_score(y_true, y_pred),
                mean_absolute_percentage_error(y_true, y_pred),
                100 - mean_absolute_percentage_error(y_true, y_pred) * 100
            )

        arima_rmse, arima_r2, arima_mape, arima_acc = safe_metrics(y_test, forecast_arima)
        lr_rmse, lr_r2, lr_mape, lr_acc = safe_metrics(y_test, lr_pred)
        rf_rmse, rf_r2, rf_mape, rf_acc = safe_metrics(y_test, rf_pred)
        lstm_rmse, lstm_r2, lstm_mape, lstm_acc = safe_metrics(y_test[-len(lstm_pred):], lstm_pred)

        st.subheader("Model Metrics")
        metrics_df = pd.DataFrame({
            "Model": ["ARIMA", "Linear Regression", "Random Forest", "LSTM"],
            "RMSE": [arima_rmse, lr_rmse, rf_rmse, lstm_rmse],
            "R²": [arima_r2, lr_r2, rf_r2, lstm_r2],
            "MAPE": [arima_mape, lr_mape, rf_mape, lstm_mape],
            "Accuracy (%)": [arima_acc, lr_acc, rf_acc, lstm_acc]
        })
        st.dataframe(metrics_df.set_index("Model"))

        st.subheader("Model Comparison: Accuracy")
        models = metrics_df['Model']
        accuracies = metrics_df['Accuracy (%)']
        fig_comp, ax_comp = plt.subplots()
        bars = ax_comp.bar(models, accuracies, color=['skyblue', 'orange', 'green', 'purple'])
        ax_comp.set_ylim([0, 100])
        ax_comp.set_ylabel("Accuracy (%)")
        ax_comp.set_title("Forecasting Model Accuracy Comparison")
        for bar in bars:
            height = bar.get_height()
            ax_comp.text(bar.get_x() + bar.get_width()/2, height - 5, f'{height:.1f}%', ha='center', va='bottom', color='white')
        st.pyplot(fig_comp)

        buf = BytesIO()
        fig_comp.savefig(buf, format="png")
        buf.seek(0)
        st.download_button(
            label="Download Comparison Chart as PNG",
            data=buf,
            file_name="model_comparison.png",
            mime="image/png"
        )

        st.subheader("Forecasted Results")
        forecast_table = pd.DataFrame({
            "Date": test.index[-len(lstm_pred):],
            "Actual Sales": y_test[-len(lstm_pred):],
            "ARIMA Forecast": forecast_arima[-len(lstm_pred):].values,
            "Linear Regression Forecast": lr_pred[-len(lstm_pred):],
            "Random Forest Forecast": rf_pred[-len(lstm_pred):],
            "LSTM Forecast": lstm_pred
        })
        st.dataframe(forecast_table.set_index("Date"))
