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
forecast_steps = st.slider("Forecast future months beyond test set", 0, 12, 0)

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
            forecast_arima = arima_fit.forecast(steps=len(test)+forecast_steps)

        # Linear Regression
        X = np.arange(len(data)).reshape(-1, 1)
        y = data['Sales'].values
        lr = LinearRegression().fit(X[:len(train)], y[:len(train)])
        lr_pred = lr.predict(np.arange(len(train), len(data) + forecast_steps).reshape(-1, 1))

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X[:len(train)], y[:len(train)])
        rf_pred = rf.predict(np.arange(len(train), len(data) + forecast_steps).reshape(-1, 1))

        # LSTM Model
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
        with st.spinner('Training LSTM model...'):
            history = model.fit(X_train_lstm, y_train_lstm, epochs=50, verbose=0)

        lstm_pred = model.predict(X_test_lstm)
        lstm_pred = scaler.inverse_transform(lstm_pred).flatten()
        lstm_dates = data.index[SEQ_LEN+split:SEQ_LEN+split+len(lstm_pred)]

        # Forecast Visualizations
        st.subheader("Forecast Visualizations")

        model_preds = [
            ("ARIMA", forecast_arima, pd.date_range(start=test.index[0], periods=len(forecast_arima), freq='MS')),
            ("Linear Regression", lr_pred, pd.date_range(start=test.index[0], periods=len(lr_pred), freq='MS')),
            ("Random Forest", rf_pred, pd.date_range(start=test.index[0], periods=len(rf_pred), freq='MS')),
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

        st.subheader("Model Comparison: Accuracy")
        y_test_range = len(y_test_lstm)

        def safe_metrics(y_true, y_pred):
            return (
                np.sqrt(mean_squared_error(y_true, y_pred)),
                r2_score(y_true, y_pred),
                mean_absolute_percentage_error(y_true, y_pred),
                100 - mean_absolute_percentage_error(y_true, y_pred) * 100
            )

        arima_rmse, arima_r2, arima_mape, arima_acc = safe_metrics(y[-y_test_range:], forecast_arima[:y_test_range])
        lr_rmse, lr_r2, lr_mape, lr_acc = safe_metrics(y[-y_test_range:], lr_pred[:y_test_range])
        rf_rmse, rf_r2, rf_mape, rf_acc = safe_metrics(y[-y_test_range:], rf_pred[:y_test_range])
        lstm_rmse, lstm_r2, lstm_mape, lstm_acc = safe_metrics(y_test_lstm, lstm_pred)

        metrics_df = pd.DataFrame({
            "Model": ["ARIMA", "Linear Regression", "Random Forest", "LSTM"],
            "RMSE": [arima_rmse, lr_rmse, rf_rmse, lstm_rmse],
            "R²": [arima_r2, lr_r2, rf_r2, lstm_r2],
            "MAPE": [arima_mape, lr_mape, rf_mape],
            "Accuracy (%)": [arima_acc, lr_acc, rf_acc, lstm_acc]
        })
        st.dataframe(metrics_df.set_index("Model"))

        fig_comp, ax_comp = plt.subplots()
        bars = ax_comp.bar(metrics_df['Model'], metrics_df['Accuracy (%)'], color=['skyblue', 'orange', 'green', 'purple'])
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
            "Date": pd.date_range(start=test.index[0], periods=len(lr_pred), freq='MS'),
            "ARIMA Forecast": forecast_arima.values,
            "Linear Regression Forecast": lr_pred,
            "Random Forest Forecast": rf_pred
        })

        lstm_table = pd.DataFrame({
            "Date": lstm_dates,
            "LSTM Forecast": lstm_pred
        })

        forecast_table = pd.merge(forecast_table, lstm_table, on="Date", how="outer").set_index("Date")
        st.dataframe(forecast_table.tail())
