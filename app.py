
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
    # Data Scaling for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Sales'].values.reshape(-1, 1))

    # Prepare data for LSTM (X, y creation)
    look_back = 10  # Number of previous days to predict next day's sales
    X_lstm, y_lstm = [], []
    for i in range(look_back, len(scaled_data)):
        X_lstm.append(scaled_data[i-look_back:i, 0])  # Taking look_back days' data
        y_lstm.append(scaled_data[i, 0])  # Target is the next day's sales
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    # Reshape X_lstm to be 3D (samples, time_steps, features)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))

    # Define LSTM Model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=False, input_shape=(X_lstm.shape[1], 1)))
    model.add(Dense(units=1))

    # Compile and Fit the Model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_lstm, y_lstm, epochs=10, batch_size=32)

    # Predict the Sales (using the same reshaped data)
    lstm_predictions = model.predict(X_lstm)

    # Inverse transform predictions to original scale
    predicted_sales_lstm = scaler.inverse_transform(lstm_predictions)

# Ensure the predictions length matches the test set length
forecast_arima = forecast_arima[:len(test)]
lr_pred = lr_pred[:len(test)]
rf_pred = rf_pred[:len(test)]
predicted_sales_lstm = predicted_sales_lstm[:len(test)]

# Forecast Visualizations (Separate for each model)
st.subheader("Forecast Visualizations")

for model_name, prediction in zip([
    "ARIMA", "Linear Regression", "Random Forest", "LSTM"],
    [forecast_arima, lr_pred, rf_pred, predicted_sales_lstm]
):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data.index, data['Sales'], label='Historical Sales', linewidth=2)
    ax.plot(test.index, prediction, label=f'{model_name} Forecast', linestyle='--')
    ax.set_title(f"{model_name} Sales Forecast")
    ax.set_xlabel('Month')
    ax.set_ylabel('Sales')
    ax.legend()
    st.pyplot(fig)
