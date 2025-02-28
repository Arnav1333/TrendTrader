import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import yfinance as yf
import io
import base64
from django.shortcuts import render

def load_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data, feature='Close', window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def geometric_brownian_motion(S0, mu, sigma, T, steps):
    dt = T / steps
    t = np.linspace(0, T, steps)
    W = np.random.normal(0, np.sqrt(dt), size=steps).cumsum()
    S0 = S0.iloc[0]
    mu = mu.iloc[0]
    sigma = sigma.iloc[0]
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return t, S

def stock_prediction_view(request):
    if request.method == 'POST':
        ticker = request.POST.get('ticker', 'MSFT')
        start_date = request.POST.get('start_date', '2015-01-01')
        end_date = request.POST.get('end_date', '2025-01-01')

        data = load_stock_data(ticker, start_date, end_date)
        X, y, scaler = preprocess_data(data, feature='Close', window_size=60)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = build_lstm_model(input_shape=(X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        S0 = data['Close'].iloc[-1]
        mu = data['Close'].pct_change().mean() * 252
        sigma = data['Close'].pct_change().std() * np.sqrt(252)
        T = 1
        steps = len(y_test)

        t, gbm_prices = geometric_brownian_motion(S0, mu, sigma, T, steps)

        gbm_prices_resized = gbm_prices[-len(predictions):]
        blended_predictions = 0.9 * predictions.flatten() + 0.1 * gbm_prices_resized

        mse_lstm = mean_squared_error(y_test_actual, predictions)
        mse_gbm = mean_squared_error(y_test_actual, gbm_prices_resized)
        mse_blended = mean_squared_error(y_test_actual, blended_predictions)

        actual_dates = data.index[-len(y_test):]

        if len(y_test_actual) != len(predictions) or len(predictions) != len(blended_predictions):
            error_message = "Length mismatch detected!"
            return render(request, 'stock_prediction.html', {'error_message': error_message})
        else:
            plt.figure(figsize=(18, 7))
            plt.plot(actual_dates, y_test_actual, color='blue', label='Actual Prices')
            plt.plot(actual_dates, predictions, color='red', label='LSTM Predictions')
            plt.plot(actual_dates, blended_predictions, color='green', label='Blended Predictions')
            plt.title(f'{ticker} Stock Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Stock Price')
            plt.legend()

            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            graphic = base64.b64encode(image_png).decode('utf-8')

            lstm_prediction = predictions[-1][0]
            gbm_prediction = gbm_prices[-1]
            blended_prediction = blended_predictions[-1]

            return render(request, 'prediction/stock_prediction.html', {
                'graphic': graphic,
                'lstm_prediction': f'{lstm_prediction:.2f}',
                'gbm_prediction': f'{gbm_prediction:.2f}',
                'blended_prediction': f'{blended_prediction:.2f}',
                'mse_lstm': f'{mse_lstm:.4f}',
                'mse_gbm': f'{mse_gbm:.4f}',
                'mse_blended': f'{mse_blended:.4f}',
            })
    return render(request, 'prediction/stock_prediction.html')