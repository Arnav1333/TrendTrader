from django.shortcuts import render
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from .forms import StockPredictionForm

# Function to calculate daily returns
def calculate_daily_returns(data):
    daily_returns = data['Close'].pct_change().dropna()
    return daily_returns

# Function for Geometric Brownian Motion (GBM)
def geometric_brownian_motion(S0, mu, sigma, T, dt, N):
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return S

# LSTM Model Functions
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

# Stock Prediction View
def stock_prediction_view(request):
    form = StockPredictionForm()
    context = {'form': form}

    if request.method == 'POST':
        form = StockPredictionForm(request.POST)
        if form.is_valid():
            stock_ticker = form.cleaned_data['stock_ticker']
            time_horizon = form.cleaned_data['time_horizon']

            try:
                # Fetch stock data
                data = yf.download(stock_ticker, start='2022-01-01', end='2024-12-14')
                price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                
                # Prepare data for LSTM
                X, y, scaler = preprocess_data(data, feature=price_column, window_size=60)
                
                # Split data for LSTM
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                # Build and train LSTM model
                model = build_lstm_model(input_shape=(X_train.shape[1], 1))
                model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)
                
                # GBM Parameters
                daily_returns = calculate_daily_returns(data)
                mu = daily_returns.mean() * 252
                sigma = daily_returns.std() * np.sqrt(252)
                S0 = data[price_column].iloc[-1]
                T = time_horizon / 12
                dt = 1 / 252
                N = int(T / dt)
                
                # Generate predictions
                gbm_predictions = geometric_brownian_motion(S0, mu, sigma, T, dt, N)
                
                # Generate LSTM predictions
                last_sequence = X[-1].reshape(1, X.shape[1], 1)
                lstm_predictions = []
                
                for _ in range(N):
                    next_pred = model.predict(last_sequence, verbose=0)
                    lstm_predictions.append(next_pred[0, 0])
                    last_sequence = np.roll(last_sequence[0], -1)
                    last_sequence[-1] = next_pred
                    last_sequence = last_sequence.reshape(1, X.shape[1], 1)
                
                # Inverse transform LSTM predictions
                lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1))
                
                # Blend predictions (90% LSTM, 10% GBM)
                blended_predictions = 0.9 * lstm_predictions.flatten() + 0.1 * gbm_predictions
                
                # Create future date index
                last_date = data.index[-1]
                future_dates = pd.date_range(last_date, periods=N, freq='B')
                
                # Create DataFrames
                predicted_data = pd.DataFrame({
                    'GBM': gbm_predictions,
                    'LSTM': lstm_predictions.flatten(),
                    'Blended': blended_predictions
                }, index=future_dates)
                
                # Calculate metrics
                historical_data = data[price_column][-N:]
                mse = mean_squared_error(historical_data, predicted_data['Blended'][:len(historical_data)])
                mape = mean_absolute_percentage_error(historical_data, predicted_data['Blended'][:len(historical_data)])
                r2 = r2_score(historical_data, predicted_data['Blended'][:len(historical_data)])
                
                # Create plots
                traces = [
                    go.Scatter(x=data.index, y=data[price_column], mode='lines', name='Historical', line=dict(color='blue')),
                    go.Scatter(x=future_dates, y=predicted_data['GBM'], mode='lines', name='GBM', line=dict(color='green')),
                    go.Scatter(x=future_dates, y=predicted_data['LSTM'], mode='lines', name='LSTM', line=dict(color='red')),
                    go.Scatter(x=future_dates, y=predicted_data['Blended'], mode='lines', name='Blended Forecast', line=dict(color='purple'))
                ]
                
                layout = go.Layout(
                    title=f'{stock_ticker} Stock Price Prediction',
                    xaxis=dict(title='Date', color='white'),
                    yaxis=dict(title='Price', color='white'),
                    legend=dict(font=dict(color='white')),
                    paper_bgcolor='rgba(30, 30, 30, 1)',
                    plot_bgcolor='rgba(30, 30, 30, 1)',
                    font=dict(color='white')
                )
                
                fig = go.Figure(data=traces, layout=layout)
                graph_div = fig.to_html(full_html=False)
                
              
                context.update({
                    'graph': graph_div,
                    'stock_ticker': stock_ticker,
                    'mse': mse,
                    'mape': mape * 100,
                    'r2': r2,
                    'lstm_forecast': lstm_predictions[-1][0],
                    'gbm_forecast': gbm_predictions[-1],
                    'blended_forecast': blended_predictions[-1]
                })

            except Exception as e:
                context['error'] = str(e)

    return render(request, 'prediction/stock_prediction.html', context)