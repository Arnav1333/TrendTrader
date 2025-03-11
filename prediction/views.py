import numpy as np
import pandas as pd
import plotly.graph_objects as go  # Import Plotly
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import yfinance as yf
import io
import base64
from django.shortcuts import render
import plotly.io as pio # import plotly io

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
    
    # Convert pandas Series to float if needed
    if hasattr(S0, 'iloc'):
        S0 = float(S0.iloc[0])
    if hasattr(mu, 'iloc'):
        mu = float(mu.iloc[0])
    if hasattr(sigma, 'iloc'):
        sigma = float(sigma.iloc[0])
        
    S = S0 * np.exp((mu - 0.5 * sigma**2) * t + sigma * W)
    return t, S

def stock_prediction_view(request):
    if request.method == 'POST':
        ticker = request.POST.get('ticker', 'MSFT')
        start_date = request.POST.get('start_date', '2015-01-01')
        end_date = request.POST.get('end_date', '2025-12-12')
        prediction_months = int(request.POST.get('prediction_months', 1))  # Get prediction months from form
        
        # Calculate future date range
        future_days = prediction_months * 30  # Approximate days in a month
        
        data = load_stock_data(ticker, start_date, end_date)
        X, y, scaler = preprocess_data(data, feature='Close', window_size=60)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = build_lstm_model(input_shape=(X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

        # Make predictions on test data
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Generate future predictions
        last_60_days = data['Close'].values[-60:].reshape(-1, 1)
        last_60_days_scaled = scaler.transform(last_60_days)
        
        # Prepare future date range
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
        
        # Generate future predictions
        future_predictions = []
        current_batch = last_60_days_scaled.reshape((1, 60, 1))
        
        for _ in range(future_days):
            # Get prediction for next day
            current_pred = model.predict(current_batch)[0]
            future_predictions.append(current_pred)
            
            # Update the batch to include the prediction
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
        
        # Convert future predictions to original scale
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Apply GBM to future predictions - using .iloc[0] to avoid warnings
        S0 = float(data['Close'].iloc[-1])
        mu = float(data['Close'].pct_change().mean().iloc[0] * 252)
        sigma = float(data['Close'].pct_change().std().iloc[0] * np.sqrt(252))
        T = prediction_months / 12  # Convert months to years
        
        t, gbm_future = geometric_brownian_motion(S0, mu, sigma, T, future_days)
        
        # Blend future predictions
        blended_future = 0.9 * future_predictions.flatten() + 0.1 * gbm_future[-len(future_predictions):]
        
        # Current predictions (test data)
        actual_dates = data.index[-len(y_test):]
        
        # Create the plot
        fig = go.Figure()
        
        # Plot historical data and predictions
        fig.add_trace(go.Scatter(x=actual_dates, y=y_test_actual.flatten(), mode='lines', 
                                 name='Historical Prices', line=dict(color='#00FFFF')))  # Cyan color for historical
        fig.add_trace(go.Scatter(x=actual_dates, y=predictions.flatten(), mode='lines', 
                                 name='Predictions'))
        
        # Add a vertical line to distinguish between historical and future data
        fig.add_vline(x=data.index[-1], line_width=1, line_dash="dash", line_color="gray")
        
        # Plot only the blended future prediction
        fig.add_trace(go.Scatter(x=future_dates, y=blended_future, mode='lines', 
                                line=dict(dash='dash', color='red'), name='Forecasted Prices'))
        
        
        fig.update_layout(
            title=f'{ticker} Stock Price Prediction', 
            xaxis_title='Date', 
            yaxis_title='Stock Price (USD)',
            hovermode='x unified',
            plot_bgcolor='#020a0e',  
            paper_bgcolor='#020a0e',  
            font=dict(
                color='white'  
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            xaxis=dict(
                gridcolor='#333333',
                zerolinecolor='#333333'
            ),
            yaxis=dict(
                gridcolor='#333333',
                zerolinecolor='#333333'
            ),
            margin=dict(l=20, r=20, t=40, b=20)
        )

        plot_div = pio.to_html(fig, full_html=False)
        
        # Get the last historical price and final predictions
        final_lstm_pred = float(future_predictions[-1][0])
        final_gbm_pred = float(gbm_future[-1])
        final_blended_pred = float(blended_future[-1])
        
        last_price = float(data['Close'].iloc[-1])
        lstm_change = ((final_lstm_pred - last_price) / last_price) * 100
        gbm_change = ((final_gbm_pred - last_price) / last_price) * 100
        blended_change = ((final_blended_pred - last_price) / last_price) * 100

        return render(request, 'prediction/stock_prediction.html', {
            'plot_div': plot_div,
            'stock_ticker': ticker,
            'lstm_prediction': f'${final_lstm_pred:.2f} ({lstm_change:+.2f}%)',
            'gbm_prediction': f'${final_gbm_pred:.2f} ({gbm_change:+.2f}%)',
            'blended_prediction': f'${final_blended_pred:.2f} ({blended_change:+.2f}%)',
            'last_price': f'${last_price:.2f}',
            'prediction_months': prediction_months
        })
    return render(request, 'prediction/stock_prediction.html')