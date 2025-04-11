# yourapp/ml_utils.py
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import timedelta, date

def fetch_crypto_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if data.empty:
        print(f"No data returned for {ticker}")
    else:
        print(f"Data sample for {ticker}: {data.head(2)}")  
    return data

def create_features(df):
    df['Target'] = df['Close'].shift(-1)
    df['PrevClose'] = df['Close'].shift(1)
    df['PrevOpen'] = df['Open'].shift(1)
    df['Returns'] = df['Close'].pct_change()
    df['MA7'] = df['Close'].rolling(window=7).mean().shift(1)
    df['MA90'] = df['Close'].rolling(window=90).mean().shift(1)  
    df['Volume_MA7'] = df['Volume'].rolling(window=7).mean().shift(1)
    
    df_features = df.dropna()
    features = ['PrevClose', 'PrevOpen', 'Returns', 'MA7', 'MA90', 'Volume_MA7']  
    X = df_features[features]
    y = df_features['Target']
    return X, y, df_features

def train_random_forest_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate model performance
    y_pred = rf_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return rf_model, scaler, (X_train, X_test, y_train, y_test), mae, r2

def forecast_future_prices(model, scaler, last_data, ticker, num_days=30):  # make sure ticker is passed
    current_features = last_data.iloc[-1][['PrevClose', 'PrevOpen', 'Returns', 'MA7', 'MA90', 'Volume_MA7']].values.reshape(1, -1)
    current_features_scaled = scaler.transform(current_features)
    future_prices = []
    last_date = last_data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(num_days)]
    last_price = current_features[0][0]
    if ticker == "BTC-USD":
        base_volatility = 0.03
    else:
        base_volatility = 0.02
    trend_factor = 1.005
    randomness_factor = 0.01
    for _ in range(num_days):
        base_prediction = model.predict(current_features_scaled)[0]
        noise = np.random.normal(0, base_volatility * base_prediction)
        trend_adjustment = (base_prediction * trend_factor - base_prediction)
        random_adjustment = np.random.uniform(-randomness_factor, randomness_factor) * base_prediction
        next_price = base_prediction + noise + trend_adjustment + random_adjustment
        future_prices.append(next_price)
        current_features = np.array([
            next_price, next_price, (next_price - last_price) / last_price,
            np.mean(future_prices[-7:]) if len(future_prices) >= 7 else next_price,
            np.mean(future_prices[-90:]) if len(future_prices) >= 90 else next_price,
            np.mean(future_prices[-7:])
        ]).reshape(1, -1)
        last_price = next_price
        current_features_scaled = scaler.transform(current_features)
    return future_dates, future_prices
