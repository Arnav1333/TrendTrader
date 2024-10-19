from django.shortcuts import render
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import plotly.graph_objs as go
from .forms import StockPredictionForm

# Function to calculate daily returns
def calculate_daily_returns(data):
    daily_returns = data['Close'].pct_change().dropna()
    return daily_returns

# Function for Geometric Brownian Motion (GBM) for price prediction
def geometric_brownian_motion(S0, mu, sigma, T, dt, N):
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Cumulative sum for Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)  # GBM formula
    return S

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
                # Fetch stock data from Yahoo Finance
                data = yf.download(stock_ticker, start='2022-01-01', end='2024-09-14')
                price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                daily_returns = calculate_daily_returns(data)

                # Parameters for GBM
                mu = daily_returns.mean() * 252  # Annualized mean return
                sigma = daily_returns.std() * np.sqrt(252)  # Annualized volatility
                S0 = data[price_column].iloc[-1]  # Last closing price
                T = time_horizon / 12  # Time horizon in years
                dt = 1 / 252  # Daily steps
                N = int(T / dt)  # Number of steps

                # Predict future stock prices using GBM
                predicted_prices = geometric_brownian_motion(S0, mu, sigma, T, dt, N)

                # Create future date index
                last_date = data.index[-1]
                future_dates = pd.date_range(last_date, periods=N, freq='B')

                # Create DataFrame for predicted data
                predicted_data = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted'])
                real_and_predicted = pd.concat([data[[price_column]], predicted_data])

                # Calculate MSE, MAPE, and R2
                overlap_period = min(len(data), len(predicted_data))
                mse = mean_squared_error(data[price_column][-overlap_period:], predicted_data['Predicted'][:overlap_period])
                mape = mean_absolute_percentage_error(data[price_column][-overlap_period:], predicted_data['Predicted'][:overlap_period])
                r2 = r2_score(data[price_column][-overlap_period:], predicted_data['Predicted'][:overlap_period])

                # Historical and simulated volatility
                historical_volatility = daily_returns.std() * np.sqrt(252)  # Annualized historical volatility
                predicted_returns = predicted_data['Predicted'].pct_change().dropna()  # Predicted returns
                simulated_volatility = predicted_returns.std() * np.sqrt(252)  # Annualized simulated volatility

                # Plot real vs predicted prices
                real_trace = go.Scatter(x=real_and_predicted.index, y=real_and_predicted[price_column],
                                        mode='lines', name='Real Prices', line=dict(color='blue'))
                predicted_trace = go.Scatter(x=predicted_data.index, y=predicted_data['Predicted'],
                                             mode='lines', name='Predicted Prices (GBM)', line=dict(color='orange'))
                layout = go.Layout(
                    title=f'{stock_ticker} Stock Price Prediction',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Price'),
                    legend=dict(x=0, y=1)
                )
                fig = go.Figure(data=[real_trace, predicted_trace], layout=layout)

                # HTML for the graph
                graph_div = fig.to_html(full_html=False)

                # Add variables to context
                context['graph'] = graph_div
                context['stock_ticker'] = stock_ticker
                context['mse'] = mse
                context['mape'] = mape * 100  # Percentage
                context['r2'] = r2
                context['historical_volatility'] = historical_volatility * 100  # Percentage
                context['simulated_volatility'] = simulated_volatility * 100  # Percentage

            except Exception as e:
                context['error'] = str(e)

    return render(request, 'prediction/stock_prediction.html', context)
