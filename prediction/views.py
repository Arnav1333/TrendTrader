from django.shortcuts import render
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error
import plotly.graph_objs as go
from .forms import StockPredictionForm

# Function to calculate daily returns
def calculate_daily_returns(data):
    daily_returns = data['Close'].pct_change().dropna()
    return daily_returns

# GBM Function
def geometric_brownian_motion(S0, mu, sigma, T, dt, N):
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)
    return S

def stock_prediction_view(request):
    form = StockPredictionForm()
    context = {'form': form}

    if request.method == 'POST':
        form = StockPredictionForm(request.POST)
        if form.is_valid():
            stock_ticker = form.cleaned_data['stock_ticker']
            time_horizon = form.cleaned_data['time_horizon']
            try:
                # Fetch historical stock data
                data = yf.download(stock_ticker, start='2022-01-01', end='2024-09-14')

                price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
                daily_returns = calculate_daily_returns(data)

                # Calculate parameters
                mu = daily_returns.mean() * 252
                sigma = daily_returns.std() * np.sqrt(252)
                S0 = data[price_column].iloc[-1]
                T = time_horizon / 12
                dt = 1 / 252
                N = int(T / dt)

                # Run GBM
                predicted_prices = geometric_brownian_motion(S0, mu, sigma, T, dt, N)

                # Generate future dates
                last_date = data.index[-1]
                future_dates = pd.date_range(last_date, periods=N, freq='B')

                # Create a DataFrame for predicted prices
                predicted_data = pd.DataFrame(predicted_prices, index=future_dates, columns=['Predicted'])
                real_and_predicted = pd.concat([data[[price_column]], predicted_data])

                # Calculate Mean Squared Error
                overlap_period = min(len(data), len(predicted_data))
                mse = mean_squared_error(data[price_column][-overlap_period:], predicted_data['Predicted'][:overlap_period])

                # Create a Plotly graph for real and predicted prices
                real_trace = go.Scatter(x=real_and_predicted.index, y=real_and_predicted[price_column], mode='lines', name='Real Prices', line=dict(color='blue'))
                predicted_trace = go.Scatter(x=predicted_data.index, y=predicted_data['Predicted'], mode='lines', name='Predicted Prices (GBM)', line=dict(color='orange'))

                layout = go.Layout(
                    title=f'{stock_ticker} Stock Price Prediction',
                    xaxis=dict(title='Date'),
                    yaxis=dict(title='Price'),
                    legend=dict(x=0, y=1)
                )

                fig = go.Figure(data=[real_trace, predicted_trace], layout=layout)

                # Convert Plotly figure to HTML
                graph_div = fig.to_html(full_html=False)

                # Pass the graph and MSE to the template
                context['graph'] = graph_div
                context['stock_ticker'] = stock_ticker

            except Exception as e:
                context['error'] = str(e)

    return render(request, 'prediction/stock_prediction.html', context)
