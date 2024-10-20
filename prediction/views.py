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
                data = yf.download(stock_ticker, start='2022-01-01', end='2024-10-14')
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

                # Calculate price drift
                drift = predicted_data['Predicted'][:overlap_period] - data[price_column][-overlap_period:]

                # Create the drift figure
                drift_trace = go.Scatter(
                    x=predicted_data.index[:overlap_period],
                    y=drift,
                    mode='lines+markers',
                    name='Price Drift',
                    line=dict(color='red'),
                )
                
                drift_fig = go.Figure(data=[drift_trace], layout=go.Layout(
                    title='Predicted Price Drift (Predicted - Actual)',
                    xaxis=dict(title='Date', color='white'),
                    yaxis=dict(title='Drift', color='white'),
                    paper_bgcolor='rgba(30, 30, 30, 1)',  # Dark background for the entire paper
                    plot_bgcolor='rgba(30, 30, 30, 1)',   # Dark background for the plot area
                    font=dict(color='white')                # Set font color for axes and title
                ))

                # HTML for the drift graph
                drift_graph_div = drift_fig.to_html(full_html=False)

                # Plot real vs predicted prices
                real_trace = go.Scatter(x=real_and_predicted.index, y=real_and_predicted[price_column],
                                        mode='lines', name='Real Prices', line=dict(color='blue'))
                predicted_trace = go.Scatter(x=predicted_data.index, y=predicted_data['Predicted'],
                                             mode='lines', name='Predicted Prices (GBM)', line=dict(color='orange'))

                # Define your layout with updated x-axis for July and January
                layout = go.Layout(
                    title=f'{stock_ticker} Stock Price Prediction',
                    xaxis=dict(
                        title='Date',
                        color='white',
                        tickvals=[
                            pd.Timestamp(year, month, day) 
                            for year in range(real_and_predicted.index.year.min(), future_dates.year.max() + 1)
                            for month, day in [(1, 1), (7, 1)]  # January 1 and July 1
                        ],
                        ticktext=[f'{month} {year}' 
                                  for year in range(real_and_predicted.index.year.min(), future_dates.year.max() + 1) 
                                  for month in ['January', 'July']],
                        range=[real_and_predicted.index.min(), future_dates[-1] + pd.DateOffset(months=3)],  # Extend range 3 months beyond prediction
                    ),
                    yaxis=dict(title='Price', color='white'),  # Set y-axis title color
                    legend=dict(x=0, y=1, font=dict(color='white')),  # Set legend font color
                    paper_bgcolor='rgba(30, 30, 30, 1)',  # Dark background for the entire paper
                    plot_bgcolor='rgba(30, 30, 30, 1)',   # Dark background for the plot area
                    font=dict(color='white')                # Set font color for axes and title
                )

                # Create the figure with updated layout
                fig = go.Figure(data=[real_trace, predicted_trace], layout=layout)

                # HTML for the graph
                graph_div = fig.to_html(full_html=False)

                # Add variables to context
                context['graph'] = graph_div
                context['drift_graph'] = drift_graph_div  # Add the drift graph to context
                context['stock_ticker'] = stock_ticker
                context['mse'] = mse
                context['mape'] = mape * 100  # Convert to percentage
                context['r2'] = r2
                context['historical_volatility'] = historical_volatility * 100  # Convert to percentage
                context['simulated_volatility'] = simulated_volatility * 100  # Convert to percentage

            except Exception as e:
                context['error'] = str(e)

    return render(request, 'prediction/stock_prediction.html', context)
