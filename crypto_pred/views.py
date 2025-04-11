import pandas as pd
import plotly.graph_objs as go
from django.shortcuts import render
from datetime import datetime, timedelta
from prophet import Prophet
import yfinance as yf

def crypto_forecast_view(request):
    
    ticker = request.GET.get("ticker", "BTC-USD")
    num_days = int(request.GET.get("days", 30))

    try:
        df = yf.download(ticker, period="max")
        df = df.reset_index()

        if len(df) == 0:
            return render(request, "crypto_graph.html", {
                "error": "No data available for this ticker",
                "ticker": ticker,
                "days": num_days
            })

        df = df[["Date", "Close"]]
        df.columns = ["ds", "y"]

        model = Prophet()
        model.fit(df)

        future = model.make_future_dataframe(periods=num_days)
        forecast = model.predict(future)

        # Plot actual and forecasted prices
        fig = go.Figure()

        # Actual price
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Actual Price', line=dict(color='cyan')))

        # Forecast start marker
        forecast_start = df['ds'].max()
        fig.add_trace(go.Scatter(x=[forecast_start, forecast_start], y=[df['y'].min(), df['y'].max()],
                                 mode='lines', name='Prediction Start', line=dict(dash='dot', color='orange')))

        # Predicted price
        fig.add_trace(go.Scatter(x=forecast['ds'][-num_days:], y=forecast['yhat'][-num_days:], mode='lines',
                                 name='Predicted Price', line=dict(color='red')))

        fig.update_layout(
            title=f'{ticker} Price Forecast ({num_days} days)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            template='plotly_dark',
            height=600
        )

        graph_html = fig.to_html(full_html=False)

        return render(request, "crypto_graph.html", {
            "graph_html": graph_html,
            "ticker": ticker,
            "days": num_days
        })

    except Exception as e:
        return render(request, "crypto_graph.html", {
            "error": str(e),
            "ticker": ticker,
            "days": num_days
        })
