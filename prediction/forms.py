from django import forms

class StockPredictionForm(forms.Form):
    stock_ticker = forms.CharField(label='Stock Ticker', max_length=20)
    time_horizon = forms.IntegerField(label='Time Horizon (in months)', min_value=1, max_value=12)
