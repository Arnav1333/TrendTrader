from django import forms

class StockTickerForm(forms.Form):
    stock_ticker = forms.CharField(max_length=10, label='Enter Stock Ticker')
