from django import forms

class PredictionForm(forms.Form):
    ticker = forms.CharField(label='Crypto Ticker (e.g., BTC-USD)', max_length=10)
    num_days = forms.IntegerField(label='Days to Predict', min_value=1, max_value=365)
