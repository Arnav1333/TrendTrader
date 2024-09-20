from django.shortcuts import render
from .forms import StockTickerForm
import requests
from textblob import TextBlob

def stock_news_view(request):
    form = StockTickerForm()
    news_articles = []
    stock_ticker = ""

    if request.method == "POST":
        form = StockTickerForm(request.POST)
        if form.is_valid():
            stock_ticker = form.cleaned_data['stock_ticker']
            api_token = 'zOYK2SiSKcWtlcFBvNiGR2EaSyGTlSc2g67BXahq'  # Replace with your actual API token
            url = f"https://api.marketaux.com/v1/news/all?symbols={stock_ticker}&filter_entities=true&api_token={api_token}"
            response = requests.get(url)
            if response.status_code == 200:
                news_articles = response.json().get('data', [])
                
                # Perform sentiment analysis on each article
                for article in news_articles:
                    description = article.get('description', '')
                    # Analyze the sentiment of the description
                    if description:
                        sentiment_analysis = TextBlob(description).sentiment
                        article['sentiment_polarity'] = sentiment_analysis.polarity  # between -1 (negative) and 1 (positive)
                        article['sentiment_subjectivity'] = sentiment_analysis.subjectivity  # between 0 (objective) and 1 (subjective)

    return render(request, 'news/stock_ticker_search.html', {
        'form': form,
        'news_articles': news_articles,
        'stock_ticker': stock_ticker
    })
