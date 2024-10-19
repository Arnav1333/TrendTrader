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
            api_token = 'zOYK2SiSKcWtlcFBvNiGR2EaSyGTlSc2g67BXahq'  
            url = f"https://api.marketaux.com/v1/news/all?symbols={stock_ticker}&filter_entities=true&api_token={api_token}"
            response = requests.get(url)
            if response.status_code == 200:
                news_articles = response.json().get('data', [])
                
              
                for article in news_articles:
                    description = article.get('description', '')
                    
                    if description:
                        sentiment_analysis = TextBlob(description).sentiment
                        article['sentiment_polarity'] = sentiment_analysis.polarity 
                        article['sentiment_subjectivity'] = sentiment_analysis.subjectivity  

    return render(request, 'news/stock_ticker_search.html', {
        'form': form,
        'news_articles': news_articles,
        'stock_ticker': stock_ticker
    })
