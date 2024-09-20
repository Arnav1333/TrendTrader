from django.urls import path
from .views import stock_news_view

urlpatterns = [
    path('news/', stock_news_view, name='stock_news'),  # The single-page for searching news and displaying results
]
