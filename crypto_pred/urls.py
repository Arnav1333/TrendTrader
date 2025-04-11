from django.urls import path
from . import views


urlpatterns = [
 path('crypto/', views.crypto_forecast_view , name="crypto_forecast"),
]
