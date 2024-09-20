from django.urls import path
from . import views

urlpatterns = [
    path('prediction/', views.stock_prediction_view, name='stock_prediction'),
]
