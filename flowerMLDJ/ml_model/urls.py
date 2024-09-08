from django.urls import path
from .views import classify

urlpatterns = [
    path('', classify, name='main')
]