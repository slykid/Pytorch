from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.register),  # views.py 에 선언된 함수를 사용해준다.
]
