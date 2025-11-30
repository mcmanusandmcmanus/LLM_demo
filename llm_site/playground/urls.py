from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("api/generate", views.generate, name="generate"),
]
