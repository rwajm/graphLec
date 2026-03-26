from django.urls import path

from . import views

urlpatterns = [
    path("", views.demo_home, name="demo_home"),
    path("api/lectures/", views.api_lectures, name="api_lectures"),
    path("api/query/", views.api_query, name="api_query"),
    path("api/graph/full/", views.api_full_graph, name="api_full_graph"),
]
