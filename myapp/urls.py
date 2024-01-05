from django.contrib import admin
from django.urls import path
from myapp import views

urlpatterns = [
    path("", views.base, name='base'),
    path("fileupload/", views.fileupload, name='fileupload'),
    path("choose_algo/", views.choose_algo, name='choose_algo'),
    path("column_select/<str:algorithm>/", views.column_select, name='column_select'),
    path("predictions/", views.predictions, name='predictions'),
    path("compute_regression_algorithm/", views.compute_regression_algorithm, name='compute_regression_algorithm'),
    path("compute_knn_algorithm/<str:algorithm>/<int:k_neighbors>/", views.compute_knn_algorithm, name='compute_knn_algorithm'),

]
