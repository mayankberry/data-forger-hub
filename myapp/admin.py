from django.contrib import admin
from .models import CsvFile

@admin.register(CsvFile)
class CsvDataAdmin(admin.ModelAdmin):
    list_display = ['file', 'id'] 
    search_fields = ['file__name']  