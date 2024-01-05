from django import forms
from .models import CsvFile

class CsvUploadForm(forms.ModelForm):
    class Meta:
        model = CsvFile
        fields = ['file']