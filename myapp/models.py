from django.db import models

# Create your models here.

class CsvFile(models.Model):
    file = models.FileField(upload_to='csv_files/')

    def __str__(self):
        return self.file.name
