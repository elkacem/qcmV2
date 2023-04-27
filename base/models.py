from django.db import models

class Section(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)

class Image(models.Model):
    section = models.ForeignKey(Section, on_delete=models.CASCADE, related_name='images')
    image = models.ImageField(upload_to='', default='pictures/None/none.jpg')
