from django.contrib import admin

# Register your models here.
from .models import Image,Section

admin.site.register(Section)
admin.site.register(Image)
