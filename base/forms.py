from django import forms
from .models import Section, Image

class SectionForm(forms.ModelForm):
    class Meta:
        model = Section
        fields = ['name','description']

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['image']
        widgets = {
            'image': forms.ClearableFileInput(attrs={'multiple': True}),
        }