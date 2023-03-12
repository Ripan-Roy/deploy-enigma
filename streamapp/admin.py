from django.contrib import admin
from .models import *
# Register your models here.

class FaceDetectionView(admin.ModelAdmin):
    list_display = ['datetime','is_face_detected','temp_field']

admin.site.register(FaceDetection,FaceDetectionView)