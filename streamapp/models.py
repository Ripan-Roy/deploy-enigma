from django.db import models

# Create your models here.
# FaceDetection

class FaceDetection(models.Model):
    datetime = models.DateTimeField(auto_now_add=True)
    is_face_detected = models.BooleanField(default=False)
    temp_field = models.CharField(max_length=100)