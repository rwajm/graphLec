from django.contrib import admin

from .models import Lecture


@admin.register(Lecture)
class LectureAdmin(admin.ModelAdmin):
    list_display = ("id", "stem", "title", "created_at")
    search_fields = ("stem", "title")
