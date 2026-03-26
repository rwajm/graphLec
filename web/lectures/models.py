from django.db import models


class Lecture(models.Model):
    """
    강의 메타. 파이프라인 산출물의 stem과 1:1로 매핑한다.
    FastAPI/LanceDB 필터는 stem 문자열을 사용한다.
    """

    stem = models.CharField(max_length=255, unique=True, db_index=True)
    title = models.CharField(max_length=500, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["id"]

    def __str__(self) -> str:
        return f"{self.title or self.stem} ({self.stem})"
