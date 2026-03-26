"""
Django settings for GraphLEC web (프록시 + 강의 메타).

로컬: USE_SQLITE=1 (기본) → db.sqlite3
운영: USE_SQLITE=0 + POSTGRES_* 환경변수
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "dev-only-change-me-in-production",
)

DEBUG = os.getenv("DJANGO_DEBUG", "1") == "1"

ALLOWED_HOSTS = [
    h.strip()
    for h in os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")
    if h.strip()
]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "lectures.apps.LecturesConfig",
    "api.apps.ApiConfig",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "graphlec_site.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "graphlec_site.wsgi.application"

USE_SQLITE = os.getenv("USE_SQLITE", "1") == "1"

if USE_SQLITE:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.getenv("POSTGRES_DB", "graphlec"),
            "USER": os.getenv("POSTGRES_USER", "graphlec"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
            "HOST": os.getenv("POSTGRES_HOST", "localhost"),
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
        }
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "ko-kr"
TIME_ZONE = "Asia/Seoul"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# FastAPI 질의 서비스 (Django가 서버-서버로 호출)
QUERY_SERVICE_URL = os.getenv("QUERY_SERVICE_URL", "http://127.0.0.1:8001")
QUERY_SERVICE_TIMEOUT = float(os.getenv("QUERY_SERVICE_TIMEOUT", "120"))

# 파이프라인 산출물 ({stem}_nodes.parquet 등). 기본: 저장소 루트의 output/
_GRAPH_DEFAULT = BASE_DIR.parent / "output"
GRAPHLEC_OUTPUT_DIR = Path(
    os.getenv("GRAPHLEC_OUTPUT_DIR", str(_GRAPH_DEFAULT))
).resolve()
