"""
API 클라이언트 및 설정 초기화
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq
from google import genai

load_dotenv()

# API 키 로드 (절대 하드코딩 금지)
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

missing_keys: list[str] = []
if not GEMINI_API_KEY:
    missing_keys.append("GOOGLE_API_KEY/GEMINI_API_KEY")
if not GROQ_API_KEY:
    missing_keys.append("GROQ_API_KEY")

if missing_keys:
    print("❌ 필요한 API 키를 환경 변수로 설정해주세요:")
    for k in missing_keys:
        print(f"   - {k}")
    sys.exit(1)

# 클라이언트 초기화
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
