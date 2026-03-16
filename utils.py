"""
공통 유틸리티 함수
"""

import subprocess
import time


def api_call_with_retry(func, max_retries=5, initial_wait=10):
    """API 호출 재시도 (429, 503, 500 에러 처리)"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e)
            retry_errors = ["429", "503", "500", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "overloaded"]
            if any(code in error_msg for code in retry_errors):
                print(f"API ERROR: {error_msg}")
                time.sleep(initial_wait * (attempt + 1))
            else:
                raise e
    raise Exception("API 호출 실패")


def get_video_duration(file_path: str) -> float:
    """영상 길이 추출 (초)"""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        file_path
    ], capture_output=True, text=True)
    return float(result.stdout.strip())
