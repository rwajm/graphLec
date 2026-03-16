# """
# slide_extractor.py
# ==================
# PPT 기반 강의 영상에서 슬라이드 + 필기 완료 시점 프레임을 추출합니다.

# Input : lecture.mp4
# Output: output_slides/
#         ├── slide_001_base.jpg          # 슬라이드 최초 등장 프레임
#         ├── slide_001_annot_01.jpg      # 필기 안정화 캡처 #1
#         ├── slide_001_annot_02.jpg      # 필기 안정화 캡처 #2 (있을 경우)
#         ├── slide_002_base.jpg
#         └── ...

# Usage:
#     python slide_extractor.py --input lecture.mp4 --output output_slides/
#     python slide_extractor.py --input lecture.mp4 --output output_slides/ --debug
# """

# import cv2
# import numpy as np
# import imagehash
# from PIL import Image
# from pathlib import Path
# import argparse
# import json
# import logging

# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# log = logging.getLogger(__name__)


# # ──────────────────────────────────────────────
# # 설정값 (튜닝 포인트)
# # ──────────────────────────────────────────────
# class Config:
#     # 슬라이드 전환 감지
#     SLIDE_CHANGE_MSE_THRESHOLD   = 500    # MSE 이 값 이상이면 슬라이드 전환 후보
#     SLIDE_CHANGE_HASH_THRESHOLD  = 10     # 퍼셉추얼 해시 hamming distance (0~64)
    
#     # 필기 감지
#     ANNOT_DIFF_THRESHOLD         = 15     # 픽셀 diff 절댓값 임계 (그레이스케일)
#     # 누적 diff: base 대비 현재 프레임의 변화 픽셀 비율 → "필기가 쌓였는가?"
#     ANNOT_CUMULATIVE_RATIO       = 0.0005  # 슬라이드 면적의 0.3% 이상 변하면 필기 있음
#     # 순간 diff: 직전 프레임 대비 변화 픽셀 비율 → "지금 쓰고 있는가?"
#     ANNOT_INSTANT_RATIO          = 0.0001 # 느린 필기도 잡기 위해 낮게 설정
    
#     # 안정화 판단
#     STABILITY_WINDOW_SEC         = 0.7    # 이 시간 동안 변화 없으면 '필기 완료'
#     MIN_ANNOT_DURATION_SEC       = 0.2    # 이보다 짧은 필기 이벤트는 노이즈로 무시
    
#     # 처리 성능
#     PROCESS_EVERY_N_FRAMES       = 2      # N 프레임마다 1번 처리 (속도 vs 정확도)
#     RESIZE_WIDTH                 = 960    # 내부 처리 해상도 (원본 저장은 별도)


# # ──────────────────────────────────────────────
# # 유틸
# # ──────────────────────────────────────────────
# def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
#     """두 프레임 간 MSE 계산 (그레이스케일)"""
#     a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
#     b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
#     return float(np.mean((a - b) ** 2))


# def compute_phash(frame: np.ndarray) -> imagehash.ImageHash:
#     """퍼셉추얼 해시 계산"""
#     pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     return imagehash.phash(pil_img)


# def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
#     """처리용 리사이즈"""
#     h, w = frame.shape[:2]
#     scale = width / w
#     return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


# def count_changed_pixels(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
#     """색상 채널 정보를 활용하여 변화한 픽셀 비율 반환"""
#     # 그레이스케일 대신 BGR 차이의 절대값을 구함
#     diff = cv2.absdiff(frame_a, frame_b)
#     # 세 채널 중 하나라도 threshold를 넘으면 변화한 것으로 간주
#     max_diff = np.max(diff, axis=2)
#     changed = np.sum(max_diff > threshold)
#     return changed / max_diff.size


# # ──────────────────────────────────────────────
# # 슬라이드 전환 감지기
# # ──────────────────────────────────────────────
# class SlideChangeDetector:
#     """
#     MSE + 퍼셉추얼 해시 이중 검증으로 슬라이드 전환 감지.
#     MSE만 쓰면 필기가 많을 때 오탐 발생하므로 phash를 보조 필터로 사용.
#     """
#     def __init__(self, cfg: Config):
#         self.cfg = cfg
#         self.prev_frame = None
#         self.prev_phash = None

#     def is_slide_change(self, frame: np.ndarray) -> bool:
#         if self.prev_frame is None:
#             self._update(frame)
#             return False

#         mse = compute_mse(self.prev_frame, frame)
        
#         # MSE 임계 미달 → 확실히 전환 아님
#         if mse < self.cfg.SLIDE_CHANGE_MSE_THRESHOLD:
#             self._update(frame)
#             return False

#         # MSE 초과 → phash로 이중 확인
#         curr_phash = compute_phash(frame)
#         hash_dist = self.prev_phash - curr_phash
        
#         is_change = hash_dist >= self.cfg.SLIDE_CHANGE_HASH_THRESHOLD
#         self._update(frame)
#         return is_change

#     def _update(self, frame):
#         self.prev_frame = frame.copy()
#         self.prev_phash = compute_phash(frame)


# # ──────────────────────────────────────────────
# # 필기 안정화 감지기
# # ──────────────────────────────────────────────
# class AnnotationStabilityDetector:
#     """
#     슬라이드 구간 내에서 필기 변화 추적 및 안정화 시점 감지.

#     두 가지 신호를 분리하여 사용:
#       - 누적 diff (base ↔ current): "필기가 쌓였는가?" → WRITING 진입 조건
#       - 순간 diff (prev ↔ current): "지금 쓰고 있는가?" → 안정화 판단 조건

#     연속 프레임 간 diff만 쓰면 느린 필기(프레임당 변화량 미미)를 감지 못하는
#     근본 문제를 해결하기 위해 누적 diff를 주 신호로 사용한다.

#     상태 머신:
#         STABLE → (누적 diff 임계 초과) → WRITING
#                → (순간 diff가 stability_window 동안 임계 미달) → STABLE + CAPTURE
#     """
#     def __init__(self, cfg: Config, fps: float):
#         self.cfg = cfg
#         self.fps = fps
#         self.stability_frames = int(cfg.STABILITY_WINDOW_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
#         self.min_annot_frames = int(cfg.MIN_ANNOT_DURATION_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
#         self.reset()

#     def reset(self, base_frame: np.ndarray = None):
#         """슬라이드 전환 시 리셋"""
#         self.state             = "STABLE"
#         self.base_frame        = base_frame   # 슬라이드 최초 클린 프레임
#         self.prev_frame        = base_frame
#         self.stable_count      = 0
#         self.writing_count     = 0
#         self.last_annot_frame  = None         # 가장 마지막으로 변화가 감지된 프레임
#         # 누적 diff가 임계를 넘은 이후 base를 갱신하지 않기 위한 플래그
#         self.annotation_started = False

#     def process(self, frame: np.ndarray) -> str:
#         """
#         프레임을 받아서 이벤트 반환.
#         반환값: "CAPTURE" | "NONE"

#         상태 전이:
#           STABLE  → (누적 diff >= CUMULATIVE_RATIO)          → WRITING
#           WRITING → (순간 diff < INSTANT_RATIO, 연속 N프레임) → CAPTURE + STABLE
#           WRITING → (누적 diff < CUMULATIVE_RATIO)           → STABLE (필기 사라짐, 이론상 없음)
#         """
#         if self.base_frame is None or self.prev_frame is None:
#             self.prev_frame = frame.copy()
#             return "NONE"

#         # ── 신호 1: 누적 diff → WRITING 진입/유지 판단 ──────────
#         cumulative_ratio = count_changed_pixels(
#             self.base_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
#         )
#         has_annotation = cumulative_ratio >= self.cfg.ANNOT_CUMULATIVE_RATIO

#         # ── 신호 2: 순간 diff → 펜이 멈췄는지 판단 ──────────────
#         instant_ratio = count_changed_pixels(
#             self.prev_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
#         )
#         is_pen_moving = instant_ratio >= self.cfg.ANNOT_INSTANT_RATIO

#         # ── 상태 전이 ─────────────────────────────────────────────
#         # WRITING 진입 로직 완화
#         if self.state == "STABLE":
#             if has_annotation:
#                 self.state = "WRITING"
#                 self.writing_count = 1 
#                 self.stable_count = 0
#                 self.last_annot_frame = frame.copy()

#         elif self.state == "WRITING":
#             if has_annotation: # 펜이 움직이든 아니든 필기가 있는 동안은 카운트 유지
#                 self.writing_count += 1
                
#                 if is_pen_moving:
#                     self.stable_count = 0
#                     self.last_annot_frame = frame.copy()
#                 else:
#                     self.stable_count += 1
                
#                 # 안정화 조건 충족 시
#                 if self.stable_count >= self.stability_frames:
#                     # min_annot_frames 조건을 삭제하거나 1로 설정하여 
#                     # 한 번이라도 찍힌 필기는 무조건 캡처되도록 함
#                     self.state = "STABLE"
#                     self.base_frame = frame.copy()
#                     return "CAPTURE"

#         self.prev_frame = frame.copy()
#         return "NONE"

#     def get_capture_frame(self, current_frame: np.ndarray) -> np.ndarray:
#         """캡처할 프레임: 마지막으로 변화가 감지된 프레임 (필기가 완성된 상태)"""
#         return self.last_annot_frame if self.last_annot_frame is not None else current_frame


# # ──────────────────────────────────────────────
# # 메인 파이프라인
# # ──────────────────────────────────────────────
# def extract_slides(input_path: str, output_dir: str, debug: bool = False):
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

#     fps        = cap.get(cv2.CAP_PROP_FPS)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     duration   = total_frames / fps

#     log.info(f"영상 로드: {input_path}")
#     log.info(f"  FPS={fps:.2f}, 총 프레임={total_frames}, 길이={duration:.1f}초")

#     out_path = Path(output_dir)
#     out_path.mkdir(parents=True, exist_ok=True)

#     cfg               = Config()
#     slide_detector    = SlideChangeDetector(cfg)
#     annot_detector    = AnnotationStabilityDetector(cfg, fps)

#     slide_idx     = 0
#     annot_idx     = 0
#     frame_no      = 0
#     first_frame   = True

#     # 메타데이터 수집
#     metadata = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_no += 1
#         if frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
#             continue

#         timestamp = frame_no / fps
#         small     = resize_frame(frame, cfg.RESIZE_WIDTH)

#         # ── 슬라이드 전환 감지 ──────────────────
#         if first_frame or slide_detector.is_slide_change(small):
#             if not first_frame:
#                 # 전환 직전 → 마지막 필기 상태가 있으면 강제 캡처
#                 if annot_detector.state == "WRITING" and annot_detector.writing_count >= annot_detector.min_annot_frames:
#                     cap_frame = annot_detector.get_capture_frame(frame)
#                     fname = f"slide_{slide_idx:03d}_annot_{annot_idx:02d}.jpg"
#                     _save(cap_frame, out_path / fname)
#                     metadata.append(_meta(fname, slide_idx, annot_idx, timestamp, "force_capture"))
#                     log.info(f"  [강제 캡처] {fname} @ {timestamp:.2f}s")
#                     annot_idx += 1

#             slide_idx += 1
#             annot_idx  = 0
#             first_frame = False

#             # 슬라이드 베이스 프레임 저장
#             fname = f"slide_{slide_idx:03d}_base.jpg"
#             _save(frame, out_path / fname)
#             metadata.append(_meta(fname, slide_idx, 0, timestamp, "base"))
#             log.info(f"[슬라이드 {slide_idx}] base @ {timestamp:.2f}s")

#             annot_detector.reset(base_frame=small)
#             continue

#         # ── 필기 안정화 감지 ────────────────────
#         event = annot_detector.process(small)
#         if event == "CAPTURE":
#             annot_idx += 1
#             cap_frame = annot_detector.get_capture_frame(frame)
#             fname = f"slide_{slide_idx:03d}_annot_{annot_idx:02d}.jpg"
#             _save(cap_frame, out_path / fname)
#             metadata.append(_meta(fname, slide_idx, annot_idx, timestamp, "annotation"))
#             log.info(f"  [필기 완료] {fname} @ {timestamp:.2f}s")

#         # 진행 상황 로그
#         if debug and frame_no % (int(fps) * 10) == 0:
#             log.debug(f"  처리 중: {timestamp:.1f}s / {duration:.1f}s")

#     cap.release()

#     # 메타데이터 저장
#     meta_path = out_path / "metadata.json"
#     with open(meta_path, "w", encoding="utf-8") as f:
#         json.dump(metadata, f, ensure_ascii=False, indent=2)

#     log.info(f"\n완료: 슬라이드 {slide_idx}개, 총 {len(metadata)}개 프레임 저장 → {output_dir}")
#     log.info(f"메타데이터: {meta_path}")
#     return metadata


# def _save(frame: np.ndarray, path: Path):
#     cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


# def _meta(fname, slide_idx, annot_idx, timestamp, capture_type):
#     return {
#         "filename":     fname,
#         "slide_index":  slide_idx,
#         "annot_index":  annot_idx,
#         "timestamp_sec": round(timestamp, 2),
#         "capture_type": capture_type,   # "base" | "annotation" | "force_capture"
#     }


# # ──────────────────────────────────────────────
# # 임계값 튜닝 도우미 (--tune 모드)
# # ──────────────────────────────────────────────
# def tune_thresholds(input_path: str, sample_sec: float = 30.0):
#     """
#     영상 앞부분 sample_sec초를 분석하여 diff 분포를 출력.
#     - instant_ratio: 연속 프레임 간 변화 (순간 필기 감지용)
#     - cumulative_ratio: 첫 프레임 대비 누적 변화 (필기 축적 감지용)
#     """
#     cap = cv2.VideoCapture(input_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     max_frames = int(sample_sec * fps)

#     instant_ratios    = []
#     cumulative_ratios = []
#     prev_small        = None
#     base_small        = None

#     log.info(f"[튜닝 모드] 앞 {sample_sec}초 분석 중...")

#     cfg = Config()
#     frame_no = 0
#     while frame_no < max_frames:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_no += 1
#         if frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
#             continue

#         small = resize_frame(frame, cfg.RESIZE_WIDTH)
#         if base_small is None:
#             base_small = small.copy()
#         if prev_small is not None:
#             instant_ratios.append(
#                 count_changed_pixels(prev_small, small, cfg.ANNOT_DIFF_THRESHOLD)
#             )
#             cumulative_ratios.append(
#                 count_changed_pixels(base_small, small, cfg.ANNOT_DIFF_THRESHOLD)
#             )
#         prev_small = small

#     cap.release()

#     ir = np.array(instant_ratios)
#     cr = np.array(cumulative_ratios)

#     print("\n===== 임계값 튜닝 가이드 =====")
#     print(f"[순간 diff]   min={ir.min():.5f}  p50={np.percentile(ir,50):.5f}"
#           f"  p95={np.percentile(ir,95):.5f}  p99={np.percentile(ir,99):.5f}  max={ir.max():.5f}")
#     print(f"  → ANNOT_INSTANT_RATIO 권장: p50~p95 사이 (현재: {cfg.ANNOT_INSTANT_RATIO})")
#     print()
#     print(f"[누적 diff]   min={cr.min():.5f}  p50={np.percentile(cr,50):.5f}"
#           f"  p95={np.percentile(cr,95):.5f}  p99={np.percentile(cr,99):.5f}  max={cr.max():.5f}")
#     print(f"  → ANNOT_CUMULATIVE_RATIO 권장: p95~p99 사이 (현재: {cfg.ANNOT_CUMULATIVE_RATIO})")
#     print("================================\n")


# # ──────────────────────────────────────────────
# # CLI
# # ──────────────────────────────────────────────
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="PPT 강의 영상 슬라이드 추출기")
#     parser.add_argument("--input",  "-i", default="lecture.mp4",    help="입력 영상 경로")
#     parser.add_argument("--output", "-o", default="output_slides/", help="출력 디렉토리")
#     parser.add_argument("--debug",  action="store_true",             help="디버그 로그 출력")
#     parser.add_argument("--tune",   action="store_true",             help="임계값 튜닝 모드 (추출 안 함)")
#     args = parser.parse_args()

#     if args.tune:
#         tune_thresholds(args.input)
#     else:
#         if args.debug:
#             logging.getLogger().setLevel(logging.DEBUG)
#         extract_slides(args.input, args.output, debug=args.debug)


"""
slide_extractor.py
==================
PPT 기반 강의 영상에서 슬라이드 + 필기 완료 시점 프레임을 추출합니다.

Input : lecture.mp4
Output: output_slides/
        ├── slide_001_base.jpg      # 슬라이드 최초 등장 프레임
        ├── slide_001_annot_01.jpg  # 필기 안정화 캡처
        ├── slide_002_base.jpg      # 전환 or PPT 애니메이션(새 콘텐츠) → 새 base
        └── ...

슬라이드 idx가 증가하는 경우:
  1. Cut 전환 (직전 프레임 대비 급격한 MSE + phash 변화)
  2. Fade 전환 (sliding window 내 oldest ↔ current 누적 변화)
  3. Slide base 이중 비교 (필기로 오염된 prev_frame 대신 클린한 base phash 비교)
  4. PPT 애니메이션 (슬라이드 내 대규모 콘텐츠 변화 → 새 base로 분리)

Usage:
    python slide_extractor.py --input lecture.mp4 --output output_slides/
    python slide_extractor.py --input lecture.mp4 --output output_slides/ --debug
    python slide_extractor.py --input lecture.mp4 --tune
"""

import cv2
import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
from collections import deque
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 설정값 (튜닝 포인트)
# ──────────────────────────────────────────────
class Config:
    # ── 슬라이드 전환 감지 (Cut) ─────────────────────────────────────
    SLIDE_CHANGE_MSE_THRESHOLD   = 500   # MSE 임계 (직전 프레임 대비)
    SLIDE_CHANGE_HASH_THRESHOLD  = 10    # phash hamming distance (0~64)

    # ── 슬라이드 전환 감지 (Fade - Sliding Window) ───────────────────
    FADE_WINDOW_SEC              = 1.0   # 페이드 감지 윈도우 크기 (초)
    FADE_MSE_THRESHOLD           = 400   # oldest ↔ current MSE 임계
    FADE_HASH_THRESHOLD          = 8     # oldest ↔ current phash 거리 임계

    # ── Slide base 이중 비교 ─────────────────────────────────────────
    # prev_frame이 필기로 오염되는 경우를 보완 (유사 레이아웃 슬라이드 구분)
    BASE_HASH_THRESHOLD          = 8     # slide_base_phash ↔ current phash 거리 임계

    # ── 필기 감지 ────────────────────────────────────────────────────
    ANNOT_DIFF_THRESHOLD         = 15    # 픽셀 변화 판정 절댓값 임계
    ANNOT_CUMULATIVE_RATIO       = 0.0005  # base 대비 0.05% 이상 변화 → 필기 시작
    ANNOT_INSTANT_RATIO          = 0.0001  # 직전 프레임 대비 변화 → 펜 움직임 여부

    # ── PPT 애니메이션 감지 (새 텍스트/이미지 등장 → 새 base) ─────────
    # 안정화 완료 시 새 slide_idx + base 파일로 저장
    # 튜닝: --tune 모드에서 누적 diff p99 이상 값 참고
    ANIM_CUMULATIVE_RATIO        = 0.05   # base 대비 5% 이상 변화 → 애니메이션

    # ── 안정화 판단 ──────────────────────────────────────────────────
    STABILITY_WINDOW_SEC         = 0.7
    MIN_ANNOT_DURATION_SEC       = 0.2

    # ── 처리 성능 ────────────────────────────────────────────────────
    PROCESS_EVERY_N_FRAMES       = 2
    RESIZE_WIDTH                 = 960


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_phash(frame: np.ndarray) -> imagehash.ImageHash:
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def count_changed_pixels(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
    diff = cv2.absdiff(frame_a, frame_b)
    max_diff = np.max(diff, axis=2)
    return np.sum(max_diff > threshold) / max_diff.size


# ──────────────────────────────────────────────
# 슬라이드 전환 감지기 (Cut + Fade + Base 이중 비교)
# ──────────────────────────────────────────────
class SlideChangeDetector:
    """
    세 가지 방식으로 슬라이드 전환을 감지:

      1. Cut:  prev_frame ↔ current MSE + phash 급등
      2. Base 이중 비교:  slide_base_phash ↔ current 비교
         - prev_frame이 필기로 오염되어 phash가 왜곡되는 경우 보완
         - 유사 레이아웃 슬라이드도 구분 가능
      3. Fade: sliding window의 oldest ↔ current 누적 변화
         - 연속 프레임 간 diff가 작아 cut을 통과하는 페이드 감지

    슬라이드 전환(또는 PPT 애니메이션 new base) 시 반드시 reset() 호출.
    """
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        buf_size = max(2, int(cfg.FADE_WINDOW_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES))
        self.frame_buffer     = deque(maxlen=buf_size)
        self.prev_frame       = None
        self.prev_phash       = None
        self.slide_base_phash = None

    def reset(self, frame: np.ndarray):
        """슬라이드 전환 후 호출: 버퍼·base 초기화"""
        phash = compute_phash(frame)
        self.prev_frame       = frame.copy()
        self.prev_phash       = phash
        self.slide_base_phash = phash
        self.frame_buffer.clear()
        self.frame_buffer.append((frame.copy(), phash))

    def is_slide_change(self, frame: np.ndarray) -> bool:
        if self.prev_frame is None:
            self.reset(frame)
            return False

        curr_phash = compute_phash(frame)

        # ── 1. Cut 전환 ────────────────────────────────────────────────
        mse = compute_mse(self.prev_frame, frame)
        if mse >= self.cfg.SLIDE_CHANGE_MSE_THRESHOLD:
            if (self.prev_phash - curr_phash) >= self.cfg.SLIDE_CHANGE_HASH_THRESHOLD:
                self._update(frame, curr_phash)
                return True

        # ── 2. Base 이중 비교 ──────────────────────────────────────────
        if self.slide_base_phash is not None:
            if (self.slide_base_phash - curr_phash) >= self.cfg.BASE_HASH_THRESHOLD:
                if mse >= self.cfg.SLIDE_CHANGE_MSE_THRESHOLD * 0.5:
                    self._update(frame, curr_phash)
                    return True

        # ── 3. Fade 전환 ───────────────────────────────────────────────
        if len(self.frame_buffer) == self.frame_buffer.maxlen:
            oldest_frame, oldest_phash = self.frame_buffer[0]
            if compute_mse(oldest_frame, frame) >= self.cfg.FADE_MSE_THRESHOLD:
                if (oldest_phash - curr_phash) >= self.cfg.FADE_HASH_THRESHOLD:
                    self._update(frame, curr_phash)
                    return True

        self._update(frame, curr_phash)
        return False

    def _update(self, frame: np.ndarray, phash: imagehash.ImageHash = None):
        if phash is None:
            phash = compute_phash(frame)
        self.prev_frame = frame.copy()
        self.prev_phash = phash
        self.frame_buffer.append((frame.copy(), phash))


# ──────────────────────────────────────────────
# 필기 안정화 감지기
# ──────────────────────────────────────────────
class AnnotationStabilityDetector:
    """
    슬라이드 내 변화를 두 단계 임계값으로 구분:

      소규모 변화 (ANNOT_RATIO ≤ ratio < ANIM_RATIO)
        → 필기 → CAPTURE_ANNOT → annot_XX 저장

      대규모 변화 (ratio ≥ ANIM_RATIO)
        → PPT 애니메이션 → NEW_BASE
          → main loop에서 슬라이드 전환과 동일하게 처리 (새 slide_idx + base 저장)

    상태 머신:
      STABLE    → (ratio ≥ ANIM_RATIO)  → ANIMATING
               → (ratio ≥ ANNOT_RATIO) → WRITING
      WRITING   → (안정화) → CAPTURE_ANNOT → STABLE (base 갱신)
      WRITING   → (ratio가 ANIM_RATIO 초과) → ANIMATING 격상
      ANIMATING → (안정화) → NEW_BASE       → (main loop이 reset 호출)
    """
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        self.stability_frames = int(cfg.STABILITY_WINDOW_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
        self.min_annot_frames = int(cfg.MIN_ANNOT_DURATION_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
        self.reset()

    def reset(self, base_frame: np.ndarray = None):
        self.state            = "STABLE"
        self.base_frame       = base_frame
        self.prev_frame       = base_frame
        self.stable_count     = 0
        self.writing_count    = 0
        self.last_annot_frame = None

    def process(self, frame: np.ndarray) -> str:
        """
        반환값:
          "CAPTURE_ANNOT"  - 필기 안정화 완료 → annot_XX 저장
          "NEW_BASE"       - 애니메이션 안정화 완료 → 새 slide_idx + base 저장
          "NONE"
        """
        if self.base_frame is None or self.prev_frame is None:
            self.prev_frame = frame.copy()
            return "NONE"

        cumulative_ratio = count_changed_pixels(
            self.base_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
        )
        instant_ratio = count_changed_pixels(
            self.prev_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
        )
        is_active = instant_ratio >= self.cfg.ANNOT_INSTANT_RATIO

        result = "NONE"

        if self.state == "STABLE":
            if cumulative_ratio >= self.cfg.ANIM_CUMULATIVE_RATIO:
                self.state         = "ANIMATING"
                self.writing_count = 1
                self.stable_count  = 0
                self.last_annot_frame = frame.copy()
            elif cumulative_ratio >= self.cfg.ANNOT_CUMULATIVE_RATIO:
                self.state         = "WRITING"
                self.writing_count = 1
                self.stable_count  = 0
                self.last_annot_frame = frame.copy()

        elif self.state in ("WRITING", "ANIMATING"):
            # WRITING 도중 변화량이 커지면 ANIMATING으로 격상
            if self.state == "WRITING" and cumulative_ratio >= self.cfg.ANIM_CUMULATIVE_RATIO:
                self.state = "ANIMATING"

            if cumulative_ratio >= self.cfg.ANNOT_CUMULATIVE_RATIO:
                self.writing_count += 1
                if is_active:
                    self.stable_count = 0
                    self.last_annot_frame = frame.copy()
                else:
                    self.stable_count += 1

                if self.stable_count >= self.stability_frames:
                    if self.state == "ANIMATING":
                        result = "NEW_BASE"
                        # main loop에서 reset()을 호출하므로 여기선 상태만 초기화
                    else:
                        result = "CAPTURE_ANNOT"
                        # 다음 필기 이벤트를 위해 base를 현재 안정 상태로 갱신
                        self.base_frame = frame.copy()
                    self.state         = "STABLE"
                    self.stable_count  = 0
                    self.writing_count = 0

        self.prev_frame = frame.copy()
        return result

    def get_capture_frame(self, current_frame: np.ndarray) -> np.ndarray:
        return self.last_annot_frame if self.last_annot_frame is not None else current_frame


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def extract_slides(input_path: str, output_dir: str, debug: bool = False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration     = total_frames / fps

    log.info(f"영상 로드: {input_path}")
    log.info(f"  FPS={fps:.2f}, 총 프레임={total_frames}, 길이={duration:.1f}초")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg            = Config()
    slide_detector = SlideChangeDetector(cfg, fps)
    annot_detector = AnnotationStabilityDetector(cfg, fps)

    slide_idx   = 0
    annot_idx   = 0
    frame_no    = 0
    first_frame = True
    metadata    = []

    def register_new_base(frame, small, timestamp, reason):
        """슬라이드 전환 or PPT 애니메이션 → 새 slide_idx + base 저장 공통 처리"""
        nonlocal slide_idx, annot_idx
        slide_idx += 1
        annot_idx  = 0
        fname = f"slide_{slide_idx:03d}_base.jpg"
        _save(frame, out_path / fname)
        metadata.append(_meta(fname, slide_idx, timestamp, "base"))
        log.info(f"[슬라이드 {slide_idx}] base ({reason}) @ {timestamp:.2f}s")
        slide_detector.reset(small)
        annot_detector.reset(base_frame=small)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
            continue

        timestamp = frame_no / fps
        small     = resize_frame(frame, cfg.RESIZE_WIDTH)

        # ── 슬라이드 전환 감지 (Cut / Fade / Base 이중 비교) ────────
        if first_frame or slide_detector.is_slide_change(small):
            if not first_frame:
                # 전환 직전 진행 중이던 필기 강제 캡처
                if annot_detector.state == "WRITING" \
                        and annot_detector.writing_count >= annot_detector.min_annot_frames:
                    cap_frame = annot_detector.get_capture_frame(frame)
                    annot_idx += 1
                    fname = f"slide_{slide_idx:03d}_annot_{annot_idx:02d}.jpg"
                    _save(cap_frame, out_path / fname)
                    metadata.append(_meta(fname, slide_idx, timestamp, "annotation"))
                    log.info(f"  [강제 캡처] {fname} @ {timestamp:.2f}s")

            first_frame = False
            register_new_base(frame, small, timestamp, "slide_change")
            continue

        # ── 필기 / 애니메이션 감지 ──────────────────────────────────
        event = annot_detector.process(small)

        if event == "CAPTURE_ANNOT":
            annot_idx += 1
            cap_frame = annot_detector.get_capture_frame(frame)
            fname = f"slide_{slide_idx:03d}_annot_{annot_idx:02d}.jpg"
            _save(cap_frame, out_path / fname)
            metadata.append(_meta(fname, slide_idx, timestamp, "annotation"))
            log.info(f"  [필기 완료] {fname} @ {timestamp:.2f}s")

        elif event == "NEW_BASE":
            # PPT 애니메이션으로 새 콘텐츠 등장 → 슬라이드 전환과 동일하게 처리
            cap_frame = annot_detector.get_capture_frame(frame)
            register_new_base(cap_frame, small, timestamp, "animation_base")

        if debug and frame_no % (int(fps) * 10) == 0:
            log.debug(f"  처리 중: {timestamp:.1f}s / {duration:.1f}s")

    cap.release()

    meta_path = out_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log.info(f"\n완료: 슬라이드 {slide_idx}개, 총 {len(metadata)}개 프레임 저장 → {output_dir}")
    return metadata


def _save(frame: np.ndarray, path: Path):
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _meta(fname: str, slide_idx: int, timestamp: float, capture_type: str) -> dict:
    return {
        "filename":      fname,
        "slide_index":   slide_idx,
        "timestamp_sec": round(timestamp, 2),
        "capture_type":  capture_type,  # base | annotation
    }


# ──────────────────────────────────────────────
# 임계값 튜닝 도우미 (--tune 모드)
# ──────────────────────────────────────────────
def tune_thresholds(input_path: str, sample_sec: float = 30.0):
    """
    영상 앞부분 sample_sec초를 분석하여 diff 분포를 출력.
    튜닝 기준:
      ANNOT_INSTANT_RATIO:    p50 ~ p95 사이
      ANNOT_CUMULATIVE_RATIO: p95 ~ p99 사이
      ANIM_CUMULATIVE_RATIO:  p99 이상 (필기보다 훨씬 큰 값)
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(sample_sec * fps)

    instant_ratios    = []
    cumulative_ratios = []
    prev_small        = None
    base_small        = None

    cfg = Config()
    frame_no = 0
    while frame_no < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
            continue
        small = resize_frame(frame, cfg.RESIZE_WIDTH)
        if base_small is None:
            base_small = small.copy()
        if prev_small is not None:
            instant_ratios.append(count_changed_pixels(prev_small, small, cfg.ANNOT_DIFF_THRESHOLD))
            cumulative_ratios.append(count_changed_pixels(base_small, small, cfg.ANNOT_DIFF_THRESHOLD))
        prev_small = small

    cap.release()

    ir = np.array(instant_ratios)
    cr = np.array(cumulative_ratios)

    print("\n===== 임계값 튜닝 가이드 =====")
    print(f"[순간 diff]   p50={np.percentile(ir,50):.5f}  p95={np.percentile(ir,95):.5f}  max={ir.max():.5f}")
    print(f"  → ANNOT_INSTANT_RATIO 권장: p50~p95 (현재: {cfg.ANNOT_INSTANT_RATIO})")
    print()
    print(f"[누적 diff]   p50={np.percentile(cr,50):.5f}  p95={np.percentile(cr,95):.5f}"
          f"  p99={np.percentile(cr,99):.5f}  max={cr.max():.5f}")
    print(f"  → ANNOT_CUMULATIVE_RATIO 권장: p95~p99  (현재: {cfg.ANNOT_CUMULATIVE_RATIO})")
    print(f"  → ANIM_CUMULATIVE_RATIO  권장: p99 이상 (현재: {cfg.ANIM_CUMULATIVE_RATIO})")
    print("================================\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPT 강의 영상 슬라이드 추출기")
    parser.add_argument("--input",  "-i", default="lecture.mp4",    help="입력 영상 경로")
    parser.add_argument("--output", "-o", default="output_slides/", help="출력 디렉토리")
    parser.add_argument("--debug",  action="store_true",             help="디버그 로그 출력")
    parser.add_argument("--tune",   action="store_true",             help="임계값 튜닝 모드")
    args = parser.parse_args()

    if args.tune:
        tune_thresholds(args.input)
    else:
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        extract_slides(args.input, args.output, debug=args.debug)