"""
slide_extractor.py
==================
PPT 기반 강의 영상에서 슬라이드 + 필기 완료 시점 프레임을 추출합니다.

Input : lecture.mp4
Output: output_slides/
        ├── slide_001_base.jpg          # 슬라이드 최초 등장 프레임
        ├── slide_001_annot_01.jpg      # 필기 안정화 캡처 #1
        ├── slide_001_annot_02.jpg      # 필기 안정화 캡처 #2 (있을 경우)
        ├── slide_002_base.jpg
        └── ...

Usage:
    python slide_extractor.py --input lecture.mp4 --output output_slides/
    python slide_extractor.py --input lecture.mp4 --output output_slides/ --debug
"""

import cv2
import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 설정값 (튜닝 포인트)
# ──────────────────────────────────────────────
class Config:
    # 슬라이드 전환 감지
    SLIDE_CHANGE_MSE_THRESHOLD   = 500    # MSE 이 값 이상이면 슬라이드 전환 후보
    SLIDE_CHANGE_HASH_THRESHOLD  = 10     # 퍼셉추얼 해시 hamming distance (0~64)
    
    # 필기 감지
    ANNOT_DIFF_THRESHOLD         = 15     # 픽셀 diff 절댓값 임계 (그레이스케일)
    # 누적 diff: base 대비 현재 프레임의 변화 픽셀 비율 → "필기가 쌓였는가?"
    ANNOT_CUMULATIVE_RATIO       = 0.0005  # 슬라이드 면적의 0.3% 이상 변하면 필기 있음
    # 순간 diff: 직전 프레임 대비 변화 픽셀 비율 → "지금 쓰고 있는가?"
    ANNOT_INSTANT_RATIO          = 0.0001 # 느린 필기도 잡기 위해 낮게 설정
    
    # 안정화 판단
    STABILITY_WINDOW_SEC         = 0.7    # 이 시간 동안 변화 없으면 '필기 완료'
    MIN_ANNOT_DURATION_SEC       = 0.2    # 이보다 짧은 필기 이벤트는 노이즈로 무시
    
    # 처리 성능
    PROCESS_EVERY_N_FRAMES       = 2      # N 프레임마다 1번 처리 (속도 vs 정확도)
    RESIZE_WIDTH                 = 960    # 내부 처리 해상도 (원본 저장은 별도)


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def compute_mse(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """두 프레임 간 MSE 계산 (그레이스케일)"""
    a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(np.mean((a - b) ** 2))


def compute_phash(frame: np.ndarray) -> imagehash.ImageHash:
    """퍼셉추얼 해시 계산"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)


def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    """처리용 리사이즈"""
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def count_changed_pixels(frame_a: np.ndarray, frame_b: np.ndarray, threshold: int) -> float:
    """색상 채널 정보를 활용하여 변화한 픽셀 비율 반환"""
    # 그레이스케일 대신 BGR 차이의 절대값을 구함
    diff = cv2.absdiff(frame_a, frame_b)
    # 세 채널 중 하나라도 threshold를 넘으면 변화한 것으로 간주
    max_diff = np.max(diff, axis=2)
    changed = np.sum(max_diff > threshold)
    return changed / max_diff.size


# ──────────────────────────────────────────────
# 슬라이드 전환 감지기
# ──────────────────────────────────────────────
class SlideChangeDetector:
    """
    MSE + 퍼셉추얼 해시 이중 검증으로 슬라이드 전환 감지.
    MSE만 쓰면 필기가 많을 때 오탐 발생하므로 phash를 보조 필터로 사용.
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.prev_frame = None
        self.prev_phash = None

    def is_slide_change(self, frame: np.ndarray) -> bool:
        if self.prev_frame is None:
            self._update(frame)
            return False

        mse = compute_mse(self.prev_frame, frame)
        
        # MSE 임계 미달 → 확실히 전환 아님
        if mse < self.cfg.SLIDE_CHANGE_MSE_THRESHOLD:
            self._update(frame)
            return False

        # MSE 초과 → phash로 이중 확인
        curr_phash = compute_phash(frame)
        hash_dist = self.prev_phash - curr_phash
        
        is_change = hash_dist >= self.cfg.SLIDE_CHANGE_HASH_THRESHOLD
        self._update(frame)
        return is_change

    def _update(self, frame):
        self.prev_frame = frame.copy()
        self.prev_phash = compute_phash(frame)


# ──────────────────────────────────────────────
# 필기 안정화 감지기
# ──────────────────────────────────────────────
class AnnotationStabilityDetector:
    """
    슬라이드 구간 내에서 필기 변화 추적 및 안정화 시점 감지.

    두 가지 신호를 분리하여 사용:
      - 누적 diff (base ↔ current): "필기가 쌓였는가?" → WRITING 진입 조건
      - 순간 diff (prev ↔ current): "지금 쓰고 있는가?" → 안정화 판단 조건

    연속 프레임 간 diff만 쓰면 느린 필기(프레임당 변화량 미미)를 감지 못하는
    근본 문제를 해결하기 위해 누적 diff를 주 신호로 사용한다.

    상태 머신:
        STABLE → (누적 diff 임계 초과) → WRITING
               → (순간 diff가 stability_window 동안 임계 미달) → STABLE + CAPTURE
    """
    def __init__(self, cfg: Config, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.stability_frames = int(cfg.STABILITY_WINDOW_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
        self.min_annot_frames = int(cfg.MIN_ANNOT_DURATION_SEC * fps / cfg.PROCESS_EVERY_N_FRAMES)
        self.reset()

    def reset(self, base_frame: np.ndarray = None):
        """슬라이드 전환 시 리셋"""
        self.state             = "STABLE"
        self.base_frame        = base_frame   # 슬라이드 최초 클린 프레임
        self.prev_frame        = base_frame
        self.stable_count      = 0
        self.writing_count     = 0
        self.last_annot_frame  = None         # 가장 마지막으로 변화가 감지된 프레임
        # 누적 diff가 임계를 넘은 이후 base를 갱신하지 않기 위한 플래그
        self.annotation_started = False

    def process(self, frame: np.ndarray) -> str:
        """
        프레임을 받아서 이벤트 반환.
        반환값: "CAPTURE" | "NONE"

        상태 전이:
          STABLE  → (누적 diff >= CUMULATIVE_RATIO)          → WRITING
          WRITING → (순간 diff < INSTANT_RATIO, 연속 N프레임) → CAPTURE + STABLE
          WRITING → (누적 diff < CUMULATIVE_RATIO)           → STABLE (필기 사라짐, 이론상 없음)
        """
        if self.base_frame is None or self.prev_frame is None:
            self.prev_frame = frame.copy()
            return "NONE"

        # ── 신호 1: 누적 diff → WRITING 진입/유지 판단 ──────────
        cumulative_ratio = count_changed_pixels(
            self.base_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
        )
        has_annotation = cumulative_ratio >= self.cfg.ANNOT_CUMULATIVE_RATIO

        # ── 신호 2: 순간 diff → 펜이 멈췄는지 판단 ──────────────
        instant_ratio = count_changed_pixels(
            self.prev_frame, frame, self.cfg.ANNOT_DIFF_THRESHOLD
        )
        is_pen_moving = instant_ratio >= self.cfg.ANNOT_INSTANT_RATIO

        # ── 상태 전이 ─────────────────────────────────────────────
        # WRITING 진입 로직 완화
        if self.state == "STABLE":
            if has_annotation:
                self.state = "WRITING"
                self.writing_count = 1 
                self.stable_count = 0
                self.last_annot_frame = frame.copy()

        elif self.state == "WRITING":
            if has_annotation: # 펜이 움직이든 아니든 필기가 있는 동안은 카운트 유지
                self.writing_count += 1
                
                if is_pen_moving:
                    self.stable_count = 0
                    self.last_annot_frame = frame.copy()
                else:
                    self.stable_count += 1
                
                # 안정화 조건 충족 시
                if self.stable_count >= self.stability_frames:
                    # min_annot_frames 조건을 삭제하거나 1로 설정하여 
                    # 한 번이라도 찍힌 필기는 무조건 캡처되도록 함
                    self.state = "STABLE"
                    self.base_frame = frame.copy()
                    return "CAPTURE"

        self.prev_frame = frame.copy()
        return "NONE"

    def get_capture_frame(self, current_frame: np.ndarray) -> np.ndarray:
        """캡처할 프레임: 마지막으로 변화가 감지된 프레임 (필기가 완성된 상태)"""
        return self.last_annot_frame if self.last_annot_frame is not None else current_frame


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────
def extract_slides(input_path: str, output_dir: str, debug: bool = False):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {input_path}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_frames / fps

    log.info(f"영상 로드: {input_path}")
    log.info(f"  FPS={fps:.2f}, 총 프레임={total_frames}, 길이={duration:.1f}초")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg               = Config()
    slide_detector    = SlideChangeDetector(cfg)
    annot_detector    = AnnotationStabilityDetector(cfg, fps)

    slide_idx     = 0
    annot_idx     = 0
    frame_no      = 0
    first_frame   = True

    # 메타데이터 수집
    metadata = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_no += 1
        if frame_no % cfg.PROCESS_EVERY_N_FRAMES != 0:
            continue

        timestamp = frame_no / fps
        small     = resize_frame(frame, cfg.RESIZE_WIDTH)

        # ── 슬라이드 전환 감지 ──────────────────
        if first_frame or slide_detector.is_slide_change(small):
            if not first_frame:
                # 전환 직전 → 마지막 필기 상태가 있으면 강제 캡처
                if annot_detector.state == "WRITING" and annot_detector.writing_count >= annot_detector.min_annot_frames:
                    cap_frame = annot_detector.get_capture_frame(frame)
                    fname = f"slide_{slide_idx:03d}_annot_{annot_idx:02d}.jpg"
                    _save(cap_frame, out_path / fname)
                    metadata.append(_meta(fname, slide_idx, annot_idx, timestamp, "force_capture"))
                    log.info(f"  [강제 캡처] {fname} @ {timestamp:.2f}s")
                    annot_idx += 1

            slide_idx += 1
            annot_idx  = 0
            first_frame = False

            # 슬라이드 베이스 프레임 저장
            fname = f"slide_{slide_idx:03d}_base.jpg"
            _save(frame, out_path / fname)
            metadata.append(_meta(fname, slide_idx, 0, timestamp, "base"))
            log.info(f"[슬라이드 {slide_idx}] base @ {timestamp:.2f}s")

            annot_detector.reset(base_frame=small)
            continue

        # ── 필기 안정화 감지 ────────────────────
        event = annot_detector.process(small)
        if event == "CAPTURE":
            annot_idx += 1
            cap_frame = annot_detector.get_capture_frame(frame)
            fname = f"slide_{slide_idx:03d}_annot_{annot_idx:02d}.jpg"
            _save(cap_frame, out_path / fname)
            metadata.append(_meta(fname, slide_idx, annot_idx, timestamp, "annotation"))
            log.info(f"  [필기 완료] {fname} @ {timestamp:.2f}s")

        # 진행 상황 로그
        if debug and frame_no % (int(fps) * 10) == 0:
            log.debug(f"  처리 중: {timestamp:.1f}s / {duration:.1f}s")

    cap.release()

    # 메타데이터 저장
    meta_path = out_path / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    log.info(f"\n완료: 슬라이드 {slide_idx}개, 총 {len(metadata)}개 프레임 저장 → {output_dir}")
    log.info(f"메타데이터: {meta_path}")
    return metadata


def _save(frame: np.ndarray, path: Path):
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _meta(fname, slide_idx, annot_idx, timestamp, capture_type):
    return {
        "filename":     fname,
        "slide_index":  slide_idx,
        "annot_index":  annot_idx,
        "timestamp_sec": round(timestamp, 2),
        "capture_type": capture_type,   # "base" | "annotation" | "force_capture"
    }


# ──────────────────────────────────────────────
# 임계값 튜닝 도우미 (--tune 모드)
# ──────────────────────────────────────────────
def tune_thresholds(input_path: str, sample_sec: float = 30.0):
    """
    영상 앞부분 sample_sec초를 분석하여 diff 분포를 출력.
    - instant_ratio: 연속 프레임 간 변화 (순간 필기 감지용)
    - cumulative_ratio: 첫 프레임 대비 누적 변화 (필기 축적 감지용)
    """
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(sample_sec * fps)

    instant_ratios    = []
    cumulative_ratios = []
    prev_small        = None
    base_small        = None

    log.info(f"[튜닝 모드] 앞 {sample_sec}초 분석 중...")

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
            instant_ratios.append(
                count_changed_pixels(prev_small, small, cfg.ANNOT_DIFF_THRESHOLD)
            )
            cumulative_ratios.append(
                count_changed_pixels(base_small, small, cfg.ANNOT_DIFF_THRESHOLD)
            )
        prev_small = small

    cap.release()

    ir = np.array(instant_ratios)
    cr = np.array(cumulative_ratios)

    print("\n===== 임계값 튜닝 가이드 =====")
    print(f"[순간 diff]   min={ir.min():.5f}  p50={np.percentile(ir,50):.5f}"
          f"  p95={np.percentile(ir,95):.5f}  p99={np.percentile(ir,99):.5f}  max={ir.max():.5f}")
    print(f"  → ANNOT_INSTANT_RATIO 권장: p50~p95 사이 (현재: {cfg.ANNOT_INSTANT_RATIO})")
    print()
    print(f"[누적 diff]   min={cr.min():.5f}  p50={np.percentile(cr,50):.5f}"
          f"  p95={np.percentile(cr,95):.5f}  p99={np.percentile(cr,99):.5f}  max={cr.max():.5f}")
    print(f"  → ANNOT_CUMULATIVE_RATIO 권장: p95~p99 사이 (현재: {cfg.ANNOT_CUMULATIVE_RATIO})")
    print("================================\n")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPT 강의 영상 슬라이드 추출기")
    parser.add_argument("--input",  "-i", default="lecture.mp4",    help="입력 영상 경로")
    parser.add_argument("--output", "-o", default="output_slides/", help="출력 디렉토리")
    parser.add_argument("--debug",  action="store_true",             help="디버그 로그 출력")
    parser.add_argument("--tune",   action="store_true",             help="임계값 튜닝 모드 (추출 안 함)")
    args = parser.parse_args()

    if args.tune:
        tune_thresholds(args.input)
    else:
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
        extract_slides(args.input, args.output, debug=args.debug)