"""
slide_extractor.py
==================
PPT 기반 강의 영상에서 슬라이드 + 필기 완료 시점 프레임을 추출합니다.

Input : input/lecture.mp4
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
    python slide_extractor.py --input input/lecture.mp4 --output output_slides/
    python slide_extractor.py --input input/lecture.mp4 --output output_slides/ --debug
    python slide_extractor.py --input input/lecture.mp4 --tune
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

    # ── 중복 슬라이드 감지 (후처리) ──────────────────────────────────
    # compute_phash_hires (256비트, hash_size=16) 기준. 최대 거리 = 256.
    # 실행 후 로그의 슬라이드 간 거리 분포를 보고 튜닝:
    #   - 실제 동일 슬라이드 쌍의 dist → 이 값보다 크게
    #   - 실제 다른 슬라이드 쌍의 dist → 이 값보다 작게
    DUPLICATE_HASH_THRESHOLD     = 30    # 초기값, 로그 확인 후 조정 필요

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
    """실시간 슬라이드 전환 감지용 (64비트, 속도 우선)"""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img)


def compute_phash_hires(frame: np.ndarray) -> imagehash.ImageHash:
    """중복 슬라이드 후처리 감지용 (256비트, 정밀도 우선).

    PPT 템플릿처럼 레이아웃이 동일한 슬라이드들은 64비트 phash로는
    콘텐츠 차이를 구분하기 어렵다. hash_size=16 (256비트)으로 세밀한
    콘텐츠 차이를 포착한다. 최대 거리는 256.
    임계값 튜닝 기준: 실제 동일 슬라이드의 거리를 로그로 확인 후 설정.
    """
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img, hash_size=16)


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

    # ── 후처리 1: 슬라이드 단위 시간 구간 추가 ──────────────────────
    metadata = add_slide_time_ranges(metadata, duration)

    # ── 후처리 2: 시각적 중복 후보 표시 ────────────────────────────
    metadata = mark_visual_duplicates(metadata, out_path, cfg)

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
# 후처리: 슬라이드 단위 시간 구간 추가
# ──────────────────────────────────────────────
def add_slide_time_ranges(metadata: list, video_duration: float) -> list:
    """
    각 프레임 레코드에 slide_start_sec / slide_end_sec 추가.

    - slide_start_sec : 해당 slide_index의 base 프레임 타임스탬프
    - slide_end_sec   : 다음 slide_index의 시작 시각 (마지막 슬라이드는 영상 길이)

    slide_classifier에서 오디오 침묵 구간과 교차할 때 이 구간을 기준으로 사용한다.
    """
    # slide_index → base 타임스탬프 수집
    slide_starts: dict[int, float] = {}
    for m in metadata:
        idx = m["slide_index"]
        if m["capture_type"] == "base" and idx not in slide_starts:
            slide_starts[idx] = m["timestamp_sec"]

    sorted_indices = sorted(slide_starts.keys())

    # 각 슬라이드의 종료 시각 = 다음 슬라이드 시작 시각
    slide_ends: dict[int, float] = {}
    for i, idx in enumerate(sorted_indices):
        if i + 1 < len(sorted_indices):
            slide_ends[idx] = slide_starts[sorted_indices[i + 1]]
        else:
            slide_ends[idx] = round(video_duration, 2)

    for m in metadata:
        idx = m["slide_index"]
        m["slide_start_sec"] = slide_starts.get(idx)
        m["slide_end_sec"]   = slide_ends.get(idx)

    return metadata


# ──────────────────────────────────────────────
# 후처리: 시각적 중복 후보 표시
# ──────────────────────────────────────────────
def mark_visual_duplicates(metadata: list, out_path: Path, cfg: Config) -> list:
    """
    모든 슬라이드의 대표 프레임(base + last_annot)을 풀에 쌓고
    전체 쌍(all-pairs)을 비교하여 is_visual_duplicate 플래그를 설정한다.

    프레임 풀 구성:
      - 각 slide_index 별로 base 프레임 + last_annot 프레임(있으면) 수집
      - 레이블: "base{idx}" / "annot{idx}"

    비교:
      - 풀 내 모든 쌍을 phash(256비트) 비교
      - 동일 slide_index 간 쌍은 건너뜀
      - min_dist < DUPLICATE_HASH_THRESHOLD → 두 슬라이드 모두 중복 후보 표시

    is_visual_duplicate 는 slide_index 단위이므로 해당 슬라이드의
    모든 프레임 레코드에 동일하게 적용된다.
    """
    from collections import defaultdict

    # slide_index별 프레임 그룹화
    groups: dict[int, list] = defaultdict(list)
    for m in metadata:
        groups[m["slide_index"]].append(m)

    # 프레임 풀 구성: label → (slide_index, filename)
    pool: dict[str, tuple[int, str]] = {}
    for idx in sorted(groups.keys()):
        frames     = groups[idx]
        base_list  = [f for f in frames if f["capture_type"] == "base"]
        annot_list = [f for f in frames if f["capture_type"] == "annotation"]

        if base_list:
            pool[f"base{idx}"] = (idx, base_list[0]["filename"])
        if annot_list:
            pool[f"annot{idx}"] = (idx, annot_list[-1]["filename"])

    # phash 계산 (256비트)
    phashes: dict[str, imagehash.ImageHash] = {}
    for label, (_, fname) in pool.items():
        img = cv2.imread(str(out_path / fname))
        if img is not None:
            phashes[label] = compute_phash_hires(resize_frame(img, cfg.RESIZE_WIDTH))
        else:
            log.warning(f"  [중복 감지] 이미지 로드 실패: {fname}")

    # 전체 쌍 비교 — slide_index별 중복 관계 수집
    labels = sorted(phashes.keys())
    # duplicate_map[idx] = 이 슬라이드와 중복으로 판정된 다른 slide_index 집합
    duplicate_map: dict[int, set[int]] = defaultdict(set)

    log.info("\n──────── 슬라이드 간 phash 거리 전체 비교 (중복 감지 튜닝용) ────────")
    log.info(f"  DUPLICATE_HASH_THRESHOLD = {cfg.DUPLICATE_HASH_THRESHOLD}  (256비트 기준, 최대 256)")
    log.info(f"  {'프레임 쌍':<30} {'dist':>5}  {'판정'}")
    log.info(f"  {'-'*30}  {'-'*5}  {'-'*10}")

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb  = labels[i], labels[j]
            idx_a   = pool[la][0]
            idx_b   = pool[lb][0]

            # 같은 슬라이드 내 base↔annot 쌍은 건너뜀
            if idx_a == idx_b:
                continue

            dist = phashes[la] - phashes[lb]
            flag = "★ 중복 후보" if dist < cfg.DUPLICATE_HASH_THRESHOLD else ""

            log.info(f"  {la:<14} ↔ {lb:<14}  {dist:>5}  {flag}")

            if dist < cfg.DUPLICATE_HASH_THRESHOLD:
                duplicate_map[idx_a].add(idx_b)
                duplicate_map[idx_b].add(idx_a)

    log.info("──────────────────────────────────────────────────────────────\n")

    # ── union-find로 전이적 중복 그룹 확정 ──────────────────────────────── #
    # 직접 쌍비교만으로는 1↔27, 27↔29가 감지돼도 1의 duplicate_of에 29가 빠질 수 있음.
    # union-find로 연결 성분을 묶으면 [1, 27, 29, 31]처럼 그룹 전체가 포함된다.
    all_indices = list(groups.keys())
    parent = {idx: idx for idx in all_indices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for idx_a, neighbors in duplicate_map.items():
        for idx_b in neighbors:
            union(idx_a, idx_b)

    # 루트별로 그룹 구성
    from collections import defaultdict as _defaultdict
    dup_groups: dict[int, set[int]] = _defaultdict(set)
    for idx in all_indices:
        dup_groups[find(idx)].add(idx)

    # 각 슬라이드의 duplicate_of = 같은 그룹의 나머지 멤버 전체 (자신 제외, 정렬)
    group_of: dict[int, set[int]] = {}
    for members in dup_groups.values():
        for idx in members:
            group_of[idx] = members

    # 플래그 삽입: duplicate_of = 중복으로 판정된 slide_index 리스트 (정렬)
    # 비어있으면 중복 없음. is_visual_duplicate 는 duplicate_of 로 대체.
    for m in metadata:
        idx = m["slide_index"]
        others = group_of.get(idx, {idx}) - {idx}
        m["duplicate_of"] = sorted(others)

    return metadata


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
    parser.add_argument("--input",  "-i", default="input/lecture.mp4",    help="입력 영상 경로")
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