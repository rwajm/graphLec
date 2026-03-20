"""
compare_slides.py
=================
이미 추출된 슬라이드 이미지를 읽어서 phash 전체 쌍 비교를 실행한다.
slide_extractor.py를 재실행하지 않고 DUPLICATE_HASH_THRESHOLD 튜닝에 사용.

Usage:
    python compare_slides.py --slides output_slides/
    python compare_slides.py --slides output_slides/ --threshold 50
    python compare_slides.py --slides output_slides/ --threshold 50 --update-metadata
"""

import cv2
import json
import argparse
import logging
import imagehash
from PIL import Image
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 30
RESIZE_WIDTH      = 960
HASH_SIZE         = 16   # 256비트


# ──────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────
def resize_frame(frame, width: int):
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def compute_phash_hires(frame) -> imagehash.ImageHash:
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_img, hash_size=HASH_SIZE)


# ──────────────────────────────────────────────
# 메인 비교 로직
# ──────────────────────────────────────────────
def compare_slides(slides_dir: str, threshold: int, update_metadata: bool):
    out_path  = Path(slides_dir)
    meta_path = out_path / "metadata.json"

    if not meta_path.exists():
        log.error(f"metadata.json 없음: {meta_path}")
        return

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    # slide_index별 그룹화
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

    log.info(f"프레임 풀: {list(pool.keys())}")

    # phash 계산
    phashes: dict[str, imagehash.ImageHash] = {}
    for label, (_, fname) in pool.items():
        img = cv2.imread(str(out_path / fname))
        if img is not None:
            phashes[label] = compute_phash_hires(resize_frame(img, RESIZE_WIDTH))
            log.info(f"  phash 계산: {label} ({fname})")
        else:
            log.warning(f"  이미지 로드 실패: {fname}")

    # 전체 쌍 비교
    labels = sorted(phashes.keys())
    duplicate_map: dict[int, set[int]] = defaultdict(set)

    print("\n" + "═" * 70)
    print(f"  슬라이드 phash 전체 쌍 비교  |  threshold={threshold}  (256비트, 최대 256)")
    print("═" * 70)
    print(f"  {'프레임 쌍':<30}  {'dist':>5}  {'판정'}")
    print(f"  {'-'*30}  {'-'*5}  {'-'*12}")

    results = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb  = labels[i], labels[j]
            idx_a   = pool[la][0]
            idx_b   = pool[lb][0]

            if idx_a == idx_b:
                continue

            dist = phashes[la] - phashes[lb]
            is_dup = dist < threshold
            flag   = "★ 중복 후보" if is_dup else ""

            print(f"  {la:<14} ↔ {lb:<14}  {dist:>5}  {flag}")
            results.append({"pair": f"{la}↔{lb}", "slide_a": idx_a, "slide_b": idx_b, "dist": dist, "duplicate": is_dup})

            if is_dup:
                duplicate_map[idx_a].add(idx_b)
                duplicate_map[idx_b].add(idx_a)

    print("═" * 70)

    # 중복 관계 요약
    if duplicate_map:
        print(f"\n  중복 관계 요약:")
        for idx in sorted(duplicate_map.keys()):
            print(f"    slide_{idx:03d}  →  duplicate_of {sorted(duplicate_map[idx])}")
    else:
        print(f"\n  threshold={threshold} 기준 중복 후보 없음")

    # 거리 분포 요약
    dists = [r["dist"] for r in results]
    if dists:
        dup_dists    = [r["dist"] for r in results if r["duplicate"]]
        nondup_dists = [r["dist"] for r in results if not r["duplicate"]]
        print(f"\n  [거리 분포]")
        print(f"    전체   min={min(dists)}  max={max(dists)}  avg={sum(dists)/len(dists):.1f}")
        if dup_dists:
            print(f"    중복   min={min(dup_dists)}  max={max(dup_dists)}")
        if nondup_dists:
            print(f"    비중복 min={min(nondup_dists)}  max={max(nondup_dists)}")
        print(f"\n  → 적정 threshold 범위: {max(dup_dists) if dup_dists else '?'} < threshold < {min(nondup_dists) if nondup_dists else '?'}")

    # metadata.json 업데이트 (--update-metadata 옵션)
    if update_metadata:
        for m in metadata:
            idx = m["slide_index"]
            m["duplicate_of"] = sorted(duplicate_map[idx])
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        log.info(f"\n  metadata.json 업데이트 완료: {meta_path}")


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="슬라이드 중복 phash 비교 (튜닝용)")
    parser.add_argument("--slides",    "-s", default="output_slides/", help="슬라이드 출력 디렉토리")
    parser.add_argument("--threshold", "-t", type=int, default=DEFAULT_THRESHOLD, help="중복 판정 임계값 (기본 30)")
    parser.add_argument("--update-metadata", action="store_true", help="비교 결과로 metadata.json의 is_visual_duplicate 업데이트")
    args = parser.parse_args()

    compare_slides(args.slides, args.threshold, args.update_metadata)