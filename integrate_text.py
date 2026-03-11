"""
텍스트 통합 파이프라인 (Stage 2)

Input:
  - slide_extracted_light.json: t1 (슬라이드 텍스트)
  - audio.json: t2 (Whisper 전사문)

Output:
  - integrated_text.json: t3 (통합 텍스트) + text_vector

수정 내역:
  [Fix1] 매칭 threshold 도입 — 저품질 세그먼트 강제 배정 제거
  [Fix2] 매칭 메타데이터 보존 — match_score, ts_score, emb_score 저장
  [Fix3] 세그먼트 단위 타임스탬프 보존 — t2_segments에 start/end 유지
  [Fix4] 동적 alpha — 슬라이드 텍스트 길이 기반으로 타임스탬프/임베딩 가중치 조정
  [Fix5] 마지막 슬라이드 종료 시점 버그 수정 — +60 제거, 실제 오디오 종료 시간 사용
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================ #
#  설정                                                                         #
# ============================================================================ #

@dataclass
class Config:
    google_api_key: str = os.getenv('GOOGLE_API_KEY', '')
    slide_json: Path = Path("./output/slide_extracted_light.json")
    audio_json: Path = Path("./audio.json")
    output_dir: Path = Path("./output")
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dim: int = 768
    alpha: float = 0.4
    # 매칭 threshold — 이 점수 미만의 세그먼트는 unmatched로 분류
    match_threshold: float = 0.55
    # 단문 필터 — 이 글자 수 미만 세그먼트는 매칭 시도 자체를 건너뜀
    min_segment_length: int = 5
    # 동적 alpha — 슬라이드 텍스트가 이 길이 미만이면 alpha_short 사용
    alpha_short_threshold: int = 30
    alpha_short: float = 0.8  # 텍스트 적은 슬라이드 → 타임스탬프 비중 확대

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================ #
#  데이터 로더                                                                   #
# ============================================================================ #

class DataLoader:
    """t1, t2 데이터 로드"""

    @staticmethod
    def load_slides(path: Path) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        slides = data.get('slides', [])
        logger.info(f"✓ Loaded {len(slides)} slides (t1)")
        return slides

    @staticmethod
    def load_transcripts(path: Path) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            segments = data
        elif "segments" in data:
            segments = data["segments"]
        else:
            raise ValueError("Unknown transcript format")

        result = []
        for seg in segments:
            result.append({
                "start": float(seg.get("start", 0)),
                "end": float(seg.get("end", 0)),
                "text": seg.get("text", "").strip()
            })

        logger.info(f"✓ Loaded {len(result)} transcript segments (t2)")
        return result


# ============================================================================ #
#  타임스탬프 + 임베딩 유사도 매칭                                                #
# ============================================================================ #

class HybridMatcher:
    """타임스탬프 + 임베딩 유사도 하이브리드 매칭"""

    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.google_api_key)

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if not text.strip():
            return None
        resp = self.client.models.embed_content(
            model=self.config.embedding_model,
            contents=text[:5000],
            config=types.EmbedContentConfig(
                output_dimensionality=self.config.embedding_dim
            )
        )
        # 즉시 np.array로 변환 — 매칭 루프에서 반복 변환 비용 제거
        return np.array(resp.embeddings[0].values, dtype=np.float32)

    def _cosine_sim(self, a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
        # import numpy 제거 — 모듈 최상단으로 이동
        if a is None or b is None:
            return 0.0
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _timestamp_score(self, seg_start: float, seg_end: float,
                         slide_start: float, slide_end: float) -> float:
        """오디오 세그먼트가 슬라이드 시간 구간에 얼마나 겹치는지 (0~1)"""
        overlap = max(0, min(seg_end, slide_end) - max(seg_start, slide_start))
        seg_duration = seg_end - seg_start
        if seg_duration == 0:
            return 0.0
        return overlap / seg_duration

    def match(self, slides: List[Dict], transcripts: List[Dict]) -> tuple[List[Dict], Dict]:
        """
        Returns:
            slides: t2_segments, t2, t2_coverage, has_audio 필드가 추가된 슬라이드 리스트
            gap_report: 매칭 품질 리포트
        """
        if not slides:
            return [], {}

        # 마지막 슬라이드 종료 시점: +60 제거 → 실제 오디오 종료 시간 사용
        audio_end = max(t["end"] for t in transcripts) if transcripts else 0.0
        for i, slide in enumerate(slides):
            if i < len(slides) - 1:
                slide["timestamp_end"] = slides[i + 1]["timestamp"]
            else:
                slide["timestamp_end"] = audio_end 

        # 슬라이드 t1 임베딩 — np.array 변환은 _embed 내부에서 처리
        logger.info("Embedding slide t1 texts...")
        for slide in slides:
            t1_combined = "\n".join(filter(None, [
                slide.get("t1", ""),
                slide.get("t1_structure", "")
            ]))
            slide["_t1_vec"] = self._embed(t1_combined)
            # 슬라이드별 동적 alpha 계산
            t1_len = len(slide.get("t1", ""))
            slide["_alpha"] = (
                self.config.alpha_short
                if t1_len < self.config.alpha_short_threshold
                else self.config.alpha
            )

        # 세그먼트별 매칭 결과를 메타데이터와 함께 저장
        # 구조: { slide_id: [ { text, start, end, match_score, ts_score, emb_score }, ... ] }
        slide_t2_map: Dict[str, List[Dict]] = {s["slide_id"]: [] for s in slides}

        # threshold 미달 세그먼트는 unmatched로 분리
        unmatched_segments: List[Dict] = []
        # 단문 필터링 카운터
        filtered_count = 0

        for seg in transcripts:
            # 단문 세그먼트 필터링 — API 비용 절감 + 매칭 오염 방지
            if len(seg["text"].strip()) < self.config.min_segment_length:
                filtered_count += 1
                continue

            best_slide_id = None
            best_score = -1.0
            best_ts_score = 0.0
            best_emb_score = 0.0

            seg_vec = self._embed(seg["text"])

            for slide in slides:
                ts_score = self._timestamp_score(
                    seg["start"], seg["end"],
                    slide["timestamp"], slide["timestamp_end"]
                )
                raw_emb = self._cosine_sim(seg_vec, slide["_t1_vec"])
                emb_score = (raw_emb + 1) / 2  # -1~1 → 0~1 정규화

                # 슬라이드별 동적 alpha 적용
                alpha = slide["_alpha"]
                score = alpha * ts_score + (1 - alpha) * emb_score

                if score > best_score:
                    best_score = score
                    best_slide_id = slide["slide_id"]
                    best_ts_score = ts_score
                    best_emb_score = emb_score

            # threshold 판단
            if best_score >= self.config.match_threshold:
                # 메타데이터 + 타임스탬프 함께 저장
                slide_t2_map[best_slide_id].append({
                    "text": seg["text"],
                    "start": seg["start"],       
                    "end": seg["end"],           
                    "match_score": round(best_score, 4),     
                    "ts_score": round(best_ts_score, 4),     
                    "emb_score": round(best_emb_score, 4),   
                })
            else:
                # 저품질 매칭은 unmatched로 분리
                unmatched_segments.append({
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "best_score": round(best_score, 4),
                    "best_slide_id": best_slide_id,  # 후보는 기록해둠
                })

        # 슬라이드에 결과 반영
        matched_count = 0
        for slide in slides:
            segments = slide_t2_map[slide["slide_id"]]
            slide["t2_segments"] = segments
            slide["t2"] = " ".join(s["text"] for s in segments)
            slide["t2_coverage"] = len(segments)
            slide["has_audio"] = len(segments) > 0
            matched_count += len(segments)
            del slide["_t1_vec"]
            del slide["_alpha"] 

        # gap_report 생성
        total_segments = len(transcripts)
        gap_report = {
            "total_segments": total_segments,
            "filtered_segments": filtered_count,        
            "matched_segments": matched_count,
            "unmatched_segments_count": len(unmatched_segments),
            "match_rate": round(matched_count / (total_segments - filtered_count), 4)
                          if (total_segments - filtered_count) > 0 else 0,
            "slide_only_slides": [
                s["slide_id"] for s in slides if not s["has_audio"]
            ],
            "avg_match_score": round(
                sum(seg["match_score"] for segs in slide_t2_map.values() for seg in segs)
                / matched_count, 4
            ) if matched_count else 0,
            "unmatched_segments": unmatched_segments,
        }

        logger.info(
            f"✓ Matching complete | "
            f"filtered: {filtered_count} | "
            f"matched: {matched_count}/{total_segments - filtered_count} "
            f"({gap_report['match_rate']*100:.1f}%) | "
            f"unmatched: {len(unmatched_segments)} | "
            f"slide-only: {len(gap_report['slide_only_slides'])}"
        )

        return slides, gap_report


# ============================================================================ #
#  텍스트 통합기                                                                 #
# ============================================================================ #

class TextIntegrator:
    """t1과 t2를 통합하여 t3 생성"""

    def integrate(self, slides: List[Dict]) -> List[Dict]:
        for slide in slides:
            t1 = slide.get("t1", "").strip()
            t2 = slide.get("t2", "").strip()
            title = slide.get("title", "")

            parts = []
            if title:
                parts.append(f"[제목] {title}")
            if t1:
                parts.append(f"[슬라이드 내용]\n{t1}")
            if t2:
                parts.append(f"[교수 설명]\n{t2}")

            slide["t3"] = "\n\n".join(parts)

        logger.info(f"✓ Generated t3 for {len(slides)} slides")
        return slides


# ============================================================================ #
#  텍스트 벡터 생성기                                                            #
# ============================================================================ #

class TextVectorizer:
    """t3 → text_vector 생성 (Gemini Embedding)"""

    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.google_api_key)
        logger.info("✓ Gemini Embedding initialized")

    def _embed(self, text: str) -> Optional[List[float]]:
        if not text.strip():
            return None
        if len(text) > 10000:
            text = text[:10000]
        resp = self.client.models.embed_content(
            model=self.config.embedding_model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.config.embedding_dim
            )
        )
        return resp.embeddings[0].values

    def vectorize(self, slides: List[Dict]) -> List[Dict]:
        logger.info(f"Generating text vectors for {len(slides)} slides...")
        for i, slide in enumerate(slides):
            t3 = slide.get("t3", "")
            slide["text_vector"] = self._embed(t3)
            logger.info(
                f"  [{i+1}/{len(slides)}] Slide {slide.get('slide_number')}: "
                f"{len(t3)} chars → {self.config.embedding_dim}d"
            )
        logger.info("✓ Text vectorization complete")
        return slides


# ============================================================================ #
#  파이프라인                                                                    #
# ============================================================================ #

class IntegrationPipeline:
    """t1 + t2 → t3 + text_vector 파이프라인"""

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def run(self) -> Dict:
        start_time = time.time()

        print("\n" + "="*70)
        print("🔗 텍스트 통합 파이프라인 (Stage 2)")
        print("="*70)
        print(f"📄 Slides (t1): {self.config.slide_json}")
        print(f"🎤 Audio (t2): {self.config.audio_json}")
        print(f"📂 Output: {self.config.output_dir}")
        print(f"⚙️  Match threshold: {self.config.match_threshold}")
        print(f"⚙️  Min segment length: {self.config.min_segment_length}자")
        print(f"⚙️  Alpha: {self.config.alpha} (텍스트 ≥{self.config.alpha_short_threshold}자) / "
              f"{self.config.alpha_short} (텍스트 <{self.config.alpha_short_threshold}자)")

        # Stage 1: 데이터 로드
        print("\n" + "-"*70)
        print("Stage 1: 데이터 로드")
        print("-"*70)
        slides = DataLoader.load_slides(self.config.slide_json)
        transcripts = DataLoader.load_transcripts(self.config.audio_json)

        # Stage 2: 하이브리드 매칭 (t2 → 슬라이드)
        print("\n" + "-"*70)
        print("Stage 2: 하이브리드 매칭")
        print("-"*70)
        slides, gap_report = HybridMatcher(self.config).match(slides, transcripts)

        # Stage 3: 텍스트 통합 (t1 + t2 → t3)
        print("\n" + "-"*70)
        print("Stage 3: 텍스트 통합 (t3 생성)")
        print("-"*70)
        slides = TextIntegrator().integrate(slides)

        # Stage 4: 텍스트 벡터 생성
        print("\n" + "-"*70)
        print("Stage 4: 텍스트 벡터 생성")
        print("-"*70)
        slides = TextVectorizer(self.config).vectorize(slides)

        # Stage 5: 결과 저장
        print("\n" + "-"*70)
        print("Stage 5: 결과 저장")
        print("-"*70)

        result = {
            "metadata": {
                "slide_json": str(self.config.slide_json),
                "audio_json": str(self.config.audio_json),
                "processing_time": time.time() - start_time,
                "total_slides": len(slides),
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.config.embedding_dim,
                "match_threshold": self.config.match_threshold,
            },
            "gap_report": gap_report, 
            "slides": [
                {
                    "slide_id": s["slide_id"],
                    "slide_number": s["slide_number"],
                    "timestamp": s["timestamp"],
                    "timestamp_end": s["timestamp_end"],
                    "timestamp_formatted": s["timestamp_formatted"],
                    "image_path": s["image_path"],
                    "title": s["title"],
                    "t1": s["t1"],
                    "t2": s["t2"],
                    "t2_segments": s["t2_segments"], 
                    "t2_coverage": s["t2_coverage"], 
                    "has_audio": s["has_audio"],     
                    "t3": s["t3"],
                    "text_vector": s["text_vector"]
                }
                for s in slides
            ]
        }

        # 전체 저장 (벡터 포함)
        output_path = self.config.output_dir / "integrated_text.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {output_path}")

        # 경량 버전 (벡터 제외)
        result_light = {
            "metadata": result["metadata"],
            "gap_report": result["gap_report"],
            "slides": [
                {k: v for k, v in s.items() if k != "text_vector"}
                for s in result["slides"]
            ]
        }
        light_path = self.config.output_dir / "integrated_text_light.json"
        with open(light_path, 'w', encoding='utf-8') as f:
            json.dump(result_light, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved (light): {light_path}")

        # 완료 리포트
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("✅ 통합 완료!")
        print("="*70)
        print(f"\n📊 매칭 결과:")
        print(f"  • 전체 세그먼트: {gap_report['total_segments']}개")
        print(f"  • 단문 필터링: {gap_report['filtered_segments']}개 (매칭 제외)")
        print(f"  • 매칭 성공: {gap_report['matched_segments']}개 ({gap_report['match_rate']*100:.1f}%)")
        print(f"  • 매칭 실패 (unmatched): {gap_report['unmatched_segments_count']}개")
        print(f"  • 오디오 없는 슬라이드: {len(gap_report['slide_only_slides'])}개")
        print(f"  • 평균 매칭 점수: {gap_report['avg_match_score']}")
        print(f"\n📊 통합 결과:")
        print(f"  • 슬라이드: {len(slides)}개")
        print(f"  • t3 생성: {sum(1 for s in slides if s['t3'])}개")
        print(f"  • 텍스트 벡터: {sum(1 for s in slides if s.get('text_vector'))}개")
        print(f"\n📁 생성된 파일:")
        print(f"  • {output_path}")
        print(f"  • {light_path}")
        print(f"\n⏱️  처리 시간: {total_time:.2f}초")

        return result


# ============================================================================ #
#  메인                                                                         #
# ============================================================================ #

def main():
    import argparse

    parser = argparse.ArgumentParser(description="t1 + t2 → t3 + text_vector")
    parser.add_argument("-s", "--slides", default="./output/slide_extracted_light.json")
    parser.add_argument("-a", "--audio", default="./audio.json")
    parser.add_argument("-o", "--output", default="./output")
    parser.add_argument("--threshold", type=float, default=0.55,
                        help="매칭 threshold (default: 0.55)")
    parser.add_argument("--min-seg-len", type=int, default=5,
                        help="단문 필터 최소 글자 수 (default: 5)")
    parser.add_argument("--alpha-short", type=float, default=0.8,
                        help="텍스트 빈약 슬라이드용 alpha (default: 0.8)")

    args = parser.parse_args()

    config = Config(
        slide_json=Path(args.slides),
        audio_json=Path(args.audio),
        output_dir=Path(args.output),
        match_threshold=args.threshold,
        min_segment_length=args.min_seg_len,
        alpha_short=args.alpha_short,
    )

    if not config.slide_json.exists():
        print(f"❌ Slide JSON not found: {config.slide_json}")
        return

    if not config.audio_json.exists():
        print(f"❌ Audio JSON not found: {config.audio_json}")
        return

    pipeline = IntegrationPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()