"""
텍스트 통합 파이프라인 (Stage 2)

Input:
  - slide_extracted_light.json: t1 (슬라이드 텍스트)
  - audio.json: t2 (Whisper 전사문)

Output:
  - integrated_text.json: t3 (통합 텍스트) + text_vector
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
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
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================ #
#  데이터 로더                                                                   #
# ============================================================================ #

class DataLoader:
    """t1, t2 데이터 로드"""
    
    @staticmethod
    def load_slides(path: Path) -> List[Dict]:
        """slide_extracted_light.json 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        slides = data.get('slides', [])
        logger.info(f"✓ Loaded {len(slides)} slides (t1)")
        return slides
    
    @staticmethod
    def load_transcripts(path: Path) -> List[Dict]:
        """audio.json 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            segments = data
        elif "segments" in data:
            segments = data["segments"]
        else:
            raise ValueError(f"Unknown transcript format")
        
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
#  타임스탬프 + 임베딩 유사도 매칭                                                               #
# ============================================================================ #

class HybridMatcher:
    """타임스탬프 + 임베딩 유사도 하이브리드 매칭"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = genai.Client(api_key=config.google_api_key)
    
    def _embed(self, text: str) -> List[float]:
        if not text.strip():
            return None
        resp = self.client.models.embed_content(
            model=self.config.embedding_model,
            contents=text[:5000],
            config=types.EmbedContentConfig(
                output_dimensionality=self.config.embedding_dim
            )
        )
        return resp.embeddings[0].values
    
    def _cosine_sim(self, a, b) -> float:
        import numpy as np
        if a is None or b is None:
            return 0.0
        a, b = np.array(a), np.array(b)
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
    
    def match(self, slides: List[Dict], transcripts: List[Dict]) -> List[Dict]:
        if not slides:
            return []
        
        # 슬라이드 시간 구간 계산
        for i, slide in enumerate(slides):
            if i < len(slides) - 1:
                slide["timestamp_end"] = slides[i + 1]["timestamp"]
            else:
                slide["timestamp_end"] = max(t["end"] for t in transcripts) if transcripts else slide["timestamp"] + 60
        
        # 슬라이드 t1 임베딩
        logger.info("Embedding slide t1 texts...")
        for slide in slides:
            slide["_t1_vec"] = self._embed(slide.get("t1", ""))
        
        # 각 오디오 세그먼트를 최적 슬라이드에 배정
        slide_t2_map = {s["slide_id"]: [] for s in slides}
        
        for seg in transcripts:
            best_slide_id = None
            best_score = -1
            
            seg_vec = self._embed(seg["text"])
            
            for slide in slides:
                # 타임스탬프 점수
                ts_score = self._timestamp_score(
                    seg["start"], seg["end"],
                    slide["timestamp"], slide["timestamp_end"]
                )
                
                # 임베딩 유사도 점수
                emb_score = self._cosine_sim(seg_vec, slide["_t1_vec"])
                # cosine sim은 -1~1이므로 0~1로 정규화
                emb_score = (emb_score + 1) / 2
                
                # 하이브리드 점수
                score = self.config.alpha * ts_score + (1 - self.config.alpha) * emb_score
                
                if score > best_score:
                    best_score = score
                    best_slide_id = slide["slide_id"]
            
            if best_slide_id:
                slide_t2_map[best_slide_id].append(seg["text"])
        
        # t2 배정
        for slide in slides:
            slide["t2"] = " ".join(slide_t2_map[slide["slide_id"]])
            del slide["_t1_vec"]  # 임시 벡터 제거
        
        logger.info(f"✓ Hybrid matched t2 to {len(slides)} slides")
        return slides


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
        logger.info(f"✓ Gemini Embedding initialized")
    
    def _embed(self, text: str) -> List[float]:
        """텍스트 → 벡터"""
        if not text.strip():
            return None
        
        # 텍스트 길이 제한
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
        """모든 슬라이드의 t3를 벡터화"""
        logger.info(f"Generating text vectors for {len(slides)} slides...")
        
        for i, slide in enumerate(slides):
            t3 = slide.get("t3", "")
            slide["text_vector"] = self._embed(t3)
            logger.info(f"  [{i+1}/{len(slides)}] Slide {slide.get('slide_number')}: {len(t3)} chars → {self.config.embedding_dim}d")
        
        logger.info(f"✓ Text vectorization complete")
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
        
        # Stage 1: 데이터 로드
        print("\n" + "-"*70)
        print("Stage 1: 데이터 로드")
        print("-"*70)
        
        slides = DataLoader.load_slides(self.config.slide_json)
        transcripts = DataLoader.load_transcripts(self.config.audio_json)
        
        # Stage 2: 타임스탬프 매칭 (t2 → 슬라이드)
        print("\n" + "-"*70)
        print("Stage 2: 타임스탬프 매칭")
        print("-"*70)
        
        slides = HybridMatcher(self.config).match(slides, transcripts)
        
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
                "embedding_dim": self.config.embedding_dim
            },
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
        print(f"\n📊 결과:")
        print(f"  • 슬라이드: {len(slides)}개")
        print(f"  • t3 생성: {sum(1 for s in slides if s['t3'])}개")
        print(f"  • 텍스트 벡터: {sum(1 for s in slides if s.get('text_vector'))}개")
        print(f"\n📁 생성된 파일:")
        print(f"  • {output_path}")
        print(f"  • {light_path}")
        print(f"\n⏱️ 처리 시간: {total_time:.2f}초")
        
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
    
    args = parser.parse_args()
    
    config = Config(
        slide_json=Path(args.slides),
        audio_json=Path(args.audio),
        output_dir=Path(args.output)
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