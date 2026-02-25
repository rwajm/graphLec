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
#  타임스탬프 매칭                                                               #
# ============================================================================ #

class TimestampMatcher:
    """슬라이드와 전사문 타임스탬프 기반 매칭"""
    
    def match(self, slides: List[Dict], transcripts: List[Dict]) -> List[Dict]:
        if not slides:
            return []
        
        # 각 슬라이드의 시간 구간 계산
        for i, slide in enumerate(slides):
            if i < len(slides) - 1:
                slide["timestamp_end"] = slides[i + 1]["timestamp"]
            else:
                if transcripts:
                    slide["timestamp_end"] = max(t["end"] for t in transcripts)
                else:
                    slide["timestamp_end"] = slide["timestamp"] + 60
        
        # 각 슬라이드에 해당하는 전사문 매칭
        for slide in slides:
            start, end = slide["timestamp"], slide["timestamp_end"]
            matched_texts = [
                seg["text"] for seg in transcripts
                if seg["end"] > start and seg["start"] < end
            ]
            slide["t2"] = " ".join(matched_texts)
        
        logger.info(f"✓ Matched t2 to {len(slides)} slides")
        return slides


# ============================================================================ #
#  텍스트 통합기 (t1 + t2 → t3)                                                  #
# ============================================================================ #

class TextIntegrator:
    """t1과 t2를 통합하여 t3 생성"""
    
    def integrate(self, slides: List[Dict]) -> List[Dict]:
        """
        t3 생성 규칙:
        - t1 (슬라이드 텍스트)과 t2 (전사문)를 구분하여 합침
        - 검색 및 임베딩에 사용할 통합 텍스트
        """
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
        
        slides = TimestampMatcher().match(slides, transcripts)
        
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