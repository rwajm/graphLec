"""
슬라이드 데이터 추출 파이프라인 (Stage 1)

Input:
  - slides/ 폴더: 슬라이드 이미지

Output:
  - slide_extracted.json: t1, image_vector 추출 결과

추출 항목:
  - t1: 슬라이드 원본 텍스트 (Gemini Vision)
  - image_vector: 이미지 벡터 (ColPali)

※ 오디오 정제는 다음 단계에서 t1을 사용하여 진행
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass
from PIL import Image
import google.generativeai as genai

try:
    import torch
    from colpali_engine.models import ColPali, ColPaliProcessor
    COLPALI_AVAILABLE = True
except ImportError:
    COLPALI_AVAILABLE = False
    print("⚠️ ColPali not installed. Image vectorization will be skipped.")

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
    slides_dir: Path = Path("./slides")
    output_dir: Path = Path("./output")
    gemini_model: str = "models/gemini-2.5-flash"
    colpali_model: str = "vidore/colpali-v1.2"
    device: str = "cuda"
    
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


# t1 추출용 프롬프트 (텍스트만)
T1_EXTRACTION_PROMPT = """
이 슬라이드 이미지에서 보이는 모든 텍스트를 추출하라.
설명 없이 JSON만 출력.

출력 형식:
{
  "title": "슬라이드 제목",
  "raw_text": "슬라이드에 보이는 모든 텍스트 (위→아래, 좌→우 순서, 줄바꿈은 \\n)"
}

추출 규칙:
- 제목, 본문, 불릿 포인트, 표 내용, 코드, 다이어그램 레이블 모두 포함
- 원문 그대로 추출 (요약하지 말 것)
- 불릿 포인트는 "- " 또는 "• "로 시작
- 줄바꿈은 \\n으로 표시
"""


# ============================================================================ #
#  슬라이드 로더                                                                 #
# ============================================================================ #

class SlideLoader:
    def __init__(self, slides_dir: Path):
        self.slides_dir = Path(slides_dir)
    
    def load(self) -> List[Dict]:
        slides = []
        pattern = re.compile(r'slide_(\d+)_(\d+\.?\d*)s\.(jpg|png)')
        
        for file in sorted(self.slides_dir.iterdir()):
            match = pattern.match(file.name)
            if match:
                slides.append({
                    "slide_number": int(match.group(1)),
                    "timestamp": float(match.group(2)),
                    "image_path": str(file),
                    "image": Image.open(file).convert("RGB")
                })
        
        logger.info(f"✓ Loaded {len(slides)} slides")
        return slides


# ============================================================================ #
#  t1 추출기 (Gemini Vision - 슬라이드 텍스트만)                                   #
# ============================================================================ #

class T1Extractor:
    """슬라이드 이미지 → t1 (원본 텍스트) 추출"""
    
    def __init__(self, config: Config):
        self.config = config
        genai.configure(api_key=config.google_api_key)
        self.model = genai.GenerativeModel(config.gemini_model)
        logger.info(f"✓ Gemini initialized for t1 extraction")
    
    def extract(self, slide: Dict) -> Dict:
        """단일 슬라이드에서 t1 추출"""
        try:
            response = self.model.generate_content([T1_EXTRACTION_PROMPT, slide["image"]])
            text = response.text
            
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            result = json.loads(text.strip())
            
            slide["title"] = result.get("title", f"Slide {slide['slide_number']}")
            slide["t1"] = result.get("raw_text", "")
            
        except Exception as e:
            logger.error(f"  ✗ t1 extraction failed for slide {slide['slide_number']}: {e}")
            slide["title"] = f"Slide {slide['slide_number']}"
            slide["t1"] = ""
        
        return slide
    
    def extract_batch(self, slides: List[Dict]) -> List[Dict]:
        logger.info(f"Extracting t1 from {len(slides)} slides...")
        
        for i, slide in enumerate(slides):
            self.extract(slide)
            logger.info(f"  [{i+1}/{len(slides)}] Slide {slide['slide_number']}: t1={len(slide['t1'])} chars")
        
        logger.info(f"✓ t1 extraction complete")
        return slides


# ============================================================================ #
#  이미지 벡터 추출기 (ColPali)                                                   #
# ============================================================================ #

class ImageVectorizer:
    """슬라이드 이미지 → image_vector 추출"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.processor = None
        
        if COLPALI_AVAILABLE:
            self._init_colpali()
    
    def _init_colpali(self):
        logger.info(f"Loading ColPali: {self.config.colpali_model}")
        self.device = self.config.device if torch.cuda.is_available() else "cpu"
        
        self.model = ColPali.from_pretrained(
            self.config.colpali_model,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
        ).to(self.device).eval()
        
        self.processor = ColPaliProcessor.from_pretrained(self.config.colpali_model)
        logger.info(f"✓ ColPali loaded on {self.device}")
    
    def vectorize(self, slides: List[Dict]) -> List[Dict]:
        if not self.model:
            logger.warning("ColPali not available, skipping image vectorization")
            for slide in slides:
                slide["image_vector"] = None
                slide["patch_vectors"] = None
            return slides
        
        logger.info(f"Extracting image vectors from {len(slides)} slides...")
        
        for i, slide in enumerate(slides):
            image = slide.get("image")
            if image is None:
                image = Image.open(slide["image_path"]).convert("RGB")
            
            batch = self.processor.process_images([image]).to(self.device)
            
            with torch.no_grad():
                embeddings = self.model(**batch)
            
            patch_vectors = embeddings.squeeze(0).cpu().numpy()
            full_vector = patch_vectors.mean(axis=0)
            
            slide["image_vector"] = full_vector.tolist()
            slide["patch_vectors"] = patch_vectors.tolist()
            
            if (i + 1) % 10 == 0:
                logger.info(f"  [{i+1}/{len(slides)}] vectorized")
        
        logger.info(f"✓ Image vectorization complete")
        return slides


# ============================================================================ #
#  파이프라인                                                                    #
# ============================================================================ #

class ExtractionPipeline:
    """t1, image_vector 추출 파이프라인"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def run(self) -> Dict:
        start_time = time.time()
        
        print("\n" + "="*70)
        print("🎓 슬라이드 데이터 추출 파이프라인 (Stage 1)")
        print("="*70)
        print(f"📁 Slides: {self.config.slides_dir}")
        print(f"📂 Output: {self.config.output_dir}")
        
        # Stage 1: 슬라이드 로드
        print("\n" + "-"*70)
        print("Stage 1: 슬라이드 로드")
        print("-"*70)
        
        slides = SlideLoader(self.config.slides_dir).load()
        
        # Stage 2: t1 추출 (슬라이드 텍스트)
        print("\n" + "-"*70)
        print("Stage 2: t1 추출 (Gemini Vision)")
        print("-"*70)
        
        slides = T1Extractor(self.config).extract_batch(slides)
        
        # Stage 3: 이미지 벡터 추출 (ColPali)
        print("\n" + "-"*70)
        print("Stage 3: 이미지 벡터 추출 (ColPali)")
        print("-"*70)
        
        slides = ImageVectorizer(self.config).vectorize(slides)
        
        # Stage 4: 결과 저장
        print("\n" + "-"*70)
        print("Stage 4: 결과 저장")
        print("-"*70)
        
        # 타임스탬프 포맷팅
        for slide in slides:
            ts = slide["timestamp"]
            mins, secs = divmod(ts, 60)
            hrs, mins = divmod(mins, 60)
            slide["timestamp_formatted"] = f"{int(hrs):02d}:{int(mins):02d}:{secs:05.2f}"
            slide["slide_id"] = f"slide_{slide['slide_number']:03d}"
        
        # 결과 구성
        result = {
            "metadata": {
                "slides_dir": str(self.config.slides_dir),
                "processing_time": time.time() - start_time,
                "total_slides": len(slides)
            },
            "slides": [
                {
                    "slide_id": s["slide_id"],
                    "slide_number": s["slide_number"],
                    "timestamp": s["timestamp"],
                    "timestamp_formatted": s["timestamp_formatted"],
                    "image_path": s["image_path"],
                    "title": s["title"],
                    "t1": s["t1"],                      # 슬라이드 원본 텍스트
                    "image_vector": s["image_vector"],  # ColPali 벡터
                    "patch_vectors": s["patch_vectors"]
                }
                for s in slides
            ]
        }
        
        # 전체 저장 (벡터 포함)
        output_path = self.config.output_dir / "slide_extracted.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved: {output_path}")
        
        # 경량 버전 (벡터 제외)
        result_light = {
            "metadata": result["metadata"],
            "slides": [
                {k: v for k, v in s.items() if k not in ["image_vector", "patch_vectors"]}
                for s in result["slides"]
            ]
        }
        light_path = self.config.output_dir / "slide_extracted_light.json"
        with open(light_path, 'w', encoding='utf-8') as f:
            json.dump(result_light, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Saved (light): {light_path}")
        
        # 완료 리포트
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("✅ 추출 완료!")
        print("="*70)
        print(f"\n📊 결과:")
        print(f"  • 슬라이드: {len(slides)}개")
        print(f"  • t1 추출: {sum(1 for s in slides if s['t1'])}개")
        print(f"  • 이미지 벡터: {sum(1 for s in slides if s.get('image_vector'))}개")
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
    
    parser = argparse.ArgumentParser(description="t1, image_vector 추출")
    parser.add_argument("-s", "--slides", default="./slides")
    parser.add_argument("-o", "--output", default="./output")
    
    args = parser.parse_args()
    
    config = Config(
        slides_dir=Path(args.slides),
        output_dir=Path(args.output)
    )
    
    if not config.slides_dir.exists():
        print(f"❌ Slides folder not found: {config.slides_dir}")
        return
    
    pipeline = ExtractionPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()