"""
json_to_graph_triples.py — fused.json → graph_triples.csv

입력: {stem}_fused.json  (fusion.py 출력)
출력: {stem}_graph_triples.csv

레이어:
  - 구조 레이어: 결정론적 (Slide, Segment, AnnotationEmphasis 노드 + 관계)
  - 개념 레이어: Gemini 기반 (Concept 노드, MENTIONS/APPEARS_IN/관계 엣지)

사용법:
  python json_to_graph_triples.py --stem os1-1
"""

import os
import json
import csv
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from google import genai
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
#  설정
# ============================================================================

@dataclass
class Config:
    stem:        str  = "lecture"
    output_dir:  Path = Path("output")
    slides_dir:  Path = Path("output_slides")

    fused_path:  Path = field(default=None)
    output_csv:  Path = field(default=None)

    google_api_key: str = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY_2', ''))
    gemini_model:   str = "models/gemini-2.5-flash"
    lecture_title:  str = "강의"

    def __post_init__(self):
        try:
            from config import output_paths
            paths = output_paths(self.stem, self.output_dir, self.slides_dir)
            if self.fused_path is None: self.fused_path = paths["fused"]
        except ImportError:
            if self.fused_path is None:
                self.fused_path = self.output_dir / f"{self.stem}_fused.json"
        if self.output_csv is None:
            self.output_csv = self.output_dir / f"{self.stem}_graph_triples.csv"


RELATION_TYPES = {
    "is_a", "part_of", "implements", "abstracts",
    "prerequisite_of", "uses", "calls",
    "compared_to", "extends", "replaces",
    "solves", "optimizes"
}

ENTITY_TYPES = {
    "concept", "agent", "system", "artifact",
    "method", "event", "phenomenon", "metric",
    "location", "time_period"
}

DOMAIN_PROMPT = """
아래 강의 내용을 보고 도메인을 분류해라. 설명 없이 JSON만 출력.

[본문]
{content}

출력 형식:
{{
  "domain": "engineering|natural_science|humanities|social_science|arts",
  "subdomain": "computer_science|physics|economics|..."
}}

도메인 기준:
- engineering: 설계·시스템·구현·최적화 중심
- natural_science: 자연 현상·법칙·실험·측정 중심
- humanities: 해석·사상·텍스트·역사·철학 중심
- social_science: 사회 구조·제도·정책·행위자·지표 중심
- arts: 작품·표현·매체·양식·창작 중심
"""

EXTRACTION_PROMPT = """
아래는 강의 전체 내용이다. 도메인: {domain} / {subdomain}
설명 없이 JSON만 출력.

각 텍스트 단위에는 [source_id] 태그가 붙어 있다.
- [slide_XXX]      : 슬라이드 제목/텍스트
- [segment/NNNN]   : 오디오 전사 세그먼트

[본문]
{content}

출력 형식:
{{
  "entities": [
    {{"name": "엔티티명", "type": "엔티티타입"}},
    ...
  ],
  "relations": [
    {{
      "from": "출발 엔티티명",
      "to": "도착 엔티티명",
      "type": "관계 타입",
      "evidence": "근거 문장"
    }}
  ],
  "mentions": {{
    "엔티티명": ["segment/0001", "slide_002"],
    ...
  }}
}}

엔티티 타입 (10가지만 사용):
- concept   : 추상 개념, 이론, 원리, 분류
- agent     : 행위 주체, 사람, 집단, 조직, 기관
- system    : 구조적으로 결합된 체계
- artifact  : 인공물, 문서, 도구, 소프트웨어
- method    : 절차, 알고리즘, 기법, 실험법
- event     : 사건, 실행, 변화, 실험, 시행
- phenomenon: 현상, 문제, 효과, 상태
- metric    : 변수, 지표, 측정값, 성능값
- location  : 장소, 영역, 공간, 구간
- time_period: 시점, 기간, 시대, 단계

엔티티 추출 규칙:
- 강의 전체에서 등장하는 모든 핵심 엔티티 추출, 동일 엔티티 통합
- 정의·용어·기술적 메커니즘·문제·현상·구체적 식별자·행위자·도구 모두 포함
- 슬라이드 레이블(코드 번호, 그림 번호), 교육 메타 표현("다음 슬라이드", "예제") 제외
- 수량 제한 없음: 강의에 등장하는 모든 의미있는 엔티티 추출
- 자기 자신과의 관계 제외

관계 타입 (12가지만 사용):
is_a, part_of, implements, abstracts, prerequisite_of, uses, calls,
compared_to, extends, replaces, solves, optimizes

mentions 작성 규칙:
- 각 엔티티가 직접 언급·설명·예시로 다뤄지는 모든 source_id 나열
- 정확한 단어 일치 불필요, 관련 설명·예시도 포함
- segment와 slide 양쪽 모두 포함 가능
- 모든 entity는 최소 1개 이상의 source_id 필수
"""


# ============================================================================
#  트리플 수집기
# ============================================================================

class TripleCollector:
    def __init__(self):
        self.triples: List[Tuple] = []
        self._edge_set: set = set()

    def add(self, subject: str, predicate: str, obj: str, properties: dict = None):
        key = (subject, predicate, obj)
        if key in self._edge_set:
            return
        self._edge_set.add(key)
        props_str = json.dumps(properties, ensure_ascii=False) if properties else ''
        self.triples.append((subject, predicate, obj, props_str))

    def write_csv(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['subject', 'predicate', 'object', 'properties'])
            for row in self.triples:
                writer.writerow(row)
        logger.info(f"✓ CSV 저장: {path} ({len(self.triples)}개 트리플)")


# ============================================================================
#  전처리: fused.json 인덱싱
# ============================================================================

class Preprocessor:
    """
    fused.json → 구조 레이어·개념 레이어에서 사용할 인덱스 생성

    인덱스:
      - slide_map:    slide_id → slide dict
      - segment_data: segment_id → segment dict (slide_id 포함)
      - annot_data:   annot_id  → annotation_summary dict (slide_id 포함)
    """

    def __init__(self, slides: List[Dict]):
        self.slides = slides
        self.slide_map:    Dict[str, Dict] = {}
        self.segment_data: Dict[str, Dict] = {}  # segment/NNNN → dict
        self.annot_data:   Dict[str, Dict] = {}  # annotation/NNNN → dict
        self.scene_data:   Dict[str, Dict] = {}  # slide_id/scene/NN → dict
        self._build()

    def _build(self):
        seg_idx = 0
        ann_idx = 0

        for slide in self.slides:
            sid = slide['slide_id']
            self.slide_map[sid] = slide

            for ctx in slide.get('contexts', []):
                scene_id = f"{sid}/scene/{ctx['context_index']:02d}"
                for seg in ctx.get('segments', []):
                    seg_id = f'segment/{seg_idx:04d}'
                    self.segment_data[seg_id] = {
                        **seg,
                        'slide_id':      sid,
                        'scene_id':      scene_id,
                        'context_index': ctx['context_index'],
                    }
                    seg_idx += 1

            for ctx in slide.get('contexts', []):
                scene_id = f"{sid}/scene/{ctx['context_index']:02d}"
                self.scene_data[scene_id] = {
                    'slide_id':      sid,
                    'context_index': ctx['context_index'],
                    'start':         ctx.get('start'),
                    'end':           ctx.get('end'),
                    'stressed':      ctx.get('stressed', False),
                }

            for ann in slide.get('annotations_summary', []):
                ann_id = f'annotation/{ann_idx:04d}'
                self.annot_data[ann_id] = {**ann, 'slide_id': sid}
                ann_idx += 1

        logger.info(f"✓ 전처리 완료: slide {len(self.slides)}개, "
                    f"segment {seg_idx}개, annotation {ann_idx}개")


# ============================================================================
#  개념 정규화
# ============================================================================

class ConceptNormalizer:

    def __init__(self):
        pass

    def normalize(self, concept: str) -> str:
        concept = concept.strip()
        if '(' in concept:
            concept = concept.split('(')[0].strip()
        return concept

    @staticmethod
    def to_slug(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def to_display_name(name: str) -> str:
        return name.strip().upper()


# ============================================================================
#  구조 레이어 빌더
# ============================================================================

class StructureLayerBuilder:

    def __init__(self, pre: Preprocessor, collector: TripleCollector, config: Config):
        self.pre = pre
        self.c   = collector
        self.cfg = config
        self.vid = f'lecture_video/{config.stem}'

    def build(self):
        self._build_root()
        self._build_slides()
        self._build_scenes()
        self._build_segments()
        self._build_annotations()
        self._build_deictic_links()
        logger.info("✓ 구조 레이어 완료")

    def _build_root(self):
        self.c.add(self.vid, 'type', 'Video', {'title': self.cfg.lecture_title, 'stem': self.cfg.stem})
        self.c.add(self.vid, 'HAS_SLIDES', f'{self.vid}/slides')
        self.c.add(f'{self.vid}/slides', 'type', 'Slides')
        self.c.add(self.vid, 'HAS_SCENES', f'{self.vid}/scenes')
        self.c.add(f'{self.vid}/scenes', 'type', 'Scenes')

    def _build_slides(self):
        for slide in self.pre.slides:
            sid = slide['slide_id']
            self.c.add(f'{self.vid}/slides', 'CONTAINS', sid)
            self.c.add(sid, 'type', 'Slide', {
                'slide_number':  slide.get('slide_number'),
                'title':         slide.get('title', ''),
                'slide_text':    slide.get('slide_text', ''),
                'role':          slide.get('role'),
                'start_sec':     slide.get('start_sec'),
                'end_sec':       slide.get('end_sec'),
                'emphasis_total': slide.get('emphasis_score', {}).get('total', 0.0),
            })

    def _build_scenes(self):
        """context 단위 Scene 노드 생성"""
        for scene_id, data in self.pre.scene_data.items():
            self.c.add(f'{self.vid}/scenes', 'CONTAINS', scene_id)
            self.c.add(data['slide_id'], 'HAS_SCENE', scene_id)
            self.c.add(scene_id, 'type', 'Scene', {
                'slide_id':      data['slide_id'],
                'context_index': data['context_index'],
                'start':         data['start'],
                'end':           data['end'],
                'stressed':      data['stressed'],
            })

    def _build_segments(self):
        """segment → Segment 노드 + Scene에 HAS_SEGMENT 연결"""
        for seg_id, data in self.pre.segment_data.items():
            self.c.add(data['scene_id'], 'HAS_SEGMENT', seg_id)
            self.c.add(seg_id, 'type', 'Segment', {
                'start':   data['start'],
                'end':     data['end'],
                'text':    data['text'],
                'stressed': data.get('stressed', False),
            })

    def _build_annotations(self):
        """annotation_summary → AnnotationEmphasis 노드 + 슬라이드에 HAS_ANNOTATION 연결"""
        for ann_id, data in self.pre.annot_data.items():
            self.c.add(data['slide_id'], 'HAS_ANNOTATION', ann_id)
            props = {
                'type':           data.get('type'),
                'target_content': data.get('target_content'),
                'score':          data.get('score', 0.0),
                'confidence':     data.get('confidence'),
                'timestamp_sec':  data.get('timestamp_sec'),
            }
            if data.get('bbox'):
                props['bbox'] = data['bbox']
            if data.get('handwritten_content'):
                props['handwritten_content'] = data['handwritten_content']
            self.c.add(ann_id, 'type', 'AnnotationEmphasis', props)

    def _build_deictic_links(self):
        """deictic_target이 있는 segment → 대상 annotation에 REFERS_TO 엣지"""
        # annotation target_content → ann_id 역인덱스 (slide 범위 내)
        slide_annot_index: Dict[str, Dict[str, str]] = defaultdict(dict)
        for ann_id, data in self.pre.annot_data.items():
            tc = data.get('target_content', '')
            if tc:
                slide_annot_index[data['slide_id']][tc] = ann_id

        for seg_id, data in self.pre.segment_data.items():
            dt = data.get('deictic_target')
            if not dt:
                continue
            tc    = dt.get('target_content', '')
            s_id  = data['slide_id']
            ann_id = slide_annot_index.get(s_id, {}).get(tc)
            if ann_id:
                self.c.add(seg_id, 'REFERS_TO', ann_id, {
                    'deictic_type': dt.get('annotation_type'),
                    'confidence':   dt.get('confidence'),
                })


# ============================================================================
#  개념 레이어 빌더 (Gemini)
# ============================================================================

class ConceptLayerBuilder:

    def __init__(self, pre: Preprocessor, collector: TripleCollector, config: Config):
        self.pre        = pre
        self.c          = collector
        self.cfg        = config
        self.normalizer = ConceptNormalizer()
        self.client     = genai.Client(api_key=config.google_api_key)

    def build(self):
        content = self._build_full_content()
        if not content.strip():
            logger.warning("전체 콘텐츠가 비어있음")
            return

        # ── 0. 도메인 감지 ───────────────────────────────────────────────────
        # 토큰 절약: 앞 3000자만 사용
        domain_result = self._call_gemini(DOMAIN_PROMPT.format(content=content[:3000]))
        domain    = domain_result.get('domain', 'engineering') if domain_result else 'engineering'
        subdomain = domain_result.get('subdomain', '')         if domain_result else ''
        logger.info(f"  도메인: {domain} / {subdomain}")

        # 도메인 노드 저장
        self.c.add('lecture_video', 'HAS_DOMAIN', f'domain/{domain}')
        self.c.add(f'domain/{domain}', 'type', 'Domain', {'name': domain, 'subdomain': subdomain})

        # ── 1. 엔티티 추출 ───────────────────────────────────────────────────
        logger.info("엔티티 레이어 추출 중...")
        result = self._call_gemini(EXTRACTION_PROMPT.format(
            domain=domain, subdomain=subdomain, content=content
        ))
        if not result:
            return

        entities_raw: List[Dict]          = result.get('entities', [])
        relations_raw: List[Dict]         = result.get('relations', [])
        mentions_raw: Dict[str, List[str]] = result.get('mentions', {})

        logger.info(f"  Gemini 반환: entities={len(entities_raw)}, "
                    f"relations={len(relations_raw)}, "
                    f"mentions_entries={len(mentions_raw)}")

        # ── 2. Entity 노드 ───────────────────────────────────────────────────
        registered: Dict[str, str] = {}  # raw_name → slug
        for entry in entities_raw:
            name      = entry.get('name', '') if isinstance(entry, dict) else entry
            ent_type  = entry.get('type', 'concept') if isinstance(entry, dict) else 'concept'
            if ent_type not in ENTITY_TYPES:
                ent_type = 'concept'
            norm  = self.normalizer.normalize(name)
            slug  = ConceptNormalizer.to_slug(norm)
            if not slug:
                continue
            registered[name] = slug
            display = ConceptNormalizer.to_display_name(norm)
            self.c.add(f'concept/{slug}', 'type', 'Concept', {
                'name':        display,
                'entity_type': ent_type,
            })

        slug_set = set(registered.values())

        # ── 3. MENTIONS / APPEARS_IN 트리플 ─────────────────────────────────
        for entity_name, source_ids in mentions_raw.items():
            norm = self.normalizer.normalize(entity_name)
            slug = ConceptNormalizer.to_slug(norm)
            if slug not in slug_set:
                continue
            for source_id in source_ids:
                if source_id.startswith('segment/'):
                    seg_data = self.pre.segment_data.get(source_id)
                    if not seg_data:
                        continue
                    self.c.add(source_id, 'MENTIONS', f'concept/{slug}', {
                        'stressed': seg_data.get('stressed', False),
                        'start':    seg_data['start'],
                        'end':      seg_data['end'],
                    })
                elif source_id.startswith('slide_'):
                    if source_id in self.pre.slide_map:
                        self.c.add(source_id, 'APPEARS_IN', f'concept/{slug}')

        # ── 4. entity↔entity 관계 ────────────────────────────────────────────
        for rel in relations_raw:
            from_raw = rel.get('from', '')
            to_raw   = rel.get('to', '')
            rel_type = rel.get('type', '')
            if rel_type not in RELATION_TYPES:
                continue
            from_slug = ConceptNormalizer.to_slug(self.normalizer.normalize(from_raw))
            to_slug   = ConceptNormalizer.to_slug(self.normalizer.normalize(to_raw))
            if not from_slug or not to_slug or from_slug == to_slug:
                continue
            if from_slug not in slug_set or to_slug not in slug_set:
                continue
            self.c.add(f'concept/{from_slug}', rel_type, f'concept/{to_slug}')

        logger.info(f"✓ 엔티티 레이어 완료: entity {len(slug_set)}개")

    def _build_full_content(self) -> str:
        """
        fused.json → [source_id] 태그 포함 전체 텍스트 조합
        슬라이드 제목([slide_XXX]) + 세그먼트([segment/NNNN])
        """
        lines = []
        seg_idx = 0

        for slide in self.pre.slides:
            sid   = slide['slide_id']
            title = slide.get('title', '')
            lines.append(f'\n--- {sid}: {title} ---')
            lines.append(f'[{sid}] {title}')

            # 강조 키워드도 슬라이드 소스로 추가
            for kw_entry in slide.get('emphasized_keywords', []):
                kw = kw_entry.get('keyword', '')
                if kw:
                    lines.append(f'[{sid}] {kw}')

            # 세그먼트 (오디오 전사)
            for ctx in slide.get('contexts', []):
                for seg in ctx.get('segments', []):
                    seg_id = f'segment/{seg_idx:04d}'
                    lines.append(f'[{seg_id}] {seg["text"]}')
                    seg_idx += 1

        return '\n'.join(lines)

    def _call_gemini(self, prompt: str) -> Optional[Dict]:
        try:
            response = self.client.models.generate_content(model=self.cfg.gemini_model, contents=prompt)
            text = response.text
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"  ✗ Gemini 호출 실패: {e}")
            return None


# ============================================================================
#  파이프라인
# ============================================================================

class GraphPipeline:

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def run(self):
        start = time.time()
        cfg = self.config

        print('\n' + '='*70)
        print('📐 그래프 트리플 생성 파이프라인')
        print('='*70)
        print(f'  fused      : {cfg.fused_path}')
        print(f'  output     : {cfg.output_csv}')

        # ── 데이터 로드 ──────────────────────────────────────────────────────
        print('\n[Step 1] 데이터 로드')
        with open(cfg.fused_path, encoding='utf-8') as f:
            fused = json.load(f)
        slides = fused['slides']
        logger.info(f"✓ slide {len(slides)}개")

        # ── 전처리 ───────────────────────────────────────────────────────────
        print('\n[Step 2] 전처리')
        pre = Preprocessor(slides)
        collector = TripleCollector()

        # ── 구조 레이어 ───────────────────────────────────────────────────────
        print('\n[Step 3] 구조 레이어 생성')
        struct_builder = StructureLayerBuilder(pre, collector, cfg)
        struct_builder.build()
        struct_count = len(collector.triples)
        logger.info(f"✓ 구조 트리플: {struct_count}개")

        # ── 개념 레이어 ───────────────────────────────────────────────────────
        print('\n[Step 4] 개념 레이어 생성 (Gemini)')
        concept_builder = ConceptLayerBuilder(pre, collector, cfg)
        concept_builder.build()
        concept_count = len(collector.triples) - struct_count
        logger.info(f"✓ 개념 트리플: {concept_count}개")

        # ── CSV 출력 ──────────────────────────────────────────────────────────
        print('\n[Step 5] CSV 출력')
        collector.write_csv(cfg.output_csv)

        elapsed = time.time() - start
        print('\n' + '='*70)
        print('✅ 완료')
        print('='*70)
        print(f'  구조 트리플 : {struct_count}개')
        print(f'  개념 트리플 : {concept_count}개')
        print(f'  전체 트리플 : {len(collector.triples)}개')
        print(f'  처리 시간   : {elapsed:.2f}초')
        print(f'  출력 파일   : {cfg.output_csv}')


# ============================================================================
#  메인
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='멀티모달 강의 그래프 트리플 생성')
    parser.add_argument('--stem',       required=True,              help='강의 파일 stem (예: os1-1)')
    parser.add_argument('--output_dir', default='output',           help='출력 디렉토리 (기본: output)')
    parser.add_argument('--slides_dir', default='output_slides',    help='슬라이드 디렉토리')
    parser.add_argument('--title',      default='강의',              help='강의 제목')
    parser.add_argument('--model',      default='models/gemini-2.5-flash')
    args = parser.parse_args()

    cfg = Config(
        stem          = args.stem,
        output_dir    = Path(args.output_dir),
        slides_dir    = Path(args.slides_dir),
        lecture_title = args.title,
        gemini_model  = args.model,
    )

    if not cfg.fused_path.exists():
        print(f'❌ fused 파일 없음: {cfg.fused_path}')
        return
    if not cfg.google_api_key:
        print('❌ GOOGLE_API_KEY 환경변수가 설정되지 않았습니다')
        return

    GraphPipeline(cfg).run()


if __name__ == '__main__':
    main()