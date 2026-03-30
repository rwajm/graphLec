"""
recommender_web.py
──────────────────
강의 추천 시스템 웹 인터페이스 (FastAPI, port 8002)

실행:
  python recommender_web.py
  python recommender_web.py --metadata_dir metadata/
  python recommender_web.py --metadata_dir metadata/ --port 8002
"""

import argparse
import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# recommender.py가 같은 디렉토리에 있다고 가정
from recommender import Recommender

# ============================================================================
#  CLI 파싱 (uvicorn 실행 전에 미리)
# ============================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--metadata_dir", default="metadata/", help="메타데이터 디렉토리")
parser.add_argument("--port", type=int, default=8002, help="포트 번호")
args, _ = parser.parse_known_args()

# ============================================================================
#  Recommender 초기화
# ============================================================================

print(f"[초기화] 메타데이터 로드 중: {args.metadata_dir}")
_recommender = Recommender(args.metadata_dir)
print(f"[초기화] 완료")

# ============================================================================
#  FastAPI 앱
# ============================================================================

app = FastAPI(title="GraphLEC 강의 추천")


# ── 요청/응답 모델 ──────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str = ""
    history: list[str] = []   # video_id 목록 (선택)
    top_k: int = 5


class LectureResult(BaseModel):
    video_id: str
    title: str
    instructor: str
    domain: str
    score: float
    reason: str
    summary: str
    keywords: list[str]
    difficulty: str
    detail: dict


# ── 엔드포인트 ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/recommend")
async def recommend(req: RecommendRequest):
    """자연어 질의 기반 강의 추천"""
    try:
        # 이력 기록
        _recommender.history.records.clear()
        for vid in req.history:
            _recommender.record_view(vid)

        results = []

        if req.query.strip():
            # 자연어 질의 추천
            raw = _recommender.recommend_from_query(req.query.strip(), top_k=req.top_k)
            mode = "query"
        elif req.history:
            # 이력 기반 추천
            raw = _recommender.recommend_from_history(top_k=req.top_k)
            mode = "history"
        else:
            return JSONResponse({"error": "질의 또는 시청 이력을 입력하세요."}, status_code=400)

        for r in raw:
            results.append(LectureResult(
                video_id   = r.video_id,
                title      = r.title,
                instructor = r.instructor,
                domain     = r.domain,
                score      = round(r.score, 4),
                reason     = r.reason,
                summary    = r.summary[:200] + ("…" if len(r.summary) > 200 else ""),
                keywords   = _recommender.collection.get(r.video_id).top_keywords[:8],
                difficulty = _recommender.collection.get(r.video_id).difficulty_level,
                detail     = r.score_detail,
            ).model_dump())

        return {"mode": mode, "query": req.query, "results": results}

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/lectures")
async def list_lectures():
    """로드된 강의 목록 반환 (이력 입력용)"""
    lecs = [
        {
            "video_id": lec.video_id,
            "title":    lec.title,
            "domain":   lec.domain,
        }
        for lec in _recommender.collection.all()
    ]
    lecs.sort(key=lambda x: (x["domain"], x["video_id"]))
    return {"lectures": lecs, "total": len(lecs)}


# ============================================================================
#  HTML 페이지 (단일 파일 인라인)
# ============================================================================

HTML_PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GraphLEC · 강의 추천</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">
<style>
  /* ── 변수 ── */
  :root {
    --bg:       #0d0f14;
    --surface:  #13161e;
    --border:   #1f2433;
    --accent:   #6ee7b7;
    --accent2:  #38bdf8;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --tag-bg:   #1a2235;
    --radius:   14px;
    --score-h:  #6ee7b7;
    --score-m:  #fbbf24;
    --score-l:  #f87171;
  }

  /* ── 기본 ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    line-height: 1.6;
    min-height: 100vh;
  }

  /* ── 배경 그래프 패턴 ── */
  body::before {
    content: '';
    position: fixed; inset: 0; z-index: 0;
    background-image:
      linear-gradient(rgba(110,231,183,.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(110,231,183,.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
  }

  /* ── 레이아웃 ── */
  .page { position: relative; z-index: 1; max-width: 820px; margin: 0 auto; padding: 48px 24px 80px; }

  /* ── 헤더 ── */
  header { margin-bottom: 48px; }
  .logo {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    letter-spacing: -.02em;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
  }
  .logo span { font-style: italic; }
  .subtitle {
    margin-top: 8px;
    color: var(--muted);
    font-size: 14px;
    font-weight: 300;
    letter-spacing: .02em;
  }

  /* ── 검색 박스 ── */
  .search-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    margin-bottom: 20px;
    transition: border-color .2s;
  }
  .search-wrap:focus-within { border-color: var(--accent); }

  .input-row { display: flex; gap: 10px; align-items: flex-start; }

  textarea#query {
    flex: 1;
    background: transparent;
    border: none;
    outline: none;
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    resize: none;
    min-height: 52px;
    max-height: 160px;
    overflow-y: auto;
    padding: 4px 0;
    line-height: 1.6;
  }
  textarea#query::placeholder { color: var(--muted); }

  button#search-btn {
    flex-shrink: 0;
    background: var(--accent);
    color: #0d0f14;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    cursor: pointer;
    transition: opacity .15s, transform .1s;
    align-self: flex-end;
  }
  button#search-btn:hover { opacity: .85; }
  button#search-btn:active { transform: scale(.97); }
  button#search-btn:disabled { opacity: .4; cursor: default; }

  /* ── 이력 토글 ── */
  .history-toggle {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 14px;
    cursor: pointer;
    color: var(--muted);
    font-size: 13px;
    user-select: none;
  }
  .history-toggle:hover { color: var(--text); }
  .chevron { transition: transform .2s; display: inline-block; }
  .chevron.open { transform: rotate(90deg); }

  .history-panel {
    display: none;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
  }
  .history-panel.open { display: block; }

  .history-label { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
  .lecture-list {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    max-height: 160px;
    overflow-y: auto;
  }
  .lec-chip {
    background: var(--tag-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
    transition: background .15s, border-color .15s;
    color: var(--text);
  }
  .lec-chip:hover { background: #1f2a3f; }
  .lec-chip.selected { border-color: var(--accent); color: var(--accent); background: rgba(110,231,183,.08); }

  .history-selected {
    margin-top: 10px;
    font-size: 12px;
    color: var(--muted);
    min-height: 18px;
  }
  .history-selected span { color: var(--accent); }

  /* ── 상태 영역 ── */
  #status {
    min-height: 28px;
    padding: 4px 0;
    font-size: 13px;
    color: var(--muted);
  }
  #status.loading { color: var(--accent); }
  #status.error   { color: var(--score-l); }

  /* ── 결과 ── */
  #results { display: flex; flex-direction: column; gap: 14px; margin-top: 8px; }

  .result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 22px 24px;
    animation: slideUp .3s ease both;
    transition: border-color .2s;
  }
  .result-card:hover { border-color: #2d3650; }

  @keyframes slideUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .card-top {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    margin-bottom: 12px;
  }

  .rank-badge {
    flex-shrink: 0;
    width: 28px; height: 28px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Serif Display', serif;
    font-size: 14px;
    color: var(--bg);
  }
  .rank-1 { background: var(--accent); }
  .rank-2 { background: var(--accent2); }
  .rank-n { background: var(--muted); }

  .card-meta { flex: 1; min-width: 0; }

  .card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: var(--text);
    line-height: 1.3;
    margin-bottom: 4px;
  }

  .card-sub {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    align-items: center;
    font-size: 12px;
    color: var(--muted);
  }
  .dot { color: #2d3650; }

  .score-pill {
    margin-left: auto;
    flex-shrink: 0;
    font-size: 13px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid currentColor;
  }

  .card-reason {
    font-size: 13px;
    color: var(--accent);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .card-reason::before {
    content: '';
    display: inline-block;
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--accent);
    flex-shrink: 0;
  }

  .card-summary {
    font-size: 13.5px;
    color: #94a3b8;
    line-height: 1.65;
    margin-bottom: 14px;
  }

  .kw-row {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .kw-tag {
    background: var(--tag-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 11.5px;
    color: var(--muted);
  }

  /* ── 신호 디테일 (토글) ── */
  .detail-toggle {
    margin-top: 12px;
    font-size: 12px;
    color: var(--muted);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .detail-toggle:hover { color: var(--text); }

  .detail-body {
    display: none;
    margin-top: 10px;
    padding: 12px;
    background: #0d0f14;
    border-radius: 8px;
    font-size: 12px;
    color: var(--muted);
    font-family: 'DM Mono', 'Courier New', monospace;
    line-height: 1.7;
  }
  .detail-body.open { display: block; }

  /* ── 빈 상태 ── */
  .empty {
    text-align: center;
    padding: 60px 24px;
    color: var(--muted);
    font-size: 14px;
  }
  .empty-icon { font-size: 40px; margin-bottom: 12px; }

  /* ── 스크롤바 ── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
</style>
</head>
<body>
<div class="page">

  <!-- 헤더 -->
  <header>
    <div class="logo">Graph<span>LEC</span></div>
    <div class="subtitle">멀티모달 강의 지식 그래프 · 자연어 추천 시스템</div>
  </header>

  <!-- 검색 -->
  <div class="search-wrap">
    <div class="input-row">
      <textarea id="query" rows="2"
        placeholder="배우고 싶은 내용을 자유롭게 입력하세요.  예) 프로세스 스케줄링이 궁금해, 신경망 학습 원리 알고 싶어"></textarea>
      <button id="search-btn" onclick="search()">추천받기</button>
    </div>

    <!-- 시청 이력 토글 -->
    <div class="history-toggle" onclick="toggleHistory()">
      <span class="chevron" id="chevron">▶</span>
      <span>시청 이력 기반 추천 추가하기</span>
    </div>
    <div class="history-panel" id="history-panel">
      <div class="history-label">시청한 강의를 선택하세요 (선택 순서대로 이력 기록)</div>
      <div class="lecture-list" id="lec-list">
        <span style="color:var(--muted);font-size:12px">로딩 중...</span>
      </div>
      <div class="history-selected" id="selected-label">선택된 강의: 없음</div>
    </div>
  </div>

  <div id="status"></div>
  <div id="results"></div>

</div>

<script>
  let allLectures = [];
  let selectedHistory = [];

  // 강의 목록 로드
  fetch('/api/lectures')
    .then(r => r.json())
    .then(data => {
      allLectures = data.lectures;
      renderLecList();
    });

  function renderLecList() {
    const container = document.getElementById('lec-list');
    // 도메인별 그룹 생략, 단순 칩 나열
    container.innerHTML = allLectures.map(lec => `
      <div class="lec-chip" data-id="${lec.video_id}" onclick="toggleLec(this, '${lec.video_id}')">
        ${lec.video_id} · ${lec.title}
      </div>
    `).join('');
  }

  function toggleLec(el, id) {
    const idx = selectedHistory.indexOf(id);
    if (idx === -1) {
      selectedHistory.push(id);
      el.classList.add('selected');
    } else {
      selectedHistory.splice(idx, 1);
      el.classList.remove('selected');
    }
    updateSelectedLabel();
  }

  function updateSelectedLabel() {
    const label = document.getElementById('selected-label');
    if (selectedHistory.length === 0) {
      label.innerHTML = '선택된 강의: 없음';
    } else {
      const names = selectedHistory.map(id => {
        const lec = allLectures.find(l => l.video_id === id);
        return `<span>${lec ? lec.title : id}</span>`;
      });
      label.innerHTML = `선택된 강의 (${selectedHistory.length}개): ` + names.join(' → ');
    }
  }

  function toggleHistory() {
    const panel = document.getElementById('history-panel');
    const chevron = document.getElementById('chevron');
    panel.classList.toggle('open');
    chevron.classList.toggle('open');
  }

  // Enter 키 → 검색
  document.getElementById('query').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); search(); }
  });

  async function search() {
    const query = document.getElementById('query').value.trim();
    const btn = document.getElementById('search-btn');
    const status = document.getElementById('status');
    const results = document.getElementById('results');

    if (!query && selectedHistory.length === 0) {
      status.className = 'error';
      status.textContent = '질의를 입력하거나 시청 이력을 선택하세요.';
      return;
    }

    btn.disabled = true;
    status.className = 'loading';
    status.textContent = query
      ? `"${query}" 분석 중…`
      : '시청 이력 기반 추천 계산 중…';
    results.innerHTML = '';

    try {
      const res = await fetch('/api/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history: selectedHistory, top_k: 5 })
      });
      const data = await res.json();

      if (data.error) {
        status.className = 'error';
        status.textContent = data.error;
        return;
      }

      status.className = '';
      if (data.mode === 'query') {
        const kws = data.results[0]?.detail?.extracted_keywords;
        status.textContent = kws?.length
          ? `추출 키워드: ${kws.join(', ')}`
          : `${data.results.length}개 강의 추천됨`;
      } else {
        status.textContent = `이력 기반 추천 ${data.results.length}개`;
      }

      if (data.results.length === 0) {
        results.innerHTML = `
          <div class="empty">
            <div class="empty-icon">🔍</div>
            질의와 일치하는 강의를 찾지 못했습니다.
          </div>`;
        return;
      }

      results.innerHTML = data.results.map((r, i) => renderCard(r, i)).join('');

    } catch (e) {
      status.className = 'error';
      status.textContent = '서버 오류: ' + e.message;
    } finally {
      btn.disabled = false;
    }
  }

  function scoreColor(score) {
    if (score >= 0.25) return 'var(--score-h)';
    if (score >= 0.10) return 'var(--score-m)';
    return 'var(--score-l)';
  }

  function renderCard(r, i) {
    const rankClass = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : 'rank-n';
    const color = scoreColor(r.score);
    const kws = (r.keywords || []).map(k => `<span class="kw-tag">${k}</span>`).join('');

    const detail = r.detail || {};
    const detailLines = Object.entries(detail)
      .filter(([k]) => !['extracted_keywords'].includes(k))
      .map(([k, v]) => {
        const val = typeof v === 'number' ? v.toFixed(4) : JSON.stringify(v);
        return `${k.padEnd(20)} ${val}`;
      }).join('\\n');

    const detailId = 'detail-' + i;

    return `
      <div class="result-card" style="animation-delay:${i * 60}ms">
        <div class="card-top">
          <div class="rank-badge ${rankClass}">${i + 1}</div>
          <div class="card-meta">
            <div class="card-title">${r.title}</div>
            <div class="card-sub">
              <span>${r.instructor}</span>
              <span class="dot">·</span>
              <span>${r.domain}</span>
              <span class="dot">·</span>
              <span>${r.video_id}</span>
              ${r.difficulty !== 'unclassified' ? `<span class="dot">·</span><span>${r.difficulty}</span>` : ''}
            </div>
          </div>
          <div class="score-pill" style="color:${color};border-color:${color}">
            ${Math.min(r.score * 100, 100).toFixed(1)}%
          </div>
        </div>

        <div class="card-reason">${r.reason}</div>
        <div class="card-summary">${r.summary}</div>
        <div class="kw-row">${kws}</div>

        ${detailLines ? `
          <div class="detail-toggle" onclick="toggleDetail('${detailId}')">
            ▶ 점수 상세 보기
          </div>
          <pre class="detail-body" id="${detailId}">${detailLines}</pre>
        ` : ''}
      </div>
    `;
  }

  function toggleDetail(id) {
    const el = document.getElementById(id);
    el.classList.toggle('open');
    const toggle = el.previousElementSibling;
    if (el.classList.contains('open')) {
      toggle.textContent = '▼ 점수 상세 닫기';
    } else {
      toggle.textContent = '▶ 점수 상세 보기';
    }
  }
</script>
</body>
</html>
"""

# ============================================================================
#  실행
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")