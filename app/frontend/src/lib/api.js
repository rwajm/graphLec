const API_BASE = '/api'

export async function uploadLecture({ file, title, category, description }) {
  const formData = new FormData();
  formData.append('video', file);
  // title, category, description 등 추가 정보도 Form으로 전송 가능
  formData.append('title', title);
  formData.append('category', category);
  formData.append('description', description);

  const res = await fetch(`${API_BASE}/jobs`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || 'Upload failed');
  }
  
  const data = await res.json();
  return { id: data.job_id, status: 'pending' };
}

export async function getLectureStatus(id) {
  const res = await fetch(`${API_BASE}/jobs/${id}`);
  if (!res.ok) throw new Error('Status check failed');
  
  const data = await res.json();
  
  // LecturesPage.jsx expectations: { lecture_status, stages: [{ stage, status }] }
  // Mapping current backend status to frontend stages
  const STAGE_KEYS = ['stt', 'voice', 'scene', 'integrate', 'summarize'];
  const stages = STAGE_KEYS.map((key) => ({
    stage: key,
    status: data.status === 'done' ? 'done' : (data.status === 'running' ? 'run' : 'wait')
  }));

  return {
    lecture_status: data.status,
    stages: stages,
    error_message: data.error_message
  };
}

export async function listLectures() {
  try {
    const res = await fetch(`${API_BASE}/jobs`);
    if (!res.ok) throw new Error('Failed to fetch lectures');
    
    const data = await res.json();
    return data.map(job => ({
      id: job.id,
      title: job.input_path.split(/[\\/]/).pop(), // 파일명 추출
      category: '컴퓨터 과학', // 기본값 (백엔드에서 아직 저장안함)
      status: job.status,
      created_at: job.created_at,
      error_message: job.error_message,
      pipeline_stages: [] // 초기 상태
    }));
  } catch (error) {
    console.error("listLectures error:", error);
    return [];
  }
}

export async function getLectureDetail(id) {
  const res = await fetch(`${API_BASE}/jobs/${id}`);
  if (!res.ok) throw new Error('Detail fetch failed');
  return res.json();
}

export async function getLectureTimeline(id) {
  const res = await fetch(`${API_BASE}/jobs/${id}/timeline`);
  if (!res.ok) {
    if (res.status === 404) return []; // No timeline yet
    throw new Error('Timeline fetch failed');
  }
  return res.json();
}

export async function deleteLecture(id) {
  const res = await fetch(`${API_BASE}/jobs/${id}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || 'Delete failed');
  }
  return res.json();
}

export async function retryLecture(id) {
  const res = await fetch(`${API_BASE}/jobs/${id}/retry`, {
    method: 'POST',
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(msg || 'Retry failed');
  }
  return res.json();
}

export async function getLectureGraph(id) {
  const res = await fetch(`${API_BASE}/jobs/${id}/graph`);
  if (!res.ok) {
    if (res.status === 404) return null; // No graph yet
    throw new Error('Graph fetch failed');
  }
  return res.json();
}

export async function askQa(lectureId, question) {
  const res = await fetch(`${API_BASE}/jobs/${lectureId}/qa`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || 'QA request failed');
  }
  return res.json();
}

export async function getQaHistory(lectureId) {
  return [];
}
