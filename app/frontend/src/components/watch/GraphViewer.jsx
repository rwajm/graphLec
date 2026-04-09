import React, { useEffect, useRef, useState } from 'react';
import { Network } from 'vis-network';
import { getLectureGraph } from '../../lib/api';

function GraphViewer({ lectureId }) {
  const containerRef = useRef(null);
  const networkRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [graphData, setGraphData] = useState(null);
  
  // 툴팁 상태 관리: 마우스 좌표를 직접 사용
  const [hoverInfo, setHoverInfo] = useState(null);

  useEffect(() => {
    if (!lectureId) return;
    setLoading(true);
    setError(null);

    getLectureGraph(lectureId)
      .then((data) => {
        if (!data || !data.graph) {
          setError('그래프 데이터가 없습니다.');
          return;
        }
        setGraphData(data);
      })
      .catch((err) => {
        console.error('Graph fetch error:', err);
        setError('그래프 데이터를 불러오는 중 오류가 발생했습니다.');
      })
      .finally(() => {
        setLoading(false);
      });
  }, [lectureId]);

  useEffect(() => {
    if (!containerRef.current || !graphData || !graphData.graph) return;

    const { nodes: rawNodes, edges: rawEdges } = graphData.graph;
    const nCount = rawNodes.length;

    const nodes = rawNodes.map((n) => ({
      id: n.id != null ? String(n.id) : 'n',
      label: (n.label || String(n.id)).slice(0, 48),
      shape: 'dot',
      size: nCount > 400 ? 12 : 16,
      color: {
        background: n.color || '#6b7280',
        border: '#374151',
        highlight: { background: n.color || '#6b7280', border: '#111' },
      },
      font: { color: '#111827', size: nCount > 400 ? 11 : 13 },
    }));

    const edges = rawEdges.map((e, i) => ({
      id: 'e' + i,
      from: String(e.from != null ? e.from : e.src_id),
      to: String(e.to != null ? e.to : e.tgt_id),
      label: e.label || e.rel_type || '',
    })).filter(e => e.from && e.to);

    const data = { nodes, edges };
    const heavy = nCount > 150;

    const options = {
      nodes: { borderWidth: 2 },
      edges: {
        arrows: 'to',
        smooth: heavy ? false : { type: 'dynamic' },
        font: { size: 10, align: 'middle' },
      },
      physics: {
        enabled: true,
        stabilization: { enabled: false },
        barnesHut: {
          gravitationalConstant: heavy ? -4000 : -2000,
          springLength: heavy ? 150 : 100,
          springConstant: 0.04,
          damping: 0.09,
        },
      },
      interaction: { 
        hover: true,
        tooltipDelay: 0,
        hideEdgesOnDrag: heavy,
      },
    };

    if (networkRef.current) {
      networkRef.current.destroy();
    }

    const network = new Network(containerRef.current, data, options);
    networkRef.current = network;

    // [개선] 마우스 이동 이벤트로 툴팁 위치를 실시간 갱신
    const handleMouseMove = (params) => {
      if (hoverInfo) {
        setHoverInfo(prev => prev ? { 
          ...prev, 
          x: params.pointer.DOM.x, 
          y: params.pointer.DOM.y 
        } : null);
      }
    };

    network.on('hoverNode', (params) => {
      const nodeId = params.node;
      const nodeData = rawNodes.find(n => String(n.id) === String(nodeId));
      if (nodeData) {
        setHoverInfo({
          id: nodeId,
          name: nodeData.label,
          type: nodeData.type,
          x: params.pointer.DOM.x,
          y: params.pointer.DOM.y
        });
      }
    });

    network.on('blurNode', () => {
      setHoverInfo(null);
    });
    
    network.on('mousemove', handleMouseMove);

    network.once('afterDrawing', () => {
      network.moveTo({ scale: 0.15, position: { x: 0, y: 0 }, animation: false });
      setTimeout(() => { network?.fit({ animation: { duration: 1000 } }); }, 500);
    });

    let fitInterval = setInterval(() => { network?.fit(); }, 500);
    setTimeout(() => clearInterval(fitInterval), 3500);

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy();
        networkRef.current = null;
      }
    };
  }, [graphData]);

  return (
    <div className="gv-container" style={{ position: 'relative' }}>
      <div className="gv-header">
        <span className="gv-stats">
          {loading ? '데이터 로딩 중...' : `노드: ${graphData?.node_count || 0}개 · 엣지: ${graphData?.edge_count || 0}개`}
        </span>
      </div>
      
      <div className="gv-network-wrapper" style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        {error && (
          <div className="gv-status">
            <p className="gv-error">{error}</p>
          </div>
        )}
        
        <div 
          className="gv-network" 
          ref={containerRef} 
          style={{ width: '100%', height: '100%', visibility: loading || error ? 'hidden' : 'visible' }}
        />

        {/* [커스텀 툴팁] 마우스 커서를 따라다님 */}
        {hoverInfo && !loading && (
          <div style={{
            position: 'absolute',
            left: hoverInfo.x + 12,
            top: hoverInfo.y + 12,
            backgroundColor: 'rgba(255, 255, 255, 0.98)',
            border: '1px solid var(--blue)',
            padding: '6px 10px',
            borderRadius: '4px',
            boxShadow: '0 4px 15px rgba(0,0,0,0.2)',
            pointerEvents: 'none',
            zIndex: 1000,
            whiteSpace: 'nowrap'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span style={{ fontSize: '14px', fontWeight: 700, color: 'var(--t1)' }}>{hoverInfo.id}</span>
              {hoverInfo.name && hoverInfo.name !== hoverInfo.id && (
                <span style={{ fontSize: '13px', color: 'var(--t2)' }}>({hoverInfo.name})</span>
              )}
              {hoverInfo.type && (
                <span style={{ fontSize: '11px', color: 'var(--blue)', background: 'var(--blue-bg)', padding: '1px 4px', borderRadius: '3px' }}>
                  {hoverInfo.type}
                </span>
              )}
            </div>
          </div>
        )}
        
        {loading && (
          <div className="gv-status" style={{ position: 'absolute', inset: 0, background: 'var(--bg-soft)', zIndex: 10, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <p>전체 지식 그래프 데이터를 불러오는 중...</p>
          </div>
        )}
      </div>
    </div>
  );
}

export default GraphViewer;
