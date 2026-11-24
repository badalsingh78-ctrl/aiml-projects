import React, { useEffect, useState } from 'react';

const API_BASE = import.meta.env.VITE_API_BASE || '/api';

const PRESETS = {
  default: { label: 'Default', contrast: 1.0, brightness: 0.0, invert: false },
  lung: { label: 'Lung Window', contrast: 1.35, brightness: -0.08, invert: false },
  soft: { label: 'Soft Tissue', contrast: 1.15, brightness: 0.08, invert: false },
  negative: { label: 'Invert / Negative', contrast: 1.1, brightness: 0.0, invert: true },
};

function statusPillClass(status) {
  if (status === 'APPROVED') return 'pill pill-emerald';
  if (status === 'REJECTED') return 'pill pill-rose';
  return 'pill pill-amber';
}

function severityPillClass(severity) {
  if (severity === 'Elevated') return 'pill pill-rose';
  if (severity === 'Moderate') return 'pill pill-amber';
  return 'pill pill-emerald';
}

export default function App() {
  const [studies, setStudies] = useState([]);
  const [selected, setSelected] = useState(null);
  const [auditLogs, setAuditLogs] = useState([]);
  const [filter, setFilter] = useState('ALL');
  const [uploadModality, setUploadModality] = useState('AUTO');

  // Viewer & Enhancement tools state
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.5);
  const [zoom, setZoom] = useState(1.0);
  const [contrast, setContrast] = useState(1.0);
  const [brightness, setBrightness] = useState(0.0);
  const [invert, setInvert] = useState(false);
  const [clahe, setClahe] = useState(false);
  const [sharpen, setSharpen] = useState(false);
  const [denoise, setDenoise] = useState(false);
  const [heatmapThreshold, setHeatmapThreshold] = useState(0.0);
  const [reanalyzing, setReanalyzing] = useState(false);
  const [deltaBanner, setDeltaBanner] = useState('');

  // Review state
  const [radiologistId, setRadiologistId] = useState('DR-REVIEWER-01');
  const [notes, setNotes] = useState('');
  const [statusMsg, setStatusMsg] = useState('');

  const loadData = async (preferredId = null) => {
    const [sRes, aRes] = await Promise.all([
      fetch(`${API_BASE}/studies`),
      fetch(`${API_BASE}/audit-log`),
    ]);
    const sJson = await sRes.json();
    const aJson = await aRes.json();
    const list = sJson.studies || [];
    setStudies(list);
    setAuditLogs(aJson.logs || []);

    const targetId = preferredId || selected?.study_id;
    const match = list.find((item) => item.study_id === targetId);
    setSelected(match || list[0] || null);
  };

  useEffect(() => {
    loadData();
  }, []);

  const resetTools = () => {
    setZoom(1.0);
    setContrast(1.0);
    setBrightness(0.0);
    setInvert(false);
    setClahe(false);
    setSharpen(false);
    setDenoise(false);
    setHeatmapThreshold(0.0);
    setDeltaBanner('');
  };

  const applyPreset = (key) => {
    const p = PRESETS[key];
    setContrast(p.contrast);
    setBrightness(p.brightness);
    setInvert(p.invert);
  };

  const handleUpload = async (evt) => {
    const file = evt.target.files?.[0];
    if (!file) return;
    const form = new FormData();
    form.append('file', file);
    form.append('modality', uploadModality);
    form.append('body_part', 'AUTO');
    const res = await fetch(`${API_BASE}/studies/upload`, { method: 'POST', body: form });
    if (res.ok) {
      const created = await res.json();
      resetTools();
      setStatusMsg(`Scanned ${file.name} (${created.modality}) -> ${created.top_finding}`);
      await loadData(created.study_id);
    }
    evt.target.value = '';
  };

  const handleReanalyze = async (overrideModality = null) => {
    if (!selected) return;
    setReanalyzing(true);
    try {
      const res = await fetch(`${API_BASE}/studies/${selected.study_id}/reanalyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          modality: overrideModality || selected.modality,
          clahe,
          sharpen,
          denoise,
          invert,
          contrast,
          brightness,
          heatmap_threshold: heatmapThreshold,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        const pctDelta = Math.round((data.confidence_delta || 0) * 100);
        const sign = pctDelta >= 0 ? '+' : '';
        setDeltaBanner(
          `Re-analyzed with ${data.enhancements_applied.length || 'default'} tool(s): ` +
          `${Math.round(data.previous_confidence * 100)}% → ${Math.round(data.study.confidence * 100)}% (${sign}${pctDelta}%)`
        );
        await loadData(selected.study_id);
      }
    } finally {
      setReanalyzing(false);
    }
  };

  const handleReview = async (decision) => {
    if (!selected) return;
    const res = await fetch(`${API_BASE}/studies/${selected.study_id}/review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ radiologist_id: radiologistId, decision, notes }),
    });
    if (res.ok) {
      setStatusMsg(`Recorded ${decision} for ${selected.study_id}`);
      setNotes('');
      await loadData(selected.study_id);
    }
  };

  const exportAuditCsv = () => {
    const header = 'id,study_id,radiologist_id,decision,ai_finding,ai_confidence,notes,decided_at\n';
    const rows = auditLogs
      .map(
        (l) =>
          `${l.id},"${l.study_id}","${l.radiologist_id}","${l.decision}","${l.ai_finding}",${l.ai_confidence},"${(l.notes || '').replace(/"/g, '""')}","${l.decided_at}"`
      )
      .join('\n');
    const blob = new Blob([header + rows], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'medical_imaging_audit_log.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const filteredStudies = studies.filter((st) => {
    if (filter === 'PENDING') return st.status === 'PENDING_REVIEW';
    if (filter === 'REVIEWED') return st.status !== 'PENDING_REVIEW';
    return true;
  });

  const cssFilter = `contrast(${Math.round(contrast * 100)}%) brightness(${Math.round((1 + brightness) * 100)}%) ${invert ? 'invert(1)' : ''}`;
  const report = selected?.report || {};

  return (
    <div>
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">AI</div>
          <h1 className="brand-title">AI-Assisted Medical Imaging Prototype</h1>
        </div>
        <div className="topbar-actions">
          <span className="pill pill-teal">{studies.length} Studies</span>
          <button className="btn-secondary" onClick={exportAuditCsv}>
            Export Audit CSV
          </button>
        </div>
      </header>

      <div className="workspace">
        {/* LEFT: Upload & Study Worklist */}
        <aside className="card">
          <div className="card-header">
            <h2 className="card-title">Study Worklist</h2>
            <span className="pill pill-slate">{filteredStudies.length}</span>
          </div>

          <div className="upload-box">
            <label style={{ fontSize: '0.74rem', color: '#94a3b8' }}>Scan Modality</label>
            <select
              className="select"
              value={uploadModality}
              onChange={(e) => setUploadModality(e.target.value)}
            >
              <option value="AUTO">Auto-Detect (X-Ray / CT / MRI)</option>
              <option value="CR">Chest X-Ray (CR)</option>
              <option value="CT">Chest CT (Axial)</option>
              <option value="MRI">Brain MRI (FLAIR)</option>
            </select>
            <label className="upload-btn">
              + Upload Scan Image (PNG / JPG)
              <input type="file" accept="image/*" onChange={handleUpload} style={{ display: 'none' }} />
            </label>
          </div>

          <div className="filter-row">
            {['ALL', 'PENDING', 'REVIEWED'].map((f) => (
              <button
                key={f}
                className={`filter-btn ${filter === f ? 'active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f === 'ALL' ? 'All' : f === 'PENDING' ? 'Pending' : 'Reviewed'}
              </button>
            ))}
          </div>

          <div className="study-list">
            {filteredStudies.map((st) => (
              <button
                key={st.study_id}
                className={`study-item ${selected?.study_id === st.study_id ? 'active' : ''}`}
                onClick={() => {
                  setSelected(st);
                  setDeltaBanner('');
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 6 }}>
                  <strong style={{ fontSize: '0.82rem' }}>{st.study_id}</strong>
                  <span className={statusPillClass(st.status)}>{st.status === 'PENDING_REVIEW' ? 'PENDING' : st.status}</span>
                </div>
                <div style={{ fontSize: '0.74rem', color: '#94a3b8', marginTop: 4 }}>
                  {st.modality} · {st.body_part}
                </div>
                <div style={{ fontSize: '0.76rem', color: '#5eead4', marginTop: 4, fontWeight: 600 }}>
                  {st.top_finding} ({Math.round((st.confidence || 0) * 100)}%)
                </div>
              </button>
            ))}
          </div>
        </aside>

        {/* CENTER: Viewer, Enhancement Tools & Diagnostic Report */}
        <main className="card">
          {selected ? (
            <>
              <div className="card-header">
                <div>
                  <h2 style={{ margin: 0, fontSize: '1.05rem' }}>
                    {selected.study_id} — {selected.body_part}
                  </h2>
                  <div style={{ fontSize: '0.76rem', color: '#94a3b8', marginTop: 2 }}>
                    Modality: {selected.modality} · Patient: {selected.patient_id}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <select
                    className="select"
                    style={{ width: 'auto', padding: '5px 8px', fontSize: '0.75rem' }}
                    value={selected.modality}
                    onChange={(e) => handleReanalyze(e.target.value)}
                    title="Switch scan modality & re-run AI"
                  >
                    <option value="CR">Mode: Chest X-Ray</option>
                    <option value="CT">Mode: Chest CT</option>
                    <option value="MRI">Mode: Brain MRI</option>
                  </select>
                  <span className={statusPillClass(selected.status)}>{selected.status}</span>
                </div>
              </div>

              <div className="viewer-frame">
                <div className="viewer-hud">
                  {selected.modality} · Zoom {zoom.toFixed(1)}x · Contrast {Math.round(contrast * 100)}%
                </div>
                <img
                  src={selected.scan_b64}
                  alt="Medical Scan"
                  style={{
                    filter: cssFilter,
                    transform: `scale(${zoom})`,
                  }}
                />
                {showOverlay && (
                  <img
                    src={selected.heatmap_b64}
                    alt="Grad-CAM Overlay"
                    style={{
                      opacity,
                      mixBlendMode: 'screen',
                      transform: `scale(${zoom})`,
                    }}
                  />
                )}
              </div>

              {/* Result Improvement & Image Enhancement Toolkit */}
              <div className="toolbar-grid">
                <div className="slider-group">
                  <div className="slider-label">
                    <label style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer', color: '#e2e8f0' }}>
                      <input
                        type="checkbox"
                        checked={showOverlay}
                        onChange={(e) => setShowOverlay(e.target.checked)}
                      />
                      Grad-CAM Overlay
                    </label>
                    <span>{Math.round(opacity * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={opacity}
                    onChange={(e) => setOpacity(Number(e.target.value))}
                  />
                </div>

                <div className="slider-group">
                  <div className="slider-label">
                    <span>Zoom Magnification</span>
                    <span>{zoom.toFixed(1)}x</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="3"
                    step="0.1"
                    value={zoom}
                    onChange={(e) => setZoom(Number(e.target.value))}
                  />
                </div>

                <div className="slider-group">
                  <div className="slider-label">
                    <span>Window Contrast</span>
                    <span>{Math.round(contrast * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.6"
                    max="2.0"
                    step="0.05"
                    value={contrast}
                    onChange={(e) => setContrast(Number(e.target.value))}
                  />
                </div>

                <div className="slider-group">
                  <div className="slider-label">
                    <span>Heatmap Focus Threshold</span>
                    <span>{Math.round(heatmapThreshold * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="0.7"
                    step="0.05"
                    value={heatmapThreshold}
                    onChange={(e) => setHeatmapThreshold(Number(e.target.value))}
                  />
                </div>

                <div style={{ gridColumn: '1 / -1', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div className="chip-row">
                    <button
                      type="button"
                      className={`tool-chip ${clahe ? 'active' : ''}`}
                      onClick={() => setClahe((v) => !v)}
                    >
                      Auto-Contrast (CLAHE)
                    </button>
                    <button
                      type="button"
                      className={`tool-chip ${sharpen ? 'active' : ''}`}
                      onClick={() => setSharpen((v) => !v)}
                    >
                      Edge Sharpen
                    </button>
                    <button
                      type="button"
                      className={`tool-chip ${denoise ? 'active' : ''}`}
                      onClick={() => setDenoise((v) => !v)}
                    >
                      Denoise Filter
                    </button>
                    <button
                      type="button"
                      className={`tool-chip ${invert ? 'active' : ''}`}
                      onClick={() => setInvert((v) => !v)}
                    >
                      Invert Grayscale
                    </button>
                    {Object.entries(PRESETS).map(([k, p]) => (
                      <button
                        key={k}
                        type="button"
                        className="tool-chip"
                        onClick={() => applyPreset(k)}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>

                  <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
                    <button
                      className="btn-primary"
                      disabled={reanalyzing}
                      onClick={() => handleReanalyze()}
                    >
                      {reanalyzing ? 'Re-Analyzing Scan...' : 'Re-Run AI with Enhanced Scan'}
                    </button>
                    <button className="btn-secondary" onClick={resetTools}>
                      Reset Tools
                    </button>
                  </div>
                  {deltaBanner && <div className="toast">{deltaBanner}</div>}
                </div>
              </div>

              {/* Plain-Language Diagnostic Report ("What This Scan Shows") */}
              <section className="report-box">
                <div className="card-header">
                  <span style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.07em', color: '#5eead4' }}>
                    What This Scan Shows — AI Diagnostic Summary
                  </span>
                  <span className={severityPillClass(report.severity || 'Moderate')}>
                    {report.severity || 'Moderate'} Concern
                  </span>
                </div>
                <h3 className="report-headline">
                  {report.headline || selected.top_finding} ({Math.round((selected.confidence || 0) * 100)}%)
                </h3>
                {report.affected_region && (
                  <div style={{ fontSize: '0.76rem', color: '#38bdf8', fontWeight: 600 }}>
                    Highlighted Region: {report.affected_region}
                  </div>
                )}
                <p className="report-text">
                  {report.summary ||
                    `Primary AI finding is ${selected.top_finding} with ${Math.round((selected.confidence || 0) * 100)}% confidence.`}
                </p>
                {report.recommendations?.length > 0 && (
                  <div>
                    <div style={{ fontSize: '0.74rem', fontWeight: 700, color: '#cbd5e1', marginBottom: 4 }}>
                      Suggested Next Steps
                    </div>
                    <ul className="rec-list">
                      {report.recommendations.map((rec, i) => (
                        <li key={i}>{rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>
            </>
          ) : (
            <p style={{ color: '#94a3b8' }}>Upload or select a scan from the worklist to begin analysis.</p>
          )}
        </main>

        {/* RIGHT: Confidence Breakdown, Radiologist Decision & Audit Log */}
        <aside className="card">
          <div className="card-header">
            <h2 className="card-title">AI Findings Breakdown</h2>
            <span className="pill pill-teal">{selected?.modality || '—'}</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {selected?.predictions?.map((p, idx) => (
              <div key={p.label}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                  <span style={{ fontWeight: idx === 0 ? 700 : 500, color: idx === 0 ? '#fff' : '#cbd5e1' }}>
                    {p.label}
                  </span>
                  <strong style={{ color: '#5eead4' }}>{Math.round(p.confidence * 100)}%</strong>
                </div>
                <div className="bar-track">
                  <div className="bar-fill" style={{ width: `${Math.round(p.confidence * 100)}%` }} />
                </div>
              </div>
            ))}
          </div>

          <div style={{ borderTop: '1px solid #1e293b', paddingTop: 14, display: 'flex', flexDirection: 'column', gap: 10 }}>
            <h2 className="card-title">Radiologist Decision</h2>
            <div>
              <label style={{ display: 'block', fontSize: '0.74rem', color: '#94a3b8', marginBottom: 4 }}>
                Radiologist ID
              </label>
              <input
                className="input"
                value={radiologistId}
                onChange={(e) => setRadiologistId(e.target.value)}
              />
            </div>
            <div>
              <label style={{ display: 'block', fontSize: '0.74rem', color: '#94a3b8', marginBottom: 4 }}>
                Clinical Review Notes
              </label>
              <textarea
                className="textarea"
                rows={3}
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Document agreement, false-positive region, or follow-up..."
              />
            </div>
            <div className="btn-row">
              <button className="btn-approve" onClick={() => handleReview('APPROVED')}>
                Approve
              </button>
              <button className="btn-reject" onClick={() => handleReview('REJECTED')}>
                Reject
              </button>
            </div>
            {statusMsg && <div className="toast">{statusMsg}</div>}
          </div>

          <div style={{ borderTop: '1px solid #1e293b', paddingTop: 14, display: 'flex', flexDirection: 'column', gap: 8 }}>
            <div className="card-header">
              <h2 className="card-title">Audit Log</h2>
              <span className="pill pill-slate">{auditLogs.length}</span>
            </div>
            <div className="audit-list">
              {auditLogs.length === 0 ? (
                <div style={{ fontSize: '0.78rem', color: '#64748b' }}>
                  No decisions logged yet. Click Approve or Reject above.
                </div>
              ) : (
                auditLogs.map((log) => (
                  <div key={log.id} className="audit-entry">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <strong>{log.study_id}</strong>
                      <span className={statusPillClass(log.decision)}>{log.decision}</span>
                    </div>
                    <div style={{ color: '#94a3b8', marginTop: 4 }}>
                      {log.radiologist_id} · {log.ai_finding} ({Math.round(log.ai_confidence * 100)}%)
                    </div>
                    {log.notes && (
                      <div style={{ color: '#cbd5e1', marginTop: 3 }}>"{log.notes}"</div>
                    )}
                  </div>
                ))
              )}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
