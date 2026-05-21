import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { IconBack, IconExt, IconSearch } from '../components/icons.jsx';
import MappingEditor from '../components/MappingEditor.jsx';

const CONFIDENCES = ['HIGH', 'MED', 'LOW'];
const STATUSES = ['pending', 'accepted', 'rejected'];

export default function Review() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [confFilter, setConfFilter] = useState(new Set(['HIGH', 'MED']));
  const [statusFilter, setStatusFilter] = useState(new Set(['pending']));
  const [search, setSearch] = useState('');
  const [selectedTicker, setSelectedTicker] = useState(null);
  const [busy, setBusy] = useState(false);
  const [toast, setToast] = useState('');
  const [showRecent, setShowRecent] = useState(false);
  const [recentAccepts, setRecentAccepts] = useState([]);

  // Multi-select state
  const [selectedSet, setSelectedSet] = useState(new Set());

  // Mapping edit state — cached per ticker
  const [editedMappings, setEditedMappings] = useState({});

  const showToast = (msg) => {
    setToast(msg);
    setTimeout(() => setToast(''), 1800);
  };

  const refresh = async () => {
    try {
      const r = await fetch('/api/data');
      const d = await r.json();
      setData(d);
    } catch {}
    setLoading(false);
  };

  const refreshRecent = async () => {
    try {
      const r = await fetch('/api/recent-accepts');
      const d = await r.json();
      setRecentAccepts(d.items || []);
    } catch {}
  };

  useEffect(() => {
    refresh();
    refreshRecent();
    const iv = setInterval(refreshRecent, 8000);
    return () => clearInterval(iv);
  }, []);

  const matches = data?.matches || [];
  const review = data?.review || { accepted: {}, rejected: {} };
  const exported = new Set(data?.exported || []);
  const existing = new Set(data?.existing || []);

  const statusOf = (t) => {
    if (review.accepted?.[t]) return 'accepted';
    if (review.rejected?.[t]) return 'rejected';
    return 'pending';
  };

  const toggleConf = (c) => {
    setConfFilter((prev) => {
      const next = new Set(prev);
      if (next.has(c)) next.delete(c); else next.add(c);
      return next;
    });
  };
  const toggleStatus = (s) => {
    setStatusFilter((prev) => {
      const next = new Set(prev);
      if (next.has(s)) next.delete(s); else next.add(s);
      return next;
    });
  };

  const filtered = useMemo(() => {
    const now = new Date();
    return matches
      .filter((m) => {
        if (existing.has(m.k_event_ticker)) return false;
        if (!confFilter.has(m.confidence)) return false;
        if (!statusFilter.has(statusOf(m.k_event_ticker))) return false;
        const end = m.p_end_date ? new Date(m.p_end_date) : m.k_end_date ? new Date(m.k_end_date) : null;
        if (end && end < now) return false;
        const cat = (m.k_category || '').toLowerCase();
        const t = m.k_title.toLowerCase();
        if (cat.includes('weather') || cat.includes('climate') || t.includes('temperature') || t.includes('weather')) return false;
        if (search) {
          const s = search.toLowerCase();
          if (!m.k_title.toLowerCase().includes(s) && !m.p_title.toLowerCase().includes(s)) return false;
        }
        return true;
      })
      .sort((a, b) => {
        const so = { pending: 0, accepted: 1, rejected: 2 };
        const co = { HIGH: 0, MED: 1, LOW: 2 };
        const ds = (so[statusOf(a.k_event_ticker)] || 0) - (so[statusOf(b.k_event_ticker)] || 0);
        if (ds) return ds;
        const dc = (co[a.confidence] || 0) - (co[b.confidence] || 0);
        if (dc) return dc;
        return b.event_score - a.event_score;
      });
  }, [matches, confFilter, statusFilter, search, review, existing]);

  // Auto-select first
  useEffect(() => {
    if (selectedTicker && !filtered.some((m) => m.k_event_ticker === selectedTicker)) {
      setSelectedTicker(filtered[0]?.k_event_ticker || null);
    }
    if (!selectedTicker && filtered[0]) {
      setSelectedTicker(filtered[0].k_event_ticker);
    }
  }, [filtered, selectedTicker]);

  const selected = filtered.find((m) => m.k_event_ticker === selectedTicker) || null;

  // Build current mapping for selected (edited > saved auto)
  const currentMapping = useMemo(() => {
    if (!selected) return {};
    const t = selected.k_event_ticker;
    if (editedMappings[t]) return editedMappings[t];
    const saved = review.accepted?.[t]?.mapping;
    if (saved && Object.keys(saved).length) return { ...saved };
    const auto = {};
    (selected.outcome_matches || []).forEach((o) => { auto[o.k_suffix] = o.p_outcome; });
    return auto;
  }, [selected, editedMappings, review]);

  const updateMapping = (newMap) => {
    if (!selected) return;
    setEditedMappings((prev) => ({ ...prev, [selected.k_event_ticker]: newMap }));
  };

  const toggleSelect = (ticker, e) => {
    e?.stopPropagation();
    setSelectedSet((prev) => {
      const next = new Set(prev);
      if (next.has(ticker)) next.delete(ticker); else next.add(ticker);
      return next;
    });
  };
  const selectAll = () => setSelectedSet(new Set(filtered.map((m) => m.k_event_ticker)));
  const selectNone = () => setSelectedSet(new Set());

  // ── Single actions ──
  const doAction = async (action, m, customMapping = null) => {
    if (!m) return;
    setBusy(true);
    const t = m.k_event_ticker;
    const body = {
      ticker: t,
      k_title: m.k_title,
      p_slug: m.p_slug,
      p_title: m.p_title,
      score: m.event_score,
    };
    try {
      if (action === 'accept') {
        body.mapping = customMapping || currentMapping;
        await fetch('/api/accept', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        await fetch('/api/export', { method: 'POST' });
        // Clear edited cache for this ticker
        setEditedMappings((prev) => { const n = { ...prev }; delete n[t]; return n; });
        showToast('Accepted & exported');
      } else if (action === 'reject') {
        await fetch('/api/reject', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        showToast('Rejected');
      } else if (action === 'reset') {
        await fetch('/api/reset', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ ticker: t }),
        });
        showToast('Reset');
      }
      await refresh();
    } catch (e) {
      showToast('Error: ' + e.message);
    }
    setBusy(false);
  };

  // ── Bulk actions ──
  const bulkAccept = async () => {
    if (!selectedSet.size) { showToast('No matches selected'); return; }
    setBusy(true);
    showToast(`Accepting ${selectedSet.size}...`);
    let count = 0;
    for (const t of selectedSet) {
      const m = matches.find((x) => x.k_event_ticker === t);
      if (!m) continue;
      const auto = {};
      (m.outcome_matches || []).forEach((o) => { auto[o.k_suffix] = o.p_outcome; });
      const mapping = editedMappings[t] || review.accepted?.[t]?.mapping || auto;
      await fetch('/api/accept', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: t, k_title: m.k_title, p_slug: m.p_slug,
          p_title: m.p_title, score: m.event_score, mapping,
        }),
      });
      count++;
    }
    await fetch('/api/export', { method: 'POST' });
    await refresh();
    setSelectedSet(new Set());
    showToast(`Accepted ${count} matches`);
    setBusy(false);
  };

  const bulkReject = async () => {
    if (!selectedSet.size) { showToast('No matches selected'); return; }
    setBusy(true);
    showToast(`Rejecting ${selectedSet.size}...`);
    let count = 0;
    for (const t of selectedSet) {
      const m = matches.find((x) => x.k_event_ticker === t);
      if (!m) continue;
      await fetch('/api/reject', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker: t, k_title: m.k_title, p_slug: m.p_slug,
          p_title: m.p_title, score: m.event_score,
        }),
      });
      count++;
    }
    await refresh();
    setSelectedSet(new Set());
    showToast(`Rejected ${count} matches`);
    setBusy(false);
  };

  const acceptAllHigh = async () => {
    const list = filtered.filter((m) => m.confidence === 'HIGH' && statusOf(m.k_event_ticker) === 'pending');
    if (!list.length) { showToast('No HIGH pending'); return; }
    setSelectedSet(new Set(list.map((m) => m.k_event_ticker)));
    setTimeout(bulkAccept, 100);
  };

  // Counts
  const counts = { pending: 0, accepted: 0, rejected: 0 };
  matches.forEach((m) => {
    if (existing.has(m.k_event_ticker)) return;
    counts[statusOf(m.k_event_ticker)]++;
  });

  return (
    <div className="app review-app">
      <header className="statusbar">
        <div className="brand">
          <div>
            <div className="brand-mark">review</div>
            <div className="brand-sub">match approval</div>
          </div>
        </div>
        <div className="statusbar-left">
          <div className="status-meta">
            <span><span className="val mono">{counts.pending}</span> pending</span>
            <span className="sep">·</span>
            <span><span className="val mono">{counts.accepted}</span> accepted</span>
            <span className="sep">·</span>
            <span><span className="val mono">{counts.rejected}</span> rejected</span>
          </div>
        </div>
        <div className="status-right">
          <button className={`link-btn ${showRecent ? 'active' : ''}`} onClick={() => { setShowRecent((v) => !v); refreshRecent(); }}>
            Recent ({recentAccepts.length})
          </button>
          <button className="btn primary" onClick={acceptAllHigh} disabled={busy}>
            Accept all HIGH
          </button>
          <Link className="link-btn" to="/dashboard">
            <IconBack size={12} /> Dashboard
          </Link>
        </div>
      </header>

      <main className="main">
        <div className="filterbar" style={{ borderBottom: 'none', padding: 0, gap: 10, flexWrap: 'wrap' }}>
          <div style={{ position: 'relative' }}>
            <div style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}>
              <IconSearch size={13} />
            </div>
            <input
              className="search"
              style={{ paddingLeft: 30, width: 240 }}
              placeholder="Search matches…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>

          <div className="row" style={{ gap: 4 }}>
            {CONFIDENCES.map((c) => (
              <button key={c} className={`filter-btn ${confFilter.has(c) ? 'active' : ''}`} onClick={() => toggleConf(c)}>
                {c}
              </button>
            ))}
          </div>

          <div className="row" style={{ gap: 4 }}>
            {STATUSES.map((s) => (
              <button key={s} className={`filter-btn ${statusFilter.has(s) ? 'active' : ''}`} onClick={() => toggleStatus(s)}>
                {s}
              </button>
            ))}
          </div>

          <div className="flex-end mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
            {filtered.length} shown
          </div>
        </div>

        {/* Bulk action bar */}
        <div className={`bulk-bar ${selectedSet.size ? 'active' : ''}`}>
          {selectedSet.size > 0 ? (
            <>
              <span className="mono" style={{ fontSize: 12, color: 'var(--text)' }}>
                {selectedSet.size} selected
              </span>
              <button className="filter-btn" onClick={selectNone}>Clear</button>
              <button className="filter-btn" onClick={selectAll}>Select all ({filtered.length})</button>
              <span className="flex-end" />
              <button className="btn primary" onClick={bulkAccept} disabled={busy}>
                ✓ Accept {selectedSet.size}
              </button>
              <button className="btn danger" onClick={bulkReject} disabled={busy}>
                ✗ Reject {selectedSet.size}
              </button>
            </>
          ) : (
            <>
              <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                Tip: check boxes for bulk actions, click row to inspect &amp; edit mapping
              </span>
              <span className="flex-end" />
              <button className="filter-btn" onClick={selectAll}>Select all</button>
            </>
          )}
        </div>

        <div className="split" style={{ gridTemplateColumns: '1fr 2fr' }}>
          {/* List */}
          <div className="panel">
            <div className="panel-head">
              <div className="panel-title">
                <span className="serif">Match Queue</span>
              </div>
            </div>
            <div className="panel-body">
              {loading && <div className="loading"><span className="spinner" /> Loading…</div>}
              {!loading && filtered.length === 0 && (
                <div className="empty">
                  <div className="empty-title">No matches to review</div>
                  <div className="empty-sub">Adjust your filters or run discovery.</div>
                </div>
              )}
              <div className="rev-list">
                {filtered.map((m) => {
                  const t = m.k_event_ticker;
                  const st = statusOf(t);
                  const isSel = selected?.k_event_ticker === t;
                  const isChecked = selectedSet.has(t);
                  return (
                    <div key={t} className={`rev-row ${isSel ? 'selected' : ''} ${isChecked ? 'checked' : ''}`}
                         onClick={() => setSelectedTicker(t)}>
                      <div className="rev-row-head">
                        <input
                          type="checkbox"
                          className="rev-check"
                          checked={isChecked}
                          onChange={(e) => toggleSelect(t, e)}
                          onClick={(e) => e.stopPropagation()}
                        />
                        <span className={`badge conf-${m.confidence?.toLowerCase()}`}>{m.confidence}</span>
                        <span className="mono" style={{ fontSize: 10, color: 'var(--text-dim)' }}>
                          {(m.event_score * 100).toFixed(0)}%
                        </span>
                        {st === 'accepted' && <span className="badge arb">✓</span>}
                        {st === 'rejected' && <span className="badge side-no">✗</span>}
                        {exported.has(t) && <span className="badge mode">📦</span>}
                        {m.confusable && <span className="badge stub">⚠</span>}
                      </div>
                      <div className="rev-title">{m.k_title}</div>
                      <div className="rev-sub">{m.p_title}</div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Detail */}
          <div className="panel">
            <div className="panel-head">
              <div className="panel-title">
                <span className="serif">Match Detail</span>
              </div>
            </div>
            <div className="panel-body">
              {!selected ? (
                <div className="empty">
                  <div className="empty-title">Select a match</div>
                  <div className="empty-sub">Pick a row to inspect outcome mapping and approve.</div>
                </div>
              ) : (
                <RevDetail
                  match={selected}
                  status={statusOf(selected.k_event_ticker)}
                  exported={exported.has(selected.k_event_ticker)}
                  busy={busy}
                  mapping={currentMapping}
                  onMappingChange={updateMapping}
                  onAction={doAction}
                  edited={!!editedMappings[selected.k_event_ticker]}
                />
              )}
            </div>
          </div>
        </div>
      </main>

      {toast && <div className="rev-toast">{toast}</div>}

      {showRecent && (
        <>
          <div className="settings-overlay" onClick={() => setShowRecent(false)} />
          <aside className="settings-panel" style={{ width: 460 }}>
            <div className="settings-head">
              <h2>Recent Accepts</h2>
              <button className="close-btn" onClick={() => setShowRecent(false)}>×</button>
            </div>
            <div className="settings-body" style={{ padding: 0 }}>
              <div style={{ padding: '10px 16px', fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
                {recentAccepts.length} total accepted matches · auto-refresh every 8s
              </div>
              {recentAccepts.length === 0 ? (
                <div className="empty">
                  <div className="empty-title">No accepts yet</div>
                  <div className="empty-sub">Accepted matches show up here as soon as you approve them.</div>
                </div>
              ) : (
                <div className="recent-list">
                  {recentAccepts.map((r) => (
                    <div key={r.ticker} className="recent-item">
                      <div className="recent-time">{r.accepted_at ? r.accepted_at.slice(11, 19) : '—'}</div>
                      <div className="recent-body">
                        <div className="recent-title">{r.k_title}</div>
                        <div className="recent-meta">
                          <span className="mono" style={{ color: 'var(--blue)' }}>{r.ticker}</span>
                          <span style={{ color: 'var(--text-faint)' }}>·</span>
                          <span className="mono" style={{ color: 'var(--purple)' }}>{(r.p_slug || '').slice(0, 30)}</span>
                          <span style={{ color: 'var(--text-faint)' }}>·</span>
                          <span className="mono">{r.mapping_count} maps</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </aside>
        </>
      )}
    </div>
  );
}

function RevDetail({ match: m, status, exported, busy, mapping, onMappingChange, onAction, edited }) {
  const mappedCount = Object.keys(mapping).length;
  const totalK = (m.all_k_outcomes || []).length;

  return (
    <div className="detail">
      <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
        <span className={`badge conf-${m.confidence?.toLowerCase()}`}>{m.confidence}</span>
        {status === 'accepted' && <span className="badge arb">Accepted</span>}
        {status === 'rejected' && <span className="badge side-no">Rejected</span>}
        {exported && <span className="badge mode">Exported</span>}
        {m.confusable && <span className="badge stub">⚠ {m.confusable}</span>}
        {edited && <span className="badge test">Edited</span>}
        <span className="flex-end mono" style={{ fontSize: 18, color: 'var(--green)' }}>
          {(m.event_score * 100).toFixed(0)}%
        </span>
      </div>

      <div className="rev-platforms">
        <div className="rev-plat">
          <div className="rev-plat-lbl">Kalshi</div>
          <div className="rev-plat-title">{m.k_title}</div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            {m.k_event_ticker}<br />
            {m.k_market_count} markets · {m.k_category || 'uncategorized'}
          </div>
          <a href={`https://kalshi.com/events/${m.k_event_ticker}`} target="_blank" rel="noreferrer"
            className="btn" style={{ marginTop: 10, display: 'inline-flex' }}>
            <IconExt size={11} /> Open
          </a>
        </div>
        <div className="rev-plat">
          <div className="rev-plat-lbl">Polymarket</div>
          <div className="rev-plat-title">{m.p_title}</div>
          <div className="mono" style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: 6 }}>
            {(m.p_slug || '').slice(0, 50)}<br />
            {m.p_market_count} markets
          </div>
          <a href={`https://polymarket.com/event/${m.p_slug}`} target="_blank" rel="noreferrer"
            className="btn" style={{ marginTop: 10, display: 'inline-flex' }}>
            <IconExt size={11} /> Open
          </a>
        </div>
      </div>

      <div className="detail-section">
        <div className="detail-label">
          Outcome Mapping
          <span style={{ marginLeft: 'auto', color: 'var(--text-dim)', textTransform: 'none', letterSpacing: 0, fontSize: 10 }}>
            {mappedCount}/{totalK} mapped · drag to connect, click to disconnect
          </span>
        </div>
        <MappingEditor match={m} mapping={mapping} onChange={onMappingChange} />
      </div>

      <div className="detail-actions" style={{ marginTop: 14 }}>
        <button className="btn primary" onClick={() => onAction('accept', m)} disabled={busy}>
          ✓ Accept &amp; Export
        </button>
        <button className="btn danger" onClick={() => onAction('reject', m)} disabled={busy}>
          ✗ Reject
        </button>
        {status !== 'pending' && (
          <button className="btn" onClick={() => onAction('reset', m)} disabled={busy}>
            ↻ Reset
          </button>
        )}
      </div>
    </div>
  );
}
