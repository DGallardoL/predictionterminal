import React, { useEffect, useMemo, useState } from 'react';
import { IconSearch, IconExt } from './icons.jsx';

const SOURCES = ['all', 'reviewed', 'main', 'discovered'];

export default function ConfigTab() {
  const [stats, setStats] = useState(null);
  const [events, setEvents] = useState([]);
  const [discovery, setDiscovery] = useState(null);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(true);
  const [discovering, setDiscovering] = useState(false);
  const [sourceFilter, setSourceFilter] = useState('all');
  const [sortBy, setSortBy] = useState('source'); // source | name | mappings
  const [sortDir, setSortDir] = useState('asc');

  const refresh = async () => {
    try {
      const [rs, re, rd] = await Promise.all([
        fetch('/api/dashboard/config-stats').then((r) => r.json()),
        fetch('/api/config-events').then((r) => r.json()),
        fetch('/api/discovery/status').then((r) => r.json()).catch(() => null),
      ]);
      setStats(rs);
      setEvents(re.events || []);
      setDiscovery(rd);
    } catch {}
    setLoading(false);
  };

  useEffect(() => {
    refresh();
  }, []);

  const runDiscovery = async () => {
    setDiscovering(true);
    try {
      await fetch('/api/discovery/run', { method: 'POST' });
      await new Promise((r) => setTimeout(r, 800));
      await refresh();
    } catch {}
    setDiscovering(false);
  };

  const sourceCounts = useMemo(() => {
    const c = { all: events.length };
    events.forEach((e) => {
      c[e.source] = (c[e.source] || 0) + 1;
    });
    return c;
  }, [events]);

  const filtered = useMemo(() => {
    let list = events;
    if (sourceFilter !== 'all') list = list.filter((e) => e.source === sourceFilter);
    if (query.trim()) {
      const q = query.toLowerCase();
      list = list.filter(
        (e) =>
          e.name?.toLowerCase().includes(q) ||
          e.kalshi_ticker?.toLowerCase().includes(q) ||
          e.poly_slug?.toLowerCase().includes(q)
      );
    }
    return list.slice().sort((a, b) => {
      let cmp = 0;
      if (sortBy === 'name') cmp = (a.name || '').localeCompare(b.name || '');
      else if (sortBy === 'mappings') cmp = Object.keys(a.mapping || {}).length - Object.keys(b.mapping || {}).length;
      else if (sortBy === 'source') cmp = (a.source || '').localeCompare(b.source || '');
      return sortDir === 'asc' ? cmp : -cmp;
    });
  }, [events, query, sourceFilter, sortBy, sortDir]);

  const toggleSort = (col) => {
    if (sortBy === col) setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    else { setSortBy(col); setSortDir('asc'); }
  };
  const arrow = (col) => (sortBy === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '');

  if (loading) {
    return (
      <div className="panel" style={{ flex: 1 }}>
        <div className="loading">
          <span className="spinner" /> Loading config…
        </div>
      </div>
    );
  }

  return (
    <div className="panel" style={{ flex: 1 }}>
      <div className="config-stats">
        <div className="config-stat">
          <div className="config-stat-lbl">Reviewed</div>
          <div className="config-stat-val">
            {stats?.reviewed?.mapped ?? 0}
            <span style={{ color: 'var(--text-muted)', fontSize: 14 }}>
              {' '}/ {stats?.reviewed?.total ?? 0}
            </span>
          </div>
        </div>
        <div className="config-stat">
          <div className="config-stat-lbl">Main</div>
          <div className="config-stat-val">
            {stats?.main?.mapped ?? 0}
            <span style={{ color: 'var(--text-muted)', fontSize: 14 }}>
              {' '}/ {stats?.main?.total ?? 0}
            </span>
          </div>
        </div>
        <div className="config-stat">
          <div className="config-stat-lbl">Total Active</div>
          <div className="config-stat-val blue">{events.length}</div>
        </div>
        {discovery && discovery.generated_at && (
          <div className="config-stat">
            <div className="config-stat-lbl">
              Discovered (HIGH)
              <button
                className="btn primary"
                style={{ float: 'right', padding: '4px 10px', fontSize: 10 }}
                onClick={runDiscovery}
                disabled={discovering}
              >
                {discovering ? '…' : 'Run'}
              </button>
            </div>
            <div className="config-stat-val">
              {discovery.high}
              <span style={{ color: 'var(--text-muted)', fontSize: 14 }}>
                {' '}/ {discovery.total}
              </span>
            </div>
          </div>
        )}
      </div>

      <div className="config-search-bar">
        <div style={{ position: 'relative', display: 'inline-block' }}>
          <div style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)', pointerEvents: 'none' }}>
            <IconSearch size={13} />
          </div>
          <input
            className="search"
            style={{ paddingLeft: 30 }}
            placeholder="Search by name, ticker, or slug…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
        </div>
        <span style={{ marginLeft: 12, color: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}>
          {filtered.length} / {events.length}
        </span>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 4 }}>
          {SOURCES.map((s) => (
            <button
              key={s}
              className={`filter-btn ${sourceFilter === s ? 'active' : ''}`}
              onClick={() => setSourceFilter(s)}
            >
              {s} {sourceCounts[s] != null && <span style={{ color: 'var(--text-muted)', marginLeft: 4 }}>{sourceCounts[s]}</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="panel-body">
        <table className="dtable">
          <thead>
            <tr>
              <th className="sortable" onClick={() => toggleSort('name')}>
                Event{arrow('name')}
              </th>
              <th>Kalshi ticker</th>
              <th>Poly slug</th>
              <th className="sortable" onClick={() => toggleSort('mappings')}>
                Mappings{arrow('mappings')}
              </th>
              <th className="sortable" onClick={() => toggleSort('source')}>
                Source{arrow('source')}
              </th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((ev, i) => {
              const mapping = ev.mapping || {};
              const entries = Object.entries(mapping);
              const preview = entries
                .slice(0, 3)
                .map(([k, v]) => `${k}→${v}`)
                .join(' · ');
              return (
                <tr key={i}>
                  <td className="event-cell" title={ev.name}>{ev.name}</td>
                  <td>
                    {ev.kalshi_ticker ? (
                      <a href={`https://kalshi.com/events/${ev.kalshi_ticker}`} target="_blank" rel="noreferrer">
                        {ev.kalshi_ticker} <IconExt size={10} />
                      </a>
                    ) : <span style={{ color: 'var(--text-faint)' }}>—</span>}
                  </td>
                  <td>
                    {ev.poly_slug ? (
                      <a href={`https://polymarket.com/event/${ev.poly_slug}`} target="_blank" rel="noreferrer">
                        {ev.poly_slug} <IconExt size={10} />
                      </a>
                    ) : <span style={{ color: 'var(--text-faint)' }}>—</span>}
                  </td>
                  <td>
                    <div>
                      <span className="mono">{entries.length}</span>
                      {preview && <div className="mapping-preview">{preview}{entries.length > 3 ? ' …' : ''}</div>}
                    </div>
                  </td>
                  <td>
                    <span className={`badge source-${ev.source || 'main'}`}>
                      {ev.source || 'main'}
                    </span>
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={5} style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                  No events match
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
