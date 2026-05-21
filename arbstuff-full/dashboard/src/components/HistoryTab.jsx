import React, { useEffect, useState, useMemo } from 'react';
import { IconExt } from './icons.jsx';

export default function HistoryTab() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterSource, setFilterSource] = useState('all');
  const [search, setSearch] = useState('');
  const [grouped, setGrouped] = useState(true);

  const refresh = async () => {
    try {
      const r = await fetch('/api/dashboard/detection-history');
      const d = await r.json();
      setItems(d.items || []);
    } catch {}
    setLoading(false);
  };

  useEffect(() => {
    refresh();
    const iv = setInterval(refresh, 5000);
    return () => clearInterval(iv);
  }, []);

  const filtered = useMemo(() => {
    return items.filter((it) => {
      if (filterSource !== 'all' && it.source !== filterSource) return false;
      if (search) {
        const s = search.toLowerCase();
        if (!(it.name || '').toLowerCase().includes(s)) return false;
      }
      return true;
    });
  }, [items, filterSource, search]);

  // Group by arb_key (or name if no key) to show how many times each arb has been detected
  const groups = useMemo(() => {
    const m = new Map();
    filtered.forEach((it) => {
      const key = it.arb_key || it.name;
      if (!m.has(key)) {
        m.set(key, {
          key,
          name: it.name,
          type: it.type,
          source: it.source,
          neg_risk: it.neg_risk,
          kalshi_event_ticker: it.kalshi_event_ticker,
          poly_slug: it.poly_slug,
          first_seen: it.ts,
          last_seen: it.ts,
          detections: 0,
          best_profit: it.profit,
          worst_profit: it.profit,
          last_profit: it.profit,
          last_volume: it.volume,
          total_volume: 0,
        });
      }
      const g = m.get(key);
      g.detections++;
      g.total_volume += it.volume || 0;
      if (it.ts > g.last_seen) {
        g.last_seen = it.ts;
        g.last_profit = it.profit;
        g.last_volume = it.volume;
      }
      if (it.ts < g.first_seen) g.first_seen = it.ts;
      g.best_profit = Math.max(g.best_profit, it.profit);
      g.worst_profit = Math.min(g.worst_profit, it.profit);
    });
    return Array.from(m.values()).sort((a, b) => (b.last_seen || '').localeCompare(a.last_seen || ''));
  }, [filtered]);

  const sourceCounts = useMemo(() => {
    const c = { all: items.length };
    items.forEach((it) => { c[it.source] = (c[it.source] || 0) + 1; });
    return c;
  }, [items]);

  if (loading) {
    return (
      <div className="panel" style={{ flex: 1 }}>
        <div className="loading">
          <span className="spinner" /> Loading detection history…
        </div>
      </div>
    );
  }

  return (
    <div className="panel" style={{ flex: 1 }}>
      <div className="filterbar">
        <input
          className="search"
          placeholder="Search detections…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          style={{ width: 240 }}
        />
        <div style={{ display: 'flex', gap: 4 }}>
          {['all', 'reviewed', 'main', 'discovered'].map((s) => (
            <button
              key={s}
              className={`filter-btn ${filterSource === s ? 'active' : ''}`}
              onClick={() => setFilterSource(s)}
            >
              {s} {sourceCounts[s] != null && <span style={{ color: 'var(--text-muted)', marginLeft: 3 }}>{sourceCounts[s]}</span>}
            </button>
          ))}
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className={`filter-btn ${grouped ? 'active' : ''}`} onClick={() => setGrouped(true)}>
            Grouped ({groups.length})
          </button>
          <button className={`filter-btn ${!grouped ? 'active' : ''}`} onClick={() => setGrouped(false)}>
            Timeline ({filtered.length})
          </button>
          <span className="mono" style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 8 }}>
            ⟳ 5s
          </span>
        </div>
      </div>

      <div className="panel-body">
        {grouped ? (
          <table className="dtable">
            <thead>
              <tr>
                <th>Last seen</th>
                <th>Event</th>
                <th>Type</th>
                <th style={{ textAlign: 'right' }}>Detections</th>
                <th style={{ textAlign: 'right' }}>Best %</th>
                <th style={{ textAlign: 'right' }}>Last %</th>
                <th style={{ textAlign: 'right' }}>Last Vol</th>
                <th>Source</th>
                <th>Links</th>
              </tr>
            </thead>
            <tbody>
              {groups.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                    No detections yet — scanner will start logging arbs as they appear
                  </td>
                </tr>
              ) : (
                groups.map((g) => (
                  <tr key={g.key}>
                    <td className="mono">{g.last_seen?.slice(11, 19) || '—'}</td>
                    <td className="event-cell" title={g.name}>
                      {g.name}
                      {g.neg_risk && <span className="badge nr" style={{ marginLeft: 6 }}>NR</span>}
                    </td>
                    <td className="mono" style={{ fontSize: 10, color: 'var(--text-dim)' }}>{g.type}</td>
                    <td className="num" style={{ textAlign: 'right', fontWeight: 600 }}>{g.detections}</td>
                    <td className="num green" style={{ textAlign: 'right' }}>{g.best_profit.toFixed(1)}%</td>
                    <td className="num green" style={{ textAlign: 'right' }}>{g.last_profit.toFixed(1)}%</td>
                    <td className="num" style={{ textAlign: 'right' }}>{(g.last_volume || 0).toLocaleString()}</td>
                    <td><span className={`badge source-${g.source || 'main'}`}>{g.source || 'main'}</span></td>
                    <td>
                      {g.kalshi_event_ticker && (
                        <a href={`https://kalshi.com/events/${g.kalshi_event_ticker}`} target="_blank" rel="noreferrer" style={{ marginRight: 6 }}>
                          K <IconExt size={9} />
                        </a>
                      )}
                      {g.poly_slug && (
                        <a href={`https://polymarket.com/event/${g.poly_slug}`} target="_blank" rel="noreferrer">
                          P <IconExt size={9} />
                        </a>
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        ) : (
          <table className="dtable">
            <thead>
              <tr>
                <th>Time</th>
                <th>Event</th>
                <th>Type</th>
                <th style={{ textAlign: 'right' }}>Profit %</th>
                <th style={{ textAlign: 'right' }}>Vol</th>
                <th style={{ textAlign: 'right' }}>Cost</th>
                <th>Source</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr><td colSpan={7} style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>No detections</td></tr>
              ) : (
                filtered.map((it, i) => (
                  <tr key={i}>
                    <td className="mono">{it.ts?.slice(11, 19)}</td>
                    <td className="event-cell" title={it.name}>{it.name}</td>
                    <td className="mono" style={{ fontSize: 10, color: 'var(--text-dim)' }}>{it.type}</td>
                    <td className="num green" style={{ textAlign: 'right' }}>{it.profit.toFixed(1)}%</td>
                    <td className="num" style={{ textAlign: 'right' }}>{(it.volume || 0).toLocaleString()}</td>
                    <td className="num" style={{ textAlign: 'right' }}>${it.cost.toFixed(3)}</td>
                    <td><span className={`badge source-${it.source || 'main'}`}>{it.source || 'main'}</span></td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
