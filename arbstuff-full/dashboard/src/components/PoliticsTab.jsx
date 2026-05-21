import React, { useEffect, useState, useMemo } from 'react';
import { IconExt, IconSearch } from './icons.jsx';

const STATE_NAMES = {
  AL: 'Alabama', AK: 'Alaska', AZ: 'Arizona', AR: 'Arkansas', CA: 'California',
  CO: 'Colorado', CT: 'Connecticut', DE: 'Delaware', FL: 'Florida', GA: 'Georgia',
  HI: 'Hawaii', ID: 'Idaho', IL: 'Illinois', IN: 'Indiana', IA: 'Iowa',
  KS: 'Kansas', KY: 'Kentucky', LA: 'Louisiana', ME: 'Maine', MD: 'Maryland',
  MA: 'Massachusetts', MI: 'Michigan', MN: 'Minnesota', MS: 'Mississippi',
  MO: 'Missouri', MT: 'Montana', NE: 'Nebraska', NV: 'Nevada', NH: 'New Hampshire',
  NJ: 'New Jersey', NM: 'New Mexico', NY: 'New York', NC: 'North Carolina',
  ND: 'North Dakota', OH: 'Ohio', OK: 'Oklahoma', OR: 'Oregon', PA: 'Pennsylvania',
  RI: 'Rhode Island', SC: 'South Carolina', SD: 'South Dakota', TN: 'Tennessee',
  TX: 'Texas', UT: 'Utah', VT: 'Vermont', VA: 'Virginia', WA: 'Washington',
  WV: 'West Virginia', WI: 'Wisconsin', WY: 'Wyoming', DC: 'DC',
};

const OFFICE_LABELS = {
  HOUSE: 'House', SEN: 'Senate', GOV: 'Governor', LTGOV: 'Lt Gov',
  AG: 'Atty General', SOS: 'Sec State', TREAS: 'Treasurer',
  PRES: 'President', MAYOR: 'Mayor',
};

function profitClass(p) {
  if (p >= 5) return 'high';
  if (p >= 2) return 'med';
  return 'low';
}

// Parse opportunity name to extract state/office for filtering
function parsePoliticalOpp(opp) {
  const name = opp.name || '';
  const m = name.match(/\b([A-Z]{2})\b/);
  const state = m ? m[1] : null;

  let office = null;
  const nl = name.toLowerCase();
  if (nl.includes('house') || /\b[A-Z]{2}-\d+\b/.test(name)) office = 'HOUSE';
  else if (nl.includes('senate')) office = 'SEN';
  else if (nl.includes('governor')) office = 'GOV';
  else if (nl.includes('attorney general')) office = 'AG';
  else if (nl.includes('president')) office = 'PRES';

  return { state, office };
}

export default function PoliticsTab({ opportunities = [] }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [discovering, setDiscovering] = useState(false);
  const [filterState, setFilterState] = useState('');
  const [filterOffice, setFilterOffice] = useState('');
  const [filterType, setFilterType] = useState('');
  const [search, setSearch] = useState('');
  const [view, setView] = useState('arbs'); // 'arbs' | 'markets'

  const refresh = async () => {
    try {
      const r = await fetch('/api/politics/events');
      const d = await r.json();
      setData(d);
    } catch {}
    setLoading(false);
  };

  useEffect(() => { refresh(); }, []);

  const runDiscovery = async () => {
    setDiscovering(true);
    try {
      await fetch('/api/politics/run', { method: 'POST' });
      await new Promise((r) => setTimeout(r, 800));
      await refresh();
    } catch {}
    setDiscovering(false);
  };

  // Live political arbs — filter opportunities from source=politics or match by title pattern
  const politicalArbs = useMemo(() => {
    return (opportunities || [])
      .filter((o) => {
        // Accept if explicitly tagged as politics
        if (o.source === 'politics') return true;
        // Or if title starts with a state name / state code
        const parsed = parsePoliticalOpp(o);
        return parsed.state != null || parsed.office != null;
      })
      .map((o) => ({ ...o, ...parsePoliticalOpp(o) }))
      .filter((o) => {
        if (filterState && o.state !== filterState) return false;
        if (filterOffice && o.office !== filterOffice) return false;
        if (search) {
          const s = search.toLowerCase();
          if (!(o.name || '').toLowerCase().includes(s)) return false;
        }
        return true;
      })
      .sort((a, b) => b.profit_pct - a.profit_pct);
  }, [opportunities, filterState, filterOffice, search]);

  const filtered = useMemo(() => {
    if (!data) return [];
    return data.events.filter((e) => {
      if (filterState && e.state !== filterState) return false;
      if (filterOffice && e.office !== filterOffice) return false;
      if (filterType && e.race_type !== filterType) return false;
      if (search) {
        const s = search.toLowerCase();
        if (!(e.name || '').toLowerCase().includes(s)) return false;
      }
      return true;
    });
  }, [data, filterState, filterOffice, filterType, search]);

  if (loading) {
    return (
      <div className="panel" style={{ flex: 1 }}>
        <div className="loading">
          <span className="spinner" /> Loading political markets…
        </div>
      </div>
    );
  }

  const stats = data?.stats || {};
  const byState = stats.by_state || {};
  const byOffice = stats.by_office || {};
  const byType = stats.by_type || {};
  const statesList = Object.entries(byState).sort((a, b) => b[1] - a[1]);
  const officeList = Object.entries(byOffice).sort((a, b) => b[1] - a[1]);

  // Top political arb
  const topArb = politicalArbs[0];

  return (
    <div className="panel" style={{ flex: 1, overflow: 'hidden' }}>
      {/* Stats row */}
      <div className="config-stats" style={{ gridTemplateColumns: 'repeat(6, 1fr)' }}>
        <div className="config-stat">
          <div className="config-stat-lbl">Tracked Markets</div>
          <div className="config-stat-val blue">{data?.total || 0}</div>
        </div>
        <div className="config-stat">
          <div className="config-stat-lbl">States Covered</div>
          <div className="config-stat-val">{Object.keys(byState).length}</div>
        </div>
        <div className="config-stat">
          <div className="config-stat-lbl" style={{ color: 'var(--green)' }}>Live Arbs</div>
          <div className="config-stat-val" style={{ color: 'var(--green)' }}>{politicalArbs.length}</div>
        </div>
        <div className="config-stat">
          <div className="config-stat-lbl">Best Profit</div>
          <div className="config-stat-val">
            {topArb ? `${topArb.profit_pct.toFixed(1)}%` : '—'}
          </div>
        </div>
        <div className="config-stat">
          <div className="config-stat-lbl">General / Primary</div>
          <div className="config-stat-val" style={{ fontSize: 16 }}>
            {byType.general || 0} / {byType.primary || 0}
          </div>
        </div>
        <div className="config-stat" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <button className="btn primary" onClick={runDiscovery} disabled={discovering}>
            {discovering ? 'Running…' : 'Re-discover'}
          </button>
        </div>
      </div>

      {/* View toggle */}
      <div className="filterbar" style={{ flexWrap: 'wrap' }}>
        <div className="sort-group">
          <button className={`sort-pill ${view === 'arbs' ? 'active' : ''}`} onClick={() => setView('arbs')}>
            Live Arbs ({politicalArbs.length})
          </button>
          <button className={`sort-pill ${view === 'markets' ? 'active' : ''}`} onClick={() => setView('markets')}>
            All Markets ({data?.total || 0})
          </button>
        </div>

        <div style={{ position: 'relative' }}>
          <div style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}>
            <IconSearch size={13} />
          </div>
          <input
            className="search"
            style={{ paddingLeft: 30, width: 200 }}
            placeholder="Search…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <select className="filter-btn" value={filterState} onChange={(e) => setFilterState(e.target.value)} style={{ padding: '5px 10px' }}>
          <option value="">All states ({Object.keys(byState).length})</option>
          {statesList.map(([code, n]) => (
            <option key={code} value={code}>{STATE_NAMES[code] || code} ({n})</option>
          ))}
        </select>

        <select className="filter-btn" value={filterOffice} onChange={(e) => setFilterOffice(e.target.value)} style={{ padding: '5px 10px' }}>
          <option value="">All offices</option>
          {officeList.map(([code, n]) => (
            <option key={code} value={code}>{OFFICE_LABELS[code] || code} ({n})</option>
          ))}
        </select>

        {view === 'markets' && (
          <div style={{ display: 'flex', gap: 4 }}>
            {['', 'general', 'primary', 'special', 'runoff'].map((t) => (
              <button key={t || 'all'}
                className={`filter-btn ${filterType === t ? 'active' : ''}`}
                onClick={() => setFilterType(t)}>
                {t || 'all'} {t && byType[t] ? <span style={{ color: 'var(--text-muted)', marginLeft: 3 }}>{byType[t]}</span> : ''}
              </button>
            ))}
          </div>
        )}

        <div className="flex-end mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {view === 'arbs'
            ? `${politicalArbs.length} live arbs`
            : `${filtered.length} / ${data?.total || 0} markets`}
        </div>
      </div>

      <div className="panel-body">
        {view === 'arbs' ? (
          <table className="dtable">
            <thead>
              <tr>
                <th>#</th>
                <th>State</th>
                <th>Race</th>
                <th style={{ textAlign: 'right' }}>Profit %</th>
                <th style={{ textAlign: 'right' }}>Vol</th>
                <th style={{ textAlign: 'right' }}>Cost</th>
                <th>Type</th>
                <th>Links</th>
              </tr>
            </thead>
            <tbody>
              {politicalArbs.length === 0 ? (
                <tr>
                  <td colSpan={8} style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                    No political arbs right now — scanner will post them here as they appear
                  </td>
                </tr>
              ) : politicalArbs.map((o, i) => {
                const kEvent = o.kalshi_event_ticker || o.kalshi_ticker || '';
                return (
                  <tr key={o.arb_key || i}>
                    <td className="mono">#{i + 1}</td>
                    <td className="mono">{o.state || '—'}</td>
                    <td>
                      <div className="event-cell" title={o.name}>
                        {o.name}
                        {o.neg_risk && <span className="badge nr" style={{ marginLeft: 6 }}>NR</span>}
                      </div>
                      <div style={{ fontSize: 10, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                        {o.type} · K ${(o.kalshi_price || 0).toFixed(3)} / P ${(o.poly_price || 0).toFixed(3)}
                      </div>
                    </td>
                    <td className={`num opp-profit ${profitClass(o.profit_pct)}`} style={{ textAlign: 'right', fontSize: 13 }}>
                      +{o.profit_pct.toFixed(1)}%
                    </td>
                    <td className="num" style={{ textAlign: 'right' }}>
                      {o.volume > 0 ? (o.volume || 0).toLocaleString() : '—'}
                    </td>
                    <td className="num" style={{ textAlign: 'right' }}>${(o.cost || 0).toFixed(3)}</td>
                    <td>
                      <span className={`badge source-${o.source || 'main'}`}>{o.source || 'main'}</span>
                    </td>
                    <td>
                      {kEvent && (
                        <a href={`https://kalshi.com/events/${kEvent}`} target="_blank" rel="noreferrer" style={{ marginRight: 6 }}>
                          K <IconExt size={9} />
                        </a>
                      )}
                      {o.poly_slug && (
                        <a href={`https://polymarket.com/event/${o.poly_slug}`} target="_blank" rel="noreferrer">
                          P <IconExt size={9} />
                        </a>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <table className="dtable">
            <thead>
              <tr>
                <th>State</th>
                <th>Office</th>
                <th>Dist</th>
                <th>Type</th>
                <th>Party</th>
                <th>Year</th>
                <th>Kalshi</th>
                <th>Polymarket</th>
                <th>Maps</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr><td colSpan={9} style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                  No political markets match the filters
                </td></tr>
              ) : filtered.map((e, i) => (
                <tr key={i}>
                  <td className="mono">{e.state || '—'}</td>
                  <td>{OFFICE_LABELS[e.office] || e.office || '—'}</td>
                  <td className="mono">{e.district != null ? `-${String(e.district).padStart(2, '0')}` : '—'}</td>
                  <td>
                    <span className={`badge ${e.race_type === 'primary' ? 'stub' : e.race_type === 'special' ? 'nr' : 'arb'}`}>
                      {e.race_type}
                    </span>
                  </td>
                  <td>
                    {e.party ? (
                      <span className={`badge ${e.party === 'D' ? 'source-main' : e.party === 'R' ? 'side-no' : 'mode'}`}>
                        {e.party}
                      </span>
                    ) : '—'}
                  </td>
                  <td className="mono">{e.year || '—'}</td>
                  <td>
                    <a href={`https://kalshi.com/events/${e.kalshi_ticker}`} target="_blank" rel="noreferrer">
                      {e.kalshi_ticker} <IconExt size={9} />
                    </a>
                  </td>
                  <td>
                    <a href={`https://polymarket.com/event/${e.poly_slug}`} target="_blank" rel="noreferrer">
                      {(e.poly_slug || '').slice(0, 28)} <IconExt size={9} />
                    </a>
                  </td>
                  <td className="num">{e.mapping_count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
