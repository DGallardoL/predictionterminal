import React, { useState, useMemo } from 'react';

export default function ScanLogTab({ log }) {
  const [filter, setFilter] = useState('all');
  const entries = log || [];

  const candidates = useMemo(
    () => entries.filter((e) => e.pass_yes || e.pass_no),
    [entries]
  );

  const shown = filter === 'candidates' ? candidates : entries;

  return (
    <div className="panel" style={{ flex: 1 }}>
      <div className="filterbar">
        <button
          className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
          onClick={() => setFilter('all')}
        >
          All ({entries.length})
        </button>
        <button
          className={`filter-btn ${filter === 'candidates' ? 'active' : ''}`}
          onClick={() => setFilter('candidates')}
        >
          Candidates ({candidates.length})
        </button>
        <div style={{ marginLeft: 'auto', fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-muted)', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
          Live scan stream
        </div>
      </div>
      <div className="panel-body">
        <table className="dtable">
          <thead>
            <tr>
              <th>Time</th>
              <th>Event</th>
              <th>Outcome</th>
              <th style={{ textAlign: 'right' }}>K_yes</th>
              <th style={{ textAlign: 'right' }}>P_no</th>
              <th style={{ textAlign: 'right' }}>Y+N</th>
              <th style={{ textAlign: 'right' }}>K_no</th>
              <th style={{ textAlign: 'right' }}>P_yes</th>
              <th style={{ textAlign: 'right' }}>N+Y</th>
              <th style={{ textAlign: 'right' }}>Thresh</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {shown.length === 0 ? (
              <tr>
                <td colSpan={11} style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                  Waiting for scan data…
                </td>
              </tr>
            ) : (
              shown.map((e, idx) => {
                const passing = e.pass_yes || e.pass_no;
                const costYesCls = e.cost_yes < e.threshold ? 'green' : '';
                const costNoCls = e.cost_no < e.threshold ? 'green' : '';
                return (
                  <tr key={idx} className={passing ? 'pass' : ''}>
                    <td>{e.t}</td>
                    <td className="event-cell" title={e.event}>{e.event}</td>
                    <td>
                      <div className="outcome-cell">
                        {e.outcome}
                        {e.neg_risk && <span className="badge nr">NR</span>}
                      </div>
                    </td>
                    <td className="num" style={{ textAlign: 'right' }}>{e.k_yes?.toFixed(3)}</td>
                    <td className="num" style={{ textAlign: 'right' }}>{e.p_no?.toFixed(3)}</td>
                    <td className={`num ${costYesCls}`} style={{ textAlign: 'right' }}>
                      {e.cost_yes?.toFixed(3)}
                    </td>
                    <td className="num" style={{ textAlign: 'right' }}>{e.k_no?.toFixed(3)}</td>
                    <td className="num" style={{ textAlign: 'right' }}>{e.p_yes?.toFixed(3)}</td>
                    <td className={`num ${costNoCls}`} style={{ textAlign: 'right' }}>
                      {e.cost_no?.toFixed(3)}
                    </td>
                    <td className="num" style={{ textAlign: 'right', color: 'var(--text-muted)' }}>
                      {e.threshold?.toFixed(3)}
                    </td>
                    <td>
                      {e.pass_yes && <span className="badge arb">YES ARB</span>}
                      {e.pass_no && <span className="badge arb">NO ARB</span>}
                      {!e.pass_yes && !e.pass_no && e.stub && (
                        <span className="badge stub">STUB</span>
                      )}
                      {!e.pass_yes && !e.pass_no && !e.stub && (
                        <span style={{ color: 'var(--text-faint)' }}>—</span>
                      )}
                    </td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
