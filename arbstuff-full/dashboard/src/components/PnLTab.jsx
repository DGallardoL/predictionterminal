import React, { useEffect, useState } from 'react';
import { IconDollar } from './icons.jsx';

function PnLChart({ trades }) {
  if (!trades || trades.length === 0) {
    return (
      <div style={{ height: 120, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: 11, fontFamily: 'var(--font-mono)' }}>
        no trades to chart
      </div>
    );
  }

  // cumulative pnl over trade index (oldest -> newest)
  const asc = [...trades].reverse();
  const cum = [];
  let total = 0;
  asc.forEach((t) => {
    total += t.guaranteed_profit || 0;
    cum.push(total);
  });

  const W = 1000;
  const H = 120;
  const pad = 6;
  const maxY = Math.max(...cum, 0.001);
  const minY = Math.min(...cum, 0);
  const range = maxY - minY || 1;

  const x = (i) => pad + (i / Math.max(cum.length - 1, 1)) * (W - 2 * pad);
  const y = (v) => H - pad - ((v - minY) / range) * (H - 2 * pad);

  const linePath = cum.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
  const areaPath = `${linePath} L${x(cum.length - 1).toFixed(2)},${H - pad} L${x(0).toFixed(2)},${H - pad} Z`;
  const zeroY = y(0);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="pnlGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#4ade80" stopOpacity="0.35" />
          <stop offset="100%" stopColor="#4ade80" stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* zero baseline */}
      {minY < 0 && (
        <line
          x1={pad}
          x2={W - pad}
          y1={zeroY}
          y2={zeroY}
          stroke="rgba(255,255,255,0.1)"
          strokeDasharray="2 3"
        />
      )}
      <path d={areaPath} fill="url(#pnlGrad)" />
      <path d={linePath} fill="none" stroke="#4ade80" strokeWidth="1.5" />
      {/* last point dot */}
      <circle
        cx={x(cum.length - 1)}
        cy={y(cum[cum.length - 1])}
        r="3"
        fill="#4ade80"
      />
    </svg>
  );
}

export default function PnLTab() {
  const [trades, setTrades] = useState([]);
  const [totalPnl, setTotalPnl] = useState(0);
  const [count, setCount] = useState(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const r = await fetch('/api/dashboard/pnl');
        const data = await r.json();
        if (cancelled) return;
        setTrades(data.trades || []);
        setTotalPnl(data.total_pnl || 0);
        setCount(data.count || 0);
      } catch {}
      setLoading(false);
    };
    load();
    const iv = setInterval(load, 8000);
    return () => {
      cancelled = true;
      clearInterval(iv);
    };
  }, []);

  const totalVol = trades.reduce((s, t) => s + (t.volume || 0), 0);
  const totalCost = trades.reduce((s, t) => s + (t.total_cost || 0), 0);
  const roi = totalCost > 0 ? (totalPnl / totalCost) * 100 : 0;

  if (loading && trades.length === 0) {
    return (
      <div className="panel" style={{ flex: 1 }}>
        <div className="loading">
          <span className="spinner" /> Loading trades…
        </div>
      </div>
    );
  }

  if (trades.length === 0) {
    return (
      <div className="panel" style={{ flex: 1 }}>
        <div className="empty" style={{ flex: 1, justifyContent: 'center' }}>
          <IconDollar size={32} />
          <div className="empty-title">No simulated trades yet</div>
          <div className="empty-sub">
            Trades appear here whenever the scanner identifies a qualifying arbitrage and the bot executes (test or live).
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="panel" style={{ flex: 1 }}>
      <div className="pnl-head">
        <div className="pnl-head-left">
          <div className="pnl-head-title">
            Trade Log <span className="badge test">Test Mode</span>
          </div>
          <div className="pnl-head-meta">
            <span className="mono">{count}</span> trades
            <span className="sep">·</span>
            <span className="mono">{totalVol.toLocaleString()}</span> contracts
            <span className="sep">·</span>
            <span className="mono">${totalCost.toFixed(2)}</span> deployed
            <span className="sep">·</span>
            <span className="mono" style={{ color: roi >= 0 ? 'var(--green)' : 'var(--red)' }}>
              {roi.toFixed(2)}%
            </span>{' '}
            ROI
          </div>
        </div>
        <div className={`pnl-value ${totalPnl >= 0 ? 'green' : ''}`}>
          {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
        </div>
      </div>

      <div className="pnl-chart">
        <PnLChart trades={trades} />
      </div>

      <div className="panel-body">
        <table className="dtable">
          <thead>
            <tr>
              <th>Time</th>
              <th>Event</th>
              <th>Outcome</th>
              <th>Side</th>
              <th style={{ textAlign: 'right' }}>Vol</th>
              <th style={{ textAlign: 'right' }}>K Price</th>
              <th style={{ textAlign: 'right' }}>P Price</th>
              <th style={{ textAlign: 'right' }}>Cost</th>
              <th style={{ textAlign: 'right' }}>Profit</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((t, i) => {
              const time = t.timestamp
                ? new Date(t.timestamp).toLocaleTimeString('en-US', { hour12: false })
                : '—';
              const sideLc = (t.side || '').toLowerCase();
              const sideCls = sideLc === 'yes' ? 'side-yes' : 'side-no';
              return (
                <tr key={i}>
                  <td>{time}</td>
                  <td className="event-cell" title={t.event}>{t.event}</td>
                  <td className="outcome-cell">{t.outcome}</td>
                  <td>
                    <span className={`badge ${sideCls}`}>{(t.side || '').toUpperCase()}</span>
                  </td>
                  <td className="num" style={{ textAlign: 'right' }}>{(t.volume || 0).toLocaleString()}</td>
                  <td className="num" style={{ textAlign: 'right' }}>${(t.k_price || 0).toFixed(3)}</td>
                  <td className="num" style={{ textAlign: 'right' }}>${(t.p_price || 0).toFixed(3)}</td>
                  <td className="num" style={{ textAlign: 'right' }}>${(t.total_cost || 0).toFixed(2)}</td>
                  <td className="num green" style={{ textAlign: 'right' }}>
                    +${(t.guaranteed_profit || 0).toFixed(2)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
