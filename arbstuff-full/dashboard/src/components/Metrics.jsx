import React from 'react';

function Metric({ label, value, sub, accentVar, accentColor }) {
  const style = accentColor ? { '--accent': accentColor } : undefined;
  return (
    <div className="metric" style={style}>
      <div className="metric-label">
        <span className="tick" />
        {label}
      </div>
      <div className={`metric-value ${accentColor ? 'accent' : ''}`}>
        {value}
      </div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

export default function Metrics({ state }) {
  const opps = state?.opportunities || [];
  const activeCount = opps.length;
  const best = opps.reduce(
    (acc, o) => (o.profit_pct > (acc?.profit_pct ?? -Infinity) ? o : acc),
    null
  );
  const totalVol = opps.reduce((s, o) => s + (o.volume || 0), 0);

  return (
    <div className="metrics">
      <Metric
        label="Active Arbs"
        value={activeCount}
        sub={activeCount ? 'live opportunities' : 'scanning…'}
        accentColor="var(--green)"
      />
      <Metric
        label="Best Profit"
        value={best ? `${best.profit_pct.toFixed(1)}%` : '—'}
        sub={best?.name || 'no matches'}
        accentColor="var(--green)"
      />
      <Metric
        label="Total Volume"
        value={totalVol.toLocaleString()}
        sub="combined contracts"
        accentColor="var(--blue)"
      />
      <Metric
        label="Scans"
        value={state?.scan_count ?? 0}
        sub="since boot"
        accentColor="var(--purple)"
      />
      <Metric
        label="Email Alerts"
        value={state?.email_enabled ? 'ON' : 'OFF'}
        sub={
          state?.email_enabled
            ? `≥ ${state?.config?.min_alert_profit ?? 1}% profit`
            : 'alerts disabled'
        }
        accentColor="var(--amber)"
      />
    </div>
  );
}
