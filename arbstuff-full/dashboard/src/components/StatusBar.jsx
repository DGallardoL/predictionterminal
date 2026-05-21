import React from 'react';
import { Link } from 'react-router-dom';
import { IconGear } from './icons.jsx';

export default function StatusBar({ state, connected, onOpenSettings }) {
  const running = state?.bot_status === 'running' && connected;
  const testMode = state?.test_mode;
  const scanMode = state?.scan_mode;
  const cfg = state?.config || {};
  const balances = state?.balances || {};
  const timestamp = state?.timestamp
    ? new Date(state.timestamp).toLocaleTimeString('en-US', { hour12: false })
    : '--:--:--';

  return (
    <header className="statusbar">
      <div className="brand">
        <div>
          <div className="brand-mark">arb</div>
          <div className="brand-sub">kalshi × poly</div>
        </div>
      </div>

      <div className="statusbar-left">
        <div className={`status-dot ${running ? 'running' : 'offline'}`} />
        <div className="status-meta">
          <span>
            <span className="val mono">{cfg.event_count ?? 0}</span>{' '}
            events
          </span>
          <span className="sep">·</span>
          <span>
            <span className="val mono">
              {state?.cycle_time_s != null ? state.cycle_time_s.toFixed(2) : '0.00'}s
            </span>
          </span>
          <span className="sep">·</span>
          <span>
            <span className="val mono">#{state?.scan_count ?? 0}</span>
          </span>
        </div>
      </div>

      <div className="status-time">{timestamp}</div>

      <div className="status-right">
        {testMode && <span className="badge test">Test</span>}
        {scanMode && <span className="badge mode">{scanMode}</span>}
        <div className="divider" />
        <div className="balance-pill" title="Kalshi balance">
          <span className="label">K</span>
          <span>${(balances.kalshi ?? 0).toFixed(2)}</span>
        </div>
        <div className="balance-pill" title="Polymarket balance">
          <span className="label">P</span>
          <span>${(balances.polymarket ?? 0).toFixed(2)}</span>
        </div>
        <div className="divider" />
        <Link className="link-btn" to="/review">
          Review
        </Link>
        <button
          className="icon-btn"
          onClick={onOpenSettings}
          aria-label="Open settings"
        >
          <IconGear size={14} />
        </button>
      </div>
    </header>
  );
}
