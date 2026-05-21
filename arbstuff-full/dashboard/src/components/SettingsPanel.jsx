import React, { useEffect, useState } from 'react';
import { IconClose } from './icons.jsx';

export default function SettingsPanel({ state, onClose }) {
  const cfg = state?.config || {};
  const [emailEnabled, setEmailEnabled] = useState(!!state?.email_enabled);
  const [scanMode, setScanMode] = useState(state?.scan_mode || 'WS');
  const [threshold, setThreshold] = useState(cfg.threshold ?? 0.94);
  const [minAlert, setMinAlert] = useState(cfg.min_alert_profit ?? 1.0);
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState(null); // 'ok' | 'err' | null

  // Close on ESC
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClose?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const save = async () => {
    setSaving(true);
    setStatus(null);
    try {
      const r = await fetch('/api/dashboard/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email_enabled: emailEnabled,
          threshold,
          min_alert_profit: minAlert,
          scan_mode: scanMode,
        }),
      });
      if (r.ok) {
        setStatus('ok');
        setTimeout(() => onClose?.(), 650);
      } else {
        setStatus('err');
      }
    } catch {
      setStatus('err');
    }
    setSaving(false);
  };

  return (
    <>
      <div className="settings-overlay" onClick={onClose} />
      <aside className="settings-panel" role="dialog" aria-label="Settings">
        <div className="settings-head">
          <h2>Settings</h2>
          <button className="close-btn" onClick={onClose} aria-label="Close">
            <IconClose size={16} />
          </button>
        </div>

        <div className="settings-body">
          <div className="settings-group">
            <div className="toggle-row">
              <div style={{ flex: 1 }}>
                <div className="settings-label">Email Alerts</div>
                <div className="settings-sub">
                  {emailEnabled
                    ? `Notifying when profit ≥ ${minAlert.toFixed(1)}%`
                    : 'Notifications are off. Turn on to get real-time alerts.'}
                </div>
              </div>
              <div
                className={`toggle ${emailEnabled ? 'on' : ''}`}
                onClick={() => setEmailEnabled((v) => !v)}
                role="switch"
                aria-checked={emailEnabled}
              />
            </div>
          </div>

          <div className="settings-group">
            <div className="settings-label">Scan Mode</div>
            <div className="mode-toggle">
              <button
                className={scanMode === 'OG' ? 'active' : ''}
                onClick={() => setScanMode('OG')}
              >
                OG
              </button>
              <button
                className={scanMode === 'WS' ? 'active' : ''}
                onClick={() => setScanMode('WS')}
              >
                WS
              </button>
            </div>
            <div className="settings-sub">
              {scanMode === 'WS'
                ? 'WebSocket streaming · lowest latency.'
                : 'Classic polling mode.'}
            </div>
          </div>

          <div className="settings-group">
            <div className="settings-label">
              Threshold
              <span className="val mono">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.85"
              max="1.00"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="slider"
            />
            <div className="settings-sub">
              Minimum combined cost to qualify. Lower = more signals.
            </div>
          </div>

          <div className="settings-group">
            <div className="settings-label">
              Min Alert Profit %
              <span className="val mono">{minAlert.toFixed(1)}%</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="10.0"
              step="0.5"
              value={minAlert}
              onChange={(e) => setMinAlert(parseFloat(e.target.value))}
              className="slider"
            />
            <div className="settings-sub">
              Only opportunities above this profit level trigger email alerts.
            </div>
          </div>
        </div>

        <div className="settings-foot">
          <button className="btn" onClick={onClose}>Cancel</button>
          <button
            className="btn primary"
            onClick={save}
            disabled={saving}
          >
            {saving ? 'Saving…' : status === 'ok' ? 'Saved ✓' : status === 'err' ? 'Retry' : 'Save'}
          </button>
        </div>
      </aside>
    </>
  );
}
