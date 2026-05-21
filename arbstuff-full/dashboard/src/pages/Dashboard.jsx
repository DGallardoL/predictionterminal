import React, { useState, useMemo } from 'react';
import { useSSE } from '../hooks/useSSE.js';
import StatusBar from '../components/StatusBar.jsx';
import Metrics from '../components/Metrics.jsx';
import OpportunitiesTab from '../components/OpportunitiesTab.jsx';
import ScanLogTab from '../components/ScanLogTab.jsx';
import PnLTab from '../components/PnLTab.jsx';
import ConfigTab from '../components/ConfigTab.jsx';
import HistoryTab from '../components/HistoryTab.jsx';
import PoliticsTab from '../components/PoliticsTab.jsx';
import SettingsPanel from '../components/SettingsPanel.jsx';

const TABS = [
  { id: 'opps', label: 'Opportunities' },
  { id: 'politics', label: 'Politics' },
  { id: 'history', label: 'History' },
  { id: 'scan', label: 'Scan Log' },
  { id: 'pnl', label: 'PnL' },
  { id: 'config', label: 'Config' },
];

export default function Dashboard() {
  const { data: state, connected } = useSSE('/api/dashboard/stream');
  const [activeTab, setActiveTab] = useState('opps');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [hiddenKeys, setHiddenKeys] = useState(() => new Set());
  const [clearing, setClearing] = useState(false);

  const opps = state?.opportunities || [];
  const log = state?.scan_log || [];

  const candidateCount = useMemo(
    () => log.filter((e) => e.pass_yes || e.pass_no).length,
    [log]
  );

  const visibleOpps = opps.filter((o) => !hiddenKeys.has(o.arb_key));

  const clearBlacklist = async () => {
    setClearing(true);
    try {
      await fetch('/api/dashboard/clear-blacklist', { method: 'POST' });
      setHiddenKeys(new Set());
    } catch {}
    setClearing(false);
  };

  return (
    <div className="app">
      <StatusBar
        state={state}
        connected={connected}
        onOpenSettings={() => setSettingsOpen(true)}
      />

      <main className="main">
        <Metrics state={state} />

        <div className="tabbar">
          {TABS.map((t) => {
            let count = null;
            if (t.id === 'opps') count = visibleOpps.length;
            if (t.id === 'scan') count = log.length;
            return (
              <button
                key={t.id}
                className={`tab ${activeTab === t.id ? 'active' : ''}`}
                onClick={() => setActiveTab(t.id)}
              >
                {t.label}
                {count != null && <span className="tab-count">{count}</span>}
              </button>
            );
          })}

          <div className="tabbar-spacer" />

          {activeTab === 'opps' && (
            <button className="filter-btn" onClick={clearBlacklist} disabled={clearing}>
              {clearing ? 'Clearing…' : 'Clear blacklist'}
            </button>
          )}
          {activeTab === 'scan' && (
            <div style={{ fontFamily: 'var(--font-mono)', fontSize: 10.5, color: 'var(--text-muted)', padding: '0 10px', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
              {candidateCount} candidates
            </div>
          )}
        </div>

        <div className="tab-content">
          {activeTab === 'opps' && (
            <OpportunitiesTab
              opportunities={opps}
              hiddenKeys={hiddenKeys}
              onHide={(key) =>
                setHiddenKeys((prev) => {
                  const next = new Set(prev);
                  next.add(key);
                  return next;
                })
              }
            />
          )}
          {activeTab === 'scan' && <ScanLogTab log={log} />}
          {activeTab === 'history' && <HistoryTab />}
          {activeTab === 'pnl' && <PnLTab />}
          {activeTab === 'config' && <ConfigTab />}
          {activeTab === 'politics' && <PoliticsTab opportunities={opps} />}
        </div>
      </main>

      {settingsOpen && (
        <SettingsPanel state={state} onClose={() => setSettingsOpen(false)} />
      )}
    </div>
  );
}
