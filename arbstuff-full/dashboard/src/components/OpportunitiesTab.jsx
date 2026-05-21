import React, { useEffect, useState } from 'react';
import { IconExt } from './icons.jsx';

function profitClass(p) {
  if (p >= 5) return 'high';
  if (p >= 2) return 'med';
  return 'low';
}

function OppRow({ opp, rank, selected, onSelect }) {
  return (
    <div
      className={`opp-row ${selected ? 'selected' : ''}`}
      onClick={() => onSelect(opp.arb_key)}
    >
      <div className="opp-rank">#{rank}</div>
      <div className="opp-name">
        <div className="opp-name-main">
          {opp.name}
          {opp.neg_risk && <span className="badge nr">NR</span>}
          {opp.source === 'discovered' && <span className="badge source-discovered" style={{ fontSize: 8 }}>DISC</span>}
        </div>
        <div className="opp-name-sub">{opp.type}</div>
      </div>
      <div className={`opp-profit ${profitClass(opp.profit_pct)}`}>
        +{opp.profit_pct.toFixed(2)}%
      </div>
      <div className="opp-num">{opp.volume.toLocaleString()}</div>
      <div className="opp-num">${opp.cost.toFixed(3)}</div>
      <div className="opp-prices">
        <span className="k">K ${opp.kalshi_price.toFixed(2)}</span>
        {' / '}
        <span className="p">P ${opp.poly_price.toFixed(2)}</span>
      </div>
      <div className="opp-actions" onClick={(e) => e.stopPropagation()}>
        <a
          className="k"
          href={`https://kalshi.com/events/${opp.kalshi_event_ticker}`}
          target="_blank"
          rel="noreferrer"
          title="Open on Kalshi"
        >
          K
        </a>
        <a
          className="p"
          href={`https://polymarket.com/event/${opp.poly_slug}`}
          target="_blank"
          rel="noreferrer"
          title="Open on Polymarket"
        >
          P
        </a>
      </div>
    </div>
  );
}

function PriceBar({ side, label, price }) {
  const pct = Math.min(100, Math.max(2, price * 100));
  return (
    <div className="price-bar-row">
      <div className="price-bar-lbl">{label}</div>
      <div className="price-bar-track">
        <div
          className={`price-bar-fill ${side}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="price-bar-val">${price.toFixed(3)}</div>
    </div>
  );
}

function Orderbook({ title, data, side }) {
  if (!data || !data.levels) {
    return (
      <div className="orderbook">
        <div className="orderbook-title">
          <span>{title}</span>
          <span>—</span>
        </div>
        <div style={{ padding: '16px 0', textAlign: 'center', color: 'var(--text-muted)', fontSize: 10 }}>
          no orderbook
        </div>
      </div>
    );
  }

  const { bids = [], asks = [], spread, mid } = data;
  const maxSize = Math.max(
    ...bids.map((l) => l.size || 0),
    ...asks.map((l) => l.size || 0),
    1
  );

  // Show top 5 asks (reversed - lowest ask near middle) and top 5 bids
  const topAsks = asks.slice(0, 5).reverse();
  const topBids = bids.slice(0, 5);

  return (
    <div className="orderbook">
      <div className="orderbook-title">
        <span>{title}</span>
        <span>{asks.length}×{bids.length}</span>
      </div>
      <div className="ob-levels">
        {topAsks.map((lvl, i) => (
          <div key={`a${i}`} className="ob-level ask">
            <span className="price">${lvl.price.toFixed(3)}</span>
            <div className="ob-bar" style={{ width: `${(lvl.size / maxSize) * 100}%` }} />
            <span className="size">{lvl.size}</span>
          </div>
        ))}
        <div className="ob-spread">
          <span>spread <span className="mono">{(spread ?? 0).toFixed(3)}</span></span>
          <span className="mid">mid ${(mid ?? 0).toFixed(3)}</span>
        </div>
        {topBids.map((lvl, i) => (
          <div key={`b${i}`} className="ob-level bid">
            <span className="price">${lvl.price.toFixed(3)}</span>
            <div className="ob-bar" style={{ width: `${(lvl.size / maxSize) * 100}%` }} />
            <span className="size">{lvl.size}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Detail({ opp, onReport }) {
  const [ob, setOb] = useState(null);
  const [obLoading, setObLoading] = useState(false);
  const [reporting, setReporting] = useState(false);

  useEffect(() => {
    if (!opp) return;
    let cancelled = false;
    setObLoading(true);
    setOb(null);
    fetch(
      `/api/dashboard/orderbook?kalshi_ticker=${encodeURIComponent(
        opp.kalshi_ticker
      )}&poly_token=${encodeURIComponent(opp.poly_token_id)}`
    )
      .then((r) => r.ok ? r.json() : null)
      .then((data) => {
        if (!cancelled) setOb(data);
      })
      .catch(() => {})
      .finally(() => {
        if (!cancelled) setObLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [opp?.arb_key]);

  if (!opp) {
    return (
      <div className="empty">
        <div className="empty-title">Select an opportunity</div>
        <div className="empty-sub">
          Choose a row on the left to inspect pricing, fees, and orderbook depth.
        </div>
      </div>
    );
  }

  const kFee = opp.kalshi_fee ?? 0.07 * opp.kalshi_price * (1 - opp.kalshi_price);
  const pFee = opp.poly_fee ?? 0.04 * opp.poly_price * (1 - opp.poly_price);
  const totalWithFees = opp.cost + kFee + pFee;

  const handleReport = async () => {
    setReporting(true);
    try {
      await fetch('/api/dashboard/blacklist', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ arb_key: opp.arb_key }),
      });
      onReport?.(opp.arb_key);
    } catch {}
    setReporting(false);
  };

  return (
    <div className="detail">
      <div className="detail-title">{opp.name}</div>

      <div className="detail-section">
        <div className="detail-label">Strategy</div>
        <div className="row">
          <div className="strategy-tag">
            {opp.type}
            {opp.neg_risk && <span className="badge nr">NR</span>}
          </div>
          <div className={`opp-profit ${profitClass(opp.profit_pct)}`} style={{ marginLeft: 'auto' }}>
            +{opp.profit_pct.toFixed(2)}%
          </div>
        </div>
      </div>

      <div className="detail-section">
        <div className="detail-label">Prices</div>
        <div className="price-bars">
          <PriceBar side="k" label="Kalshi" price={opp.kalshi_price} />
          <PriceBar side="p" label="Poly" price={opp.poly_price} />
        </div>
      </div>

      <div className="detail-section">
        <div className="detail-label">Cost & Fees</div>
        <dl className="fee-table">
          <dt>Combined cost</dt>
          <dd>${opp.cost.toFixed(4)}</dd>
          <dt>Spread</dt>
          <dd>{(opp.spread ?? 0).toFixed(3)}</dd>
          <dt>Kalshi fee <span style={{ color: 'var(--text-faint)' }}>0.07p(1−p)</span></dt>
          <dd>${kFee.toFixed(4)}</dd>
          <dt>Poly fee <span style={{ color: 'var(--text-faint)' }}>0.04p(1−p)</span></dt>
          <dd>${pFee.toFixed(4)}</dd>
          <dt className="total">Total w/ fees</dt>
          <dd className="total">${totalWithFees.toFixed(4)}</dd>
          <dt>Profit</dt>
          <dd className="green">+{opp.profit_pct.toFixed(2)}%</dd>
          {opp.volume > 0 && (
            <>
              <dt>Volume</dt>
              <dd>{opp.volume.toLocaleString()}</dd>
            </>
          )}
        </dl>
      </div>

      <div className="detail-section">
        <div className="detail-label">
          Orderbook
          {obLoading && <span className="spinner" style={{ marginLeft: 8 }} />}
        </div>
        <div className="orderbook-wrap">
          <Orderbook title="Kalshi" data={ob?.kalshi} side="k" />
          <Orderbook title="Polymarket" data={ob?.polymarket} side="p" />
        </div>
      </div>

      <div className="detail-actions">
        <a
          className="btn"
          href={`https://kalshi.com/events/${opp.kalshi_event_ticker}`}
          target="_blank"
          rel="noreferrer"
        >
          <IconExt size={12} /> Open Kalshi
        </a>
        <a
          className="btn"
          href={`https://polymarket.com/event/${opp.poly_slug}`}
          target="_blank"
          rel="noreferrer"
        >
          <IconExt size={12} /> Open Polymarket
        </a>
        <button
          className="btn danger"
          onClick={handleReport}
          disabled={reporting}
          style={{ marginLeft: 'auto' }}
        >
          {reporting ? 'Reporting…' : 'Report as not arb'}
        </button>
      </div>
    </div>
  );
}

export default function OpportunitiesTab({ opportunities, hiddenKeys, onHide }) {
  const [selectedKey, setSelectedKey] = useState(null);
  const [sortBy, setSortBy] = useState('profit'); // profit | cost | volume
  const [sortDir, setSortDir] = useState('desc'); // desc | asc
  const [sourceTab, setSourceTab] = useState('manual'); // 'manual' | 'discovered' | 'all'

  const all = (opportunities || []).filter((o) => !hiddenKeys.has(o.arb_key));
  const manualOpps = all.filter((o) => o.source !== 'discovered');
  const discoveredOpps = all.filter((o) => o.source === 'discovered');

  const sourceFiltered =
    sourceTab === 'all' ? all
    : sourceTab === 'discovered' ? discoveredOpps
    : manualOpps;

  const visible = sourceFiltered
    .slice()
    .sort((a, b) => {
      let cmp = 0;
      if (sortBy === 'profit') cmp = a.profit_pct - b.profit_pct;
      else if (sortBy === 'cost') cmp = a.cost - b.cost;
      else if (sortBy === 'volume') cmp = (a.volume || 0) - (b.volume || 0);
      return sortDir === 'asc' ? cmp : -cmp;
    });

  // Keep selection valid across updates
  useEffect(() => {
    if (selectedKey && !visible.some((o) => o.arb_key === selectedKey)) {
      setSelectedKey(null);
    }
    if (!selectedKey && visible.length > 0) {
      setSelectedKey(visible[0].arb_key);
    }
  }, [visible, selectedKey]);

  const selected = visible.find((o) => o.arb_key === selectedKey) || null;

  const toggleSort = (col) => {
    if (sortBy === col) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortBy(col);
      // cost defaults to ascending (cheapest = best arb), others to descending
      setSortDir(col === 'cost' ? 'asc' : 'desc');
    }
  };

  const arrow = (col) => (sortBy === col ? (sortDir === 'asc' ? ' ↑' : ' ↓') : '');

  return (
    <div className="split">
      <div className="panel">
        <div className="panel-head" style={{ flexDirection: 'column', alignItems: 'stretch', gap: 10 }}>
          <div className="row" style={{ width: '100%' }}>
            <div className="panel-title">
              <span className="serif">Live Opportunities</span>
            </div>
            <div className="row flex-end" style={{ gap: 8 }}>
              <div className="sort-group">
                <button
                  className={`sort-pill ${sortBy === 'profit' ? 'active' : ''}`}
                  onClick={() => toggleSort('profit')}
                >
                  Profit{arrow('profit')}
                </button>
                <button
                  className={`sort-pill ${sortBy === 'volume' ? 'active' : ''}`}
                  onClick={() => toggleSort('volume')}
                >
                  Vol{arrow('volume')}
                </button>
                <button
                  className={`sort-pill ${sortBy === 'cost' ? 'active' : ''}`}
                  onClick={() => toggleSort('cost')}
                >
                  Cost{arrow('cost')}
                </button>
              </div>
              <div className="mono" style={{ fontSize: 11, color: 'var(--text-dim)' }}>
                {visible.length} found
              </div>
            </div>
          </div>
          <div className="source-tabs">
            <button
              className={`source-tab ${sourceTab === 'manual' ? 'active' : ''}`}
              onClick={() => setSourceTab('manual')}
            >
              Manual <span className="source-tab-count">{manualOpps.length}</span>
            </button>
            <button
              className={`source-tab ${sourceTab === 'discovered' ? 'active discovered' : ''}`}
              onClick={() => setSourceTab('discovered')}
            >
              Discovered <span className="source-tab-count">{discoveredOpps.length}</span>
            </button>
            <button
              className={`source-tab ${sourceTab === 'all' ? 'active' : ''}`}
              onClick={() => setSourceTab('all')}
            >
              All <span className="source-tab-count">{all.length}</span>
            </button>
          </div>
        </div>
        <div className="panel-body">
          {visible.length === 0 ? (
            <div className="empty">
              <div className="empty-title">No arbitrage opportunities right now</div>
              <div className="empty-sub">
                The scanner will update this list as soon as a cross-market edge appears.
              </div>
            </div>
          ) : (
            <div className="opp-table">
              {visible.map((o, i) => (
                <OppRow
                  key={o.arb_key}
                  opp={o}
                  rank={i + 1}
                  selected={selectedKey === o.arb_key}
                  onSelect={setSelectedKey}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <div className="panel-title">
            <span className="serif">Detail</span>
          </div>
        </div>
        <div className="panel-body">
          <Detail opp={selected} onReport={onHide} />
        </div>
      </div>
    </div>
  );
}
