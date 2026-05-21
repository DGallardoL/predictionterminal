import React, { useState, useRef, useLayoutEffect } from 'react';

const hue = (i) => (i * 137.5) % 360;
const colorLine = (h) => `hsl(${h},65%,55%)`;
const colorBg = (h) => `hsla(${h},60%,55%,0.12)`;
const colorBdr = (h) => `hsla(${h},55%,60%,0.45)`;

/**
 * MappingEditor - drag & drop interface to map Kalshi outcomes to Polymarket outcomes.
 * - Click & drag a Kalshi item onto a Polymarket item to connect.
 * - Click on a connected item to disconnect.
 * - Click on a connection line to remove it.
 *
 * Props:
 *   match: { all_k_outcomes: [{suffix, name, ticker}], all_p_outcomes: [{name}] }
 *   mapping: { [k_suffix]: p_outcome_name }
 *   onChange: (newMapping) => void
 */
export default function MappingEditor({ match, mapping, onChange }) {
  const kItems = match?.all_k_outcomes || [];
  const pItemsRaw = match?.all_p_outcomes || [];
  const areaRef = useRef(null);
  const [positions, setPositions] = useState({});
  const [dragSuffix, setDragSuffix] = useState(null);
  const [mouse, setMouse] = useState(null);
  const dragRef = useRef(null);

  // Reorder Poly: connected items align to their Kalshi row, unconnected at the bottom
  const connectedPNames = new Set(Object.values(mapping));
  const unconnectedP = pItemsRaw.filter((p) => !connectedPNames.has(p.name));
  let unIdx = 0;
  const finalP = [];
  kItems.forEach((k) => {
    const pName = mapping[k.suffix];
    if (pName) {
      const pItem = pItemsRaw.find((p) => p.name === pName);
      finalP.push(pItem || null);
    } else {
      finalP.push(unIdx < unconnectedP.length ? unconnectedP[unIdx++] : null);
    }
  });
  while (unIdx < unconnectedP.length) finalP.push(unconnectedP[unIdx++]);

  // Color per connection
  const entries = Object.entries(mapping);
  const kHue = {};
  const pHue = {};
  entries.forEach(([ks, pn], i) => {
    kHue[ks] = hue(i);
    pHue[pn] = hue(i);
  });

  // Measure positions after render
  useLayoutEffect(() => {
    const el = areaRef.current;
    if (!el) return;
    const t = setTimeout(() => {
      const r = el.getBoundingClientRect();
      const pos = {};
      el.querySelectorAll('[data-mid]').forEach((item) => {
        const ir = item.getBoundingClientRect();
        pos[item.dataset.mid] = {
          right: ir.right - r.left,
          left: ir.left - r.left,
          cy: ir.top + ir.height / 2 - r.top,
        };
      });
      setPositions(pos);
    }, 30);
    return () => clearTimeout(t);
  }, [mapping, kItems.length, pItemsRaw.length]);

  // SVG lines
  const lines = entries
    .map(([ks, pn], ci) => {
      const kp = positions['k-' + ks];
      const pp = positions['p-' + pn];
      if (!kp || !pp) return null;
      const x1 = kp.right;
      const y1 = kp.cy;
      const x2 = pp.left;
      const y2 = pp.cy;
      return {
        d: `M${x1},${y1} C${x1 + 35},${y1} ${x2 - 35},${y2} ${x2},${y2}`,
        x1, y1, x2, y2, ks,
        color: colorLine(hue(ci)),
      };
    })
    .filter(Boolean);

  // Drag temp line
  let dragLine = null;
  if (dragSuffix && mouse) {
    const kp = positions['k-' + dragSuffix];
    if (kp) {
      const x1 = kp.right;
      const y1 = kp.cy;
      dragLine = `M${x1},${y1} C${x1 + 25},${y1} ${mouse.x - 25},${mouse.y} ${mouse.x},${mouse.y}`;
    }
  }

  function onKDown(suffix, e) {
    e.preventDefault();
    if (mapping[suffix]) {
      const next = { ...mapping };
      delete next[suffix];
      onChange(next);
      return;
    }
    dragRef.current = suffix;
    setDragSuffix(suffix);
  }

  function onPUp(name) {
    const src = dragRef.current;
    if (src) {
      const next = { ...mapping, [src]: name };
      // If another K was already mapped to this P, remove that link
      Object.keys(next).forEach((k) => {
        if (k !== src && next[k] === name) delete next[k];
      });
      dragRef.current = null;
      setDragSuffix(null);
      setMouse(null);
      onChange(next);
    }
  }

  function onAreaMove(e) {
    if (!dragRef.current || !areaRef.current) return;
    const r = areaRef.current.getBoundingClientRect();
    setMouse({ x: e.clientX - r.left, y: e.clientY - r.top });
  }

  function onAreaUp() {
    dragRef.current = null;
    setDragSuffix(null);
    setMouse(null);
  }

  function removeLine(ks) {
    const next = { ...mapping };
    delete next[ks];
    onChange(next);
  }

  const svgW = areaRef.current?.offsetWidth || 0;
  const svgH = areaRef.current?.offsetHeight || 0;

  return (
    <div ref={areaRef} className="map-area" onMouseMove={onAreaMove} onMouseUp={onAreaUp} onMouseLeave={onAreaUp}>
      <div className="map-col">
        <div className="map-lbl">Kalshi</div>
        {kItems.map((k) => {
          const h = kHue[k.suffix];
          const conn = h != null;
          return (
            <div
              key={k.suffix}
              data-mid={`k-${k.suffix}`}
              className={`mi${conn ? ' connected' : ''}${dragSuffix === k.suffix ? ' dragging' : ''}`}
              style={conn ? { background: colorBg(h), borderColor: colorBdr(h), color: colorLine(h) } : undefined}
              onMouseDown={(e) => onKDown(k.suffix, e)}
            >
              <span className="tag" style={conn ? { background: colorLine(h) } : undefined}>{k.suffix}</span>
              <span className="mi-name">{k.name}</span>
            </div>
          );
        })}
      </div>

      <div className="map-gap" />

      <div className="map-col">
        <div className="map-lbl">Polymarket</div>
        {finalP.map((p, i) => {
          if (!p) return <div key={`empty-${i}`} className="mi" style={{ visibility: 'hidden' }} />;
          const h = pHue[p.name];
          const conn = h != null;
          return (
            <div
              key={`p-${i}-${p.name}`}
              data-mid={`p-${p.name}`}
              className={`mi${conn ? ' connected' : ''}`}
              style={conn ? { background: colorBg(h), borderColor: colorBdr(h), color: colorLine(h) } : undefined}
              onMouseUp={() => onPUp(p.name)}
              onMouseEnter={(e) => dragSuffix && e.currentTarget.classList.add('drop-over')}
              onMouseLeave={(e) => e.currentTarget.classList.remove('drop-over')}
            >
              <span className="mi-name">{p.name}</span>
            </div>
          );
        })}
      </div>

      <svg className="map-svg" width={svgW} height={svgH} style={{ width: svgW, height: svgH }}>
        {lines.map((l) => (
          <g key={l.ks}>
            <path
              className="conn"
              d={l.d}
              stroke={l.color}
              style={{ filter: `drop-shadow(0 1px 3px ${l.color}55)` }}
              onClick={() => removeLine(l.ks)}
            />
            <circle className="dot" cx={l.x1} cy={l.y1} r={3.5} fill={l.color} />
            <circle className="dot" cx={l.x2} cy={l.y2} r={3.5} fill={l.color} />
          </g>
        ))}
        {dragLine && <path className="drag-line" d={dragLine} />}
      </svg>
    </div>
  );
}
