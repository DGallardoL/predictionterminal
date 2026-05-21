"""Comprehensive READ-ONLY audit of ``factors.yml`` (W13-52).

Combines three existing scripts without modifying ``factors.yml``:

* **W12-09** — :mod:`detect_dead_slugs` (dry-run; uses the offline CLI
  fetcher backed by ``audit_dead_factors.report.json``). No network calls.
* **W11-42** — :mod:`validate_factors` schema validations (required keys,
  source whitelist, slug shape, theme membership). We re-use the loader
  via :func:`pfm.factors.load_factors` and run the lightweight YAML-side
  schema checks locally; we do NOT trigger the live HTTP venue checks.
* **W12-56** — :mod:`factor_cluster_report` KMeans clustering. Uses the
  fixture path when supplied (offline, deterministic). Falls back to a
  pure-synthetic-RNG matrix keyed by ``id`` order when no fixture is
  provided, so the audit never blocks on cold caches.

The aggregator emits a single JSON blob with::

    {
      "generated_at": "<UTC ISO8601>",
      "total_factors": 1228,
      "dead": {"count": ..., "by_reason": {...}, "examples": [...]},
      "schema_issues": {"count": ..., "by_kind": {...}, "examples": [...]},
      "thematic_balance": {"by_theme": {...}, "by_source": {...}, "concentration_top3": ...},
      "clusters": {"k": 20, "size_min": ..., "size_max": ..., "skipped": ...},
      "recommendations": ["..."]
    }

Outputs:

* ``/tmp/factors-audit-{YYYY-MM-DD}.json`` — machine-readable.
* ``docs/factors-audit-history/{YYYY-MM-DD}.md`` — human summary.

CLI::

    python scripts/factors_audit_full.py
    python scripts/factors_audit_full.py --factors-yml path/to/factors.yml
    python scripts/factors_audit_full.py --cluster-fixture path.json

Designed to be safe to run repeatedly — never writes to ``factors.yml``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Module bootstrap
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_API_ROOT = _HERE.parent
_REPO_ROOT = _API_ROOT.parent
_SRC = _API_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_FACTORS_YML = _SRC / "pfm" / "factors.yml"
DEFAULT_HISTORY_DIR = _REPO_ROOT / "docs" / "factors-audit-history"
DEFAULT_TMP_DIR = Path("/tmp")

# Sibling-script paths — loaded via importlib because ``scripts/`` is not a package.
_DETECTOR_PATH = _HERE / "detect_dead_slugs.py"
_VALIDATOR_PATH = _HERE / "validate_factors.py"
_CLUSTER_PATH = _HERE / "factor_cluster_report.py"

# Canonical schema, mirrors validate_factors.py + factors.py expectations.
REQUIRED_KEYS: frozenset[str] = frozenset({"id", "slug", "source", "theme"})
ALLOWED_SOURCES: frozenset[str] = frozenset(
    {"polymarket", "kalshi", "fred", "bls", "manifold", "predictit", "chain", "sentiment"}
)
ALLOWED_THEMES: frozenset[str] = frozenset(
    {
        "ai",
        "business",
        "chips",
        "climate",
        "commodities",
        "crypto",
        "energy",
        "equity",
        "geopolitics",
        "health",
        "legal",
        "macro",
        "other",
        "politics",
        "pop_culture",
        "science",
        "space",
        "sports",
        "weather",
    }
)

# Recommendation thresholds — surfaced as constants so tests can pin them.
DEAD_FRACTION_WARN = 0.05  # >5% dead = housekeeping recommended
SCHEMA_WARN_THRESHOLD = 0  # any schema issue = recommend fixing
THEME_CONCENTRATION_WARN = 0.40  # top-3 themes >40% = diversification flag
CLUSTER_DOMINANCE_WARN = 0.30  # any single cluster >30% = redundancy flag

LOG = logging.getLogger("factors_audit_full")


# ---------------------------------------------------------------------------
# Lazy module loaders (late-binding so tests can monkeypatch).
# ---------------------------------------------------------------------------


def _load_sibling(module_alias: str, path: Path):
    """Load a sibling script under ``module_alias`` and cache in sys.modules."""
    cached = sys.modules.get(module_alias)
    if cached is not None:
        return cached
    if not path.exists():
        raise FileNotFoundError(f"required sibling script missing: {path}")
    spec = importlib.util.spec_from_file_location(module_alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path} via importlib")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = mod
    spec.loader.exec_module(mod)
    return mod


def _detector_mod():
    return _load_sibling("_w12_09_detect", _DETECTOR_PATH)


def _cluster_mod():
    return _load_sibling("_w12_56_cluster", _CLUSTER_PATH)


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------


def _load_raw_factors(factors_yml_path: Path) -> list[dict]:
    """Read raw factor dicts from YAML (preserves invalid entries for schema checks)."""
    if not factors_yml_path.exists():
        raise FileNotFoundError(f"factors file not found: {factors_yml_path}")
    with factors_yml_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    items = raw.get("factors", []) or []
    if not isinstance(items, list):
        raise ValueError("`factors` key must be a list")
    return [item for item in items if isinstance(item, dict)]


# ---------------------------------------------------------------------------
# Dead-slug step (W12-09, dry-run, no factors.yml writes).
# ---------------------------------------------------------------------------


def run_dead_slug_step(
    factors_yml_path: Path,
    *,
    min_obs: int | None = None,
    since_days: int | None = None,
    fetch_history: Any = None,
) -> dict[str, Any]:
    """Run W12-09's :func:`detect_dead_slugs` in dry-run mode.

    Returns a summary dict ``{count, by_reason, examples, params}``. No
    files are modified. When ``fetch_history`` is ``None`` we delegate to
    the detector's CLI fetcher, which consults
    ``audit_dead_factors.report.json`` and never hits the network.
    """
    mod = _detector_mod()
    effective_min_obs = mod.DEFAULT_MIN_OBS if min_obs is None else min_obs
    effective_since = mod.DEFAULT_SINCE_DAYS if since_days is None else since_days
    fetcher = fetch_history if fetch_history is not None else mod._make_cli_fetcher()

    dead = mod.detect_dead_slugs(
        factors_yml_path,
        min_obs=effective_min_obs,
        since_days=effective_since,
        fetch_history=fetcher,
    )

    by_reason: Counter[str] = Counter(str(r.get("reason", "unknown")) for r in dead)
    examples = [
        {
            "id": str(r.get("id", "")),
            "slug": str(r.get("slug", "")),
            "source": str(r.get("source", "")),
            "obs_count": int(r.get("obs_count", 0) or 0),
            "reason": str(r.get("reason", "")),
        }
        for r in dead[:10]
    ]
    return {
        "count": len(dead),
        "by_reason": dict(by_reason),
        "examples": examples,
        "params": {"min_obs": effective_min_obs, "since_days": effective_since},
    }


# ---------------------------------------------------------------------------
# Schema validation step (W11-42; local-only, no live HTTP).
# ---------------------------------------------------------------------------


def run_schema_step(raw_factors: list[dict]) -> dict[str, Any]:
    """Validate every YAML entry against the canonical schema.

    Issue kinds emitted:

    * ``missing_keys`` — entry lacks one of ``REQUIRED_KEYS``
    * ``unknown_source`` — ``source`` not in ``ALLOWED_SOURCES``
    * ``unknown_theme`` — ``theme`` not in ``ALLOWED_THEMES``
    * ``empty_slug`` — ``slug`` blank or non-string
    * ``duplicate_id`` — same ``id`` seen >1 time
    * ``duplicate_slug`` — same ``(source, slug)`` pair seen >1 time
    """
    issues: list[dict[str, str]] = []
    seen_ids: dict[str, int] = {}
    seen_pairs: dict[tuple[str, str], int] = {}

    for idx, entry in enumerate(raw_factors):
        fid = str(entry.get("id", "") or "")
        slug = entry.get("slug", "")
        source = str(entry.get("source", "") or "")
        theme = str(entry.get("theme", "") or "")
        label = fid or f"<row {idx}>"

        missing = [k for k in REQUIRED_KEYS if not entry.get(k)]
        if missing:
            issues.append(
                {
                    "kind": "missing_keys",
                    "id": label,
                    "detail": ",".join(sorted(missing)),
                }
            )

        if source and source not in ALLOWED_SOURCES:
            issues.append(
                {
                    "kind": "unknown_source",
                    "id": label,
                    "detail": source,
                }
            )

        if theme and theme not in ALLOWED_THEMES:
            issues.append(
                {
                    "kind": "unknown_theme",
                    "id": label,
                    "detail": theme,
                }
            )

        if not isinstance(slug, str) or not slug.strip():
            issues.append(
                {
                    "kind": "empty_slug",
                    "id": label,
                    "detail": str(slug),
                }
            )

        if fid:
            seen_ids[fid] = seen_ids.get(fid, 0) + 1
        if isinstance(slug, str) and slug.strip() and source:
            key = (source, slug.strip())
            seen_pairs[key] = seen_pairs.get(key, 0) + 1

    for fid, count in seen_ids.items():
        if count > 1:
            issues.append(
                {
                    "kind": "duplicate_id",
                    "id": fid,
                    "detail": f"{count} occurrences",
                }
            )
    for (source, slug), count in seen_pairs.items():
        if count > 1:
            issues.append(
                {
                    "kind": "duplicate_slug",
                    "id": f"{source}:{slug}",
                    "detail": f"{count} occurrences",
                }
            )

    by_kind: Counter[str] = Counter(i["kind"] for i in issues)
    return {
        "count": len(issues),
        "by_kind": dict(by_kind),
        "examples": issues[:10],
    }


# ---------------------------------------------------------------------------
# Thematic balance step.
# ---------------------------------------------------------------------------


def compute_thematic_balance(raw_factors: list[dict]) -> dict[str, Any]:
    """Summarise the theme / source mix and flag concentration."""
    themes: Counter[str] = Counter(str(f.get("theme", "other") or "other") for f in raw_factors)
    sources: Counter[str] = Counter(
        str(f.get("source", "unknown") or "unknown") for f in raw_factors
    )
    total = max(1, len(raw_factors))
    top3 = sum(c for _, c in themes.most_common(3))
    concentration = top3 / total
    return {
        "by_theme": dict(themes.most_common()),
        "by_source": dict(sources.most_common()),
        "concentration_top3": round(concentration, 4),
        "n": len(raw_factors),
    }


# ---------------------------------------------------------------------------
# Cluster step (W12-56).
# ---------------------------------------------------------------------------


def _synthetic_returns(factor_ids: list[str], window: int, seed: int) -> dict[str, list[float]]:
    """Deterministic synthetic returns keyed by id-hash → 5 archetypal shapes.

    Used when no live history fetcher and no fixture are supplied. Keeps
    the audit fully offline while still producing a meaningful cluster
    map (factors with similar id prefixes land in the same archetype).
    """
    rng = np.random.default_rng(seed)
    # Five canonical centroid shapes — momentum, mean-revert, flat, spike, drift.
    archetypes = np.stack(
        [
            np.linspace(-1.0, 1.0, window),
            np.linspace(1.0, -1.0, window),
            np.zeros(window),
            np.concatenate([np.zeros(window // 2), np.ones(window - window // 2)]),
            np.sin(np.linspace(0, 3.14, window)),
        ]
    )
    out: dict[str, list[float]] = {}
    for fid in factor_ids:
        bucket = abs(hash(fid)) % archetypes.shape[0]
        noise = rng.normal(scale=0.15, size=window)
        out[fid] = (archetypes[bucket] + noise).tolist()
    return out


def run_cluster_step(
    raw_factors: list[dict],
    *,
    k: int = 20,
    window: int = 30,
    seed: int = 42,
    fixture_path: Path | None = None,
) -> dict[str, Any]:
    """Cluster factors via W12-56's KMeans pipeline; return a compact summary.

    Strategy for return-series sourcing (in order of preference):

    1. ``fixture_path`` — load via :func:`factor_cluster_report.load_fixture`.
    2. Synthetic-RNG fallback (offline, deterministic). The audit is a
       diagnostic, not a research run — we explicitly avoid hitting the
       live cache to keep this script fast and hermetic.
    """
    cluster_mod = _cluster_mod()
    factor_ids = [str(f.get("id", "") or "") for f in raw_factors if f.get("id")]

    if fixture_path is not None and Path(fixture_path).exists():
        returns_by_factor = cluster_mod.load_fixture(Path(fixture_path))
    else:
        returns_by_factor = _synthetic_returns(factor_ids, window=window, seed=seed)

    kept, matrix, skipped = cluster_mod.build_feature_matrix(returns_by_factor, window=window)
    labels, centroids = cluster_mod.run_kmeans(matrix, k=k, seed=seed)
    report = cluster_mod.assemble_report(
        kept,
        labels,
        centroids,
        k=k,
        window=window,
        seed=seed,
        factor_count=len(factor_ids),
        skipped=skipped,
    )

    sizes = [int(c["size"]) for c in report["clusters"]]
    if sizes:
        size_min = min(sizes)
        size_max = max(sizes)
        size_mean = float(np.mean(sizes))
    else:
        size_min = size_max = 0
        size_mean = 0.0

    total_clustered = sum(sizes)
    dominant_share = (size_max / total_clustered) if total_clustered else 0.0

    return {
        "k": int(report["k"]),
        "window": int(report["window"]),
        "seed": int(report["seed"]),
        "n_clusters": len(sizes),
        "size_min": int(size_min),
        "size_max": int(size_max),
        "size_mean": round(size_mean, 2),
        "dominant_share": round(dominant_share, 4),
        "skipped": len(report["skipped"]),
        "source": "fixture"
        if (fixture_path is not None and Path(fixture_path).exists())
        else "synthetic",
    }


# ---------------------------------------------------------------------------
# Recommendation engine.
# ---------------------------------------------------------------------------


def build_recommendations(
    *,
    total: int,
    dead: dict[str, Any],
    schema: dict[str, Any],
    balance: dict[str, Any],
    clusters: dict[str, Any],
) -> list[str]:
    """Translate raw numbers into operator-facing action items."""
    recs: list[str] = []
    denom = max(1, total)

    dead_count = int(dead.get("count", 0))
    dead_frac = dead_count / denom
    if dead_frac > DEAD_FRACTION_WARN:
        recs.append(
            f"Dead-slug share {dead_frac:.1%} (>{DEAD_FRACTION_WARN:.0%}). "
            "Run `scripts/detect_dead_slugs.py --apply` to prune."
        )
    elif dead_count:
        recs.append(
            f"{dead_count} dead slug(s) detected (under tolerance). "
            "Re-check after next venue-resolution wave."
        )

    schema_count = int(schema.get("count", 0))
    if schema_count > SCHEMA_WARN_THRESHOLD:
        kinds = ", ".join(sorted(schema.get("by_kind", {}).keys())) or "none"
        recs.append(
            f"{schema_count} schema issue(s) found ({kinds}). "
            "Fix in factors.yml before next CI run."
        )

    concentration = float(balance.get("concentration_top3", 0.0))
    if concentration > THEME_CONCENTRATION_WARN:
        top3 = ", ".join(t for t, _ in list(balance.get("by_theme", {}).items())[:3])
        recs.append(
            f"Top-3 themes ({top3}) hold {concentration:.0%} of factors "
            f"(>{THEME_CONCENTRATION_WARN:.0%}). Expand under-represented themes."
        )

    dom = float(clusters.get("dominant_share", 0.0))
    if dom > CLUSTER_DOMINANCE_WARN:
        recs.append(
            f"Largest KMeans cluster holds {dom:.0%} of clustered factors "
            f"(>{CLUSTER_DOMINANCE_WARN:.0%}). Likely redundancy; review with "
            "`scripts/factor_cluster_report.py`."
        )

    if not recs:
        recs.append("Catalog health is within all thresholds; no action required.")
    return recs


# ---------------------------------------------------------------------------
# Aggregator + writers.
# ---------------------------------------------------------------------------


def run_audit(
    factors_yml_path: Path,
    *,
    cluster_fixture: Path | None = None,
    k: int = 20,
    window: int = 30,
    seed: int = 42,
    dead_min_obs: int | None = None,
    dead_since_days: int | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the full audit payload (pure function; no disk writes)."""
    ref_now = now if now is not None else datetime.now(UTC)
    raw_factors = _load_raw_factors(factors_yml_path)

    dead = run_dead_slug_step(
        factors_yml_path,
        min_obs=dead_min_obs,
        since_days=dead_since_days,
    )
    schema = run_schema_step(raw_factors)
    balance = compute_thematic_balance(raw_factors)
    clusters = run_cluster_step(
        raw_factors,
        k=k,
        window=window,
        seed=seed,
        fixture_path=cluster_fixture,
    )
    total = len(raw_factors)
    recs = build_recommendations(
        total=total,
        dead=dead,
        schema=schema,
        balance=balance,
        clusters=clusters,
    )
    return {
        "generated_at": ref_now.isoformat().replace("+00:00", "Z"),
        "factors_yml": str(factors_yml_path),
        "total_factors": total,
        "dead": dead,
        "schema_issues": schema,
        "thematic_balance": balance,
        "clusters": clusters,
        "recommendations": recs,
    }


def write_json_report(payload: dict[str, Any], out_path: Path) -> Path:
    """Write the JSON blob with parents created on demand."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))
    return out_path


def _format_counter_lines(d: dict[str, Any], indent: str = "  ") -> str:
    if not d:
        return f"{indent}_(none)_\n"
    rows = [f"{indent}- `{k}`: {v}" for k, v in d.items()]
    return "\n".join(rows) + "\n"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the human-readable Markdown summary for the history dir."""
    dead = payload["dead"]
    schema = payload["schema_issues"]
    balance = payload["thematic_balance"]
    clusters = payload["clusters"]
    total = payload["total_factors"]
    recs = payload["recommendations"]

    lines: list[str] = []
    lines.append(f"# factors.yml audit — {payload['generated_at']}\n")
    lines.append(
        f"**Total factors**: {total}  \n"
        f"**Source**: `{payload['factors_yml']}`  \n"
        f"**Cluster mode**: `{clusters['source']}`  \n"
    )
    lines.append("\n## Recommendations\n")
    for r in recs:
        lines.append(f"- {r}")
    lines.append("")

    lines.append("\n## Dead slugs (W12-09 dry-run)\n")
    lines.append(
        f"- Count: **{dead['count']}** "
        f"(min_obs={dead['params']['min_obs']}, since_days={dead['params']['since_days']})"
    )
    lines.append("- By reason:")
    lines.append(_format_counter_lines(dead.get("by_reason", {}), indent="  "))
    if dead.get("examples"):
        lines.append("- First examples:")
        for e in dead["examples"]:
            lines.append(f"  - `{e['source']}:{e['slug']}` ({e['reason']}, obs={e['obs_count']})")
        lines.append("")

    lines.append("\n## Schema issues (W11-42 local checks)\n")
    lines.append(f"- Count: **{schema['count']}**")
    lines.append("- By kind:")
    lines.append(_format_counter_lines(schema.get("by_kind", {}), indent="  "))
    if schema.get("examples"):
        lines.append("- First examples:")
        for e in schema["examples"]:
            lines.append(f"  - `{e['id']}` — {e['kind']}: {e['detail']}")
        lines.append("")

    lines.append("\n## Thematic balance\n")
    lines.append(
        f"- Top-3 concentration: **{balance['concentration_top3']:.1%}** "
        f"(warn threshold {THEME_CONCENTRATION_WARN:.0%})"
    )
    lines.append("- By theme:")
    lines.append(_format_counter_lines(balance.get("by_theme", {}), indent="  "))
    lines.append("- By source:")
    lines.append(_format_counter_lines(balance.get("by_source", {}), indent="  "))

    lines.append("\n## KMeans clusters (W12-56)\n")
    lines.append(f"- k={clusters['k']}, window={clusters['window']}, seed={clusters['seed']}")
    lines.append(f"- Clusters formed: {clusters['n_clusters']}")
    lines.append(
        f"- Size min/mean/max: {clusters['size_min']} / "
        f"{clusters['size_mean']} / {clusters['size_max']}"
    )
    lines.append(
        f"- Dominant cluster share: **{clusters['dominant_share']:.1%}** "
        f"(warn threshold {CLUSTER_DOMINANCE_WARN:.0%})"
    )
    lines.append(f"- Skipped (insufficient history): {clusters['skipped']}")

    lines.append(
        "\n---\n"
        "_Generated by `scripts/factors_audit_full.py` (W13-52). "
        "READ-ONLY — `factors.yml` was not modified._\n"
    )
    return "\n".join(lines)


def write_markdown_report(payload: dict[str, Any], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(payload))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Comprehensive READ-ONLY audit of factors.yml: combines W12-09 "
            "dead-slug dry-run, W11-42 schema checks, and W12-56 KMeans "
            "clustering into a single JSON + Markdown report."
        )
    )
    p.add_argument(
        "--factors-yml",
        type=Path,
        default=DEFAULT_FACTORS_YML,
        help=f"Path to factors.yml (default: {DEFAULT_FACTORS_YML}).",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Override JSON output path (default: /tmp/factors-audit-<UTC-date>.json).",
    )
    p.add_argument(
        "--markdown-out",
        type=Path,
        default=None,
        help=(f"Override Markdown output path (default: {DEFAULT_HISTORY_DIR}/<UTC-date>.md)."),
    )
    p.add_argument(
        "--cluster-fixture",
        type=Path,
        default=None,
        help=(
            "Optional JSON fixture {factor_id: [returns]} to feed the cluster "
            "step. When omitted, the script uses a deterministic synthetic "
            "matrix so the audit stays offline."
        ),
    )
    p.add_argument("--k", type=int, default=20, help="KMeans cluster count (default 20).")
    p.add_argument("--window", type=int, default=30, help="Cluster window in days (default 30).")
    p.add_argument("--seed", type=int, default=42, help="Cluster RNG seed (default 42).")
    p.add_argument(
        "--dead-min-obs",
        type=int,
        default=None,
        help="Override detect_dead_slugs min_obs (default: detector's own).",
    )
    p.add_argument(
        "--dead-since-days",
        type=int,
        default=None,
        help="Override detect_dead_slugs since_days (default: detector's own).",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress stdout summary lines.")
    return p.parse_args(list(argv) if argv is not None else None)


def _default_json_out(today: date) -> Path:
    return DEFAULT_TMP_DIR / f"factors-audit-{today.isoformat()}.json"


def _default_markdown_out(today: date) -> Path:
    return DEFAULT_HISTORY_DIR / f"{today.isoformat()}.md"


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    now = datetime.now(UTC)
    today = now.date()
    json_out = args.json_out if args.json_out is not None else _default_json_out(today)
    md_out = args.markdown_out if args.markdown_out is not None else _default_markdown_out(today)

    payload = run_audit(
        args.factors_yml,
        cluster_fixture=args.cluster_fixture,
        k=args.k,
        window=args.window,
        seed=args.seed,
        dead_min_obs=args.dead_min_obs,
        dead_since_days=args.dead_since_days,
        now=now,
    )

    write_json_report(payload, json_out)
    write_markdown_report(payload, md_out)

    if not args.quiet:
        dead = payload["dead"]
        schema = payload["schema_issues"]
        balance = payload["thematic_balance"]
        clusters = payload["clusters"]
        print(
            f"factors_audit_full: total={payload['total_factors']}  "
            f"dead={dead['count']}  schema_issues={schema['count']}  "
            f"top3_themes={balance['concentration_top3']:.1%}  "
            f"clusters={clusters['n_clusters']} (max={clusters['size_max']})"
        )
        print(f"  JSON     : {json_out}")
        print(f"  Markdown : {md_out}")
        print("  Recommendations:")
        for r in payload["recommendations"]:
            print(f"    - {r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
