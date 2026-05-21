"""Audit current live arb pairs for false positives.

Task T78 — Track-I (Arb matching quality).

For every active arb pair (sourced from ``arbstuff/dashboard_state.json`` when
available, otherwise from :func:`pfm.arb_scanner.top_arbs`), builds a
``MarketDesc`` for each leg, scores the pair via the new
:func:`pfm.arb_matching.event_similarity.score_match` matcher (T76 + T77),
and writes a confusion-matrix CSV at
``/tmp/arb-match-audit-YYYYMMDD.csv``.

Usage::

    python api/scripts/audit_arb_matches.py
    python api/scripts/audit_arb_matches.py --apply-blacklist
    python api/scripts/audit_arb_matches.py --source arbstuff
    python api/scripts/audit_arb_matches.py --source top_arbs --top-n 25

Outputs::

    /tmp/arb-match-audit-YYYYMMDD.csv          (always)
    /tmp/arb-blacklist-proposals.json          (only with --apply-blacklist)

A non-zero exit code means at least one rejected pair was found AND the
script was invoked with ``--fail-on-reject`` (CI use). The default behaviour
is to print the audit summary and exit 0 — humans review the proposals.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Hard dependencies — defer import so the script can print a helpful error if
# the T76/T77 modules have not been written yet.
# ---------------------------------------------------------------------------


def _import_matchers():
    """Import T76+T77 matchers, raising a *helpful* error if they're missing.

    Returns a 3-tuple ``(MarketDesc, score_match, build_market_desc)``. The
    third element is the T77-native ``build_market_desc(raw_payload, venue)``
    helper; we prefer it over hand-constructing the dataclass because T77 may
    evolve its fields.
    """
    try:
        from pfm.arb_matching.event_similarity import (  # type: ignore[import-not-found]
            MarketDesc,
            build_market_desc,
            score_match,
        )
    except Exception as exc:  # pragma: no cover — depends on T76/T77 land order
        msg = (
            "audit_arb_matches.py depends on pfm.arb_matching.event_similarity\n"
            "(task T77) which exposes MarketDesc + score_match + "
            "build_market_desc. T77 has not landed yet (or its import is "
            "broken). Error: " + repr(exc) + "\n\n"
            "Once T77 lands, this script will run without changes."
        )
        raise SystemExit(msg) from exc
    return MarketDesc, score_match, build_market_desc


# ---------------------------------------------------------------------------
# Data-source loaders
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_STATE_PATH = REPO_ROOT / "arbstuff" / "dashboard_state.json"


def load_pairs_from_dashboard_state(path: Path) -> list[dict[str, Any]]:
    """Load arb pairs from ``arbstuff/dashboard_state.json``.

    Returns a list of dicts shaped uniformly for downstream scoring::

        {pair_id, poly_title, kalshi_title, poly_slug, kalshi_ticker,
         profit_pct, cost, source}
    """
    if not path.exists():
        return []
    try:
        with path.open() as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    opps = state.get("opportunities") or []
    pairs: list[dict[str, Any]] = []
    for o in opps:
        name = (o.get("name") or "").strip()
        poly_slug = o.get("poly_slug") or ""
        k_ticker = o.get("kalshi_ticker") or ""
        # The dashboard_state format only stores one combined "name". We
        # split heuristically; the matcher will read both via build_market_desc.
        kalshi_title = name or k_ticker
        # Use poly slug as best-effort title fallback (slugs are kebab-case
        # questions, which the matcher tokenises adequately).
        poly_title = name or poly_slug.replace("-", " ")
        pairs.append(
            {
                "pair_id": (o.get("arb_key") or f"{k_ticker}|{poly_slug}")[:80],
                "poly_title": poly_title,
                "kalshi_title": kalshi_title,
                "poly_slug": poly_slug,
                "kalshi_ticker": k_ticker,
                "profit_pct": float(o.get("profit_pct") or 0.0),
                "cost": float(o.get("cost") or 0.0),
                "source": "dashboard_state",
            }
        )
    return pairs


def load_pairs_from_top_arbs(top_n: int = 25) -> list[dict[str, Any]]:
    """Fallback: call :func:`pfm.arb_scanner.top_arbs` directly."""
    try:
        from pfm.arb_scanner import top_arbs  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise SystemExit("Could not import pfm.arb_scanner.top_arbs: " + repr(exc)) from exc
    try:
        arbs = top_arbs(n=top_n)
    except Exception as exc:  # pragma: no cover — network-dependent
        print(f"[warn] top_arbs() raised: {exc!r}; returning empty list", file=sys.stderr)
        return []
    pairs: list[dict[str, Any]] = []
    for a in arbs:
        # top_arbs returns 4-way concept arbs with prices per venue.
        label = a.get("label") or a.get("concept_id") or ""
        legs = a.get("prices") or {}
        venues = list(legs.keys())
        # Best-effort 2-leg projection for matching purposes
        a_v = venues[0] if venues else "polymarket"
        b_v = venues[1] if len(venues) > 1 else "kalshi"
        pairs.append(
            {
                "pair_id": a.get("concept_id") or label[:60],
                "poly_title": f"{label} ({a_v})",
                "kalshi_title": f"{label} ({b_v})",
                "poly_slug": "",
                "kalshi_ticker": "",
                "profit_pct": float(a.get("max_spread_pct") or 0.0),
                "cost": 0.0,
                "source": "top_arbs",
            }
        )
    return pairs


# ---------------------------------------------------------------------------
# MarketDesc builder — bridges raw pair fields to T77's contract.
# ---------------------------------------------------------------------------


def build_market_desc(
    MarketDescCls,
    title: str,
    *,
    venue: str,
    slug: str = "",
    description: str = "",
    build_helper: Any | None = None,
) -> Any:
    """Construct a ``MarketDesc`` for a leg, preferring the T77 helper.

    When ``build_helper`` is the T77 ``build_market_desc(raw_payload, venue)``
    function we delegate to it (preserves date/threshold/entity/jurisdiction
    extraction). When it is ``None`` we fall back to constructing the
    dataclass directly with neutral defaults — useful for unit tests that
    only have a ``MarketDescCls`` and a title.
    """
    if build_helper is not None:
        payload = {"title": title, "description": description, "slug": slug}
        return build_helper(payload, venue)
    # Fallback: construct the dataclass with neutral defaults.
    try:
        return MarketDescCls(
            title=title,
            description=description,
            venue=(venue or "").strip().lower(),
            resolution_window=None,
            threshold=None,
            entities=(),
            jurisdiction=None,
        )
    except TypeError as exc:  # pragma: no cover — schema drift
        raise SystemExit(
            "Could not construct MarketDesc with neutral defaults: " + repr(exc)
        ) from exc


# ---------------------------------------------------------------------------
# Scoring normalisation — T77's score_match return shape may evolve.
# ---------------------------------------------------------------------------


def _normalise_score_result(raw: Any) -> dict[str, Any]:
    """Coerce a ``score_match`` return value into the expected envelope.

    Expected output keys::

        score (float in [0,1])
        rejected (bool)
        reason (str | "")
        breakdown: {jaccard, entity_jaccard, topic, window_overlap}
    """
    out = {
        "score": 0.0,
        "rejected": True,
        "reason": "",
        "breakdown": {
            "jaccard": None,
            "entity_jaccard": None,
            "topic": None,
            "window_overlap": None,
        },
    }
    if raw is None:
        out["reason"] = "score_match returned None"
        return out
    if isinstance(raw, (int, float)):
        out["score"] = float(raw)
        out["rejected"] = float(raw) < 0.4
        return out
    # T77's SimilarityScore dataclass: has .total, .components, .rejected_reason
    if hasattr(raw, "total") and hasattr(raw, "components"):
        out["score"] = float(getattr(raw, "total", 0.0) or 0.0)
        reason = getattr(raw, "rejected_reason", None)
        out["reason"] = str(reason) if reason else ""
        out["rejected"] = bool(reason) or out["score"] < 0.4
        comps = getattr(raw, "components", {}) or {}
        # Map T77's component keys to our CSV column names.
        out["breakdown"]["jaccard"] = comps.get("title_jaccard")
        out["breakdown"]["entity_jaccard"] = comps.get("entity_jaccard")
        out["breakdown"]["topic"] = comps.get("topic_overlap")
        out["breakdown"]["window_overlap"] = comps.get("window_center")
        return out
    if isinstance(raw, dict):
        out["score"] = float(raw.get("score", raw.get("total", 0.0)))
        # Hard reject either via an explicit flag or score below threshold.
        out["rejected"] = bool(raw.get("rejected", out["score"] < 0.4))
        out["reason"] = str(raw.get("reason") or raw.get("rejected_reason") or "")
        bd = raw.get("breakdown") or raw.get("scoring_breakdown") or raw.get("components") or {}
        if isinstance(bd, dict):
            for k_dst, k_src in (
                ("jaccard", ("jaccard", "title_jaccard")),
                ("entity_jaccard", ("entity_jaccard",)),
                ("topic", ("topic", "topic_overlap")),
                ("window_overlap", ("window_overlap", "window_center")),
            ):
                for k in k_src:
                    if k in bd:
                        out["breakdown"][k_dst] = bd[k]
                        break
        return out
    # tuple-ish (score, rejected, reason)?
    try:
        score = float(raw[0])
        out["score"] = score
        if len(raw) > 1:
            out["rejected"] = bool(raw[1])
        if len(raw) > 2:
            out["reason"] = str(raw[2])
    except Exception:
        out["reason"] = f"unrecognised score_match return: {type(raw).__name__}"
    return out


# ---------------------------------------------------------------------------
# Audit core
# ---------------------------------------------------------------------------


def audit_pairs(pairs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score every pair; return list of result rows."""
    MarketDescCls, score_match, build_helper = _import_matchers()
    results: list[dict[str, Any]] = []
    for p in pairs:
        a = build_market_desc(
            MarketDescCls,
            p["poly_title"],
            venue="polymarket",
            slug=p.get("poly_slug", ""),
            build_helper=build_helper,
        )
        b = build_market_desc(
            MarketDescCls,
            p["kalshi_title"],
            venue="kalshi",
            slug=p.get("kalshi_ticker", ""),
            build_helper=build_helper,
        )
        try:
            raw = score_match(a, b)
        except Exception as exc:
            raw = {"score": 0.0, "rejected": True, "reason": f"score_match raised: {exc!r}"}
        norm = _normalise_score_result(raw)
        row = {
            "pair_id": p["pair_id"],
            "poly_title": p["poly_title"],
            "kalshi_title": p["kalshi_title"],
            "score": round(float(norm["score"]), 4),
            "rejected": bool(norm["rejected"]),
            "reason": norm["reason"],
            "jaccard": norm["breakdown"]["jaccard"],
            "entity_jaccard": norm["breakdown"]["entity_jaccard"],
            "topic": norm["breakdown"]["topic"],
            "window_overlap": norm["breakdown"]["window_overlap"],
            "profit_pct": p.get("profit_pct"),
            "cost": p.get("cost"),
            "source": p.get("source"),
        }
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Output formatters
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "pair_id",
    "poly_title",
    "kalshi_title",
    "score",
    "rejected",
    "reason",
    "jaccard",
    "entity_jaccard",
    "topic",
    "window_overlap",
]


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_blacklist_proposals(rows: list[dict[str, Any]], path: Path) -> int:
    """Write proposed blacklist entries for rejected pairs to ``path``.

    NEVER touches the live ``arbstuff/blacklist.json`` or ``arb_blacklist.json``.
    Humans review the proposals before promotion.
    """
    proposals = [
        {
            "pair_id": r["pair_id"],
            "reason": r["reason"] or "low_score",
            "score": r["score"],
            "poly_title": r["poly_title"],
            "kalshi_title": r["kalshi_title"],
            "proposed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        }
        for r in rows
        if r["rejected"]
    ]
    path.write_text(json.dumps(proposals, indent=2))
    return len(proposals)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    high_conf = sum(1 for r in rows if r["score"] > 0.7 and not r["rejected"])
    borderline = sum(1 for r in rows if 0.4 <= r["score"] <= 0.7 and not r["rejected"])
    rejected = sum(1 for r in rows if r["rejected"])

    # "Top-10 worst": highest cost (or profit_pct as proxy) but lowest score.
    def _priority(r: dict[str, Any]) -> float:
        cost = float(r.get("cost") or 0.0)
        profit = float(r.get("profit_pct") or 0.0)
        priced = cost or profit
        # high priced, low score = high audit priority
        return priced * (1.0 - float(r["score"]))

    worst = sorted(rows, key=_priority, reverse=True)[:10]
    return {
        "total": total,
        "high_confidence": high_conf,
        "borderline": borderline,
        "rejected": rejected,
        "top10_worst": [
            {
                "pair_id": r["pair_id"],
                "score": r["score"],
                "profit_pct": r.get("profit_pct"),
                "reason": r["reason"],
            }
            for r in worst
        ],
    }


def print_summary(summary: dict[str, Any], csv_path: Path) -> None:
    print(f"audit_arb_matches: wrote {csv_path}")
    print(f"  total pairs:        {summary['total']}")
    print(f"  high-conf (>0.7):   {summary['high_confidence']}")
    print(f"  borderline (.4-.7): {summary['borderline']}")
    print(f"  rejected (<0.4):    {summary['rejected']}")
    print("  top-10 worst (priced × (1-score)):")
    for w in summary["top10_worst"]:
        print(
            f"    - {w['pair_id'][:46]:46s}  "
            f"score={w['score']:.2f}  profit_pct={w['profit_pct']}  "
            f"reason={(w['reason'] or '-')[:40]}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        choices=["auto", "arbstuff", "top_arbs"],
        default="auto",
        help="Where to load pairs from. 'auto' prefers dashboard_state.json.",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="When falling back to top_arbs(), how many to fetch.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override CSV output path (default /tmp/arb-match-audit-YYYYMMDD.csv).",
    )
    p.add_argument(
        "--apply-blacklist",
        action="store_true",
        help="Write proposed blacklist entries to /tmp/arb-blacklist-proposals.json.",
    )
    p.add_argument(
        "--fail-on-reject",
        action="store_true",
        help="Exit non-zero if any pair is rejected (for CI).",
    )
    return p.parse_args(argv)


def _resolve_csv_path(arg_out: Path | None) -> Path:
    if arg_out is not None:
        return arg_out
    today = datetime.datetime.now().strftime("%Y%m%d")
    return Path(f"/tmp/arb-match-audit-{today}.csv")


def load_pairs(source: str, top_n: int) -> list[dict[str, Any]]:
    if source == "arbstuff":
        return load_pairs_from_dashboard_state(DASHBOARD_STATE_PATH)
    if source == "top_arbs":
        return load_pairs_from_top_arbs(top_n=top_n)
    # auto
    pairs = load_pairs_from_dashboard_state(DASHBOARD_STATE_PATH)
    if pairs:
        return pairs
    return load_pairs_from_top_arbs(top_n=top_n)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pairs = load_pairs(args.source, args.top_n)
    if not pairs:
        print("audit_arb_matches: no pairs found (empty dashboard_state and top_arbs)")
        return 0
    rows = audit_pairs(pairs)
    csv_path = _resolve_csv_path(args.out)
    write_csv(rows, csv_path)
    summary = summarise(rows)
    print_summary(summary, csv_path)
    if args.apply_blacklist:
        prop_path = Path("/tmp/arb-blacklist-proposals.json")
        n_props = write_blacklist_proposals(rows, prop_path)
        print(f"  wrote {n_props} blacklist proposals -> {prop_path}")
    if args.fail_on_reject and summary["rejected"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    # Ensure the repo's src/ is on sys.path when invoked directly.
    src_path = REPO_ROOT / "api" / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    raise SystemExit(main())
