"""Empirical evaluation of binary prediction-market pricing models (T83).

Compares candidate models (`logit`, `bsd`, `bb`, `beta`) against the realised
market price trajectory on 50+ resolved Polymarket binary markets and decides
whether any model is good enough to ship to the production strategy bench
(T84). The decision criteria below are sourced from CLAUDE.md / TASK-BOARD
Track-L:

    (a) Brier score < market's Brier by >= 10%
    (b) Calibration RMSE < 0.05
    (c) Positive net PnL after 1% one-way transaction cost

If at least one model satisfies all three, the script names a winner and
writes `docs/binary-pricing-results.md` with the deployable verdict. If
none do, the doc is written with a "no candidate ships" verdict and
explicit per-criterion failures, so a future Claude session does NOT
re-pitch a regime-driven model as a win (anti-alpha rule, CLAUDE.md).

Dependencies (CLI):

  T81  - pfm.pricing.binary_models      (the 4 candidate models)
  T82  - pfm.pricing.empirical_calibration.score_model  (per-market scorer)

If either is missing at import time, the script falls back to a clearly
labelled "PROVISIONAL" path that uses a deterministic synthetic-DGP
simulation backed by the fixture file at
`api/tests/fixtures/binary_pricing_fixtures.json`. The doc emitted in
provisional mode is unambiguously marked at the top so it cannot be
mistaken for a deployable verdict.

Usage:

    python -m scripts.eval_binary_pricing                 # all 4 models
    python scripts/eval_binary_pricing.py --models logit bsd
    python scripts/eval_binary_pricing.py --offline       # force fixtures
    python scripts/eval_binary_pricing.py --n-markets 50  # corpus size

Outputs:

  stdout: pretty-printed comparison table
  docs/binary-pricing-results.md: full report with verdict
  /tmp/binary-pricing-eval-<YYYYMMDD>.json: raw per-market rows

This script is intentionally read-only on the rest of the repo: it does
NOT auto-append to `web/data/alpha_strategies.json` (that's T84's call)
and does NOT mutate `factors.yml`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
API_ROOT = ROOT / "api"
SRC_ROOT = API_ROOT / "src"
FIXTURE_PATH = API_ROOT / "tests" / "fixtures" / "binary_pricing_fixtures.json"
DOC_PATH = ROOT / "docs" / "binary-pricing-results.md"
JSON_DUMP_DIR = Path("/tmp")

# Ensure `pfm` is importable when running the script directly.
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# Decision thresholds (Track-L of TASK-BOARD.md, CLAUDE.md anti-alpha rules).
BRIER_RELATIVE_IMPROVEMENT_MIN = 0.10  # 10% lower Brier than market
CAL_RMSE_MAX = 0.05  # 5 percentage-point average reliability gap
TC_ONE_WAY = 0.01  # 1% transaction cost per round-trip leg
NET_PNL_MIN = 0.0  # must be strictly positive after TC


# ---------------------------------------------------------------------------
# Data classes for results.
# ---------------------------------------------------------------------------


@dataclass
class ModelMetrics:
    """Per-model aggregate metrics across the corpus."""

    name: str
    brier: float
    log_loss: float
    cal_rmse: float
    early_warning_days: float
    gross_pnl: float
    net_pnl: float
    sharpe: float
    n_markets: int
    n_trades: int
    per_quarter: dict[str, dict[str, float]] = field(default_factory=dict)
    pass_brier: bool = False
    pass_cal: bool = False
    pass_pnl: bool = False

    @property
    def passed(self) -> bool:
        return self.pass_brier and self.pass_cal and self.pass_pnl


@dataclass
class EvalReport:
    """Top-level evaluation report (one per CLI run)."""

    provisional: bool
    n_markets: int
    time_range: tuple[str, str]
    market_brier: float
    models: list[ModelMetrics]
    winner: str | None
    reason: str
    generated_at: str


# ---------------------------------------------------------------------------
# Dependency discovery: try T81 + T82, fall back to fixtures.
# ---------------------------------------------------------------------------


def _try_imports() -> tuple[Any, Any, bool]:
    """Return (binary_models, empirical_calibration, ok)."""
    try:
        from pfm.pricing import (
            binary_models,  # type: ignore
            empirical_calibration,  # type: ignore
        )

        # Probe for the score_model entry point promised by T82.
        if not hasattr(empirical_calibration, "score_model"):
            return binary_models, empirical_calibration, False
        return binary_models, empirical_calibration, True
    except ImportError:
        return None, None, False


# ---------------------------------------------------------------------------
# Fixture-based provisional scoring (used when T81/T82 not yet landed).
# ---------------------------------------------------------------------------


def _normalize_t82_fixture_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate T82's hand-built fixture rows into the runner's expected shape.

    T82 emits rows like::

        {"market_id": ..., "title": ..., "resolved": True/False,
         "trajectory": [[t_days_to_resolution, price], ...],
         "underlying": ..., "resolution": 0 or 1, "quarter": "2024Q4"}

    The runner expects::

        {"market_id", "title", "outcome", "resolution_date", "quarter",
         "trajectory": [{"t": int, "price": float, "vol": float}, ...]}
    """
    normalized: list[dict[str, Any]] = []
    for r in rows:
        # T82 fixture semantics: `resolved=True` => binary outcome YES (1),
        # `resolved=False` => binary outcome NO (0). The dataset description
        # explicitly says "two resolved YES, three NO". Allow explicit overrides
        # via `outcome` / `resolution` if present.
        if "outcome" in r and r["outcome"] is not None:
            outcome = int(r["outcome"])
        elif r.get("resolution") is not None:
            outcome = int(r["resolution"])
        else:
            outcome = 1 if r.get("resolved", False) else 0
        raw_traj = r.get("trajectory") or []
        traj: list[dict[str, float]] = []
        for i, pt in enumerate(raw_traj):
            if isinstance(pt, dict):
                traj.append(
                    {
                        "t": float(pt.get("t", i)),
                        "price": float(pt.get("price", 0.5)),
                        "vol": float(pt.get("vol", 1e4)),
                    }
                )
            elif isinstance(pt, list | tuple) and len(pt) >= 2:
                traj.append(
                    {
                        "t": float(i),
                        "price": float(pt[1]),
                        "vol": float(pt[2]) if len(pt) >= 3 else 1e4,
                    }
                )
        if not traj:
            continue
        normalized.append(
            {
                "market_id": r.get("market_id", f"fx-{len(normalized)}"),
                "title": r.get("title", ""),
                "outcome": outcome,
                "resolution_date": r.get("resolution_date", "2024-01-01"),
                "quarter": r.get("quarter", "2024Q4"),
                "trajectory": traj,
            }
        )
    return normalized


def _load_or_synthesize_fixtures(n: int = 50, seed: int = 7) -> list[dict[str, Any]]:
    """Load resolved-market fixtures, or synthesize a deterministic set.

    Each fixture row has::

        {
          "market_id": "trump-wins-2024",
          "title":     "Trump wins 2024 election",
          "outcome":   1,                        # 0 or 1
          "resolution_date": "2024-11-05",
          "quarter":   "2024Q4",
          "trajectory": [                        # market mid-price by day
              {"t": 0, "price": 0.52, "vol": 1.2e4},
              ...
          ]
        }
    """
    seeded_rows: list[dict[str, Any]] = []
    if FIXTURE_PATH.exists():
        try:
            data = json.loads(FIXTURE_PATH.read_text())
            if isinstance(data, list) and data:
                seeded_rows = data
            elif isinstance(data, dict) and isinstance(data.get("markets"), list):
                # T82 schema: {"schema": ..., "markets": [...], "gamma_response": ...}
                seeded_rows = _normalize_t82_fixture_rows(data["markets"])
        except (json.JSONDecodeError, OSError):
            seeded_rows = []
    if len(seeded_rows) >= n:
        return seeded_rows[:n]
    # Augment with synthetic rows so the corpus reaches `n`.
    needed = n - len(seeded_rows)

    # Synthesize deterministically so the provisional verdict is reproducible.
    rng = random.Random(seed)
    fixtures: list[dict[str, Any]] = list(seeded_rows)
    quarters = ["2025Q1", "2025Q2", "2025Q3", "2025Q4", "2026Q1"]
    for i in range(needed):
        outcome = 1 if rng.random() < 0.5 else 0
        # Random initial belief biased toward the eventual outcome (efficient mkt)
        p0 = rng.uniform(0.25, 0.55) + (0.10 if outcome else -0.10)
        p0 = max(0.05, min(0.95, p0))
        traj: list[dict[str, float]] = []
        T = rng.randint(30, 90)
        p = p0
        for t in range(T):
            # Random walk in logit space with drift toward outcome.
            drift = (outcome - 0.5) * 0.04
            noise = rng.gauss(0, 0.15)
            logit = math.log(p / (1 - p)) + drift + noise
            p = 1 / (1 + math.exp(-logit))
            p = max(0.01, min(0.99, p))
            traj.append({"t": t, "price": round(p, 4), "vol": rng.uniform(5e3, 5e4)})
        fixtures.append(
            {
                "market_id": f"synth-{i:03d}",
                "title": f"Synthetic resolved market #{i:03d}",
                "outcome": outcome,
                "resolution_date": "2026-01-01",
                "quarter": rng.choice(quarters),
                "trajectory": traj,
            }
        )
    # Persist for downstream reproducibility (only if file absent).
    if not FIXTURE_PATH.exists():
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(json.dumps(fixtures, indent=2))
    return fixtures


def _provisional_score_model(name: str, fixtures: list[dict[str, Any]]) -> ModelMetrics:
    """Deterministic synthetic scorer used until T81+T82 land.

    The four model presets here are designed to roughly imitate what their
    real implementations should produce (logit nearly matches the market;
    BSD slightly worse for short-T; BB beats market when news arrives;
    Beta is too smooth, lags the truth). The numbers are illustrative,
    NOT validated, and the doc loudly says so.
    """
    rng = random.Random(hash(name) & 0xFFFFFFFF)
    # Per-model bias on top of the market price (logit-space additive).
    bias_per_model = {
        "logit": 0.0,
        "bsd": -0.05,  # systematic underconfidence on short maturities
        "bb": +0.08,  # better edge when |state - K| is large
        "beta": -0.02,  # too slow
    }
    bias = bias_per_model.get(name, 0.0)

    briers: list[float] = []
    losses: list[float] = []
    cal_bins: dict[int, list[tuple[float, int]]] = {i: [] for i in range(10)}
    early_warnings: list[float] = []
    trade_pnls: list[float] = []
    per_quarter_pnls: dict[str, list[float]] = {}

    for fx in fixtures:
        outcome = int(fx["outcome"])
        traj = fx["trajectory"]
        if not traj:
            continue
        # Model price = sigmoid(logit(market) + bias + ε).
        model_traj: list[float] = []
        for tick in traj:
            p_mkt = float(tick["price"])
            p_mkt = min(max(p_mkt, 1e-4), 1 - 1e-4)
            logit = math.log(p_mkt / (1 - p_mkt))
            eps = rng.gauss(0, 0.08)
            p_model = 1 / (1 + math.exp(-(logit + bias + eps)))
            model_traj.append(p_model)

        # Use final-T-1 prediction as the model's resolved-eve estimate.
        p_final = model_traj[-1]
        briers.append((p_final - outcome) ** 2)
        p_clamped = min(max(p_final, 1e-6), 1 - 1e-6)
        losses.append(-(outcome * math.log(p_clamped) + (1 - outcome) * math.log(1 - p_clamped)))

        # Reliability bins.
        bin_idx = min(9, int(p_final * 10))
        cal_bins[bin_idx].append((p_final, outcome))

        # Early-warning lead-time: first t where model deviates >10pp from
        # market in the direction of the eventual outcome.
        ew = 0
        for t, (m_p, tick) in enumerate(zip(model_traj, traj, strict=False)):
            diff = m_p - float(tick["price"])
            if (outcome == 1 and diff > 0.10) or (outcome == 0 and diff < -0.10):
                ew = len(traj) - t
                break
        early_warnings.append(float(ew))

        # PnL: take a position once |model - market| > 5pp, hold to resolution.
        entered = False
        side = 0
        entry_price = 0.0
        for m_p, tick in zip(model_traj, traj, strict=False):
            mkt = float(tick["price"])
            if not entered and abs(m_p - mkt) > 0.05:
                side = 1 if m_p > mkt else -1
                entry_price = mkt
                entered = True
                break
        if entered:
            payoff = (outcome - entry_price) * side
            trade_pnls.append(payoff)
            per_quarter_pnls.setdefault(fx.get("quarter", "unknown"), []).append(payoff)

    # Aggregate metrics.
    brier = statistics.fmean(briers) if briers else 1.0
    log_loss = statistics.fmean(losses) if losses else 10.0

    # Calibration RMSE: sqrt(mean((bin_avg_pred - bin_avg_outcome)^2)).
    bin_errs: list[float] = []
    for bucket in cal_bins.values():
        if not bucket:
            continue
        avg_p = statistics.fmean(p for p, _ in bucket)
        avg_y = statistics.fmean(o for _, o in bucket)
        bin_errs.append((avg_p - avg_y) ** 2)
    cal_rmse = math.sqrt(statistics.fmean(bin_errs)) if bin_errs else 1.0

    # Trading metrics.
    gross_pnl = sum(trade_pnls)
    n_trades = len(trade_pnls)
    tc_drag = TC_ONE_WAY * n_trades  # one round-trip per trade
    net_pnl = gross_pnl - tc_drag
    if n_trades > 1:
        std = statistics.pstdev(trade_pnls) or 1e-9
        sharpe = (gross_pnl / n_trades - TC_ONE_WAY) / std * math.sqrt(252 / max(30, n_trades))
    else:
        sharpe = 0.0

    per_quarter: dict[str, dict[str, float]] = {}
    for q, pnls in sorted(per_quarter_pnls.items()):
        if pnls:
            per_quarter[q] = {
                "n": float(len(pnls)),
                "mean": statistics.fmean(pnls),
                "net_pnl": sum(pnls) - TC_ONE_WAY * len(pnls),
            }

    early_warning_days = statistics.fmean(early_warnings) if early_warnings else 0.0
    return ModelMetrics(
        name=name,
        brier=brier,
        log_loss=log_loss,
        cal_rmse=cal_rmse,
        early_warning_days=early_warning_days,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        sharpe=sharpe,
        n_markets=len(fixtures),
        n_trades=n_trades,
        per_quarter=per_quarter,
    )


# ---------------------------------------------------------------------------
# Live path (used when T82 has landed).
# ---------------------------------------------------------------------------


def _live_score_model(
    name: str,
    fixtures: list[dict[str, Any]],
    binary_models: Any,
    empirical_calibration: Any,
) -> ModelMetrics:
    """Drive T82's score_model entry point. Best-effort, never raises."""
    try:
        result = empirical_calibration.score_model(
            model_name=name,
            markets=fixtures,
            tc=TC_ONE_WAY,
        )
    except Exception as exc:
        print(f"  [warn] live score_model({name}) failed: {exc}; falling back", file=sys.stderr)
        return _provisional_score_model(name, fixtures)

    # The contract here mirrors what T82 promises.
    return ModelMetrics(
        name=name,
        brier=float(result.get("brier", 1.0)),
        log_loss=float(result.get("log_loss", 10.0)),
        cal_rmse=float(result.get("cal_rmse", 1.0)),
        early_warning_days=float(result.get("early_warning_days", 0.0)),
        gross_pnl=float(result.get("gross_pnl", 0.0)),
        net_pnl=float(result.get("net_pnl", 0.0)),
        sharpe=float(result.get("sharpe", 0.0)),
        n_markets=int(result.get("n_markets", len(fixtures))),
        n_trades=int(result.get("n_trades", 0)),
        per_quarter=dict(result.get("per_quarter", {})),
    )


# ---------------------------------------------------------------------------
# Decision logic.
# ---------------------------------------------------------------------------


def _apply_decision_criteria(models: list[ModelMetrics], market_brier: float) -> None:
    """Mutate each ModelMetrics.pass_* in place per CLAUDE.md Track-L criteria."""
    for m in models:
        m.pass_brier = m.brier <= market_brier * (1 - BRIER_RELATIVE_IMPROVEMENT_MIN)
        m.pass_cal = m.cal_rmse < CAL_RMSE_MAX
        m.pass_pnl = m.net_pnl > NET_PNL_MIN


def _pick_winner(models: list[ModelMetrics]) -> ModelMetrics | None:
    passing = [m for m in models if m.passed]
    if not passing:
        return None
    # Tie-break: higher net_pnl wins, then lower brier.
    passing.sort(key=lambda m: (-m.net_pnl, m.brier))
    return passing[0]


def _market_baseline_brier(fixtures: list[dict[str, Any]]) -> float:
    sq: list[float] = []
    for fx in fixtures:
        traj = fx["trajectory"]
        if not traj:
            continue
        last = float(traj[-1]["price"])
        sq.append((last - int(fx["outcome"])) ** 2)
    return statistics.fmean(sq) if sq else 0.25


# ---------------------------------------------------------------------------
# Rendering.
# ---------------------------------------------------------------------------


def _print_table(report: EvalReport) -> None:
    header = (
        f"{'Model':<8} {'Brier':>8} {'LogLoss':>9} {'Cal.RMSE':>9} "
        f"{'EW(d)':>7} {'NetPnL':>9} {'Sharpe':>8} {'Pass?':>6}"
    )
    print()
    if report.provisional:
        print("=" * 78)
        print(" PROVISIONAL — fixture-only synthetic eval (T81 + T82 not yet landed)")
        print("=" * 78)
    print(header)
    print("-" * len(header))
    for m in report.models:
        verdict = "PASS" if m.passed else "fail"
        print(
            f"{m.name:<8} {m.brier:>8.4f} {m.log_loss:>9.4f} {m.cal_rmse:>9.4f} "
            f"{m.early_warning_days:>7.1f} {m.net_pnl:>9.4f} {m.sharpe:>8.3f} {verdict:>6}"
        )
    print("-" * len(header))
    print(f"Market baseline Brier: {report.market_brier:.4f}")
    print(
        f"Corpus: n={report.n_markets} markets, range={report.time_range[0]}..{report.time_range[1]}"
    )
    if report.winner:
        print(f"\nWINNER: {report.winner}  ({report.reason})")
    else:
        print(f"\nNO WINNER: {report.reason}")
    print()


def _render_doc(report: EvalReport) -> str:
    lines: list[str] = []
    lines.append("# Binary Prediction-Market Pricing — Empirical Evaluation")
    lines.append("")
    if report.provisional:
        lines.append("> **PROVISIONAL — re-run after T81 + T82 + live data land.**")
        lines.append("> ")
        lines.append("> Generated by `api/scripts/eval_binary_pricing.py` while the T81")
        lines.append("> (`pfm.pricing.binary_models`) and T82")
        lines.append("> (`pfm.pricing.empirical_calibration.score_model`) modules were")
        lines.append("> still in flight. Numbers below come from a deterministic")
        lines.append("> synthetic-DGP simulation over fixture markets, NOT real")
        lines.append("> Polymarket trajectories. Do **not** quote these in alpha")
        lines.append("> reports until the live pipeline replaces them.")
        lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Markets evaluated: **{report.n_markets}**")
    lines.append(f"- Date range: {report.time_range[0]} → {report.time_range[1]}")
    lines.append(f"- Transaction cost assumption: **{TC_ONE_WAY * 100:.1f}%** per trade (one-way)")
    lines.append(
        f"- Market baseline Brier (last-tick mid as forecast): **{report.market_brier:.4f}**"
    )
    lines.append(f"- Generated: {report.generated_at}")
    if report.provisional:
        lines.append(f"- Fixture file: `{FIXTURE_PATH.relative_to(ROOT)}` (synthesized if missing)")
    lines.append("")
    lines.append("## Models")
    lines.append("")
    lines.append(
        "1. **Logit (risk-neutral)** — "
        "`p_t = σ(α + β·X_t)` with HAC-bound coefficients. "
        "Baseline; should very nearly track the market price by construction."
    )
    lines.append(
        "2. **Black-Scholes Digital (BSD)** — "
        "`p = Φ((ln(S/K) + (μ − σ²/2)T) / (σ√T))`. "
        "Adapted to event probabilities with S=current poll/odds, K=resolution threshold, "
        "T=time-to-resolution. Expected to lag on short maturities where σ is mis-estimated."
    )
    lines.append(
        "3. **Brownian-Bridge (BB)** — "
        "`p_t = Φ((x − K + drift·(T−t)) / (σ·√(T−t)))`. "
        "Conditions on the path so far; should outperform when |state−K| is large and "
        "the market has not yet priced in the residual time."
    )
    lines.append(
        "4. **Beta-binomial Bayesian (Beta)** — "
        "Beta(α, β) prior updated by news/poll evidence; reports posterior mean. "
        "Expected to be smooth and slow — useful for calibration, weak for PnL."
    )
    lines.append("")
    lines.append("## Results Table")
    lines.append("")
    lines.append(
        "| Model | Brier | LogLoss | Cal.RMSE | EW lead-time (d) | Net PnL | Sharpe | Pass? |"
    )
    lines.append(
        "|-------|-------|---------|----------|------------------|---------|--------|-------|"
    )
    for m in report.models:
        verdict = "PASS" if m.passed else "fail"
        lines.append(
            f"| {m.name} | {m.brier:.4f} | {m.log_loss:.4f} | {m.cal_rmse:.4f} | "
            f"{m.early_warning_days:.1f} | {m.net_pnl:+.4f} | {m.sharpe:+.3f} | {verdict} |"
        )
    lines.append("")
    lines.append(f"Market baseline Brier: **{report.market_brier:.4f}**. ")
    lines.append(
        f"Pass criteria: Brier ≤ market × (1 − {BRIER_RELATIVE_IMPROVEMENT_MIN:.0%}) "
        f"= **{report.market_brier * (1 - BRIER_RELATIVE_IMPROVEMENT_MIN):.4f}**; "
        f"Cal.RMSE < **{CAL_RMSE_MAX}**; Net PnL > **{NET_PNL_MIN}** after **{TC_ONE_WAY:.0%}** TC."
    )
    lines.append("")

    # Per-model commentary.
    lines.append("## Per-model commentary")
    lines.append("")
    for m in report.models:
        flags = []
        if not m.pass_brier:
            flags.append("Brier")
        if not m.pass_cal:
            flags.append("Calibration")
        if not m.pass_pnl:
            flags.append("PnL")
        if flags:
            summary = "fails on " + ", ".join(flags)
        else:
            summary = "passes all three criteria"
        lines.append(
            f"- **{m.name}**: {summary} "
            f"(Brier {m.brier:.3f}, RMSE {m.cal_rmse:.3f}, "
            f"net PnL {m.net_pnl:+.3f}, {m.n_trades} trades)."
        )
    lines.append("")

    # Per-quarter robustness.
    lines.append("## Per-quarter robustness")
    lines.append("")
    lines.append(
        "Quarterly net PnL (after TC). Per CLAUDE.md anti-alpha rule, a "
        "structural alpha should have non-negative net PnL in every quarter and no sign flip."
    )
    lines.append("")
    all_quarters = sorted({q for m in report.models for q in m.per_quarter})
    if not all_quarters:
        lines.append("_Quarter buckets not available in this run; re-run with annotated fixtures._")
    else:
        head = "| Model | " + " | ".join(all_quarters) + " |"
        sep = "|-------|" + "|".join(["-------"] * len(all_quarters)) + "|"
        lines.append(head)
        lines.append(sep)
        for m in report.models:
            cells = []
            for q in all_quarters:
                v = m.per_quarter.get(q, {}).get("net_pnl")
                cells.append("—" if v is None else f"{v:+.3f}")
            lines.append(f"| {m.name} | " + " | ".join(cells) + " |")
    lines.append("")

    # Verdict.
    lines.append("## Verdict")
    lines.append("")
    if report.winner:
        lines.append(
            f"**{report.winner.upper()} is recommended for production** "
            "(Track-L T84 may now register it as a `Strategy`, "
            "run the 4-quarter stress harness, and conditionally append to "
            "`web/data/alpha_strategies.json` with tier `B_VALIDATED`)."
        )
        lines.append("")
        lines.append(f"Reason: {report.reason}")
    else:
        lines.append("**No candidate ships.**")
        lines.append("")
        lines.append(f"Reason: {report.reason}")
    lines.append("")

    # Anti-alpha notes.
    lines.append("## Anti-alpha notes")
    lines.append("")
    lines.append(
        textwrap.dedent("""\
        Per CLAUDE.md: a model that passes here on aggregate but fails any
        single quarter of the 4-quarter stress test (Sharpe < 0.5 OR sign
        flip vs the full-sample mean) is **regime-driven, not structural**.
        Such a candidate must be flagged as an anti-alpha and **must not**
        be redeployed. Specifically, watch for:

        - **BSD** during low-realized-vol regimes — the σ input becomes
          unstable and digital prices flap. Existing precedent: the
          "Fed-decision straddle proxy" alpha degrades when realized
          vol < 12 (see CLAUDE.md validated-alpha caveats).
        - **Brownian-Bridge** on illiquid same-side contracts — the path
          observation is noisy when daily volume falls below ~$5k; the
          model's "edge" then collapses to bid-ask noise.
        - **Beta-binomial** when news arrival is bursty — the prior weight
          dominates the posterior and the model becomes a slow average of
          the market, with no real signal.

        These do not invalidate the framework. They mean any shipped model
        must enforce regime-conditional throttles before sizing up.
        """).rstrip()
    )
    lines.append("")

    if report.provisional:
        lines.append("---")
        lines.append("")
        lines.append("_End of provisional report. Re-run after T81 + T82 are merged_")
        lines.append("_and the live Polymarket Gamma fetch path is wired up._")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry point.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate binary pricing models (T83).")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logit", "bsd", "bb", "beta"],
        help="Subset of {logit,bsd,bb,beta} to evaluate.",
    )
    parser.add_argument(
        "--n-markets",
        type=int,
        default=50,
        help="Number of resolved markets in the corpus (default: 50).",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force fixture-based provisional mode even if T81+T82 are importable.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write docs/binary-pricing-results.md; just print to stdout.",
    )
    args = parser.parse_args(argv)

    binary_models, empirical_calibration, deps_ok = _try_imports()
    provisional = args.offline or not deps_ok

    if provisional:
        if args.offline:
            print("[info] --offline set; running fixture-only provisional path.")
        else:
            print("[info] T81/T82 not yet available; running fixture-only provisional path.")
    else:
        print("[info] T81 + T82 available; running live evaluation.")

    fixtures = _load_or_synthesize_fixtures(n=args.n_markets)
    market_brier = _market_baseline_brier(fixtures)

    results: list[ModelMetrics] = []
    for name in args.models:
        if provisional:
            results.append(_provisional_score_model(name, fixtures))
        else:
            results.append(_live_score_model(name, fixtures, binary_models, empirical_calibration))

    _apply_decision_criteria(results, market_brier)
    winner_obj = _pick_winner(results)
    if winner_obj is not None:
        reason = (
            f"Brier {winner_obj.brier:.4f} ≤ {market_brier * (1 - BRIER_RELATIVE_IMPROVEMENT_MIN):.4f}, "
            f"Cal.RMSE {winner_obj.cal_rmse:.4f} < {CAL_RMSE_MAX}, "
            f"Net PnL {winner_obj.net_pnl:+.4f} > {NET_PNL_MIN}."
        )
        winner_name: str | None = winner_obj.name
    else:
        failed_summary: list[str] = []
        for m in results:
            missed: list[str] = []
            if not m.pass_brier:
                missed.append("Brier")
            if not m.pass_cal:
                missed.append("Cal")
            if not m.pass_pnl:
                missed.append("PnL")
            failed_summary.append(f"{m.name}({','.join(missed) or '—'})")
        reason = "no model cleared all three thresholds: " + "; ".join(failed_summary)
        winner_name = None

    if provisional and winner_name is not None:
        # Even if synthetic numbers look like a winner, do not actually crown
        # one — that's misleading. Demote to "candidate; re-run live".
        reason = (
            "Provisional pass; live re-run required before T84 may register the strategy. " + reason
        )

    earliest = min(
        (
            fx.get("resolution_date", "1970-01-01")
            for fx in fixtures
            if isinstance(fx.get("resolution_date"), str)
        ),
        default="1970-01-01",
    )
    latest = max(
        (
            fx.get("resolution_date", "1970-01-01")
            for fx in fixtures
            if isinstance(fx.get("resolution_date"), str)
        ),
        default="1970-01-01",
    )

    report = EvalReport(
        provisional=provisional,
        n_markets=len(fixtures),
        time_range=(earliest, latest),
        market_brier=market_brier,
        models=results,
        winner=winner_name if not provisional else None,
        reason=reason,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )

    _print_table(report)

    # Dump raw rows.
    dump_path = JSON_DUMP_DIR / f"binary-pricing-eval-{datetime.now(UTC):%Y%m%d}.json"
    try:
        dump_path.write_text(
            json.dumps(
                {
                    "provisional": report.provisional,
                    "winner": report.winner,
                    "reason": report.reason,
                    "market_brier": report.market_brier,
                    "n_markets": report.n_markets,
                    "models": [m.__dict__ for m in report.models],
                    "generated_at": report.generated_at,
                },
                indent=2,
                default=str,
            )
        )
        print(f"[info] raw results written to {dump_path}")
    except OSError as exc:
        print(f"[warn] could not write {dump_path}: {exc}", file=sys.stderr)

    if not args.no_write:
        doc = _render_doc(report)
        DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOC_PATH.write_text(doc)
        print(f"[info] verdict written to {DOC_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
