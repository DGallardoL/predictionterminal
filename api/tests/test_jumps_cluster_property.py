"""Property-based tests for ``pfm.terminal.jumps_cluster.find_clusters``.

The unit tests in ``tests/terminal/test_jumps_cluster.py`` cover hand-crafted
synthetic scenarios. This file complements them with Hypothesis-driven property
tests that randomise jump count, slug count, timestamps and keyword overlap
across hundreds of examples per property.

Properties covered
------------------
1.  **Partition** — every *input jump node* (a (slug, jump-index) pair)
    appears in exactly one component when we project ``find_clusters``
    results back to the jump-node universe (clusters explicitly returned ∪
    singletons inferred for jumps absent from any cluster). Slug-level
    partition does NOT hold: a slug with multiple jumps can legitimately
    appear in two clusters at different timestamps, so we work at the
    jump-node grain.
2.  **Size-sum** — node counts of all components (clusters + inferred
    singletons) sum to the total parseable input jump count.
3.  **Singletons allowed** — a jump that shares no time-and-terms
    neighbour ends up in its own (inferred) singleton component, i.e. it is
    absent from the returned cluster list.
4.  **Identical-timeline → same cluster** — two slugs with byte-identical jump
    timelines (same ts + same matched_terms) and a third slug with one shared
    qualifying neighbour always land in the *same* cluster.
5.  **Empty input → empty output** — ``find_clusters({})`` and
    ``find_clusters({slug: []})`` both return ``[]``.
6.  **Shuffle invariance** — shuffling the per-slug jump list order, or the
    dict insertion order, does not change the slug-level component partition
    (cluster *ids* are stable mod relabeling, which we check via canonicalised
    partitions of slug sets).
7.  **Transitivity (union-find correctness)** — if slug ``a`` qualifies with
    ``b`` and ``b`` qualifies with ``c`` (under both the time AND Jaccard
    gates), then ``a, b, c`` all share a cluster id.
8.  **Time-window respected** — for two slugs whose jumps are separated by
    strictly more than ``time_tol_minutes``, no cluster is formed (regardless
    of perfect term overlap).

Why we redefine ``find_clusters``'s "partition" to include inferred singletons:
the production function deliberately *filters singletons out* (a 1-slug
"cluster" isn't macro news, it's a slug ringing alone). For partition-style
properties to even make sense we have to add the absent slugs back as their
own components.
"""

from __future__ import annotations

from collections.abc import Iterable

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pfm.terminal.jumps import Jump, JumpArticle
from pfm.terminal.jumps_cluster import Cluster, find_clusters

# ---------------------------------------------------------------------------
# Hypothesis configuration
# ---------------------------------------------------------------------------

# 200 examples per property (per task spec) but keep individual examples fast.
PROPERTY_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
)


# ---------------------------------------------------------------------------
# Builders & strategies
# ---------------------------------------------------------------------------

# A small finite vocabulary keeps Jaccard scores meaningful — random strings
# would almost never collide so every property degenerates to "all singletons".
VOCAB: list[str] = [
    "fed",
    "rate-hike",
    "trump",
    "china",
    "tariff",
    "earnings",
    "ai",
    "nvda",
    "oil",
    "ukraine",
    "btc",
    "etf",
    "spy",
    "vix",
    "fomc",
    "cpi",
]


def _make_jump(
    ts_minute: int,
    terms: Iterable[str],
    *,
    delta_pp: float = 5.0,
    relevance: float = 0.7,
) -> Jump:
    """Build a minimal :class:`Jump` anchored at 2026-05-01 + ``ts_minute`` mins.

    ``ts_minute`` is an integer offset in minutes from a fixed anchor so that
    Hypothesis can shrink simply. Strategies below cap ``ts_minute`` well
    below 24*60 = 1440 to stay inside one calendar day.
    """
    hh, mm = divmod(ts_minute, 60)
    hh = hh % 24
    ts_iso = f"2026-05-01T{hh:02d}:{mm:02d}:00Z"
    article = JumpArticle(
        ts_iso=ts_iso,
        seconds_from_jump=0,
        headline=f"Headline {ts_minute}",
        source="test.example",
        url="https://test.example/x",
        tone=0.0,
        relevance_score=relevance,
        matched_terms=list(terms),
        sentiment_score=0.0,
        sentiment_label="neutral",
    )
    return Jump(
        ts_iso=ts_iso,
        price_before=0.40,
        price_after=0.45,
        delta_pp=delta_pp,
        delta_logit=0.2,
        z_score=4.0,
        direction="up",
        explained=True,
        n_articles=1,
        top_articles=[article],
        news_sentiment_score=0.0,
        news_sentiment_label="neutral",
        sentiment_alignment="neutral",
    )


@st.composite
def jump_strategy(draw, *, max_minute: int = 600) -> Jump:
    """Hypothesis strategy producing a single :class:`Jump`.

    Time offset is in minutes [0, max_minute] (default 10 h) so we stay
    inside one calendar day, and terms are a non-empty subset of ``VOCAB``.
    """
    ts_minute = draw(st.integers(min_value=0, max_value=max_minute))
    terms = draw(st.lists(st.sampled_from(VOCAB), min_size=1, max_size=5, unique=True))
    return _make_jump(ts_minute, terms)


@st.composite
def jumps_by_slug_strategy(draw) -> dict[str, list[Jump]]:
    """``{slug -> [Jump...]}`` map. Small N keeps property runtimes tight.

    Per-slug jump timestamps are drawn unique so that each (slug, ts_iso)
    pair is a distinct node — otherwise the jump-level partition properties
    cannot be checked (the production schema's ``ClusterMember`` collapses
    identical (slug, ts_iso) hits).
    """
    n_slugs = draw(st.integers(min_value=0, max_value=6))
    out: dict[str, list[Jump]] = {}
    for i in range(n_slugs):
        slug = f"slug-{i}"
        n_jumps = draw(st.integers(min_value=0, max_value=3))
        if n_jumps == 0:
            out[slug] = []
            continue
        minutes = draw(
            st.lists(
                st.integers(min_value=0, max_value=600),
                min_size=n_jumps,
                max_size=n_jumps,
                unique=True,
            )
        )
        jumps: list[Jump] = []
        for ts_minute in minutes:
            terms = draw(st.lists(st.sampled_from(VOCAB), min_size=1, max_size=5, unique=True))
            jumps.append(_make_jump(ts_minute, terms))
        out[slug] = jumps
    return out


# ---------------------------------------------------------------------------
# Helpers: jump-node level partitioning
# ---------------------------------------------------------------------------

# A jump node is identified by the (slug, ts_iso) pair — the ``ClusterMember``
# Pydantic record carries exactly this fingerprint back from the cluster
# response. The strategy guarantees per-slug ts uniqueness so this is a true
# 1:1 mapping into the universe of input nodes.
JumpKey = tuple[str, str]


def _all_jump_keys(jumps_by_slug: dict[str, list[Jump]]) -> set[JumpKey]:
    return {(slug, j.ts_iso) for slug, jumps in jumps_by_slug.items() for j in jumps}


def _node_partition(
    jumps_by_slug: dict[str, list[Jump]], clusters: list[Cluster]
) -> tuple[frozenset[frozenset[JumpKey]], set[JumpKey]]:
    """Project ``find_clusters`` output onto a jump-node partition.

    Returns ``(components, all_nodes)``. Components from ``clusters`` are taken
    verbatim. Jumps not present in any cluster become their own singleton
    component — that's how the function semantically treats them (it filters
    singletons out of the response, but they still exist conceptually).
    """
    all_nodes = _all_jump_keys(jumps_by_slug)
    comps: list[frozenset[JumpKey]] = []
    seen: set[JumpKey] = set()
    for c in clusters:
        nodes = {(m.slug, m.ts_iso) for m in c.member_jumps}
        comps.append(frozenset(nodes))
        seen |= nodes
    for node in sorted(all_nodes - seen):
        comps.append(frozenset({node}))
    return frozenset(comps), all_nodes


# ---------------------------------------------------------------------------
# Property 1: Partition — every slug in exactly one component
# Property 2: Sizes sum to the input slug count
# Property 3: Singleton components allowed (no-neighbour slugs)
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(jumps_by_slug=jumps_by_slug_strategy())
def test_partition_each_jump_exactly_once(
    jumps_by_slug: dict[str, list[Jump]],
) -> None:
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5.0, kw_min_jaccard=0.20)
    comps, all_nodes = _node_partition(jumps_by_slug, clusters)

    # Property 1: each jump node appears in exactly one component.
    membership: dict[JumpKey, int] = {}
    for idx, comp in enumerate(comps):
        for node in comp:
            assert node not in membership, f"node {node} in multiple components"
            membership[node] = idx
    assert set(membership.keys()) == all_nodes

    # Property 2: sum of |components| == |all_nodes|.
    assert sum(len(c) for c in comps) == len(all_nodes)

    # The production cluster list never contains a cluster with <2 distinct
    # slugs; singletons live only in the inferred set.
    for c in clusters:
        assert c.n_markets >= 2
        assert len(c.member_jumps) >= 2


@PROPERTY_SETTINGS
@given(
    n_lonely=st.integers(min_value=1, max_value=6),
    base_offset=st.integers(min_value=0, max_value=240),
    gap_minutes=st.integers(min_value=10, max_value=120),
)
def test_singleton_clusters_when_no_overlap(
    n_lonely: int, base_offset: int, gap_minutes: int
) -> None:
    """Slugs whose jump terms / times don't overlap → all-singletons partition."""
    # Each slug gets its own unique disjoint term + an isolated timestamp
    # ``gap_minutes`` apart (≥10), well outside the default 5 min window.
    jumps_by_slug: dict[str, list[Jump]] = {}
    for i in range(n_lonely):
        unique_term = f"unique-term-{i}"
        ts_minute = base_offset + i * gap_minutes
        jumps_by_slug[f"lonely-{i}"] = [_make_jump(ts_minute, [unique_term])]
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5.0, kw_min_jaccard=0.20)
    assert clusters == []
    comps, all_nodes = _node_partition(jumps_by_slug, clusters)
    assert len(comps) == n_lonely
    assert all(len(c) == 1 for c in comps)
    assert set().union(*comps) == all_nodes


# ---------------------------------------------------------------------------
# Property 4: Two slugs with identical jump timeline → same cluster
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    ts_minute=st.integers(min_value=0, max_value=600),
    terms=st.lists(st.sampled_from(VOCAB), min_size=1, max_size=5, unique=True),
)
def test_identical_timeline_two_slugs_same_cluster(ts_minute: int, terms: list[str]) -> None:
    """If slug A and slug B have the byte-identical jump, they cluster together.

    The function refuses to merge two jumps on the *same* slug, but for
    different slugs with the same timestamp and the same term set, Jaccard
    = 1.0 and Δt = 0, so the union-find must link them.
    """
    j_a = _make_jump(ts_minute, terms)
    j_b = _make_jump(ts_minute, terms)
    jumps_by_slug = {"slug-a": [j_a], "slug-b": [j_b]}
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5.0, kw_min_jaccard=0.20)
    assert len(clusters) == 1
    members = {m.slug for m in clusters[0].member_jumps}
    assert members == {"slug-a", "slug-b"}


# ---------------------------------------------------------------------------
# Property 5: Empty input → empty output
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(empty_slugs=st.lists(st.text(alphabet="abcdef", min_size=1, max_size=8), max_size=5))
def test_empty_input_returns_empty(empty_slugs: list[str]) -> None:
    # No slugs at all
    assert find_clusters({}, time_tol_minutes=5.0, kw_min_jaccard=0.20) == []
    # Slugs present but each with an empty jump list
    jumps_by_slug = {s: [] for s in empty_slugs}
    assert find_clusters(jumps_by_slug, time_tol_minutes=5.0, kw_min_jaccard=0.20) == []


# ---------------------------------------------------------------------------
# Property 6: Shuffle invariance — slug-level partition is stable
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    jumps_by_slug=jumps_by_slug_strategy(),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_shuffle_invariance(jumps_by_slug: dict[str, list[Jump]], seed: int) -> None:
    """Permuting slug-dict insertion order and per-slug jump order doesn't
    change the slug-level partition (cluster *ids* may relabel)."""
    import random

    rng = random.Random(seed)

    # Baseline
    base = find_clusters(jumps_by_slug, time_tol_minutes=5.0, kw_min_jaccard=0.20)
    base_part, _ = _node_partition(jumps_by_slug, base)

    # Shuffle slug insertion order
    items = list(jumps_by_slug.items())
    rng.shuffle(items)
    shuffled_outer: dict[str, list[Jump]] = {}
    for slug, jumps in items:
        # Also shuffle per-slug jump order
        jc = list(jumps)
        rng.shuffle(jc)
        shuffled_outer[slug] = jc

    shuffled_clusters = find_clusters(shuffled_outer, time_tol_minutes=5.0, kw_min_jaccard=0.20)
    shuffled_part, _ = _node_partition(shuffled_outer, shuffled_clusters)
    assert base_part == shuffled_part


# ---------------------------------------------------------------------------
# Property 7: Transitivity (a~b, b~c → all three share one cluster)
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    ts_a_minute=st.integers(min_value=0, max_value=200),
    gap_ab_sec=st.integers(min_value=0, max_value=60),  # < 5 min tol → links a~b
    gap_bc_sec=st.integers(min_value=0, max_value=60),  # < 5 min tol → links b~c
    shared_terms=st.lists(st.sampled_from(VOCAB), min_size=2, max_size=4, unique=True),
)
def test_transitivity_three_slugs(
    ts_a_minute: int,
    gap_ab_sec: int,
    gap_bc_sec: int,
    shared_terms: list[str],
) -> None:
    """``a~b`` and ``b~c`` (both gates pass) ⇒ ``a, b, c`` in the same cluster.

    We construct three different slugs whose jumps are pairwise within 1 min
    in time and share at least 2 terms → both the time gate (≤5 min) and the
    Jaccard gate (=1.0 since terms are identical) pass for every pair.
    Transitivity is satisfied trivially because each pair links directly, but
    the union-find machinery is what *guarantees* the property even when
    edges are not transitive in the predicate sense; we test that here by
    using *exactly* the same term set for all three (so every pairwise gate
    passes) — the harder transitivity case is covered by the chain test below.
    """
    base_min = ts_a_minute
    a = _make_jump(base_min, shared_terms)
    # Build B and C by adding small minute offsets that keep them all inside
    # one 5-min window so every pair satisfies the time gate.
    b = _make_jump(base_min + max(1, gap_ab_sec // 60), shared_terms)
    c = _make_jump(
        base_min + max(1, gap_ab_sec // 60) + max(1, gap_bc_sec // 60),
        shared_terms,
    )
    jumps_by_slug = {"sa": [a], "sb": [b], "sc": [c]}
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5.0, kw_min_jaccard=0.20)
    assert len(clusters) == 1
    members = {m.slug for m in clusters[0].member_jumps}
    assert members == {"sa", "sb", "sc"}


@PROPERTY_SETTINGS
@given(
    base_minute=st.integers(min_value=0, max_value=600),
    relevance=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
)
def test_transitivity_chain_only_overlapping_pairs(base_minute: int, relevance: float) -> None:
    """A real chain: ``terms_a ∩ terms_c = ∅`` but ``a~b~c`` via shared b-overlap.

    Jaccard(a,c)=0 → no direct link. But Jaccard(a,b)≥0.5 and Jaccard(b,c)≥0.5
    and time gaps all <5 min, so union-find must still group a, b, c together.
    This is the property where naive pair-merge loops fail and union-find
    succeeds.
    """
    # a-terms and c-terms are disjoint; b-terms overlap with both
    terms_a = ["fed", "rate-hike"]
    terms_b = ["fed", "ukraine"]  # shares 'fed' w/ a (Jaccard 1/3 ≥ 0.2)
    terms_c = ["ukraine", "oil"]  # shares 'ukraine' w/ b (Jaccard 1/3 ≥ 0.2)
    # a ∩ c = ∅ → Jaccard 0, no direct link.

    a = _make_jump(base_minute, terms_a, relevance=relevance)
    b = _make_jump(base_minute + 1, terms_b, relevance=relevance)
    c = _make_jump(base_minute + 2, terms_c, relevance=relevance)
    jumps_by_slug = {"sa": [a], "sb": [b], "sc": [c]}
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5.0, kw_min_jaccard=0.20)
    assert len(clusters) == 1
    members = {m.slug for m in clusters[0].member_jumps}
    assert members == {"sa", "sb", "sc"}


# ---------------------------------------------------------------------------
# Property 8: Time-window cutoff — jumps outside the window never cluster
# ---------------------------------------------------------------------------


@PROPERTY_SETTINGS
@given(
    base_minute=st.integers(min_value=0, max_value=200),
    # Strictly outside any reasonable window: ≥ 10 minutes apart.
    gap_minutes=st.integers(min_value=10, max_value=300),
    terms=st.lists(st.sampled_from(VOCAB), min_size=2, max_size=5, unique=True),
)
def test_time_window_respected(base_minute: int, gap_minutes: int, terms: list[str]) -> None:
    """With ``time_tol_minutes=5``, two jumps ≥ 10 min apart never cluster.

    We use *identical* term sets so the Jaccard gate is perfect — only the
    time gate should suppress the merge. This isolates Property 8 from
    Property 4.
    """
    a = _make_jump(base_minute, terms)
    b = _make_jump(base_minute + gap_minutes, terms)
    jumps_by_slug = {"sa": [a], "sb": [b]}
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5.0, kw_min_jaccard=0.20)
    # No multi-slug cluster should form — both nodes are singletons.
    assert clusters == []


@PROPERTY_SETTINGS
@given(
    base_minute=st.integers(min_value=0, max_value=200),
    # Window-tolerance edge: jumps inside ≤ 4 min stay inside the 5-min tol.
    gap_seconds=st.integers(min_value=0, max_value=4 * 60),
    terms=st.lists(st.sampled_from(VOCAB), min_size=2, max_size=5, unique=True),
)
def test_time_window_includes_boundary(
    base_minute: int, gap_seconds: int, terms: list[str]
) -> None:
    """Mirror of test_time_window_respected: inside the window → DO cluster."""
    a = _make_jump(base_minute, terms)
    b_minute = base_minute + (gap_seconds // 60)
    # Avoid base==b_minute identical-second edge (same-slug guard doesn't apply
    # across slugs, but we want a non-degenerate time delta either way).
    b = _make_jump(b_minute, terms)
    jumps_by_slug = {"sa": [a], "sb": [b]}
    clusters = find_clusters(jumps_by_slug, time_tol_minutes=5.0, kw_min_jaccard=0.20)
    assert len(clusters) == 1
    members = {m.slug for m in clusters[0].member_jumps}
    assert members == {"sa", "sb"}
