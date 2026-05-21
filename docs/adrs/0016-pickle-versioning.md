# ADR-0016: Pickle Envelope Versioning for Redis L2 Cache

- **Status:** Accepted
- **Date:** 2026-05-16
- **Authors:** Damian Gallardo
- **References:** ADR-0004 (redis cache TTL), ADR-0008 (cache tiering), ADR-0011 (cache stampede single-flight)

## Context

The L2 (Redis) tier of our cache hierarchy stores serialised Python values produced by many heterogeneous code paths: `pfm.terminal.TTLCache` (jumps, leaderboard, sentiment, peer-scanner — Wave-2/Wave-3 era) and `pfm.cache_pool.CachePool` (T16, factor prewarm and `/alpha-hub/strategy/{pair_id}` envelopes). These caches share a single Redis instance to keep ops simple, but they evolved independently. Values include `pd.Series`, `np.ndarray`, dataclasses, nested dicts with `Timestamp` keys, and Pydantic models — none of which round-trip safely through `json.dumps(default=str)` (the original Wave-1 encoder), which silently stringified series and produced `dict[str, str]` on the read side, crashing downstream `.iloc` calls.

We switched to `pickle` for fidelity. But `pickle` is format-fragile: payloads written by one Python version (3.12 with `protocol=5`) loaded by another, or by a *different module* expecting a different shape, can silently corrupt or, worse, succeed-with-wrong-types. As schemas drift over time (model adds a field, dataclass renames a column) we also need to distinguish "this is an old layout" from "this is a fresh layout I should trust." A bare `pickle.loads()` cannot answer either question.

## Problem

1. **Legacy entries** written by the json-encoder era survived in Redis after we cut over to pickle. The first pickle reader on a legacy entry raised `UnpicklingError` inside hot request paths.
2. **Cross-prefix bleed**: two `CachePool` namespaces, or a `TTLCache` and a `CachePool` sharing a Redis key by accident, could each `pickle.loads()` the other's payload and unwrap into a wrong-typed object — no exception, just downstream mystery.
3. **Schema drift**: there is no in-band way to say "this entry's data layout is v1; bump to v2 means invalidate."

## Decision

Every L2 payload is wrapped in a fixed **magic-byte prefix** + **version byte** envelope before pickling. Two independent magics are shipped, one per cache class:

| Layout | Magic | Module | Wave |
|---|---|---|---|
| Terminal TTLCache pickled cache | `b"PFMTC1\x00"` | `pfm/terminal/__init__.py` | Wave-3 fix |
| CachePool L2 pickle | `b"PFMCP1\x00"` | `pfm/cache_pool.py` | T16 |

Both share the same envelope shape inside the pickle: `{"v": <int>, "data": <any>}`. The magic identifies *which cache wrote it*; the `v` identifies *which schema*. A read-side mismatch on either field raises `ValueError`, which callers treat as a cache miss and proceed to recompute. Stale entries thereby age out naturally.

### Implementation snippets

```python
# pfm/terminal/__init__.py
_L2_PAYLOAD_VERSION: int = 1
_L2_MAGIC: bytes = b"PFMTC1\x00"

@staticmethod
def _encode_l2(value: Any) -> bytes:
    envelope = {"v": _L2_PAYLOAD_VERSION, "data": value}
    return _L2_MAGIC + pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)

@staticmethod
def _decode_l2(raw: bytes | str) -> Any:
    if isinstance(raw, str):
        raw = raw.encode()
    if not raw.startswith(_L2_MAGIC):
        raise ValueError("legacy or unknown L2 payload (missing magic)")
    envelope = pickle.loads(raw[len(_L2_MAGIC):])
    if not isinstance(envelope, dict) or envelope.get("v") != _L2_PAYLOAD_VERSION:
        raise ValueError(f"unsupported L2 payload version: {envelope!r}")
    return envelope["data"]
```

```python
# pfm/cache_pool.py
_L2_MAGIC: bytes = b"PFMCP1\x00"
_L2_PAYLOAD_VERSION: int = 1
# encode/decode identical in shape; magic differs so the two caches cannot
# unwrap each other's blobs even on a key-collision.
```

## Round-trip discipline (W12-06 tests)

For every type we cache (`pd.Series`, `np.ndarray`, dataclasses, plain dicts, Pydantic models) the test suite asserts:

1. `set(k, v)` followed by `get(k)` returns a value equal to `v`.
2. `decode(encode(v)) == v` directly on the static methods.
3. An entry written by `TTLCache._encode_l2` is rejected by `CachePool._decode_l2` and vice versa — cross-prefix isolation is enforced, not assumed.

## Legacy compatibility

Pre-magic entries (json-encoded strings from Wave-1) are detected by the missing prefix check and rejected as `ValueError`. The caller treats this as a miss and overwrites the slot on the next `set`. No explicit Redis flush is required — natural migration completes within one TTL cycle (15 min for terminal; 1 h for prewarm).

## Forward compatibility

When a breaking schema change is required (e.g. a dataclass field is renamed), we bump `_L2_PAYLOAD_VERSION` from `1` to `2`. The decode path already has a version-mismatch branch that raises `ValueError`; v1 entries become misses and are rewritten as v2 on the next access. **Bumping the version requires an explicit invalidation note in the change PR** — adding a field with a default does not justify a bump; removing or retyping a field does.

A `version=2` path exists only as a stub today (the equality check `envelope.get("v") != _L2_PAYLOAD_VERSION` will trip on anything other than `1`). The "PFMTC2"/"PFMCP2" magics are reserved for a future *layout* change (e.g. adopting `msgpack` or framed pickle) — distinct from version bumps inside the current layout.

## Cross-prefix isolation

The two magics (`PFMTC1` vs `PFMCP1`) make wrong-cache reads loud rather than silent. If a `CachePool` instance reads a Redis key whose value happens to start with `PFMTC1`, the prefix check in `CachePool._decode_l2` raises immediately — there is no possibility of `pickle.loads()` succeeding into a structurally-similar but semantically-wrong object. This was the silent corruption mode we feared most.

## Path forward when we need PFMTC2 / PFMCP2

1. Add the new magic constant (e.g. `_L2_MAGIC_V2 = b"PFMTC2\x00"`).
2. Promote the decoder to a chain: try v2, fall back to v1, finally treat as miss. Encoder always writes the newest.
3. Add a migration test that loads a fixture written with v1 magic and confirms graceful miss on a v2-only reader.
4. Bump the constants in lockstep across `terminal/__init__.py` and `cache_pool.py` only if the schema change spans both caches; otherwise keep them independent.
5. Document the cutover in a follow-up ADR (ADR-0014+) explaining what specifically changed and the expected migration window.

## Consequences

- **Pro:** Silent corruption from cross-cache key collisions or stale json blobs is impossible; failures are loud `ValueError`s converted to cache misses.
- **Pro:** Future schema bumps have a well-defined protocol with explicit invalidation semantics.
- **Con:** Each entry pays a 7-byte overhead. Negligible vs typical 1–50 KB payloads.
- **Con:** Two magics to maintain. The cost is small (constants in two modules) and the isolation guarantee is worth it.
