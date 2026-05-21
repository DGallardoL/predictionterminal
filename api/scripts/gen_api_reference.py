"""Generate ``docs/API_REFERENCE.md`` from the live (or cached) OpenAPI schema.

Usage
-----
    # 1) Hit a running API on :8000 (default)
    python scripts/gen_api_reference.py

    # 2) Hit a different URL
    python scripts/gen_api_reference.py --openapi-url http://localhost:8001/openapi.json

    # 3) Read from a previously saved file
    python scripts/gen_api_reference.py --from-file docs/openapi.json

    # 4) Write somewhere other than docs/API_REFERENCE.md
    python scripts/gen_api_reference.py --out docs/API_REFERENCE.md

The script groups endpoints by the **first path segment** (``/terminal/*``,
``/strategies/*``, ``/alpha-hub/*``, ...). Each group gets a Table of Contents
and a per-endpoint section with summary, parameters, response schema and an
example ``curl`` invocation.

CI integration (separate task; documented here so the next maintainer can wire
it up):

* Add a step to ``.github/workflows/ci.yml`` (e.g. a new ``docs-up-to-date``
  job) that:
    1. Boots the API or imports the FastAPI app in-process.
    2. Runs ``python scripts/gen_api_reference.py --out /tmp/API_REFERENCE.md``
       (or with ``--from-file`` against a freshly dumped schema).
    3. ``diff -u docs/API_REFERENCE.md /tmp/API_REFERENCE.md``
    4. Fails the build if the diff is non-empty (i.e. the doc is stale).
* Recommended one-liner inside CI when the API is not running::

    python -c "from pfm.main import app; import json; print(json.dumps(app.openapi()))" \
        > /tmp/openapi.json
    python scripts/gen_api_reference.py --from-file /tmp/openapi.json \
        --out /tmp/API_REFERENCE.md
    diff -u docs/API_REFERENCE.md /tmp/API_REFERENCE.md

If the diff fails, contributors regenerate locally with ``python
scripts/gen_api_reference.py`` and commit the updated ``docs/API_REFERENCE.md``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_from_url(url: str, timeout: float = 10.0) -> dict[str, Any]:
    """Fetch and parse the OpenAPI schema from ``url``."""
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def load_from_file(path: Path) -> dict[str, Any]:
    """Read OpenAPI schema from a local JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_from_app() -> dict[str, Any]:
    """Last-resort: import the FastAPI app and call ``app.openapi()``."""
    try:
        from pfm.main import app  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on env
        raise SystemExit(
            "Could not import pfm.main; run from api/ with PYTHONPATH=src, "
            f"or pass --from-file. Original error: {exc!r}"
        ) from exc
    return app.openapi()


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head", "options")


def first_segment(path: str) -> str:
    """Return the first path segment for grouping, e.g. ``/terminal/foo`` -> ``terminal``."""
    parts = [p for p in path.split("/") if p]
    return parts[0] if parts else "root"


def slugify_anchor(method: str, path: str) -> str:
    """Build a stable GitHub-flavoured Markdown anchor for ``METHOD PATH``."""
    raw = f"{method.lower()} {path.lower()}"
    out = []
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_"):
            out.append("-")
        # drop everything else: braces, slashes, dots, commas...
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def resolve_ref(schema: dict[str, Any], ref: str) -> dict[str, Any]:
    """Resolve a ``$ref`` like ``#/components/schemas/Foo`` against ``schema``."""
    if not ref.startswith("#/"):
        return {}
    node: Any = schema
    for part in ref[2:].split("/"):
        if not isinstance(node, dict) or part not in node:
            return {}
        node = node[part]
    return node if isinstance(node, dict) else {}


def example_for_schema(node: dict[str, Any], schema: dict[str, Any], depth: int = 0) -> Any:
    """Synthesize a tiny example value from a JSON-Schema node (depth-capped)."""
    if depth > 4 or not isinstance(node, dict):
        return None
    if "$ref" in node:
        return example_for_schema(resolve_ref(schema, node["$ref"]), schema, depth + 1)
    if "example" in node:
        return node["example"]
    if "default" in node:
        return node["default"]
    if node.get("enum"):
        return node["enum"][0]
    if "anyOf" in node:
        for sub in node["anyOf"]:
            if isinstance(sub, dict) and sub.get("type") != "null":
                return example_for_schema(sub, schema, depth + 1)
        return None
    if node.get("oneOf"):
        return example_for_schema(node["oneOf"][0], schema, depth + 1)
    if "allOf" in node:
        merged: dict[str, Any] = {}
        for sub in node["allOf"]:
            if isinstance(sub, dict):
                merged.update(sub)
        return example_for_schema(merged, schema, depth + 1)

    t = node.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t == "object":
        props = node.get("properties", {}) or {}
        return {k: example_for_schema(v, schema, depth + 1) for k, v in list(props.items())[:6]}
    if t == "array":
        return [example_for_schema(node.get("items", {}), schema, depth + 1)]
    if t == "string":
        return node.get("format") or "string"
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return True
    return None


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

GROUP_TITLES: dict[str, str] = {
    "terminal": "Terminal",
    "strategies": "Strategies",
    "alpha-hub": "Alpha Hub",
    "alpha": "Alpha",
    "archive": "Archive",
    "factors": "Factors",
    "auth": "Auth",
    "macro": "Macro",
    "arb": "Arbitrage",
    "reverse-finder": "Reverse Finder",
    "advanced-model": "Advanced Model",
    "event-model": "Event Model",
    "multi-event": "Multi-Event",
    "news": "News",
    "indices": "Indices",
    "quant": "Quant",
    "lab": "Lab",
    "signals": "Signals",
    "alerts": "Alerts",
    "embed": "Embed",
    "replay": "Replay",
    "fit": "Fit",
    "attribution": "Attribution",
    "health": "Health",
    "root": "Root",
}


def group_title(key: str) -> str:
    return GROUP_TITLES.get(key, key.replace("-", " ").title())


def render_params(params: list[dict[str, Any]], schema: dict[str, Any]) -> str:
    """Render the Parameters table for one endpoint."""
    if not params:
        return "**Parameters**: (none)\n"
    lines = [
        "**Parameters**:",
        "",
        "| Name | In | Type | Required | Description |",
        "| --- | --- | --- | --- | --- |",
    ]
    for p in params:
        if "$ref" in p:
            p = resolve_ref(schema, p["$ref"])
        name = p.get("name", "")
        loc = p.get("in", "")
        pschema = p.get("schema") or {}
        if "$ref" in pschema:
            pschema = resolve_ref(schema, pschema["$ref"])
        ptype = (
            pschema.get("type") or pschema.get("anyOf", [{}])[0].get("type", "any")
            if pschema
            else ""
        )
        req = "yes" if p.get("required") else "no"
        desc = (p.get("description") or "").strip().replace("\n", " ").replace("|", "\\|")
        if not desc:
            desc = ""
        lines.append(f"| `{name}` | {loc} | {ptype or 'any'} | {req} | {desc} |")
    lines.append("")
    return "\n".join(lines)


def render_request_body(op: dict[str, Any], schema: dict[str, Any]) -> str:
    """Render the Request body section if present."""
    body = op.get("requestBody")
    if not body:
        return ""
    content = body.get("content") or {}
    if not content:
        return ""
    media, media_obj = next(iter(content.items()))
    body_schema = media_obj.get("schema") or {}
    if "$ref" in body_schema:
        body_schema = resolve_ref(schema, body_schema["$ref"])
    example = example_for_schema(body_schema, schema)
    lines = [
        "**Request Body** (`" + media + "`):",
        "",
        "```json",
        json.dumps(example, indent=2, default=str),
        "```",
        "",
    ]
    return "\n".join(lines)


def render_response(op: dict[str, Any], schema: dict[str, Any]) -> str:
    """Render the Response 200 section (or first 2xx)."""
    responses = op.get("responses") or {}
    if not responses:
        return "**Response**: (no schema declared)\n"
    # Prefer 200, then any 2xx, then default
    code = None
    for candidate in ("200", "201", "202", "204"):
        if candidate in responses:
            code = candidate
            break
    if code is None:
        for k in responses:
            if k.startswith("2"):
                code = k
                break
    if code is None:
        code = next(iter(responses))
    resp = responses.get(code) or {}
    content = resp.get("content") or {}
    if not content:
        desc = (resp.get("description") or "").strip()
        return f"**Response {code}**: {desc or '(no body)'}\n"
    media, media_obj = next(iter(content.items()))
    body_schema = media_obj.get("schema") or {}
    if "$ref" in body_schema:
        body_schema = resolve_ref(schema, body_schema["$ref"])
    example = example_for_schema(body_schema, schema)
    return (
        f"**Response {code}** (`{media}`):\n\n"
        "```json\n" + json.dumps(example, indent=2, default=str) + "\n```\n"
    )


def render_curl(method: str, path: str, op: dict[str, Any], schema: dict[str, Any]) -> str:
    """Render a representative ``curl`` invocation."""
    base = "http://localhost:8000"
    # Substitute path params with placeholders
    rendered_path = path
    query_pieces: list[str] = []
    for p in op.get("parameters") or []:
        if "$ref" in p:
            p = resolve_ref(schema, p["$ref"])
        if p.get("in") == "path":
            name = p.get("name")
            if name:
                rendered_path = rendered_path.replace(f"{{{name}}}", f"<{name}>")
        elif p.get("in") == "query" and p.get("required"):
            name = p.get("name") or "q"
            pschema = p.get("schema") or {}
            if "$ref" in pschema:
                pschema = resolve_ref(schema, pschema["$ref"])
            ex = example_for_schema(pschema, schema)
            if ex is None:
                ex = f"<{name}>"
            query_pieces.append(f"{name}={ex}")
    qs = ("?" + "&".join(query_pieces)) if query_pieces else ""
    url = f"{base}{rendered_path}{qs}"

    parts = ["curl"]
    method_upper = method.upper()
    if method_upper != "GET":
        parts += ["-X", method_upper]
    parts += [shlex.quote(url)]

    body = op.get("requestBody")
    if body:
        content = body.get("content") or {}
        if "application/json" in content:
            body_schema = content["application/json"].get("schema") or {}
            if "$ref" in body_schema:
                body_schema = resolve_ref(schema, body_schema["$ref"])
            example = example_for_schema(body_schema, schema)
            parts += [
                "-H",
                shlex.quote("Content-Type: application/json"),
                "-d",
                shlex.quote(json.dumps(example, default=str)),
            ]

    return "```bash\n" + " ".join(parts) + "\n```\n"


def render_endpoint(method: str, path: str, op: dict[str, Any], schema: dict[str, Any]) -> str:
    """Render one full endpoint section."""
    method_u = method.upper()
    header = f"### {method_u} {path}\n"
    summary = (op.get("summary") or "").strip()
    description = (op.get("description") or "").strip()
    lines = [header]
    if summary:
        lines.append(f"**Summary**: {summary}\n")
    if description and description != summary:
        # Keep description compact: first paragraph only.
        first_para = description.split("\n\n", 1)[0].strip()
        lines.append(first_para + "\n")
    lines.append(render_params(op.get("parameters") or [], schema))
    rb = render_request_body(op, schema)
    if rb:
        lines.append(rb)
    lines.append(render_response(op, schema))
    lines.append("**Example**:\n")
    lines.append(render_curl(method, path, op, schema))
    return "\n".join(lines)


def render_document(schema: dict[str, Any], source: str) -> str:
    """Render the full Markdown document."""
    paths = schema.get("paths") or {}

    # Group endpoints by first segment
    groups: dict[str, list[tuple[str, str, dict[str, Any]]]] = defaultdict(list)
    for path, ops in sorted(paths.items()):
        if not isinstance(ops, dict):
            continue
        for method, op in ops.items():
            if method.lower() not in _HTTP_METHODS:
                continue
            if not isinstance(op, dict):
                continue
            groups[first_segment(path)].append((method.lower(), path, op))

    # Sort groups: known order first, then alphabetical
    known_order = [
        "terminal",
        "strategies",
        "alpha-hub",
        "alpha",
        "archive",
        "factors",
        "auth",
        "macro",
        "arb",
        "reverse-finder",
        "advanced-model",
        "event-model",
        "multi-event",
        "news",
        "indices",
        "quant",
        "lab",
        "signals",
        "alerts",
        "embed",
        "replay",
        "fit",
        "attribution",
        "health",
    ]
    ordered_keys = [k for k in known_order if k in groups]
    ordered_keys += sorted(k for k in groups if k not in known_order)

    total_endpoints = sum(len(v) for v in groups.values())
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    api_title = schema.get("info", {}).get("title", "Prediction Terminal API")
    api_version = schema.get("info", {}).get("version", "")

    out: list[str] = []
    out.append(f"# {api_title} Reference\n")
    out.append(
        f"Auto-generated from `{source}` on {now}. "
        f"Do not edit by hand — re-run `python scripts/gen_api_reference.py`.\n"
    )
    if api_version:
        out.append(f"**API version**: `{api_version}`  ")
    out.append(f"**Total endpoints**: {total_endpoints}  ")
    out.append(f"**Groups**: {len(ordered_keys)}\n")

    # Group-level TOC
    out.append("## Endpoints by Group\n")
    for key in ordered_keys:
        items = groups[key]
        anchor = f"#{slugify_anchor('', key)}-" + f"{len(items)}-endpoints"
        out.append(f"### {group_title(key)} ({len(items)} endpoints)")
        for method, path, op in sorted(items, key=lambda x: (x[1], x[0])):
            summary = (op.get("summary") or "").strip()
            link = "#" + slugify_anchor(method, path)
            tail = f" — {summary}" if summary else ""
            out.append(f"- [`{method.upper()} {path}`]({link}){tail}")
        out.append("")

    # Endpoint Details
    out.append("## Endpoint Details\n")
    for key in ordered_keys:
        items = groups[key]
        out.append(f"## {group_title(key)}\n")
        for method, path, op in sorted(items, key=lambda x: (x[1], x[0])):
            out.append(render_endpoint(method, path, op, schema))
            out.append("")

    return "\n".join(out).rstrip() + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--openapi-url",
        default="http://localhost:8000/openapi.json",
        help="URL of the running API's /openapi.json (default: %(default)s).",
    )
    src.add_argument(
        "--from-file",
        type=Path,
        help="Path to a local openapi.json instead of fetching over HTTP.",
    )
    src.add_argument(
        "--from-app",
        action="store_true",
        help="Import pfm.main:app in-process and call app.openapi(). Requires PYTHONPATH=src.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "docs" / "API_REFERENCE.md",
        help="Output Markdown path (default: docs/API_REFERENCE.md).",
    )
    args = parser.parse_args(argv)

    if args.from_file is not None:
        schema = load_from_file(args.from_file)
        source = str(args.from_file)
    elif args.from_app:
        schema = load_from_app()
        source = "pfm.main:app.openapi()"
    else:
        try:
            schema = load_from_url(args.openapi_url)
            source = args.openapi_url
        except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
            print(
                f"[gen_api_reference] could not reach {args.openapi_url}: {exc}; "
                "falling back to in-process app.openapi()...",
                file=sys.stderr,
            )
            schema = load_from_app()
            source = "pfm.main:app.openapi() (fallback)"

    text = render_document(schema, source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    paths = schema.get("paths") or {}
    n_endpoints = sum(
        1
        for ops in paths.values()
        if isinstance(ops, dict)
        for m in ops
        if m.lower() in _HTTP_METHODS
    )
    print(
        f"[gen_api_reference] wrote {args.out} "
        f"({len(paths)} paths, {n_endpoints} endpoints) from {source}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
