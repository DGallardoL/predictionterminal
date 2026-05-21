#!/usr/bin/env bash
# Regenerate the golden-file snapshots used by tests/test_golden_endpoints.py.
#
# Workflow
# --------
#   1. Wipe everything under api/tests/golden/.
#   2. Run the golden suite once.  Every test fails-on-first-run because the
#      helper writes the new golden and aborts the test (deliberate, so we
#      never silently accept a regenerated snapshot).
#   3. Run the golden suite a second time.  All tests should now pass.
#   4. Inspect with `git diff api/tests/golden/` and commit if intentional.
#
# Usage:  ./scripts/regenerate_golden.sh
#
# This script lives at scripts/regenerate_golden.sh and is referenced from
# api/tests/golden_helper.py — keep both in sync if you rename it.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
API_DIR="${REPO_ROOT}/api"
GOLDEN_DIR="${API_DIR}/tests/golden"

if [[ ! -d "${API_DIR}/.venv" ]]; then
    echo "error: ${API_DIR}/.venv not found — create it first (python -m venv .venv && pip install -r requirements.txt)" >&2
    exit 1
fi

PYTHON="${API_DIR}/.venv/bin/python"

echo "[1/3] wiping ${GOLDEN_DIR}/*.json"
rm -f "${GOLDEN_DIR}"/*.json
mkdir -p "${GOLDEN_DIR}"

echo "[2/3] first run — pytest writes the new goldens (each test will FAIL on purpose)"
cd "${API_DIR}"
"${PYTHON}" -m pytest tests/test_golden_endpoints.py --tb=line -q || true

echo "[3/3] second run — verify the new goldens reproduce"
"${PYTHON}" -m pytest tests/test_golden_endpoints.py --tb=short -q

echo
echo "done.  Inspect changes with: git diff -- api/tests/golden/"
