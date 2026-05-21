#!/usr/bin/env bash
# backup.sh — snapshot critical project state into a single tar.gz.
#
# Captures coordination state, the factor catalog, alpha hub data, live signals,
# and the cross-venue arb dashboard state (if present). Output lands in
# /tmp/pfm-backup-<UTC-timestamp>.tar.gz.
#
# Usage:
#   bash api/scripts/backup.sh [output_path]
#
# Exits non-zero only on tar failure. Missing optional files are skipped with a
# warning so a backup can still be taken in a partially-initialised repo.

set -euo pipefail

# Resolve repo root (parent of api/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
DEFAULT_OUT="/tmp/pfm-backup-${TS}.tar.gz"
OUT="${1:-${DEFAULT_OUT}}"

# Files to back up. Order is intentional: coordination first, then domain data.
CANDIDATES=(
  ".coordination/active-edits.json"
  ".coordination/active-edits-archive.jsonl"
  "api/src/pfm/factors.yml"
  "web/data/alpha_strategies.json"
  "web/data/alpha_graveyard.json"
  "web/data/live_signals.json"
  "arbstuff/dashboard_state.json"
)

INCLUDED=()
MISSING=()
for f in "${CANDIDATES[@]}"; do
  if [[ -f "${f}" ]]; then
    INCLUDED+=("${f}")
  else
    MISSING+=("${f}")
  fi
done

if [[ ${#INCLUDED[@]} -eq 0 ]]; then
  echo "backup.sh: no candidate files found under ${REPO_ROOT}; aborting." >&2
  exit 1
fi

# Stage a MANIFEST.txt at repo root (deleted immediately after the archive is
# written). Using a relative path keeps the in-archive name portable across
# GNU and BSD tar without --transform / -s gymnastics.
MANIFEST_PATH="${REPO_ROOT}/MANIFEST.txt"
if [[ -e "${MANIFEST_PATH}" ]]; then
  echo "backup.sh: refusing to overwrite existing ${MANIFEST_PATH}" >&2
  exit 1
fi
trap 'rm -f "${MANIFEST_PATH}"' EXIT
{
  echo "pfm-backup manifest"
  echo "created_utc=${TS}"
  echo "repo_root=${REPO_ROOT}"
  echo "host=$(hostname)"
  echo "user=$(whoami)"
  echo "---"
  echo "included:"
  for f in "${INCLUDED[@]}"; do
    size=$(wc -c <"${f}" | tr -d ' ')
    echo "  ${f} (${size} bytes)"
  done
  if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "missing (skipped):"
    for f in "${MISSING[@]}"; do
      echo "  ${f}"
    done
  fi
} >"${MANIFEST_PATH}"

tar -czf "${OUT}" "MANIFEST.txt" "${INCLUDED[@]}"

SIZE=$(wc -c <"${OUT}" | tr -d ' ')
echo "backup.sh: wrote ${OUT} (${SIZE} bytes, ${#INCLUDED[@]} files)"
if [[ ${#MISSING[@]} -gt 0 ]]; then
  echo "backup.sh: skipped ${#MISSING[@]} missing files: ${MISSING[*]}" >&2
fi
