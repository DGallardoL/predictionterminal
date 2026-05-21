#!/usr/bin/env bash
# restore.sh — restore critical project state from a pfm-backup tar.gz.
#
# Before extracting, this script runs backup.sh to snapshot the current state
# (so a botched restore is always reversible). It then extracts the supplied
# archive into the repo root, preserving relative paths.
#
# Usage:
#   bash api/scripts/restore.sh <path-to-pfm-backup.tar.gz>
#
# Exit codes:
#   1 — missing or invalid argument
#   2 — archive not found / unreadable
#   3 — pre-restore backup failed
#   4 — tar extraction failed

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "restore.sh: missing archive path" >&2
  echo "usage: bash api/scripts/restore.sh <pfm-backup.tar.gz>" >&2
  exit 1
fi

ARCHIVE="$1"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "restore.sh: archive not found: ${ARCHIVE}" >&2
  exit 2
fi

if ! tar -tzf "${ARCHIVE}" >/dev/null 2>&1; then
  echo "restore.sh: archive is not a readable gzipped tar: ${ARCHIVE}" >&2
  exit 2
fi

# Resolve repo root (parent of api/scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
PRE_RESTORE_OUT="/tmp/pfm-prerestore-${TS}.tar.gz"

echo "restore.sh: taking pre-restore snapshot -> ${PRE_RESTORE_OUT}"
if ! bash "${SCRIPT_DIR}/backup.sh" "${PRE_RESTORE_OUT}"; then
  echo "restore.sh: pre-restore backup failed; aborting before extraction." >&2
  exit 3
fi

# Show what is about to be restored.
echo "restore.sh: archive contents:"
tar -tzf "${ARCHIVE}" | sed 's/^/  /'

# Extract into repo root. The archive uses repo-relative paths already.
# --no-same-owner avoids permission errors when restoring across machines.
EXTRACT_ARGS=(-xzf "${ARCHIVE}" -C "${REPO_ROOT}" --no-same-owner)
if ! tar "${EXTRACT_ARGS[@]}"; then
  echo "restore.sh: extraction failed. Pre-restore snapshot is at ${PRE_RESTORE_OUT}" >&2
  exit 4
fi

# MANIFEST.txt is part of the archive metadata; move it out of the repo so it
# does not pollute the tree.
if [[ -f "${REPO_ROOT}/MANIFEST.txt" ]]; then
  mv "${REPO_ROOT}/MANIFEST.txt" "/tmp/pfm-restore-manifest-${TS}.txt"
  echo "restore.sh: manifest moved to /tmp/pfm-restore-manifest-${TS}.txt"
fi

echo "restore.sh: restore complete from ${ARCHIVE}"
echo "restore.sh: pre-restore snapshot preserved at ${PRE_RESTORE_OUT}"
