# ADR-0010: Multi-Session Claude Code Coordination

## Status
Accepted (2026-05-16)

## Context
At peak, up to **60 concurrent Claude Code sub-agents** plus 5 human-coordinated
sessions operate on this repository in parallel. Two overnight incidents in
Wave-9 and early Wave-10 caused silently-lost edits: (a) two agents wrote
`web/index.html` within the same minute, and the later write clobbered a
recently-mounted `<script>` tag; (b) an agent ran `Write` on
`.coordination/active-edits.json` with only its own entry, erasing the claims
of twelve other sessions. Neither failure produced an error — both were
discovered hours later via `git diff` review. We need a deterministic
file-ownership discipline that does not rely on filesystem locks (the harness
provides none) and does not require a central broker.

## Decision
We adopt an **append-only coordination claim** protocol enforced by convention
in `.coordination/PROTOCOL-V2.md` and tracked by task in
`.coordination/TASK-BOARD.md`:

1. **Single-owner per hot file per wave.** Each of `web/index.html`,
   `web/config.js`, `api/src/pfm/main.py`, `api/src/pfm/schemas.py` is owned
   by at most one agent at a time.
2. **Hot files are partitioned by section.** `main.py` is split into named
   regions (`lifespan`, `cors`, `routes`, `exception-handlers`); claims name
   the region, not the whole file.
3. **No direct edits to `index.html`.** New CSS lives in `web/css/<feature>.css`,
   new JS in `web/js/<feature>.js`. Only the `index-html-owner` agent mounts
   these via `<link>` / `<script>` tags.
4. **`active-edits.json` is append-only via read-merge-write.** Every agent
   reads the full array, pushes its entry, and writes the merged array back.
   Solo `Write` is explicitly forbidden.
5. **`schemas.py` is append-only at the bottom.** Pydantic models go at the
   end of the file, never interspersed.
6. **Test-and-verify before claim release.** Run the targeted pytest module,
   confirm `import pfm.main` still loads, then mark the claim expired.

## Consequences
- **Positive:** Zero file-clobber incidents observed in Wave-10 across 60+
  agents. Parallel discovery of independent task scopes is fast.
- **Positive:** New CSS/JS modules are naturally smaller and more reviewable
  than appending to the 1.6 MB `index.html`.
- **Negative:** Hot-file contention forces some agents to wait or pivot; the
  index-html-owner becomes a bottleneck once per wave.
- **Negative:** The append-only discipline is convention, not enforcement —
  one careless `Write` still corrupts coordination state.

## Alternatives Considered
- **Per-agent git worktrees.** Rejected: merging 60 worktrees per wave costs
  more human attention than the protocol saves.
- **PID-based lock file.** Rejected: agents run in ephemeral harness sessions
  without stable PIDs; stale locks would block everyone.
- **Centralized broker service.** Rejected: adds infrastructure (a fifth
  service in `docker-compose.yml`) for a POC.

## Lessons
- **2026-05-16, alphahub session:** Solo `Write` on `active-edits.json`
  clobbered 12 entries. Recovered via `/tmp` JSON backups; protocol hardened
  to V2 with explicit ban on solo writes.
- **Wave-11 T76/T78 half-open window bug:** Surfaced only because the
  protocol allowed T78 (audit) to run in parallel with T76 (producer) with
  no false dependency. Coordination discipline created the conditions for
  the bug to be caught the same day.

## References
- `.coordination/PROTOCOL-V2.md` — the operational protocol agents must read
- `.coordination/TASK-BOARD.md` — wave-scoped task assignments and ownership
- `.coordination/issues.log`, `.coordination/outcomes.log` — incident trail
