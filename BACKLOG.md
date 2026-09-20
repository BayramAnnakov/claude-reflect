# Backlog

Known work that is real, understood, and not done. Each entry says **why it
matters** - the cost of leaving it - not just what it is. Reproduced, measured
or found-in-review earns a place here; speculation does not.

---

## 1. Rank repeat corrections above one-off redirects in `/reflect`

**Measured, 2026-09-19.** A census of 102 captured learnings on one heavy
user's machine (every `learnings-queue.json` under `~/.claude/projects/`,
not a sample):

| bucket | count |
|---|---|
| `"no, <one-off task redirect>"` | 55 |
| `"actually, <redirect>"` | 13 |
| already rejected by current `main` (the `"no need"` lane) | 20 |
| genuinely reusable rules | ~16 |

So roughly **20% of what reaches the queue is a reusable rule**. The rest are
corrections that were true for one moment and are worthless as memory.

The signal that separates them is already in the data and we throw it away:
**`"use unipile mcp"` appears four times**, in four different wordings, across
months. The detector caught it every time. It never reached a CLAUDE.md, so the
user kept retyping it. A one-off redirect almost never recurs; a real rule
recurs until it is written down.

**Why it matters:** `/reflect` currently presents the queue in timestamp order,
so a user wades through 80 one-offs to find the 16 rules. A queue that is
reliably mostly noise trains people to stop running `/reflect` at all - and
then the genuine corrections are discarded along with the noise. Clustering
near-duplicate learnings and surfacing the recurring ones first would raise the
hit rate without touching detection.

**Shape:** cluster queue items by semantic similarity at `/reflect` time (the
semantic layer is already there in `scripts/lib/semantic_detector.py`), show
clusters of size >= 2 first, and label them with the recurrence count.

---

## 2. Regex cannot separate a rule from a moment - stop trying to

Related to #1, and a constraint to hold in mind whenever a new
`FALSE_POSITIVE_PATTERNS` entry is proposed.

`"no, use unipile mcp"` (a rule) and `"no, lets discuss first, show me"` (a
moment) are structurally identical. No opener, length or punctuation heuristic
tells them apart - only meaning does.

**Measured against the same 102 items:** the three false-positive PRs open in
September 2026 (#37, #44, and the fix for #43) between them would have rejected
**1** of those 102. Each targets a shape that was real in its reporter's
project and absent from this one. They are still worth merging - they cost
nothing and remove genuine noise - but the lesson is that the returns are
per-user and small.

**Why it matters:** each new regex adds a false-negative risk for somebody
else's phrasing and moves the precision needle by about one item per hundred.
Effort is better spent on the semantic pass and on #1.

---

## 3. No test runs the hooks the way Claude Code runs them

CI's smoke step pipes `{"prompt":"test"}` into each hook. `"test"` is not a
correction, so detection returns `None` and **the entire capture-and-write path
is never executed in CI** - queue creation, folder encoding, `save_queue`, and
the confirmation `print()` are all unreached.

**Why it matters:** this is exactly how two shipped bugs survived. The
`WinError 267` folder-encoding crash (#38, #41) and the `UnicodeEncodeError` on
the `📝` confirmation both live past the branch that smoke test takes. Both
were reported by users, twice each, while CI stayed green.

**Shape:** pipe a real correction (`{"prompt":"no, use python not python3"}`)
through `capture_learning.py` against a throwaway `HOME`, on all three CI
platforms, and assert the queue file exists at the expected path with the
expected item, and that stderr is empty.

**Found by the clean-slate run, 2026-09-19:** the suite also had an undeclared
`jq` dependency. The GitHub runners ship `jq`, so CI never saw it; on a bare
Linux box six tests failed and - worse - the `test_bash_ignores_*` tests
*passed* for the wrong reason, because a `jq`-less script emits nothing either
way. Now skipped explicitly when `jq` is absent. The remaining gap is above:
nothing runs the Python hooks end to end.

---

## 4. Duplicate-project-path collisions after the encoder fix

`_encode_project_path` maps every non-alphanumeric character to `-`, matching
Claude Code. That means `/Users/bob/my_app` and `/Users/bob/my-app` collide on
one folder, and their queues merge.

This is Claude Code's own behaviour and we must match it, so the collision is
not ours to fix - but queue items carry a `project` field, so `/reflect` could
warn when one queue holds items from more than one project path.

**Why it matters:** low frequency, but the failure is confusing when it lands -
a learning from one repo is offered as a target in another.

---

## 5. `remember:` is unused in practice

Zero of the 102 captured items on the audited machine used the explicit
`remember:` marker, the one path with 0.90 confidence and no false-positive
risk. Every capture came from implicit detection.

**Why it matters:** the highest-precision input we have is invisible to users.
Worth one line in `SessionStart` output or the README before investing further
in implicit detection.

---

## 6. Findings from the 2026-09-19 review gate that were NOT fixed

codex (GPT-6) and Grok 4.6 reviewed the v3.2.0 change independently. The ship
blockers are fixed; these are real, verified, and deliberately left.

**Concurrent captures can still drop one.** `load_queue` → append →
`save_queue` has no lock, so two Claude Code sessions in the same project can
each read, append and write, and the second overwrites the first. Writes are
now atomic (no torn file), but the read-modify-write race predates this change
and is not closed. Fixing it needs a lockfile or an append-only journal.
*Cost of leaving it:* a lost learning, silently, only when two sessions in one
project capture within the same instant.

**`~/.claude` is inside the inclusion-graph allowlist.** A cloned repo's
`CLAUDE.md` can write `@~/.claude/projects/<other-project>/memory/general.md`
and `/reflect` will offer that file as a write target. The tests call this a
deliberate trust boundary, but the trusting party is whoever cloned the repo.
*Cost of leaving it:* a hostile repo can steer a learning into another
project's memory. Needs a decision on whether `~/.claude` belongs in the
allowlist at all.

**No timeout on inclusion reads.** `Path.resolve()`, `is_file()` and `open()`
have no deadline, so a FIFO named `notes.md` or a link to an unreachable UNC
share can hang `/reflect`. Claude Code itself had to special-case FIFOs.
*Cost of leaving it:* `/reflect` hangs with no message, in a repo the user
did not write.

**`max_nodes` counts successes, not attempts.** codex measured 5,000 path
resolutions with `max_nodes=1` when references are repeatedly missing.
*Cost of leaving it:* a slow `/reflect` on a pathological repo; bounded, not
unbounded.

**`_EXTERNAL_SCHEME_RE` treats `C:/Users/...` as a URL scheme**, so Windows
absolute markdown links are silently skipped.

**Inclusion traversal reaches `node_modules` and `.git`.** `EXCLUDED_DIRS`
applies to the `os.walk` discovery pass but not to `@`-includes and markdown
links, so `[pkg](node_modules/pkg/README.md)` and `@.git/notes.md` both surface
as `/reflect` write targets (Fable, executed).
*Cost of leaving it:* `/reflect` can be steered into writing a learning into a
dependency file or into `.git`.

**Auto-memory is keyed on cwd, Claude Code keys it on the git root.** Claude
Code resolves the canonical worktree root for the project; `get_auto_memory_path`
uses cwd. A session started in a subdirectory or a worktree writes memory where
Claude never reads it, and the migration moves memory files to that same wrong
place (Fable, from the 2.1.278 binary; not independently probed).
*Cost of leaving it:* auto-memory silently does nothing for subdirectory
sessions - the same failure class as the folder-encoding bug, one level up.

**Standalone `AGENTS.md` is not an inclusion-graph seed** (codex #12).

**`ensure_utf8_io` has no test and does not set `errors="replace"`.** It is a
no-op on Python 3.6, which the README still claims to support. The capture
confirmation is ASCII now, but `session_start_reminder.py` still prints
`⚠️ 📚 💡`.
*Cost of leaving it:* the function can be gutted and CI stays green.

**Two projects whose paths differ only by a character that encodes to a dash
share one folder** — see entry 4. Real on this machine today:
`-Users-bayramannakov-GH-darwin_new` (plugin-only) will migrate into
`-Users-bayramannakov-GH-darwin-new` (128+ session files) on the next
`load_queue()` there. That is the intended, Claude-Code-matching behaviour,
but `/reflect` should warn when one queue holds items from several project
paths.
