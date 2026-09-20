# Changelog

All notable changes to claude-reflect will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.2.0] - 2026-09-19

Bug-fix release with one new feature. The headline is that **`/reflect` was
reading the wrong file** and **the queue was being written to the wrong
folder** — two independent faults that each made the plugin silently do
nothing for a large set of users. Neither errored.

### Added
- **Referenced docs as `/reflect` targets** (#35, thanks @alonl; closes #34)
  - `find_claude_files()` now follows `@`-includes and inline markdown links out of your memory files and offers the `.md` docs it reaches as routing targets, so a learning that belongs in `docs/standards.md` no longer drifts into `CLAUDE.md`.
  - Bounded: depth 3, 200 nodes, 1 MiB per file, cycle-safe, `.md` only, and resolved paths confined to the project root plus `~/.claude`.
- **`scripts/project_paths.py`** — prints the resolved queue, session and memory paths for a project as JSON. Slash commands call this instead of deriving paths in shell.
- **`scripts/clear_queue.py`** — clears the per-project queue through `save_queue()`.

### Fixed
- **`/reflect` read and cleared the pre-3.1 *global* queue**
  - `commands/reflect.md` still named `~/.claude/learnings-queue.json` in five places — a file `migrate_global_queue()` deletes on first access. So `/reflect` processed nothing, "cleared" an empty file, and left the real per-project queue intact; the same learnings reappeared every run. The per-project change (#21, v3.1.0) moved the Python and left the skill behind.
  - `/reflect-skills` encoded the session folder with `sed 's|/|-|g'`, and `--scan-history` grepped `~/.claude/projects` by basename with a `tr '_' '-'` guess. Three hand-rolled encoders, all drifted. All now call `project_paths.py`.
- **Project folder encoding sent queues to a folder Claude Code never reads** (#41, thanks @George-tmm; Windows crash also reported in #38 by @keitaemsden-lab)
  - `get_project_folder_name()` replaced only path separators. Claude Code replaces *every* non-alphanumeric character with `-`, verified against 209 of its own session folders (May–Sep 2026).
  - Any project path holding `_`, `.` or a space (`/Users/bob/my_app`) wrote its queue and auto-memory to `-Users-bob-my_app` while sessions lived in `-Users-bob-my-app`. Nothing errored — `--scan-history` searched the wrong folder, found no `*.jsonl`, and reported nothing. Affected macOS and Linux, not just Windows.
  - On Windows the drive colon also survived (`-C:-Users-bob-app`), making `mkdir` raise `WinError 267`. The hook's top-level handler swallowed it, so capture failed silently and no queue was ever created.
  - Three further divergences, each established with a live `claude -p` probe: the cwd is normalized to **NFC** first (an NFD path from Finder or unzip is one code point longer and mis-encodes); the substitution counts **UTF-16 code units**, so an emoji becomes two dashes; and a name over **200 characters** is truncated and given a base-36 hash, which is now reproduced.
  - The capture hook now takes the folder from the hook payload's `transcript_path` rather than deriving it, since the transcript already lives in the folder Claude Code chose.
  - Queues and auto-memory left in a mis-encoded folder are migrated automatically on first access. Session files are never touched, unreadable queues are never deleted or overwritten, and the old folder is removed only once empty.
- **Fewer false positives, without losing real corrections** (#37 thanks @dirtpan, #44 thanks @gztes)
  - Slash-command invocations, contentless praise (`perfect!` with no referent) and task pivots (`Perfect! Now let's add the column`) no longer queue.
  - Both guards were then narrowed after review, because each was dropping real feedback: the slash guard also ate absolute paths (`/etc/hosts is wrong, use 127.0.0.1 not localhost`, and `remember:` after a path, breaking the promise that `remember:` is always processed), and the pivot guard matched `please` / `we need to` anywhere, eating `Nailed it! Please keep using this pattern`.
  - The `no`-opener detection is a deny-list rather than an allowlist of continuations. The allowlist dropped project rules — `no semicolons in this codebase`, `no emojis in commit messages`, `no typescript any, ever` — while still admitting `no it works now thanks`, because `it`/`this`/`you`/`i` are the words a benign reply opens with. Detection favours recall; the semantic pass at `/reflect` time is what filters.
- **Slash commands could not find their own scripts** (#40, thanks @Ike-li)
  - `/view-queue` and `/skip-reflect` called `python3 scripts/read_queue.py` with a bare relative path resolved against the user's cwd, so `/skip-reflect` reported `Queue count: 0` for a queue it could not read. `/reflect` used `readlink -f "$0"`, where `$0` is the command's first argument. All bundled scripts are now referenced through `${CLAUDE_PLUGIN_ROOT}`, and the 11 Python snippets in `reflect.md` bootstrap `sys.path` instead of assuming the plugin checkout is the cwd.
- **Queue writes are atomic.** `write_text` truncates first, so a crash or a concurrent reader saw an empty file — which the loaders read as "no learnings".
- **Hook I/O is forced to UTF-8** (`ensure_utf8_io`, stdout half from #38, stdin half from #41). Windows consoles default to the locale codepage: non-ASCII prompts were stored as mojibake, and printing the `📝` acknowledgement raised `UnicodeEncodeError`, replacing every capture confirmation with a stderr warning. The confirmation is ASCII now as well.
- **`UnicodeDecodeError` no longer escapes the queue loaders.** It is a `ValueError`, not a `JSONDecodeError`, so a non-UTF-8 queue file propagated out of `load_queue()` and killed every capture for that project on every prompt.
- **Inclusion parsing no longer backtracks catastrophically.** A line of unmatched `[` took 2.0s at 80k characters, under the 1 MiB cap and uninterrupted by the depth and node limits. Memory-file reads are size-capped too; previously only the link-extraction pass was.

### Testing
- 322 tests, green on macOS, Linux and Windows across Python 3.8 and 3.11.
- New: `TestProjectPathEncoding`, `TestLongFolderNameResolution`, `TestLegacyFolderMigration`, `TestInclusionParserHardening`, `TestReviewGateRegressions`, `TestFableGateRegressions`. The last two hold the exact prompt strings three independent reviewers found were being dropped.
- The encoder cases exercise the pure encoder rather than `get_project_folder_name()`, so they run on Windows — the previous assertions were skipped on the one platform where the encoder crashed, and the one test that did run there never checked for a colon.
- Legacy bash tests skip explicitly when `jq` is absent. Without it six failed and, less visibly, the `test_bash_ignores_*` tests *passed* for the wrong reason: a `jq`-less script emits nothing and they assert an absence.

### Known issues
See [BACKLOG.md](BACKLOG.md) entry 6 for verified findings deliberately left in
this release, each with the cost of leaving it — including a read-modify-write
race between concurrent sessions in one project, `~/.claude` being inside the
inclusion allowlist, and auto-memory being keyed on cwd where Claude Code keys
it on the git root.

## [3.1.0] - 2026-03-16

### Added
- **Comprehensive CJK support** — Pattern detection now works for Chinese, Japanese, and Korean (#24, thanks @shohu and @yulin0629)
  - 13 CJK correction patterns: Japanese (8), Chinese (3), Korean (2)
  - Full-width `？` and CJK question particles (`嗎吗呢か까`) in false positive filter
  - CJK-aware short message threshold (2 chars for CJK vs 4 for ASCII)
  - Non-correction English phrase filter (`No problem`, `don't worry`, `never mind`, etc.)
- **Per-project queue scoping** — Learnings queue is now stored per-project to prevent cross-contamination (#21, thanks @marcodelpin)
  - Queue files at `~/.claude/projects/<encoded>/learnings-queue.json`
  - Automatic migration from legacy global queue on first access
  - Updated `/view-queue`, `/skip-reflect`, `/reflect` commands

### Fixed
- Windows CI: path encoding tests now platform-agnostic (no longer hardcode Unix paths)

## [3.0.1] - 2026-02-12

### Added
- **`--model` flag** — Control which model is used for semantic analysis during `/reflect` (#16)
  - Default: `sonnet` (cost-effective for classification tasks)
  - Usage: `/reflect --model haiku` for faster/cheaper, `/reflect --model opus` for maximum accuracy
  - Previously defaulted to user's CLI model (often Opus), burning expensive tokens on simple classification

### Changed
- `DEFAULT_MODEL = "sonnet"` in `semantic_detector.py` — all semantic analysis functions (`semantic_analyze`, `validate_tool_error`, `detect_contradictions`) now default to sonnet instead of inheriting the user's CLI model

## [3.0.0] - 2026-02-12

### Added
- **Full Memory Hierarchy Integration** — `/reflect` now supports all 6 Claude Code memory tiers:
  - `.claude/rules/*.md` — Modular rule files with optional YAML `paths:` frontmatter for path-scoping
  - `~/.claude/rules/*.md` — User-level global rule files
  - `CLAUDE.local.md` — Personal, gitignored learnings
  - Auto memory (`~/.claude/projects/<project>/memory/*.md`) — Low-confidence staging area
- **Hierarchy-Aware Routing** — `suggest_claude_file()` now routes by learning type:
  - Guardrails → `.claude/rules/guardrails.md`
  - Model preferences → model-preferences rule file or global CLAUDE.md
  - Low-confidence (0.60-0.74) → auto memory for later promotion
  - Path-scoped rules → matching rule file by `paths:` frontmatter
- **`--organize` command** — Analyze memory hierarchy and suggest reorganization:
  - Detects overgrown files, wrong-tier entries, scattered topics, promotion candidates
  - Presents issues with suggested fixes, applies with user approval
- **Auto Memory Enrichment (Step 1.6)** — During `/reflect`, scans auto memory for promotion candidates and routes low-confidence items to auto memory
- **Cross-Tier Duplicate Detection** — Step 4 now searches all memory tiers (CLAUDE.md, rules, local, auto memory)
- **Rule File Mapping** — Learning type → suggested rule file mapping table in reflect.md
- **New utilities in `reflect_utils.py`**:
  - `_parse_rule_frontmatter()` — Line-based YAML parser for rule frontmatter (no PyYAML dependency)
  - `get_project_folder_name()` — Claude Code folder name encoding
  - `get_auto_memory_path()` — Auto memory directory resolution
  - `read_auto_memory()` — Read all auto memory topic files
  - `suggest_auto_memory_topic()` — Keyword-based topic filename suggestion
  - `read_all_memory_entries()` — Cross-tier entry reader for deduplication
- **New test file** `tests/test_memory_hierarchy.py` with 28 tests covering all new functionality
- Backward-compat tests added to `tests/test_reflect_utils.py`

### Changed
- `find_claude_files()` now discovers `CLAUDE.local.md`, `.claude/rules/*.md`, and `~/.claude/rules/*.md`
- `suggest_claude_file()` accepts optional `learning_type` parameter for smarter routing
- `--targets` display updated with full hierarchy view (rules, local, auto memory)
- Step 7 (Apply Changes) now handles rule files and auto memory destinations

## [2.6.0] - 2026-02-12

### Added
- **Session retention warning** - SessionStart hook warns when `cleanupPeriodDays` is not configured
  - Claude Code deletes sessions after 30 days by default, which affects `/reflect --scan-history` and `/reflect-skills`
  - Self-resolving: warning disappears once user adds `{"cleanupPeriodDays": 99999}` to `~/.claude/settings.json`
  - New `get_cleanup_period_days()` utility in reflect_utils.py
- **README tip #7** - Documents the recommended `cleanupPeriodDays` setting

## [2.5.1] - 2026-02-04

### Fixed
- **False positive filtering** - System content (`<task-notification>`, `<system-reminder>`, session continuations) no longer triggers false pattern matches (#15)
  - Added `should_include_message()` filter before pattern detection
  - Added `MAX_CAPTURE_PROMPT_LENGTH` (500 chars) guard — real corrections are short, system content is long
  - Explicit `remember:` markers bypass length filter
  - Thanks to @DmitryBMsk for the contribution!

### Changed
- Made `_should_include_message` public as `should_include_message()` (backward-compatible alias preserved)
- Test count increased from 141 to 160

## [2.5.0] - 2026-01-25

### Added
- **Session Start Reminder** - New SessionStart hook shows pending learnings when you start a session (#13)
  - Displays up to 5 learnings with confidence scores
  - Reminds to run `/reflect` at the right time
  - Can be disabled via `CLAUDE_REFLECT_REMINDER=false` environment variable
  - Thanks to @xqliu for the contribution!

## [2.4.0] - 2026-01-23

### Added
- **Capture Feedback** - Hooks now output confirmation when learnings are captured (#10)
  - Example: `📝 Learning captured: 'no, use gpt-5.1 not gpt-5' (confidence: 85%)`
  - Claude acknowledges captures in real-time
- **Confidence in /view-queue** - Queue display now shows confidence scores, patterns, and relative timestamps
  - Format: `[0.85] "message preview..." (pattern-name) - 2 days ago`
- **Guardrail Pattern Detection** - New pattern type for "don't do X" constraints
  - Detects: "don't add X unless", "only change what I asked", "stop refactoring unrelated", etc.
  - Higher confidence (0.85-0.90) for constraint-based corrections
  - Routes to new `## Guardrails` section in CLAUDE.md
- **Contradiction Detection** - Semantic analysis to find conflicting CLAUDE.md entries
  - New `detect_contradictions()` function in semantic_detector.py
  - Integrated into `/reflect --dedupe` workflow
  - Resolution options: keep first, keep second, merge, or keep both

### Changed
- `/reflect --dedupe` now checks for contradictions before similarity grouping
- Added `## Guardrails` to standard section headers

## [2.1.1] - 2026-01-06

### Fixed
- **Plugin installation error** - Removed duplicate hooks declaration from plugin.json (#9)
  - The `hooks/hooks.json` file is auto-loaded by Claude Code; explicitly declaring it in manifest caused "Duplicate hooks file detected" error

## [2.1.0] - 2026-01-05

### Added
- **Tool Error Extraction** - Scan session files for repeated tool execution errors and convert to CLAUDE.md guidelines (#7)
  - Extracts connection errors, environment issues, module not found errors
  - Filters out Claude Code guardrails and one-off errors
  - Usage: `/reflect --scan-history --include-tool-errors`
- **Mandatory TodoWrite Tracking** - `/reflect` workflow now uses TodoWrite to track all phases

### Changed
- Improved workflow visibility with real-time progress tracking

## [2.0.0] - 2026-01-04

### Added
- **Windows Support** - Native Python scripts replace bash, no WSL required (#1)
- **Semantic AI Detection** - Multi-language support via `claude -p` (#2, #3)
- **UserPromptSubmit Hook** - Automatic capture now properly registered
- **GitHub Actions CI** - Automated testing on Windows, macOS, Linux (Python 3.8 & 3.11)
- **Comparison Tool** - `scripts/compare_detection.py` for testing detection accuracy
- **90 Unit Tests** - Comprehensive test coverage with mocked Claude CLI calls

### Changed
- Hooks now use Python scripts instead of bash for cross-platform compatibility
- `/reflect` command validates queue items with semantic AI before presenting
- Detection uses hybrid approach: regex patterns (fast, real-time) + semantic AI (accurate, during /reflect)
- Updated documentation (README.md, CLAUDE.md) with new architecture

### Deprecated
- Bash scripts moved to `scripts/legacy/` (still available for reference)

### Fixed
- Hooks failing on Windows due to bash dependency (#1)
- False positives from English-only regex patterns (#2)
- Multi-language corrections not being detected (#3)
- UserPromptSubmit hook not registered in hooks.json

## [1.4.1] - 2025-12-xx

### Fixed
- Critical jq filter bug in distribution files
- Historical scan now ensures matches are always presented to user
- Queue items being ignored during history scan

## [1.4.0] - 2025-12-xx

### Added
- Confidence scoring for learnings (0.60-0.90)
- Positive feedback pattern detection
- AGENTS.md sync support
- Semantic deduplication (`/reflect --dedupe`)

## [1.3.5] - 2025-12-xx

### Changed
- PreCompact hook now informs and backs up instead of blocking

## [1.3.4] - 2025-12-xx

### Fixed
- Restored UserPromptSubmit hook for automatic capture
