# claude-reflect

[![GitHub stars](https://img.shields.io/github/stars/BayramAnnakov/claude-reflect?style=flat-square)](https://github.com/BayramAnnakov/claude-reflect/stargazers)
[![Version](https://img.shields.io/badge/version-2.2.0-blue?style=flat-square)](https://github.com/BayramAnnakov/claude-reflect/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-130%20passing-brightgreen?style=flat-square)](https://github.com/BayramAnnakov/claude-reflect/actions)
[![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey?style=flat-square)](https://github.com/BayramAnnakov/claude-reflect#platform-support)

A self-learning system for Claude Code that captures corrections, positive feedback, and preferences — then syncs them to CLAUDE.md and AGENTS.md.

## What it does

When you correct Claude Code during a session ("no, use gpt-5.1 not gpt-5", "use database for caching"), these corrections are captured and can be added to your CLAUDE.md files so Claude remembers them in future sessions.

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  You correct    │ ──► │  Hook captures  │ ──► │  /reflect adds  │
│  Claude Code    │     │  to queue       │     │  to CLAUDE.md   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
      (automatic)            (automatic)            (manual review)
```

## Installation

```bash
# Add the marketplace
claude plugin marketplace add bayramannakov/claude-reflect

# Install the plugin
claude plugin install claude-reflect@claude-reflect-marketplace

# IMPORTANT: Restart Claude Code to activate the plugin
```

After installation, **restart Claude Code** (exit and reopen). Then hooks auto-configure and commands are ready.

> **First run?** When you run `/reflect` for the first time, you'll be prompted to scan your past sessions for learnings.

### Prerequisites

- [Claude Code](https://claude.ai/code) CLI installed
- Python 3.6+ (included on most systems)

### Platform Support

- **macOS**: Fully supported
- **Linux**: Fully supported
- **Windows**: Fully supported (native Python, no WSL required)

## Commands

| Command | Description |
|---------|-------------|
| `/reflect` | Process queued learnings with human review |
| `/reflect --scan-history` | Scan ALL past sessions for missed learnings |
| `/reflect --dry-run` | Preview changes without applying |
| `/reflect --targets` | Show detected config files (CLAUDE.md, AGENTS.md) |
| `/reflect --review` | Show queue with confidence scores and decay status |
| `/reflect --dedupe` | Find and consolidate similar entries in CLAUDE.md |
| `/reflect-skills` | **NEW** Discover skill candidates from repeating patterns |
| `/skip-reflect` | Discard all queued learnings |
| `/view-queue` | View pending learnings without processing |

## How It Works

![claude-reflect in action](assets/reflect-demo.jpg)

### Two-Stage Process

**Stage 1: Capture (Automatic)**

Hooks run automatically to detect and queue corrections:

| Hook | Trigger | Purpose |
|------|---------|---------|
| `capture_learning.py` | Every prompt | Detects correction patterns and queues them |
| `check_learnings.py` | Before compaction | Backs up queue and informs user |
| `post_commit_reminder.py` | After git commit | Reminds to run /reflect after completing work |

**Stage 2: Process (Manual)**

Run `/reflect` to review and apply queued learnings to CLAUDE.md.

### Detection Methods

Claude-reflect uses a **hybrid detection approach**:

**1. Regex patterns (real-time capture)**

Fast pattern matching during sessions detects:

- **Corrections**: `"no, use X"` / `"don't use Y"` / `"actually..."` / `"that's wrong"`
- **Positive feedback**: `"Perfect!"` / `"Exactly right"` / `"Great approach"`
- **Explicit markers**: `"remember:"` — highest confidence

**2. Semantic AI validation (during /reflect)**

When you run `/reflect`, an AI-powered semantic filter:
- **Multi-language support** — understands corrections in any language
- **Better accuracy** — filters out false positives from regex
- **Cleaner learnings** — extracts concise, actionable statements

Example: A Spanish correction like `"no, usa Python"` is correctly detected even though it doesn't match English patterns.

Each captured learning has a **confidence score** (0.60-0.95). The final score is the higher of regex and semantic confidence.

### Human Review

When you run `/reflect`, Claude presents a summary table with options:
- **Apply** - Accept the learning and add to CLAUDE.md
- **Edit before applying** - Modify the learning text first
- **Skip** - Don't apply this learning

### Multi-Target Sync

Approved learnings are synced to:
- `~/.claude/CLAUDE.md` (global - applies to all projects)
- `./CLAUDE.md` (project-specific)
- `./**/CLAUDE.md` (subdirectories - auto-discovered)
- `./commands/*.md` (skill files - when correction relates to a skill)
- `AGENTS.md` (if exists - works with Codex, Cursor, Aider, Jules, Zed, Factory)

Run `/reflect --targets` to see which files will be updated.

### Skill Discovery (NEW in v2.2.0)

Run `/reflect-skills` to discover repeating patterns in your sessions that could become reusable skills:

```
/reflect-skills              # Analyze last 14 days
/reflect-skills --days 30    # Analyze last 30 days
```

Features:
- **AI-powered detection** — uses reasoning, not regex, to find patterns
- **Semantic similarity** — detects same intent across different phrasings
- **Generates skill files** — creates draft skills in `./commands/`

### Skill Improvement Routing (NEW in v2.2.0)

When you correct Claude while using a skill (e.g., `/deploy`), the correction can be routed back to the skill file itself:

```
User: /deploy
Claude: [deploys without running tests]
User: "no, always run tests before deploying"

→ /reflect detects this relates to /deploy
→ Offers to add learning to commands/deploy.md
→ Skill file updated with new step
```

This makes skills smarter over time, not just CLAUDE.md.

## Uninstall

```bash
claude plugin uninstall claude-reflect@claude-reflect-marketplace
```

## File Structure

```
claude-reflect/
├── .claude-plugin/
│   └── plugin.json         # Plugin manifest (auto-registers hooks)
├── commands/
│   ├── reflect.md          # Main command
│   ├── skip-reflect.md     # Discard queue
│   └── view-queue.md       # View queue
├── hooks/
│   └── hooks.json          # Auto-configured when plugin installed
├── scripts/
│   ├── lib/
│   │   ├── reflect_utils.py      # Shared utilities
│   │   └── semantic_detector.py  # AI-powered semantic analysis
│   ├── capture_learning.py       # Hook: detect corrections
│   ├── check_learnings.py        # Hook: pre-compact check
│   ├── post_commit_reminder.py   # Hook: post-commit reminder
│   ├── compare_detection.py      # Compare regex vs semantic detection
│   ├── extract_session_learnings.py
│   ├── extract_tool_rejections.py
│   └── legacy/                   # Bash scripts (deprecated)
├── tests/                  # Test suite
└── SKILL.md                # Skill context for Claude
```

## Features

### Historical Scan

First time using claude-reflect? Run:

```bash
/reflect --scan-history
```

This scans all your past sessions for corrections you made, so you don't lose learnings from before installation.

### Smart Filtering

Claude filters out:
- Questions (not corrections)
- One-time task instructions
- Context-specific requests
- Vague/non-actionable feedback

Only reusable learnings are kept.

### Duplicate Detection

Before adding a learning, existing CLAUDE.md content is checked. If similar content exists, you can:
- Merge with existing entry
- Replace the old entry
- Skip the duplicate

### Semantic Deduplication

Over time, CLAUDE.md can accumulate similar entries. Run `/reflect --dedupe` to:
- Find semantically similar entries (even with different wording)
- Propose consolidated versions
- Clean up redundant learnings

Example:
```
Before:
  - Use gpt-5.1 for complex tasks
  - Prefer gpt-5.1 for reasoning
  - gpt-5.1 is better for hard problems

After:
  - Use gpt-5.1 for complex reasoning tasks
```

## Tips

1. **Use explicit markers** for important learnings:
   ```
   remember: always use venv for Python projects
   ```

2. **Run /reflect after git commits** - The hook reminds you, but make it a habit

3. **Historical scan on new machines** - When setting up a new dev environment:
   ```
   /reflect --scan-history --days 90
   ```

4. **Project vs Global** - Model names and general patterns go global; project-specific conventions stay in project CLAUDE.md

## Contributing

Pull requests welcome! Please read the contributing guidelines first.

## License

MIT
