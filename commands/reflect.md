---
description: Reflect on session corrections and update CLAUDE.md (with human review)
allowed-tools: Read, Edit, Write, Glob, Bash, Grep, AskUserQuestion
---

## Arguments
- `--dry-run`: Preview all changes without prompting or writing.
- `--scan-history`: Scan ALL past sessions for corrections (useful for first-time setup or cold start).
- `--days N`: Limit history scan to last N days (default: 30). Only used with `--scan-history`.

## Context
- Project CLAUDE.md: @CLAUDE.md
- Global CLAUDE.md: @~/.claude/CLAUDE.md
- Learnings queue: !`cat ~/.claude/learnings-queue.json 2>/dev/null || echo "[]"`
- Current project: !`pwd`

## Your Task

### First-Run Detection (Per-Project)

Check if /reflect has been run in THIS project before:

```bash
PROJECT_FOLDER=$(ls ~/.claude/projects/ 2>/dev/null | grep -F "$(pwd | tr '/' '-' | cut -c2-)" | head -1)
if [ -n "$PROJECT_FOLDER" ]; then
  test -f ~/.claude/projects/$PROJECT_FOLDER/.reflect-initialized && echo "initialized" || echo "first-run"
else
  echo "first-run"
fi
```

**If "first-run" for this project AND user did NOT pass `--scan-history`:**

Use AskUserQuestion to recommend historical scan:
```json
{
  "questions": [{
    "question": "First time running /reflect in this project. Scan past sessions for learnings?",
    "header": "First run",
    "multiSelect": false,
    "options": [
      {"label": "Yes, scan history (Recommended)", "description": "Find corrections from past sessions in this project"},
      {"label": "No, just process queue", "description": "Only process learnings captured by hooks"}
    ]
  }]
}
```

If user chooses "Yes, scan history", proceed as if `--scan-history` was passed.

### Step 0: Check Arguments

**If user passed `--dry-run`:**
- Process all learnings with project filtering
- Show proposed changes with line numbers
- Do NOT prompt for actions, do NOT write
- End with: "Dry run complete. Run /reflect without --dry-run to apply."

**If user passed `--scan-history`:**
- Skip the queue (Step 1) and current session analysis
- Instead, scan ALL historical sessions for this project
- Proceed to Step 0.5: Historical Scan

### Step 0.5: Historical Scan (only with --scan-history)

Scan past sessions for corrections missed by hooks. Useful for:
- First-time /reflect installation (cold start)
- Periodic deep review of past learnings

**0.5a. Find ALL session files for this project:**

1. First, list project folders to find the correct path pattern:
   ```bash
   ls ~/.claude/projects/ | grep -i "$(basename $(pwd))"
   ```

2. Then list ALL session files in that folder:
   ```bash
   ls ~/.claude/projects/[PROJECT_FOLDER]/*.jsonl
   ```

Note: Project paths have `/` replaced with `-`. For `/Users/bob/code/myapp`, look for `-Users-bob-code-myapp`.

**IMPORTANT**: With `--scan-history`, process ALL session files (not just recent ones). This includes:
- Main session files (UUID format like `fa5ae539-d170-4fa8-a8d2-bf50b3ec2861.jsonl`)
- Agent files (`agent-*.jsonl`) - these may contain corrections too
- Apply `--days N` filter by checking file modification times if specified

**0.5b. Extract corrections from session files:**

Session files are JSONL (one JSON object per line). Use this single command to extract user corrections:

```bash
# Extract user messages with correction patterns from a session file
python3 << 'PYEOF'
import json, sys, re
patterns = r'(?i)(no,?\s*(use|don.t)|actually|remember:|instead.?\s+use|should be|not\s+\w+.?\s+use|don.t use)'
for line in open(sys.argv[1]):
    try:
        obj = json.loads(line)
        if obj.get('type') == 'user':
            msg = obj.get('message', {})
            if isinstance(msg, dict):
                for c in msg.get('content', []):
                    if isinstance(c, dict) and c.get('type') == 'text':
                        text = c.get('text', '')
                        if re.search(patterns, text) and len(text) > 10 and len(text) < 500:
                            print('---')
                            print(text[:300])
    except: pass
PYEOF
```

Run this for each session file found in 0.5a, passing the file path as argument.

**Alternative (if python unavailable):** Use jq + grep:
```bash
cat "$SESSION_FILE" | jq -r 'select(.type=="user") | .message.content[]? | select(.type=="text") | .text' 2>/dev/null | grep -iE "(no,? use|don't use|actually|remember:|instead)"
```

**0.5b-extra. Extract tool rejections (HIGH confidence):**

Tool rejections are when user stopped a tool and gave feedback. Look for "user said:" pattern:
```bash
grep -o "user said:[^\"]*" "$SESSION_FILE" 2>/dev/null | sed 's/user said:\\n//' | head -10
```

**0.5c. Apply date filter if `--days N` specified:**
- Check file modification time
- Skip files older than N days

**0.5d. LLM Filter (Inline):**

For each extracted correction, evaluate whether it's a REUSABLE learning:

**REJECT if the correction is:**
- A question (ends with "?" or asks for information)
- A one-time task instruction ("save this to folder X", "create file named Y")
- Context-specific with no reusable pattern (references specific data/names)
- Too vague to be actionable ("more concise pls", "fix it")
- Already completed task confirmation ("yes", "ok", "done")

**ACCEPT if the correction is:**
- A tool/technology recommendation ("use X for Y", "don't use Z")
- An architecture/design pattern ("use database for caching")
- An environment setup rule ("run on port X", "use venv")
- A coding convention or best practice
- A rate limiting or API usage pattern
- A model name correction ("use gpt-5.1 not gpt-5")

For each ACCEPTED correction, create:
1. An actionable learning in imperative form (e.g., "Use Gemini Flash for extraction tasks")
2. Suggested scope: "global" (applies to all projects) or "project" (specific to this codebase)
3. Confidence: high (explicit correction) or medium (implicit pattern)

**0.5e. Deduplicate:**
- Collect all accepted corrections
- Remove exact duplicates
- For similar corrections, keep the most recent

**0.5f. Build working list:**
- Use the actionable learning you created as the proposed entry
- Use the scope suggestion (global/project) as default
- Mark source as "history-scan" or "tool-rejection"
- Continue to Step 3 (Project-Aware Filtering)

### Step 1: Load and Validate
- Read the queue
- If empty, ask if there are any learnings from this session to capture manually
- If no learnings, exit

### Step 2: Session Reflection (Enhanced with History Analysis)

**Note**: This step is for analyzing the CURRENT session only (when NOT using `--scan-history`).
If `--scan-history` was passed, skip to Step 3 with results from Step 0.5.

Analyze the current session for corrections missed by real-time hooks:

**2a. Find current session file:**

List session files for this project (most recent first):
```bash
ls -lt ~/.claude/projects/ | grep -i "$(basename $(pwd))"
```

Then list files in that folder and pick the most recent non-agent file:
```bash
ls -lt ~/.claude/projects/[PROJECT_FOLDER]/*.jsonl | head -5
```

Agent files (`agent-*.jsonl`) are sub-conversations; focus on main session files for current session analysis.

**2b. Extract tool rejections (HIGH confidence corrections):**

Tool rejections contain "The user doesn't want to proceed...user said: [correction]". Extract them:

```bash
# Find tool rejections - look for "user said:" after rejection message
grep -o "user said:[^\"]*" "$SESSION_FILE" 2>/dev/null | sed 's/user said:\\n//' | head -10
```

Alternative with more context:
```bash
grep "doesn't want to proceed" "$SESSION_FILE" | grep -o "user said:[^}]*" | head -10
```

**2c. Extract user messages with correction patterns:**

Use the same python/jq approach from Step 0.5b on the current session file.

**2d. Also reflect on conversation context:**
- Were there any corrections or patterns not explicitly queued?
- Model names, API patterns, tool usage mistakes, project conventions?
- Implicit corrections (e.g., "Actually, the API returns...")

**2e. LLM Filter (Inline):**
If there are extracted corrections from 2b or 2c, evaluate each using the same criteria as Step 0.5d:
- REJECT questions, one-time tasks, context-specific items, vague feedback
- ACCEPT tool recommendations, patterns, conventions, model corrections
- Create actionable learnings in imperative form with scope suggestions

**2f. Add findings to working list:**
For each ACCEPTED learning:
- Use the actionable learning you created as the proposed entry
- Use the scope suggestion (global/project) as default
- Add to working list alongside queued items
- Mark source type:
  - "queued" — from hooks/explicit remember:
  - "session-scan" — from message pattern matching
  - "tool-rejection" — from tool rejections (HIGH confidence)

### Step 3: Project-Aware Filtering

Get current project path. For each queue item, compare `item.project` with current project:

**CASE A: Same project**
- Show normally
- Offer: [a]pprove | [e]dit | [s]kip
- If approve, ask scope: [p]roject | [g]lobal | [b]oth

**CASE B: Different project, looks GLOBAL**
(message contains: gpt-*, claude-*, model names, general patterns like "always/never")
- Show with warning: "⚠️ FROM DIFFERENT PROJECT"
- Show: "Captured in: [original-project]"
- Offer: [g]lobal | [s]kip (NOT project - wrong context)

**CASE C: Different project, looks PROJECT-SPECIFIC**
(message contains: specific DB names, file paths, project-specific tools)
- Auto-skip with note: "Skipping project-specific learning from [other-project]"
- Offer: [f]orce to add to global anyway

**Heuristics:**
- `gpt-[0-9]` or `claude-` → GLOBAL (model name)
- `always|never|don't` + generic verb → GLOBAL (general rule)
- Specific tool/DB/service names → PROJECT-SPECIFIC
- File paths → PROJECT-SPECIFIC

### Step 4: Duplicate Detection with Line Numbers

For each learning kept after filtering, search BOTH CLAUDE.md files:

```bash
grep -n -i "keyword" ~/.claude/CLAUDE.md
grep -n -i "keyword" CLAUDE.md
```

If duplicate found:
- Show: "⚠️ SIMILAR in [global/project] CLAUDE.md: Line [N]: [content]"
- Offer: [m]erge | [r]eplace | [a]dd anyway | [s]kip

### Step 5: Present Summary and Get User Decision

**5a. Display condensed summary table:**

Show all learnings in a compact table format:

```
════════════════════════════════════════════════════════════
LEARNINGS SUMMARY — [N] items found
════════════════════════════════════════════════════════════

┌────┬─────────────────────────────────────────┬──────────┬────────┐
│ #  │ Learning                                │ Scope    │ Status │
├────┼─────────────────────────────────────────┼──────────┼────────┤
│ 1  │ Use DB for persistent storage           │ project  │ ✓ new  │
│ 2  │ Backoff on actual errors only           │ global   │ ✓ new  │
│ ...│ ...                                     │ ...      │ ...    │
└────┴─────────────────────────────────────────┴──────────┴────────┘

Destinations: [N] → Global, [M] → Project
Duplicates: [K] items will be merged with existing entries
```

**5b. Use AskUserQuestion for strategy:**

Use the AskUserQuestion tool:
```json
{
  "questions": [{
    "question": "How would you like to process these [N] learnings?",
    "header": "Action",
    "multiSelect": false,
    "options": [
      {"label": "Apply all (Recommended)", "description": "Add [X] new entries, merge [K] duplicates with recommended scopes"},
      {"label": "Select which to apply", "description": "Choose specific learnings from grouped lists"},
      {"label": "Review details first", "description": "Show full details for each learning before deciding"},
      {"label": "Skip all", "description": "Don't apply any learnings, clear the queue"}
    ]
  }]
}
```

**5c. Handle user selection:**

- **"Apply all"** → Proceed to Step 6 (Final Confirmation)
- **"Select which to apply"** → Go to Step 5.1 (Selection Mode)
- **"Review details first"** → Show full learning cards (format below), then return to 5b
- **"Skip all"** → Go to Step 8 (Clear Queue)

**Full learning card format (for "Review details first"):**
```
════════════════════════════════════════════════════════════
LEARNING [N] of [TOTAL] — [source: queued/session-scan/tool-rejection]
════════════════════════════════════════════════════════════
Original message:
  "[the user's original text]"

Proposed addition:
┌──────────────────────────────────────────────────────────┐
│ ## [Section Name]                                        │
│ - [Exact bullet point that will be added]                │
└──────────────────────────────────────────────────────────┘

Duplicate check:
  ✓ None found
  OR
  ⚠️ SIMILAR in [global/project] CLAUDE.md:
     Line [N]: "[existing content]"
════════════════════════════════════════════════════════════
```

### Step 5.1: Selection Mode (if user chose "Select which to apply")

Group learnings by destination and use AskUserQuestion with multiSelect.

**Rules:**
- Split into multiple questions if >4 items per destination
- Use short labels: "#{N} {short_title}" (max 20 chars)
- Use descriptions for full learning text (max 80 chars)

**Example for GLOBAL learnings:**
```json
{
  "questions": [
    {
      "question": "Select GLOBAL learnings to apply:",
      "header": "Global",
      "multiSelect": true,
      "options": [
        {"label": "#2 Backoff errors", "description": "Implement backoff only on actual errors, not artificial delays"},
        {"label": "#3 DB cache", "description": "Use local database cache to minimize data fetching"},
        {"label": "#4 Batch+delays", "description": "Use batching with stochastic delays for API rate limits"},
        {"label": "#5 Gemini model", "description": "Use gemini-3-flash-preview for LLM tasks"}
      ]
    }
  ]
}
```

**If >4 global items:** Add second question with header "Global+"

**Example for PROJECT learnings:**
```json
{
  "questions": [
    {
      "question": "Select PROJECT learnings to apply:",
      "header": "Project",
      "multiSelect": true,
      "options": [
        {"label": "#1 DB storage", "description": "Use database for persistent tracking data"},
        {"label": "#6 DB ports", "description": "Assign unique ports per database instance"}
      ]
    }
  ]
}
```

**Selection rules:**
- Items NOT selected will be skipped
- Continue to Step 6 with selected items only

### Step 6: Final Confirmation

**6a. Show summary of changes:**
```
════════════════════════════════════════════════════════════
SUMMARY: [N] changes ready to apply
════════════════════════════════════════════════════════════

Project CLAUDE.md ([path]):
  Line [N]: UPDATE "[old]" → "[new]"
  After line [N]: ADD "[new entry]"

Global CLAUDE.md (~/.claude/CLAUDE.md):
  Line [N]: REPLACE "[old]" → "[new]"
  After line [N]: ADD "[new entry]"

Skipped: [N] learnings (including [M] from other projects)
════════════════════════════════════════════════════════════
```

**6b. Use AskUserQuestion for confirmation:**
```json
{
  "questions": [{
    "question": "Apply [N] learnings to CLAUDE.md files?",
    "header": "Confirm",
    "multiSelect": false,
    "options": [
      {"label": "Yes, apply all", "description": "[X] to Global, [Y] to Project CLAUDE.md"},
      {"label": "Go back", "description": "Return to selection to adjust"},
      {"label": "Cancel", "description": "Don't apply anything, keep queue"}
    ]
  }]
}
```

**6c. Handle response:**
- **"Yes, apply all"** → Proceed to Step 7
- **"Go back"** → Return to Step 5b
- **"Cancel"** → Exit without changes (keep queue intact)

### Step 7: Apply Changes

Only after final confirmation:
1. Read current CLAUDE.md files
2. Use Edit tool with precise old_string from detected line numbers
3. For new entries, add after the relevant section header

### Step 8: Clear Queue

```bash
echo "[]" > ~/.claude/learnings-queue.json
```

### Step 9: Confirm

```
════════════════════════════════════════════════════════════
DONE: Applied [N] learnings
════════════════════════════════════════════════════════════
  Project: [N] entries added/updated
  Global:  [N] entries added/updated
  Skipped: [N]
════════════════════════════════════════════════════════════
```

### Step 10: Mark Initialized (Per-Project)

Create marker file for THIS project so first-run detection won't trigger again:

```bash
PROJECT_FOLDER=$(ls ~/.claude/projects/ 2>/dev/null | grep -F "$(pwd | tr '/' '-' | cut -c2-)" | head -1)
if [ -n "$PROJECT_FOLDER" ]; then
  touch ~/.claude/projects/$PROJECT_FOLDER/.reflect-initialized
fi
```

## Formatting Rules

- **Bullets, not prose**: Keep entries as single bullet points
- **Actionable**: "Use X for Y" not "X is better than Y"
- **Concise**: Max 2 lines per entry
- **Examples when helpful**: `(e.g., gpt-5.2 not gpt-5.1)`

## Section Headers

Use these standard headers:
- `## LLM Model Recommendations` — model names, versions
- `## Tool Usage` — MCP, APIs, which tool for what
- `## Project Conventions` — coding style, patterns
- `## Common Errors to Avoid` — gotchas, mistakes
- `## Environment Setup` — venv, configs, paths

## Size Check

If CLAUDE.md exceeds 150 lines, warn:
```
Note: CLAUDE.md is [N] lines. Consider consolidating entries.
```
