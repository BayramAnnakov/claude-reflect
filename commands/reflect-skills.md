---
description: Discover skill candidates from repeating session patterns
allowed-tools: Read, Write, Bash, Glob, Grep, AskUserQuestion, TodoWrite
---

## Arguments
- `--days N`: Analyze sessions from last N days (default: 14)
- `--project <path>`: Only analyze sessions from a specific project
- `--dry-run`: Show analysis without generating skill files

## Context
- Current project: !`pwd`
- Session files location: `~/.claude/projects/`

## Your Task

You are analyzing session history to discover **repeating patterns** that could become reusable skills.

### IMPORTANT: AI-Powered Detection

**DO NOT use hardcoded patterns, regex, or keyword matching.**

Your job is to **reason** about the sessions and identify:
1. **Workflow patterns** - Multi-step sequences the user requests repeatedly
2. **Misunderstanding patterns** - Corrections that keep happening (could become skill guardrails)
3. **Prompt sequences** - Similar intents expressed in different words

Use your semantic understanding. The same intent might appear as:
- "search for X on linkedin"
- "find X's linkedin profile"
- "lookup X on linkedin"

These are the **same pattern** despite different wording.

---

## Workflow

### Step 1: Initialize Task Tracking

Use TodoWrite to track progress:
```json
{
  "todos": [
    {"content": "Parse arguments", "status": "in_progress", "activeForm": "Parsing command arguments"},
    {"content": "Gather session data", "status": "pending", "activeForm": "Reading session files"},
    {"content": "Analyze for patterns", "status": "pending", "activeForm": "Analyzing sessions for patterns"},
    {"content": "Propose skill candidates", "status": "pending", "activeForm": "Proposing skill candidates"},
    {"content": "Get user approval", "status": "pending", "activeForm": "Getting user approval"},
    {"content": "Generate skill files", "status": "pending", "activeForm": "Generating skill files"}
  ]
}
```

### Step 2: Parse Arguments

Check for:
- `--days N` → Limit to last N days of sessions (default: 14)
- `--project <path>` → Filter to specific project
- `--dry-run` → Analysis only, no file generation

### Step 3: Gather Session Data

Find and read recent session files:

```bash
# List session directories
ls -la ~/.claude/projects/ 2>/dev/null | head -20
```

For each relevant project, find session files:
```bash
# Example: Find recent sessions (adjust date filter based on --days)
find ~/.claude/projects/ -name "*.jsonl" -mtime -14 -type f 2>/dev/null | head -20
```

Extract user messages from each session file using the existing extraction logic.

**What to extract:**
- User prompts (natural language requests)
- Sequences of tool calls that followed
- Any corrections or clarifications

### Step 4: Analyze for Patterns (AI-Powered)

**This is the core step. Use your reasoning to identify patterns.**

Read the extracted session data and think:

1. **Workflow Repetition**
   - "I see the user asked for [X] multiple times"
   - "Each time, I performed steps: A → B → C"
   - "This could be automated as a skill"

2. **Semantic Similarity**
   - "These 3 requests have different wording but same intent"
   - "User wants to [accomplish Y] but phrases it differently"
   - "A skill with good trigger detection would help"

3. **Correction Patterns**
   - "User corrected me twice about [Z]"
   - "This should be a guardrail in the skill"
   - "Next time, the skill should do [Z] by default"

**Output your analysis as structured findings:**

```
PATTERN 1: [Name]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Evidence:
- Session [date]: "user prompt..."
- Session [date]: "similar prompt..."
- Session [date]: "another variant..."

Intent: [What the user is trying to accomplish]

Typical Steps:
1. [First action]
2. [Second action]
3. [Third action]

Corrections Applied: [Any guardrails learned from corrections]

Suggested Skill Name: /[skill-name]
Confidence: [High/Medium/Low]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Step 5: Propose Skill Candidates

Present discovered patterns to the user:

```
════════════════════════════════════════════════════════════
SKILL CANDIDATES DISCOVERED
════════════════════════════════════════════════════════════

Found [N] potential skills from analyzing [M] sessions:

1. /[skill-name] (Confidence: High)
   → [One-line description]
   Evidence: [N] similar requests found

2. /[skill-name] (Confidence: Medium)
   → [One-line description]
   Evidence: [N] similar requests found

════════════════════════════════════════════════════════════
```

Use AskUserQuestion to get feedback:
- Which patterns should become skills?
- Any patterns to skip?
- Any name changes?

### Step 6: Generate Skill Files

For each approved skill candidate, generate a skill file in `./commands/`:

**Skill File Template:**
```markdown
---
description: [One-line description]
allowed-tools: [Relevant tools based on workflow]
---

## Context
[Any context the skill needs]

## Your Task

[Clear description of what the skill does]

### Steps

1. [First step]
2. [Second step]
3. [Third step]

### Guardrails
[Any learned corrections/constraints]

---
*Generated by /reflect-skills from [N] session patterns*
```

Write to: `./commands/[skill-name].md`

### Step 7: Summary

```
════════════════════════════════════════════════════════════
SKILL GENERATION COMPLETE
════════════════════════════════════════════════════════════

Created [N] new skill(s):
- /[skill-1]: [path]
- /[skill-2]: [path]

Next steps:
- Review generated skills in ./commands/
- Test with: /[skill-name]
- Iterate on skill content as needed

════════════════════════════════════════════════════════════
```

---

## Key Principles

1. **Semantic over Syntactic** - Match intent, not keywords
2. **Evidence-Based** - Show the user WHY you think it's a pattern
3. **Human-in-the-Loop** - User approves before generation
4. **Minimal but Useful** - Only propose skills that would genuinely save time
5. **Learn from Corrections** - Build guardrails from past mistakes

---

## Example Analysis

Given these session excerpts:

```
Session 1: "find john smith on linkedin and get his email"
Session 2: "linkedin lookup for sarah jones, need her contact info"
Session 3: "search linkedin for the CTO of acme corp"
```

Your analysis should reason:

> "I see 3 requests that are semantically similar - all are about finding
> someone on LinkedIn. The specific names and details vary, but the workflow
> is consistent: search LinkedIn → find profile → extract contact info.
> This is a strong candidate for a /linkedin-lookup skill."

NOT:

> "I found 3 messages containing 'linkedin' - potential skill."

The first is semantic reasoning. The second is keyword matching.
