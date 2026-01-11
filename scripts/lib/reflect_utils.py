#!/usr/bin/env python3
"""Shared utilities for claude-reflect hooks and scripts.

Cross-platform compatible (Windows, macOS, Linux).
"""
import json
import re
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple

# =============================================================================
# Path utilities
# =============================================================================

def get_queue_path() -> Path:
    """Get path to learnings queue file."""
    return Path.home() / ".claude" / "learnings-queue.json"


def get_backup_dir() -> Path:
    """Get path to learnings backup directory."""
    return Path.home() / ".claude" / "learnings-backups"


def get_claude_dir() -> Path:
    """Get path to .claude directory."""
    return Path.home() / ".claude"


# Directories to exclude when searching for CLAUDE.md files
EXCLUDED_DIRS = {
    'node_modules', '.git', '.svn', '.hg', 'venv', '.venv', 'env', '.env',
    '__pycache__', '.pytest_cache', '.mypy_cache', 'dist', 'build',
    '.next', '.nuxt', 'coverage', '.coverage', 'htmlcov',
    'vendor', 'target', 'out', 'bin', 'obj',
}


def find_claude_files(root_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Find all CLAUDE.md files in the project tree.

    Args:
        root_dir: Root directory to search from (defaults to cwd)

    Returns:
        List of dicts with {path, relative_path, type} for each CLAUDE.md found.
        Type is 'root', 'subdirectory', or 'global'.
    """
    root = Path(root_dir) if root_dir else Path.cwd()
    results = []

    # Always include global CLAUDE.md
    global_claude = get_claude_dir() / "CLAUDE.md"
    if global_claude.exists():
        results.append({
            "path": str(global_claude),
            "relative_path": "~/.claude/CLAUDE.md",
            "type": "global",
        })

    # Check root CLAUDE.md
    root_claude = root / "CLAUDE.md"
    if root_claude.exists():
        results.append({
            "path": str(root_claude),
            "relative_path": "./CLAUDE.md",
            "type": "root",
        })

    # Search for CLAUDE.md in subdirectories
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        # Skip root (already handled)
        if Path(dirpath) == root:
            continue

        if "CLAUDE.md" in filenames:
            full_path = Path(dirpath) / "CLAUDE.md"
            rel_path = full_path.relative_to(root)
            results.append({
                "path": str(full_path),
                "relative_path": f"./{rel_path}",
                "type": "subdirectory",
            })

    return results


def suggest_claude_file(learning: str, claude_files: List[Dict[str, Any]]) -> Optional[str]:
    """
    Suggest which CLAUDE.md file a learning should go to.

    This is a hint for Claude to use when reasoning about placement.
    Returns the relative_path of the suggested file, or None to let Claude decide.

    Note: This is intentionally simple - Claude should use its reasoning
    to make the final decision, not rely on this heuristic.
    """
    learning_lower = learning.lower()

    # Global indicators (model names, general patterns)
    global_indicators = ['gpt-', 'claude-', 'always ', 'never ', 'prefer ']
    if any(ind in learning_lower for ind in global_indicators):
        return "~/.claude/CLAUDE.md"

    # Check if learning mentions a specific directory
    for cf in claude_files:
        if cf["type"] == "subdirectory":
            # Extract directory name from path
            dir_name = Path(cf["relative_path"]).parent.name.lower()
            if dir_name in learning_lower:
                return cf["relative_path"]

    # Default: let Claude decide (return None)
    return None


# =============================================================================
# Queue operations
# =============================================================================

def load_queue() -> List[Dict[str, Any]]:
    """Load learnings queue from file."""
    path = get_queue_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, IOError):
        return []


def save_queue(items: List[Dict[str, Any]]) -> None:
    """Save learnings queue to file."""
    path = get_queue_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def append_to_queue(item: Dict[str, Any]) -> None:
    """Append a single item to the queue."""
    items = load_queue()
    items.append(item)
    save_queue(items)


# =============================================================================
# Timestamp utilities
# =============================================================================

def iso_timestamp() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def backup_timestamp() -> str:
    """Get timestamp for backup filenames."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


# =============================================================================
# Pattern definitions (from capture-learning.sh)
# =============================================================================

# Explicit marker patterns (highest confidence)
EXPLICIT_PATTERNS = [
    (r"remember:", "remember:", 0.90, 120),  # pattern, name, confidence, decay_days
]

# Positive feedback patterns
POSITIVE_PATTERNS = [
    (r"perfect!|exactly right|that's exactly", "perfect", 0.70, 90),
    (r"that's what I wanted|great approach", "great-approach", 0.70, 90),
    (r"keep doing this|love it|excellent|nailed it", "keep-doing", 0.70, 90),
]

# Correction patterns (conservative set to minimize false positives)
# Format: (regex_pattern, pattern_name, is_strong)
CORRECTION_PATTERNS = [
    (r"no[,. ]+use", "no,use", True),
    (r"don't use|do not use", "don't-use", True),
    (r"stop using|never use", "stop/never-use", True),
    (r"that's (wrong|incorrect)|that is (wrong|incorrect)", "that's-wrong", True),
    (r"not right|not correct", "not-right", False),
    (r"^actually[,. ]|[.!?] actually[,. ]", "actually", False),
    (r"I meant|I said", "I-meant/said", True),
    (r"I told you|I already told", "I-told-you", True),  # Higher confidence
    (r"you (should|need to|must) use", "you-should-use", False),
    (r"use .+ not|not .+, use", "use-X-not-Y", True),
]


def detect_patterns(text: str) -> Tuple[Optional[str], str, float, str, int]:
    """
    Detect patterns in text and return classification.

    Returns:
        Tuple of (type, matched_patterns, confidence, sentiment, decay_days)
        type: "explicit", "positive", "auto", or None
        matched_patterns: Space-separated pattern names
        confidence: 0.0 to 1.0
        sentiment: "correction" or "positive"
        decay_days: Number of days until decay
    """
    # Check for explicit "remember:"
    for pattern, name, confidence, decay in EXPLICIT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return ("explicit", name, confidence, "correction", decay)

    # Check for positive patterns
    matched_positive = []
    for pattern, name, confidence, decay in POSITIVE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            matched_positive.append(name)

    if matched_positive:
        return ("positive", " ".join(matched_positive), 0.70, "positive", 90)

    # Check for correction patterns
    matched_corrections = []
    pattern_count = 0
    has_strong_pattern = False
    has_i_told_you = False

    for pattern, name, is_strong in CORRECTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            matched_corrections.append(name)
            pattern_count += 1
            if is_strong:
                has_strong_pattern = True
            if name == "I-told-you":
                has_i_told_you = True

    if matched_corrections:
        # Calculate confidence based on pattern count and type
        if has_i_told_you:
            confidence = 0.85
            decay_days = 120
        elif pattern_count >= 3:
            confidence = 0.85
            decay_days = 120
        elif pattern_count >= 2:
            confidence = 0.75
            decay_days = 90
        else:
            confidence = 0.60
            decay_days = 60

        return ("auto", " ".join(matched_corrections), confidence, "correction", decay_days)

    return (None, "", 0.0, "correction", 90)


def create_queue_item(
    message: str,
    item_type: str,
    patterns: str,
    confidence: float,
    sentiment: str,
    decay_days: int,
    project: Optional[str] = None
) -> Dict[str, Any]:
    """Create a properly formatted queue item."""
    return {
        "type": item_type,
        "message": message,
        "timestamp": iso_timestamp(),
        "project": project or os.getcwd(),
        "patterns": patterns,
        "confidence": confidence,
        "sentiment": sentiment,
        "decay_days": decay_days,
    }


# =============================================================================
# Session file utilities
# =============================================================================

def extract_user_messages(session_file: Path, corrections_only: bool = False) -> List[str]:
    """
    Extract user messages from a Claude Code session file (JSONL format).

    Args:
        session_file: Path to the session JSONL file
        corrections_only: If True, only return messages matching correction patterns

    Returns:
        List of user message texts
    """
    if not session_file.exists():
        return []

    messages = []

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter: type=user, not isMeta
                if entry.get("type") != "user":
                    continue
                if entry.get("isMeta"):
                    continue

                # Extract text from content array
                content = entry.get("message", {}).get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if text:
                                # Apply filters (same as bash script)
                                if _should_include_message(text):
                                    messages.append(text)
    except IOError:
        return []

    if corrections_only:
        # Filter for correction patterns
        correction_pattern = (
            r"(no,? use|don't use|stop using|never use|that's wrong|that's incorrect|"
            r"not right|not correct|actually[,. ]|I meant|I said|I told you|"
            r"I already told|you should use|you need to use|use .+ not|not .+, use|remember:)"
        )
        messages = [m for m in messages if re.search(correction_pattern, m, re.IGNORECASE)]

    return messages


def _should_include_message(text: str) -> bool:
    """Check if a message should be included (apply filters from bash script)."""
    # Skip empty lines
    if not text.strip():
        return False

    # Skip lines starting with certain patterns
    skip_patterns = [
        r"^<",              # XML tags
        r"^\[",             # Brackets
        r"^\{",             # JSON
        r"tool_result",
        r"tool_use_id",
        r"<command-",
        r"This session is being continued",
        r"^Analysis:",
        r"^\*\*",           # Bold text
        r"^   -",           # Indented lists
    ]

    for pattern in skip_patterns:
        if re.search(pattern, text):
            return False

    return True


def extract_tool_rejections(session_file: Path) -> List[str]:
    """
    Extract user corrections from tool rejections in session files.

    Matches the behavior of the legacy bash script which looks for:
    - type == "user" entries
    - message.content[] array with type == "tool_result"
    - is_error == true
    - content containing "The user doesn't want to proceed"

    Args:
        session_file: Path to the session JSONL file

    Returns:
        List of user correction texts from tool rejections
    """
    if not session_file.exists():
        return []

    rejections = []

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Must be a user entry (matches bash: select(.type=="user"))
                if entry.get("type") != "user":
                    continue

                # Get message.content array (matches bash: select(.message.content | type == "array"))
                content = entry.get("message", {}).get("content", [])
                if not isinstance(content, list):
                    continue

                # Look for tool_result items in content array
                for item in content:
                    if not isinstance(item, dict):
                        continue

                    # Must be type == "tool_result" (matches bash: select(.type=="tool_result"))
                    if item.get("type") != "tool_result":
                        continue

                    # Must have is_error == true (matches bash: select(.is_error==true))
                    if not item.get("is_error"):
                        continue

                    # Get the content string
                    tool_content = item.get("content", "")
                    if not isinstance(tool_content, str):
                        continue

                    # Must contain rejection message (matches bash: select(.content | contains(...)))
                    if "The user doesn't want to proceed" not in tool_content:
                        continue

                    # Extract text after "the user said:" (matches bash: awk '/the user said:/{getline; print}')
                    # Note: bash uses lowercase "the user said:", let's be case-insensitive
                    lower_content = tool_content.lower()
                    if "the user said:" in lower_content:
                        # Find the position case-insensitively
                        idx = lower_content.find("the user said:")
                        after_marker = tool_content[idx + len("the user said:"):]
                        # Get the next line (bash uses getline)
                        lines = after_marker.strip().split("\n")
                        if lines and lines[0].strip():
                            rejections.append(lines[0].strip())

    except IOError:
        return []

    return rejections


# =============================================================================
# Tool execution error patterns
# =============================================================================

# EXCLUDE: Claude Code guardrails AND global Claude behavior (not project-specific)
TOOL_ERROR_EXCLUDE_PATTERNS = [
    # Claude Code guardrails - system enforcing its rules
    r"File has not been read yet",
    r"exceeds maximum allowed tokens",
    r"InputValidationError",
    r"not valid JSON",
    r"The user doesn't want to proceed",  # User rejections handled separately
    # Global Claude behavior issues - not project-specific
    r"unexpected EOF while looking for matching",  # Bash quoting
    r"EISDIR|illegal operation on a directory",    # File vs dir confusion
    r"syntax error.*eval",                          # Bash syntax errors
]

# PROJECT-SPECIFIC error patterns that reveal env/config/structure issues
# Format: (error_type, regex_pattern, suggested_guideline_template)
PROJECT_SPECIFIC_ERROR_PATTERNS = [
    # Connection/service errors - often reveal env/config issues
    ("connection_refused",
     r"Connection refused|ECONNREFUSED|connect ECONNREFUSED",
     "Check .env for service URLs - don't assume localhost"),
    ("env_undefined",
     r"(\w+_URL|DATABASE_URL|API_KEY|SECRET).*undefined|not set|is not defined",
     "Load .env file before accessing environment variables"),
    # Database-specific errors
    ("supabase_error",
     r"supabase|Supabase|SUPABASE",
     "Check SUPABASE_URL and SUPABASE_KEY in .env"),
    ("postgres_error",
     r"postgres|PostgreSQL|PGHOST|:5432|password authentication failed",
     "Check DATABASE_URL in .env for PostgreSQL connection"),
    ("redis_error",
     r"redis|REDIS|:6379",
     "Check REDIS_URL in .env for Redis connection"),
    # Path/module errors - reveal project structure
    ("module_not_found",
     r"ModuleNotFoundError|Cannot find module|No module named",
     "Check import paths - verify project structure"),
    ("venv_not_found",
     r"venv.*No such file|activate: No such file|\.venv.*not found",
     "Check virtual environment location"),
    # Port/service conflicts
    ("port_in_use",
     r"address already in use|EADDRINUSE|port.*already.*use",
     "Check if service is already running on this port"),
]


def extract_tool_errors(
    session_file: Path,
    project_specific_only: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract tool execution errors from session files.

    Unlike extract_tool_rejections(), this captures TECHNICAL errors where:
    - is_error == true
    - NOT a user rejection (no "doesn't want to proceed")
    - Optionally filtered for project-specific patterns only

    Args:
        session_file: Path to the session JSONL file
        project_specific_only: If True, only return errors matching project-specific patterns

    Returns:
        List of dicts with {error_type, content, project, timestamp, suggested_guideline}
    """
    if not session_file.exists():
        return []

    errors = []

    try:
        with open(session_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Must be a user entry (tool results come back as user messages)
                if entry.get("type") != "user":
                    continue

                # Get message.content array
                content = entry.get("message", {}).get("content", [])
                if not isinstance(content, list):
                    continue

                # Look for tool_result items with is_error
                for item in content:
                    if not isinstance(item, dict):
                        continue

                    if item.get("type") != "tool_result":
                        continue

                    if not item.get("is_error"):
                        continue

                    tool_content = item.get("content", "")
                    if not isinstance(tool_content, str):
                        continue

                    # Skip if matches exclude patterns
                    should_exclude = False
                    for exclude_pattern in TOOL_ERROR_EXCLUDE_PATTERNS:
                        if re.search(exclude_pattern, tool_content, re.IGNORECASE):
                            should_exclude = True
                            break

                    if should_exclude:
                        continue

                    # If project_specific_only, check for matching patterns
                    error_type = "unknown"
                    suggested_guideline = None

                    for etype, pattern, guideline in PROJECT_SPECIFIC_ERROR_PATTERNS:
                        if re.search(pattern, tool_content, re.IGNORECASE):
                            error_type = etype
                            suggested_guideline = guideline
                            break

                    # Skip unknown errors if project_specific_only
                    if project_specific_only and error_type == "unknown":
                        continue

                    errors.append({
                        "error_type": error_type,
                        "content": tool_content[:500],  # Truncate long errors
                        "project": str(session_file.parent.name),
                        "timestamp": entry.get("timestamp", ""),
                        "suggested_guideline": suggested_guideline,
                    })

    except IOError:
        return []

    return errors


def aggregate_tool_errors(
    errors: List[Dict[str, Any]],
    min_occurrences: int = 2
) -> List[Dict[str, Any]]:
    """
    Group errors by type and return those with multiple occurrences.

    Only repeated errors are valuable for CLAUDE.md - one-off errors are noise.

    Args:
        errors: List of error dicts from extract_tool_errors()
        min_occurrences: Minimum times an error type must occur

    Returns:
        List of aggregated errors with {error_type, count, suggested_guideline,
        confidence, sample_errors}
    """
    from collections import Counter

    # Count by error type
    type_counts = Counter(e["error_type"] for e in errors)

    # Group errors by type
    errors_by_type: Dict[str, List[Dict]] = {}
    for error in errors:
        etype = error["error_type"]
        if etype not in errors_by_type:
            errors_by_type[etype] = []
        errors_by_type[etype].append(error)

    # Build aggregated results for types meeting threshold
    aggregated = []
    for error_type, count in type_counts.items():
        if count < min_occurrences:
            continue

        samples = errors_by_type[error_type][:3]  # Keep up to 3 samples
        suggested_guideline = samples[0].get("suggested_guideline") if samples else None

        # Higher confidence for more occurrences
        if count >= 5:
            confidence = 0.90
        elif count >= 3:
            confidence = 0.85
        else:
            confidence = 0.70

        aggregated.append({
            "error_type": error_type,
            "count": count,
            "suggested_guideline": suggested_guideline,
            "confidence": confidence,
            "decay_days": 180,  # Tool error learnings decay slower
            "sample_errors": [s["content"][:200] for s in samples],
        })

    # Sort by count descending
    aggregated.sort(key=lambda x: x["count"], reverse=True)

    return aggregated
