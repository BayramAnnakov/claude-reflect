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

                # Look for tool rejection pattern
                tool_result = entry.get("toolUseResult", "")
                if not tool_result:
                    continue

                if "The user doesn't want to proceed" not in tool_result:
                    continue

                # Extract text after "user said:"
                if "user said:" in tool_result:
                    parts = tool_result.split("user said:")
                    if len(parts) > 1:
                        feedback = parts[1].strip()
                        if feedback:  # Skip empty feedback
                            rejections.append(feedback)
    except IOError:
        return []

    return rejections
