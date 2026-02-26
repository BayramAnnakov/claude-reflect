#!/usr/bin/env python3
"""Capture session activity summary at session end. SessionEnd hook.

Reads the completed session JSONL file and creates a structured activity
log. Output is written to a local file and/or sent to a configurable
webhook/command for integration with external knowledge systems.

Cross-platform compatible (Windows, macOS, Linux).

Environment variables:
  CLAUDE_REFLECT_SESSION_LOG         "true" (default) or "false" to disable
  CLAUDE_REFLECT_SESSION_LOG_WEBHOOK URL to POST session summary JSON to
  CLAUDE_REFLECT_SESSION_LOG_COMMAND Shell command that receives JSON on stdin

Output format (local file: ~/.claude/session-logs/YYYY-MM-DD-HHMMSS.md):
  # Session Log: YYYY-MM-DD HH:MM
  **Project:** /path/to/project
  **Duration:** ~N messages

  ## Topics Covered
  - bullet list derived from user prompts

  ## Tools Used
  - Bash (N calls), Edit (N calls), ...

  ## Corrections Captured
  - learning 1
  - learning 2
"""
import sys
import os
import json
import re
import urllib.request
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.reflect_utils import (
    get_claude_dir,
    get_project_folder_name,
    detect_patterns,
    should_include_message,
)


# ---------------------------------------------------------------------------
# Session file helpers
# ---------------------------------------------------------------------------

def find_session_file(project_dir: str, session_id: str) -> Optional[Path]:
    """Locate the JSONL file for the current session."""
    folder_name = get_project_folder_name(project_dir)
    sessions_dir = get_claude_dir() / "projects" / folder_name

    if not sessions_dir.exists():
        return None

    # Prefer exact session_id match
    if session_id:
        candidate = sessions_dir / f"{session_id}.jsonl"
        if candidate.exists():
            return candidate

    # Fall back to most recently modified JSONL
    jsonl_files = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return jsonl_files[-1] if jsonl_files else None


def parse_session(session_file: Path) -> Dict[str, Any]:
    """Parse a session JSONL file and extract a structured summary."""
    user_messages: List[str] = []
    tool_calls: List[str] = []
    corrections: List[str] = []
    message_count = 0

    try:
        with open(session_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg_type = entry.get("type", "")
                message_count += 1

                # User messages → topic extraction + correction detection
                if msg_type == "user" and not entry.get("isMeta"):
                    content = entry.get("message", {}).get("content", [])
                    texts = _extract_texts(content)
                    for text in texts:
                        if should_include_message(text):
                            user_messages.append(text)
                            _, _, confidence, _, _ = detect_patterns(text)
                            if confidence >= 0.75:
                                corrections.append(text)

                # Tool use events → count by tool name
                if msg_type == "assistant":
                    content = entry.get("message", {}).get("content", [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "tool_use":
                                tool_calls.append(item.get("name", "unknown"))

    except IOError:
        pass

    tool_counts = Counter(tool_calls)
    return {
        "message_count": message_count,
        "user_messages": user_messages,
        "tool_counts": dict(tool_counts.most_common()),
        "corrections": corrections,
    }


def _extract_texts(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out.append(item.get("text", ""))
        return out
    return []


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def derive_topics(user_messages: List[str]) -> List[str]:
    """Extract short topic descriptions from user messages (heuristic)."""
    topics = []
    seen: set = set()

    for msg in user_messages:
        # Take first line / first sentence, strip noise, cap at 80 chars
        first_line = msg.split("\n")[0].strip()
        first_line = re.sub(r"<[^>]+>", "", first_line)  # remove XML tags
        first_line = first_line[:80].strip()
        if len(first_line) > 10:
            key = first_line.lower()[:40]
            if key not in seen:
                seen.add(key)
                topics.append(first_line)
        if len(topics) >= 10:
            break

    return topics


def format_markdown(
    project_dir: str,
    session_id: str,
    timestamp: str,
    data: Dict[str, Any],
) -> str:
    """Format the session summary as Markdown."""
    dt = datetime.now()
    human_ts = dt.strftime("%Y-%m-%d %H:%M")

    topics = derive_topics(data["user_messages"])
    tool_summary = ", ".join(
        f"{name} ({count})" for name, count in data["tool_counts"].items()
    )

    lines = [
        f"# Session Log: {human_ts}",
        f"",
        f"**Project:** {project_dir}",
        f"**Session:** {session_id}",
        f"**Messages:** ~{data['message_count']}",
        f"",
    ]

    if topics:
        lines += ["## Topics Covered", ""]
        for t in topics:
            lines.append(f"- {t}")
        lines.append("")

    if tool_summary:
        lines += ["## Tools Used", "", f"- {tool_summary}", ""]

    if data["corrections"]:
        lines += ["## Corrections Captured", ""]
        for c in data["corrections"][:5]:
            short = c[:100].replace("\n", " ")
            lines.append(f"- {short}")
        lines.append("")

    return "\n".join(lines)


def format_json(
    project_dir: str,
    session_id: str,
    timestamp: str,
    data: Dict[str, Any],
) -> str:
    """Format the session summary as JSON for webhook delivery."""
    payload = {
        "event": "session_end",
        "timestamp": timestamp,
        "session_id": session_id,
        "project_dir": project_dir,
        "message_count": data["message_count"],
        "topics": derive_topics(data["user_messages"]),
        "tool_counts": data["tool_counts"],
        "corrections": data["corrections"][:5],
        "user_message_count": len(data["user_messages"]),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Output handlers
# ---------------------------------------------------------------------------

def write_local(markdown: str, timestamp: str) -> Optional[Path]:
    """Write summary to ~/.claude/session-logs/."""
    log_dir = get_claude_dir() / "session-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts_safe = timestamp.replace(":", "").replace("T", "-").replace("Z", "")
    log_file = log_dir / f"{ts_safe}.md"
    log_file.write_text(markdown, encoding="utf-8")
    return log_file


def send_webhook(json_payload: str, url: str) -> bool:
    """POST JSON payload to a webhook URL. Returns True on success."""
    try:
        data = json_payload.encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "claude-reflect/session-log"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status < 400
    except Exception:
        return False


def run_command(json_payload: str, command: str) -> bool:
    """Pipe JSON payload into a shell command. Returns True on success."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            input=json_payload.encode("utf-8"),
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Main entry point."""
    # Respect opt-out
    if os.environ.get("CLAUDE_REFLECT_SESSION_LOG", "true").lower() == "false":
        return 0

    # Read hook input from stdin
    input_data = sys.stdin.read()
    data = {}
    if input_data:
        try:
            data = json.loads(input_data)
        except json.JSONDecodeError:
            pass

    session_id = data.get("session_id", "")
    project_dir = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR", "")

    if not project_dir:
        return 0

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Find and parse session file
    session_file = find_session_file(project_dir, session_id)
    if not session_file:
        return 0

    parsed = parse_session(session_file)

    # Skip sessions with almost no content (e.g. accidental opens)
    if parsed["message_count"] < 4 or not parsed["user_messages"]:
        return 0

    markdown = format_markdown(project_dir, session_id, timestamp, parsed)
    json_payload = format_json(project_dir, session_id, timestamp, parsed)

    # Always write local log
    log_file = write_local(markdown, timestamp)

    # Optional: send to webhook
    webhook_url = os.environ.get("CLAUDE_REFLECT_SESSION_LOG_WEBHOOK", "")
    if webhook_url:
        send_webhook(json_payload, webhook_url)

    # Optional: pipe to custom command
    log_command = os.environ.get("CLAUDE_REFLECT_SESSION_LOG_COMMAND", "")
    if log_command:
        run_command(json_payload, log_command)

    if log_file:
        print(f"\n📋 Session logged → {log_file}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Never block the session on errors
        print(f"Warning: session_end_log.py error: {e}", file=sys.stderr)
        sys.exit(0)
