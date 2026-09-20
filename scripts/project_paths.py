#!/usr/bin/env python3
"""Print the resolved claude-reflect paths for a project, as JSON.

Slash commands call this instead of re-deriving the project folder in shell.
Three separate copies of that shell logic had drifted out of sync with
lib/reflect_utils.py, each in a way that failed silently:

  * /reflect read and cleared ~/.claude/learnings-queue.json, the pre-3.1
    global queue, which migrate_global_queue() deletes on first access -- so
    it processed nothing and cleared nothing.
  * /reflect-skills encoded the folder with `sed 's|/|-|g'`, which leaves
    "_", "." and spaces alone and so pointed at a folder Claude Code never
    writes to.
  * /reflect --scan-history grepped ~/.claude/projects by basename and
    guessed at underscores with `tr '_' '-'`.

There is one encoder. This exposes it.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.reflect_utils import (  # noqa: E402
    ensure_utf8_io,
    get_auto_memory_path,
    get_claude_dir,
    get_project_folder_name,
    get_queue_path,
)


def main() -> int:
    ensure_utf8_io()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=None,
                        help="Project directory (default: cwd)")
    args = parser.parse_args()

    project = args.project or os.getcwd()
    folder = get_project_folder_name(project)
    session_dir = get_claude_dir() / "projects" / folder

    sessions = []
    if session_dir.is_dir():
        sessions = sorted(
            (str(p) for p in session_dir.glob("*.jsonl")),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )

    print(json.dumps({
        "project_dir": str(project),
        "folder_name": folder,
        "session_dir": str(session_dir),
        "session_dir_exists": session_dir.is_dir(),
        "session_files": sessions,
        "queue_path": str(get_queue_path(project)),
        "memory_dir": str(get_auto_memory_path(project)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # never let a slash command die on this
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
