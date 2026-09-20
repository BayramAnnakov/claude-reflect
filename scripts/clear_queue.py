#!/usr/bin/env python3
"""Clear the current project's learnings queue. Helper for slash commands.

/reflect used to do this with `echo "[]" > ~/.claude/learnings-queue.json`,
which wrote the pre-3.1 global file and left the real per-project queue
untouched -- so the same learnings reappeared on the next run.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.reflect_utils import ensure_utf8_io, load_queue, save_queue  # noqa: E402

if __name__ == "__main__":
    ensure_utf8_io()
    project = sys.argv[1] if len(sys.argv) > 1 else None
    count = len(load_queue(project))
    save_queue([], project)
    print(f"Cleared {count} queued learning(s).")
