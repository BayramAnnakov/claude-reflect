#!/usr/bin/env python3
"""Tests for session_end_log module."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from session_end_log import (
    parse_session,
    derive_topics,
    format_markdown,
    format_json,
    write_local,
    _extract_texts,
)


def _make_session_jsonl(messages: list) -> str:
    """Build a minimal JSONL session file from a list of message dicts."""
    return "\n".join(json.dumps(m) for m in messages)


class TestExtractTexts(unittest.TestCase):
    def test_string_content(self):
        self.assertEqual(_extract_texts("hello"), ["hello"])

    def test_list_content(self):
        content = [{"type": "text", "text": "hello"}, {"type": "image"}]
        self.assertEqual(_extract_texts(content), ["hello"])

    def test_empty(self):
        self.assertEqual(_extract_texts([]), [])
        self.assertEqual(_extract_texts(None), [])


class TestParseSession(unittest.TestCase):
    def _write_session(self, messages: list) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        for msg in messages:
            tmp.write(json.dumps(msg) + "\n")
        tmp.close()
        return Path(tmp.name)

    def tearDown(self):
        # Clean up temp files
        pass

    def test_counts_messages(self):
        messages = [
            {"type": "user", "isMeta": False, "message": {"content": [{"type": "text", "text": "hello"}]}},
            {"type": "assistant", "message": {"content": []}},
            {"type": "user", "isMeta": False, "message": {"content": [{"type": "text", "text": "world"}]}},
        ]
        f = self._write_session(messages)
        result = parse_session(f)
        self.assertEqual(result["message_count"], 3)
        self.assertEqual(len(result["user_messages"]), 2)
        f.unlink()

    def test_skips_meta_messages(self):
        messages = [
            {"type": "user", "isMeta": True, "message": {"content": [{"type": "text", "text": "/reflect --dry-run"}]}},
            {"type": "user", "isMeta": False, "message": {"content": [{"type": "text", "text": "real message"}]}},
        ]
        f = self._write_session(messages)
        result = parse_session(f)
        self.assertEqual(len(result["user_messages"]), 1)
        self.assertEqual(result["user_messages"][0], "real message")
        f.unlink()

    def test_counts_tool_calls(self):
        messages = [
            {"type": "assistant", "message": {"content": [
                {"type": "tool_use", "name": "Bash"},
                {"type": "tool_use", "name": "Bash"},
                {"type": "tool_use", "name": "Read"},
            ]}},
        ]
        f = self._write_session(messages)
        result = parse_session(f)
        self.assertEqual(result["tool_counts"]["Bash"], 2)
        self.assertEqual(result["tool_counts"]["Read"], 1)
        f.unlink()

    def test_detects_corrections(self):
        messages = [
            {"type": "user", "isMeta": False, "message": {"content": [
                {"type": "text", "text": "no, use gpt-5.1 not gpt-4"}
            ]}},
            {"type": "user", "isMeta": False, "message": {"content": [
                {"type": "text", "text": "looks good"}
            ]}},
        ]
        f = self._write_session(messages)
        result = parse_session(f)
        self.assertEqual(len(result["corrections"]), 1)
        self.assertIn("gpt-5.1", result["corrections"][0])
        f.unlink()

    def test_nonexistent_file(self):
        result = parse_session(Path("/nonexistent/file.jsonl"))
        self.assertEqual(result["message_count"], 0)
        self.assertEqual(result["user_messages"], [])

    def test_invalid_json_lines_skipped(self):
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        )
        tmp.write("not json\n")
        tmp.write(json.dumps({"type": "user", "isMeta": False, "message": {"content": "hi"}}) + "\n")
        tmp.close()
        result = parse_session(Path(tmp.name))
        self.assertEqual(result["message_count"], 1)
        Path(tmp.name).unlink()


class TestDeriveTopics(unittest.TestCase):
    def test_extracts_first_lines(self):
        messages = ["fix the login bug", "add dark mode", "update dependencies"]
        topics = derive_topics(messages)
        self.assertEqual(len(topics), 3)
        self.assertIn("fix the login bug", topics)

    def test_deduplicates_similar(self):
        # Messages share the same first 40 chars → deduplicated
        base = "fix the authentication login flow bug in"  # exactly 40 chars
        messages = [base + " production", base + " staging", "something else entirely"]
        topics = derive_topics(messages)
        self.assertEqual(len(topics), 2)

    def test_strips_xml_tags(self):
        messages = ["<system-reminder>ignore this</system-reminder>fix the bug"]
        topics = derive_topics(messages)
        self.assertFalse(any("<" in t for t in topics))

    def test_caps_at_ten(self):
        messages = [f"task number {i}" for i in range(20)]
        topics = derive_topics(messages)
        self.assertLessEqual(len(topics), 10)

    def test_skips_short_messages(self):
        messages = ["ok", "yes", "no", "this is a real message worth keeping"]
        topics = derive_topics(messages)
        short_included = [t for t in topics if len(t) <= 10]
        self.assertEqual(short_included, [])


class TestFormatMarkdown(unittest.TestCase):
    def _sample_data(self):
        return {
            "message_count": 20,
            "user_messages": ["fix the auth bug", "add dark mode", "update docs"],
            "tool_counts": {"Bash": 5, "Edit": 3},
            "corrections": ["no, use gpt-5.1 not gpt-4"],
        }

    def test_contains_project(self):
        md = format_markdown("/my/project", "abc123", "2026-01-01T00:00:00Z", self._sample_data())
        self.assertIn("/my/project", md)

    def test_contains_tool_summary(self):
        md = format_markdown("/p", "s", "t", self._sample_data())
        self.assertIn("Bash (5)", md)
        self.assertIn("Edit (3)", md)

    def test_contains_corrections(self):
        md = format_markdown("/p", "s", "t", self._sample_data())
        self.assertIn("Corrections Captured", md)
        self.assertIn("gpt-5.1", md)

    def test_no_corrections_section_when_empty(self):
        data = self._sample_data()
        data["corrections"] = []
        md = format_markdown("/p", "s", "t", data)
        self.assertNotIn("Corrections Captured", md)

    def test_no_tools_section_when_empty(self):
        data = self._sample_data()
        data["tool_counts"] = {}
        md = format_markdown("/p", "s", "t", data)
        self.assertNotIn("Tools Used", md)


class TestFormatJson(unittest.TestCase):
    def test_valid_json(self):
        data = {
            "message_count": 10,
            "user_messages": ["hello", "world"],
            "tool_counts": {"Bash": 2},
            "corrections": [],
        }
        payload = format_json("/project", "sess123", "2026-01-01T00:00:00Z", data)
        parsed = json.loads(payload)
        self.assertEqual(parsed["event"], "session_end")
        self.assertEqual(parsed["session_id"], "sess123")
        self.assertEqual(parsed["message_count"], 10)
        self.assertIn("topics", parsed)

    def test_corrections_capped_at_five(self):
        data = {
            "message_count": 10,
            "user_messages": [],
            "tool_counts": {},
            "corrections": [f"correction {i}" for i in range(10)],
        }
        payload = format_json("/p", "s", "t", data)
        parsed = json.loads(payload)
        self.assertLessEqual(len(parsed["corrections"]), 5)


class TestWriteLocal(unittest.TestCase):
    def test_writes_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("session_end_log.get_claude_dir", return_value=Path(tmpdir)):
                path = write_local("# Test content\n", "2026-01-01T120000Z")
                self.assertIsNotNone(path)
                self.assertTrue(path.exists())
                self.assertEqual(path.read_text(), "# Test content\n")

    def test_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            claude_dir = Path(tmpdir) / "new_subdir"
            with patch("session_end_log.get_claude_dir", return_value=claude_dir):
                path = write_local("content", "2026-01-01T120000Z")
                self.assertTrue(path.parent.exists())


if __name__ == "__main__":
    unittest.main()
