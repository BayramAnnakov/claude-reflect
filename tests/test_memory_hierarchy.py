#!/usr/bin/env python3
"""Tests for memory hierarchy integration (v3.0.0).

Tests for: _parse_rule_frontmatter, find_claude_files (rules/local/user-rules),
suggest_claude_file (enhanced routing), auto memory utilities, read_all_memory_entries.
"""
import json
import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lib.reflect_utils import (
    _parse_rule_frontmatter,
    _parse_inclusions,
    _resolve_inclusion,
    _follow_inclusion_graph,
    find_claude_files,
    suggest_claude_file,
    get_project_folder_name,
    get_auto_memory_path,
    read_auto_memory,
    suggest_auto_memory_topic,
    read_all_memory_entries,
)


class TestParseRuleFrontmatter(unittest.TestCase):
    """Tests for _parse_rule_frontmatter()."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simple_paths(self):
        """Test parsing frontmatter with a simple paths list."""
        f = Path(self.temp_dir) / "rule.md"
        f.write_text("---\npaths:\n  - src/\n  - lib/\n---\n\n# Rule content\n")
        result = _parse_rule_frontmatter(f)
        self.assertIsNotNone(result)
        self.assertEqual(result["paths"], ["src/", "lib/"])

    def test_multi_paths_with_quotes(self):
        """Test parsing paths with quoted values."""
        f = Path(self.temp_dir) / "rule.md"
        f.write_text('---\npaths:\n  - "src/api/"\n  - \'lib/utils/\'\n---\n\nContent\n')
        result = _parse_rule_frontmatter(f)
        self.assertIsNotNone(result)
        self.assertEqual(result["paths"], ["src/api/", "lib/utils/"])

    def test_no_frontmatter(self):
        """Test file without frontmatter returns None."""
        f = Path(self.temp_dir) / "rule.md"
        f.write_text("# Just a regular markdown file\n\n- Some content\n")
        result = _parse_rule_frontmatter(f)
        self.assertIsNone(result)

    def test_malformed_frontmatter(self):
        """Test frontmatter without closing delimiter returns None."""
        f = Path(self.temp_dir) / "rule.md"
        f.write_text("---\npaths:\n  - src/\nSome content without closing\n")
        result = _parse_rule_frontmatter(f)
        self.assertIsNone(result)

    def test_empty_frontmatter(self):
        """Test empty frontmatter returns None."""
        f = Path(self.temp_dir) / "rule.md"
        f.write_text("---\n---\n\nContent\n")
        result = _parse_rule_frontmatter(f)
        self.assertIsNone(result)

    def test_scalar_value(self):
        """Test frontmatter with scalar key-value pair."""
        f = Path(self.temp_dir) / "rule.md"
        f.write_text("---\ndescription: My rule\n---\n\nContent\n")
        result = _parse_rule_frontmatter(f)
        self.assertIsNotNone(result)
        self.assertEqual(result["description"], "My rule")

    def test_nonexistent_file(self):
        """Test nonexistent file returns None."""
        result = _parse_rule_frontmatter(Path("/nonexistent/rule.md"))
        self.assertIsNone(result)


class TestFindClaudeFilesRules(unittest.TestCase):
    """Tests for find_claude_files() with rules, local, and user-rules."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discovers_project_rules(self):
        """Test that .claude/rules/*.md files are discovered."""
        rules_dir = Path(self.temp_dir) / ".claude" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "guardrails.md").write_text("# Guardrails\n- Don't over-engineer\n")
        (rules_dir / "coding-style.md").write_text("# Style\n- Use 2-space indent\n")

        files = find_claude_files(self.temp_dir)
        rule_files = [f for f in files if f["type"] == "rule"]
        self.assertEqual(len(rule_files), 2)
        names = sorted(Path(f["path"]).name for f in rule_files)
        self.assertEqual(names, ["coding-style.md", "guardrails.md"])

    def test_rule_frontmatter_parsing(self):
        """Test that rule files have frontmatter parsed."""
        rules_dir = Path(self.temp_dir) / ".claude" / "rules"
        rules_dir.mkdir(parents=True)
        (rules_dir / "api.md").write_text("---\npaths:\n  - src/api/\n---\n\n# API Rules\n")

        files = find_claude_files(self.temp_dir)
        rule_files = [f for f in files if f["type"] == "rule"]
        self.assertEqual(len(rule_files), 1)
        self.assertIsNotNone(rule_files[0]["frontmatter"])
        self.assertEqual(rule_files[0]["frontmatter"]["paths"], ["src/api/"])

    @patch("lib.reflect_utils.get_claude_dir")
    def test_discovers_user_rules(self, mock_claude_dir):
        """Test that ~/.claude/rules/*.md files are discovered."""
        fake_claude_dir = Path(self.temp_dir) / "fake_claude"
        fake_claude_dir.mkdir()
        mock_claude_dir.return_value = fake_claude_dir

        user_rules = fake_claude_dir / "rules"
        user_rules.mkdir()
        (user_rules / "model-prefs.md").write_text("# Models\n- Use gpt-5.1\n")

        files = find_claude_files(self.temp_dir)
        user_rule_files = [f for f in files if f["type"] == "user-rule"]
        self.assertEqual(len(user_rule_files), 1)
        self.assertIn("model-prefs.md", user_rule_files[0]["relative_path"])

    def test_discovers_local_claude(self):
        """Test that CLAUDE.local.md is discovered."""
        (Path(self.temp_dir) / "CLAUDE.local.md").write_text("# Local\n- My setting\n")

        files = find_claude_files(self.temp_dir)
        local_files = [f for f in files if f["type"] == "local"]
        self.assertEqual(len(local_files), 1)
        self.assertEqual(local_files[0]["relative_path"], "./CLAUDE.local.md")

    @patch("lib.reflect_utils.get_claude_dir")
    def test_all_types_together(self, mock_claude_dir):
        """Test discovering all file types in one call."""
        fake_claude_dir = Path(self.temp_dir) / "fake_claude"
        fake_claude_dir.mkdir()
        mock_claude_dir.return_value = fake_claude_dir

        # Global CLAUDE.md
        (fake_claude_dir / "CLAUDE.md").write_text("# Global\n")
        # User rules
        (fake_claude_dir / "rules").mkdir()
        (fake_claude_dir / "rules" / "user-rule.md").write_text("# User Rule\n")

        # Project root
        (Path(self.temp_dir) / "CLAUDE.md").write_text("# Root\n")
        (Path(self.temp_dir) / "CLAUDE.local.md").write_text("# Local\n")

        # Project rules
        proj_rules = Path(self.temp_dir) / ".claude" / "rules"
        proj_rules.mkdir(parents=True)
        (proj_rules / "style.md").write_text("# Style\n")

        # Subdirectory
        sub = Path(self.temp_dir) / "src"
        sub.mkdir()
        (sub / "CLAUDE.md").write_text("# Src\n")

        files = find_claude_files(self.temp_dir)
        types = set(f["type"] for f in files)
        self.assertIn("global", types)
        self.assertIn("root", types)
        self.assertIn("local", types)
        self.assertIn("subdirectory", types)
        self.assertIn("rule", types)
        self.assertIn("user-rule", types)

    def test_excluded_dirs_still_work(self):
        """Test that excluded dirs are still excluded for new discovery."""
        nm = Path(self.temp_dir) / "node_modules"
        nm.mkdir()
        (nm / "CLAUDE.md").write_text("# Should be excluded\n")

        nm_rules = nm / ".claude" / "rules"
        nm_rules.mkdir(parents=True)
        (nm_rules / "bad.md").write_text("# Should not be found\n")

        files = find_claude_files(self.temp_dir)
        all_paths = [f["path"] for f in files]
        self.assertFalse(any("node_modules" in p for p in all_paths))

    def test_no_rules_dir_no_error(self):
        """Test that missing .claude/rules/ doesn't cause errors."""
        files = find_claude_files(self.temp_dir)
        rule_files = [f for f in files if f["type"] in ("rule", "user-rule")]
        # May find user rules depending on system, but should not error
        self.assertIsInstance(files, list)


class TestSuggestClaudeFileEnhanced(unittest.TestCase):
    """Tests for enhanced suggest_claude_file() with learning_type."""

    def setUp(self):
        self.files = [
            {"path": "/home/.claude/CLAUDE.md", "relative_path": "~/.claude/CLAUDE.md", "type": "global"},
            {"path": "/project/CLAUDE.md", "relative_path": "./CLAUDE.md", "type": "root"},
            {"path": "/project/.claude/rules/guardrails.md", "relative_path": "./.claude/rules/guardrails.md",
             "type": "rule", "frontmatter": None},
            {"path": "/project/.claude/rules/api.md", "relative_path": "./.claude/rules/api.md",
             "type": "rule", "frontmatter": {"paths": ["src/api/"]}},
        ]

    def test_guardrail_routes_to_rule_file(self):
        """Test guardrail learning routes to guardrails.md."""
        result = suggest_claude_file(
            "don't add docstrings unless asked",
            self.files,
            learning_type="guardrail",
        )
        self.assertEqual(result, "./.claude/rules/guardrails.md")

    def test_guardrail_creates_path_when_no_file(self):
        """Test guardrail suggests creating guardrails.md if not found."""
        files_no_guardrails = [f for f in self.files if "guardrails" not in f.get("path", "")]
        result = suggest_claude_file(
            "don't add docstrings unless asked",
            files_no_guardrails,
            learning_type="guardrail",
        )
        self.assertEqual(result, "./.claude/rules/guardrails.md")

    def test_model_routing_global(self):
        """Test model-related learning routes to global CLAUDE.md."""
        result = suggest_claude_file("use gpt-5.1 for reasoning", self.files)
        self.assertEqual(result, "~/.claude/CLAUDE.md")

    def test_backward_compat_no_learning_type(self):
        """Test backward compatibility — no learning_type still works."""
        result = suggest_claude_file("always use venv", self.files)
        self.assertEqual(result, "~/.claude/CLAUDE.md")

    def test_path_scoped_rule_match(self):
        """Test learning mentioning a directory matches path-scoped rule."""
        result = suggest_claude_file("In the src/api/ module, use REST", self.files)
        self.assertEqual(result, "./.claude/rules/api.md")

    def test_ambiguous_returns_none(self):
        """Test ambiguous learning returns None."""
        result = suggest_claude_file("use database pooling", self.files)
        self.assertIsNone(result)


class TestAutoMemoryPath(unittest.TestCase):
    """Tests for auto memory path utilities."""

    @unittest.skipIf(platform.system() == "Windows", "Unix-specific path encoding")
    def test_folder_name_encoding_unix(self):
        """Test project folder name encoding for Unix paths."""
        result = get_project_folder_name("/Users/bob/myapp")
        self.assertEqual(result, "-Users-bob-myapp")

    @unittest.skipIf(platform.system() == "Windows", "Unix-specific path encoding")
    def test_folder_name_encoding_deep(self):
        """Test project folder name encoding for deep paths."""
        result = get_project_folder_name("/Users/bob/code/projects/myapp")
        self.assertEqual(result, "-Users-bob-code-projects-myapp")

    def test_folder_name_encoding_structure(self):
        """Test folder name encoding produces valid structure on any platform."""
        result = get_project_folder_name(tempfile.gettempdir())
        self.assertTrue(result.startswith("-"))
        self.assertNotIn("/", result)
        self.assertNotIn("\\", result)

    @unittest.skipIf(platform.system() == "Windows", "Unix-specific path encoding")
    @patch("lib.reflect_utils.get_claude_dir")
    def test_auto_memory_path_resolution(self, mock_claude_dir):
        """Test auto memory path is correctly resolved."""
        mock_claude_dir.return_value = Path("/home/user/.claude")
        path = get_auto_memory_path("/Users/bob/myapp")
        self.assertEqual(path, Path("/home/user/.claude/projects/-Users-bob-myapp/memory"))

    def test_read_auto_memory_empty(self):
        """Test reading auto memory from nonexistent directory."""
        result = read_auto_memory("/nonexistent/path")
        self.assertEqual(result, [])

    def test_read_auto_memory_with_files(self):
        """Test reading auto memory with actual files."""
        temp_dir = tempfile.mkdtemp()
        try:
            with patch("lib.reflect_utils.get_auto_memory_path") as mock_path:
                memory_dir = Path(temp_dir) / "memory"
                memory_dir.mkdir()
                (memory_dir / "general.md").write_text("# General\n- Entry one\n- Entry two\n")
                (memory_dir / "tools.md").write_text("# Tools\n- Use MCP\n")
                mock_path.return_value = memory_dir

                result = read_auto_memory()
                self.assertEqual(len(result), 2)
                names = sorted(r["name"] for r in result)
                self.assertEqual(names, ["general", "tools"])
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_suggest_topic_model(self):
        """Test topic suggestion for model-related learning."""
        topic = suggest_auto_memory_topic("use gpt-5.1 for reasoning")
        self.assertEqual(topic, "model-preferences")

    def test_suggest_topic_tool(self):
        """Test topic suggestion for tool-related learning."""
        topic = suggest_auto_memory_topic("configure the MCP server plugin")
        self.assertEqual(topic, "tool-usage")

    def test_suggest_topic_general(self):
        """Test topic suggestion falls back to general."""
        topic = suggest_auto_memory_topic("something very generic")
        self.assertEqual(topic, "general")

    def test_suggest_topic_environment(self):
        """Test topic suggestion for environment-related learning."""
        topic = suggest_auto_memory_topic("always use venv for Python projects")
        self.assertEqual(topic, "environment")

    def test_suggest_topic_workflow(self):
        """Test topic suggestion for workflow-related learning."""
        topic = suggest_auto_memory_topic("run tests before deploying")
        self.assertEqual(topic, "workflow")


class TestReadAllMemoryEntries(unittest.TestCase):
    """Tests for read_all_memory_entries()."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("lib.reflect_utils.get_claude_dir")
    def test_multi_tier_reads(self, mock_claude_dir):
        """Test reading entries from multiple tiers."""
        fake_claude = Path(self.temp_dir) / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        # Global CLAUDE.md
        (fake_claude / "CLAUDE.md").write_text("# Global\n- Use gpt-5.1\n- Always test\n")

        # Project CLAUDE.md
        (Path(self.temp_dir) / "CLAUDE.md").write_text("# Project\n- Use postgres\n")

        entries = read_all_memory_entries(self.temp_dir)
        texts = [e["text"] for e in entries]
        self.assertIn("Use gpt-5.1", texts)
        self.assertIn("Always test", texts)
        self.assertIn("Use postgres", texts)

    @patch("lib.reflect_utils.get_claude_dir")
    def test_source_tracking(self, mock_claude_dir):
        """Test that entries track their source file and type."""
        fake_claude = Path(self.temp_dir) / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        (fake_claude / "CLAUDE.md").write_text("# Global\n- Use gpt-5.1\n")
        (Path(self.temp_dir) / "CLAUDE.md").write_text("# Project\n- Use postgres\n")

        entries = read_all_memory_entries(self.temp_dir)
        global_entries = [e for e in entries if e["source_type"] == "global"]
        root_entries = [e for e in entries if e["source_type"] == "root"]
        self.assertTrue(len(global_entries) > 0)
        self.assertTrue(len(root_entries) > 0)
        self.assertEqual(global_entries[0]["source_file"], "~/.claude/CLAUDE.md")

    @patch("lib.reflect_utils.get_claude_dir")
    def test_missing_files_no_error(self, mock_claude_dir):
        """Test that missing files don't cause errors."""
        fake_claude = Path(self.temp_dir) / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        # No files exist
        entries = read_all_memory_entries(self.temp_dir)
        self.assertEqual(entries, [])

    @patch("lib.reflect_utils.get_claude_dir")
    def test_includes_referenced_file_bullets(self, mock_claude_dir):
        """Bullets from inclusion-graph-discovered docs feed the dedup pool.

        This is the load-bearing integration: /reflect's cross-tier dedup
        only reaches docs that find_claude_files() surfaces.
        """
        fake_claude = Path(self.temp_dir) / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        (Path(self.temp_dir) / "CLAUDE.md").write_text(
            "# Project\n- Use postgres\n\nSee @standards.md\n"
        )
        (Path(self.temp_dir) / "standards.md").write_text(
            "# Standards\n- Follow REST conventions\n- Async by default\n"
        )

        entries = read_all_memory_entries(self.temp_dir)
        ref_entries = [e for e in entries if e["source_type"] == "referenced"]
        texts = sorted(e["text"] for e in ref_entries)
        self.assertEqual(texts, ["Async by default", "Follow REST conventions"])
        self.assertEqual(ref_entries[0]["source_file"], "./standards.md")

    @patch("lib.reflect_utils.get_claude_dir")
    def test_rule_file_can_be_inclusion_source(self, mock_claude_dir):
        """A doc referenced from a .claude/rules/*.md file is reachable."""
        fake_claude = Path(self.temp_dir) / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        (Path(self.temp_dir) / "CLAUDE.md").write_text("# P\n")
        rules = Path(self.temp_dir) / ".claude" / "rules"
        rules.mkdir(parents=True)
        (rules / "python.md").write_text(
            "# Python\nSee [Style](../../docs/style.md)\n"
        )
        docs = Path(self.temp_dir) / "docs"
        docs.mkdir()
        (docs / "style.md").write_text("# Style\n- 4-space indent\n")

        from lib.reflect_utils import find_claude_files
        files = find_claude_files(self.temp_dir)
        ref = next(f for f in files if f["type"] == "referenced")
        self.assertEqual(Path(ref["path"]).name, "style.md")
        self.assertIn("python.md", ref["referenced_from"])


class TestParseInclusions(unittest.TestCase):
    """Tests for _parse_inclusions() — extracting @-includes and md-links."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _write(self, name, content):
        path = Path(self.temp_dir) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_at_include_simple(self):
        """@filename.md is captured as an include."""
        f = self._write("CLAUDE.md", "- See @AGENTS.md for details\n")
        self.assertEqual(_parse_inclusions(f), ["AGENTS.md"])

    def test_at_include_with_path(self):
        """@-includes capture ~ and relative path forms verbatim."""
        f = self._write(
            "CLAUDE.md",
            "- Project: @CLAUDE.md\n- Global: @~/.claude/CLAUDE.md\n- Sub: @./docs/standards.md\n",
        )
        refs = _parse_inclusions(f)
        self.assertIn("CLAUDE.md", refs)
        self.assertIn("~/.claude/CLAUDE.md", refs)
        self.assertIn("./docs/standards.md", refs)

    def test_at_include_skips_email_addresses(self):
        """@-pattern doesn't match email addresses ending in .md."""
        f = self._write("CLAUDE.md", "Contact: foo@bar.md is not an include\n")
        self.assertEqual(_parse_inclusions(f), [])

    def test_md_link_simple(self):
        """[text](path.md) is captured."""
        f = self._write("CLAUDE.md", "See [Standards](standards.md) for details.\n")
        self.assertEqual(_parse_inclusions(f), ["standards.md"])

    def test_md_link_with_title(self):
        """Link with title attribute is captured (title stripped)."""
        f = self._write("CLAUDE.md", '[Doc](./docs/api.md "API Doc")\n')
        self.assertEqual(_parse_inclusions(f), ["./docs/api.md"])

    def test_md_link_skips_external_urls(self):
        """https://, http://, mailto:, and other schemes are ignored."""
        f = self._write(
            "CLAUDE.md",
            "[GitHub](https://github.com/x.md)\n"
            "[Site](http://example.com/y.md)\n"
            "[Mail](mailto:foo@bar.com)\n",
        )
        self.assertEqual(_parse_inclusions(f), [])

    def test_md_link_skips_anchor_only(self):
        """[text](#section) is a same-file anchor and skipped."""
        f = self._write("CLAUDE.md", "[Top](#top)\n[Section](#section-1)\n")
        self.assertEqual(_parse_inclusions(f), [])

    def test_md_link_strips_in_page_anchor(self):
        """[text](foo.md#section) → foo.md (anchor stripped)."""
        f = self._write("CLAUDE.md", "[Section](standards.md#api)\n")
        self.assertEqual(_parse_inclusions(f), ["standards.md"])

    def test_skips_fenced_code_blocks(self):
        """References inside ``` and ~~~ fences are not captured."""
        f = self._write(
            "CLAUDE.md",
            "Real: @real.md\n"
            "```\n@fake.md\n[ignored](nope.md)\n```\n"
            "Mid: @mid.md\n"
            "~~~\n@tilde.md\n~~~\n"
            "After: @after.md\n",
        )
        refs = _parse_inclusions(f)
        self.assertIn("real.md", refs)
        self.assertIn("mid.md", refs)
        self.assertIn("after.md", refs)
        self.assertNotIn("fake.md", refs)
        self.assertNotIn("nope.md", refs)
        self.assertNotIn("tilde.md", refs)

    def test_unreadable_file_returns_empty(self):
        """Missing or unreadable file yields no refs (pins error branch)."""
        self.assertEqual(_parse_inclusions(Path("/nonexistent/file.md")), [])


class TestResolveInclusion(unittest.TestCase):
    """Tests for _resolve_inclusion() — path resolution + safety checks."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_resolve_relative_to_source_dir(self):
        """Relative target resolves against the source file's directory."""
        sub = Path(self.temp_dir) / "sub"
        sub.mkdir()
        (sub / "AGENTS.md").write_text("# A")
        (sub / "standards.md").write_text("# S")
        result = _resolve_inclusion("standards.md", sub / "AGENTS.md")
        self.assertEqual(result, (sub / "standards.md").resolve())

    def test_resolve_absolute_path(self):
        """Absolute target is used as-is."""
        target = Path(self.temp_dir) / "abs.md"
        target.write_text("# X")
        result = _resolve_inclusion(str(target), Path(self.temp_dir) / "src.md")
        self.assertEqual(result, target.resolve())

    def test_resolve_rejects_non_md(self):
        """Non-.md targets return None even if the file exists."""
        sub = Path(self.temp_dir) / "sub"
        sub.mkdir()
        (sub / "script.py").write_text("print('x')")
        result = _resolve_inclusion("script.py", sub / "CLAUDE.md")
        self.assertIsNone(result)

    def test_resolve_rejects_missing_file(self):
        """Targets pointing at non-existent files return None."""
        result = _resolve_inclusion(
            "missing.md", Path(self.temp_dir) / "src.md"
        )
        self.assertIsNone(result)

    def test_resolve_rejects_directory(self):
        """Directories named foo.md are not memory targets."""
        d = Path(self.temp_dir) / "weird.md"
        d.mkdir()
        result = _resolve_inclusion("weird.md", Path(self.temp_dir) / "src.md")
        self.assertIsNone(result)


class TestFollowInclusionGraph(unittest.TestCase):
    """Tests for _follow_inclusion_graph() — bounded BFS traversal."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _seed(self, path):
        return [{"path": str(path), "relative_path": "./CLAUDE.md", "type": "root"}]

    def test_one_hop_at_include(self):
        """CLAUDE.md → @AGENTS.md is discovered."""
        claude = self.root / "CLAUDE.md"
        agents = self.root / "AGENTS.md"
        claude.write_text("- See @AGENTS.md\n")
        agents.write_text("# Agents\n")

        discovered = _follow_inclusion_graph(self._seed(claude), self.root)
        self.assertEqual(len(discovered), 1)
        entry = discovered[0]
        self.assertEqual(entry["type"], "referenced")
        self.assertEqual(entry["depth"], 1)
        self.assertEqual(entry["relative_path"], "./AGENTS.md")
        self.assertEqual(entry["referenced_from"], "./CLAUDE.md")

    def test_one_hop_md_link(self):
        """CLAUDE.md → [Standards](standards.md) is discovered."""
        claude = self.root / "CLAUDE.md"
        standards = self.root / "standards.md"
        claude.write_text("[Standards](standards.md)\n")
        standards.write_text("# S\n")

        discovered = _follow_inclusion_graph(self._seed(claude), self.root)
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0]["relative_path"], "./standards.md")

    def test_two_hops(self):
        """CLAUDE.md → AGENTS.md → standards.md surfaces both."""
        claude = self.root / "CLAUDE.md"
        agents = self.root / "AGENTS.md"
        standards = self.root / "standards.md"
        claude.write_text("- @AGENTS.md\n")
        agents.write_text("- [Standards](standards.md)\n")
        standards.write_text("# Standards\n")

        discovered = _follow_inclusion_graph(self._seed(claude), self.root)
        paths = sorted(d["relative_path"] for d in discovered)
        self.assertEqual(paths, ["./AGENTS.md", "./standards.md"])

        depth_by_name = {Path(d["path"]).name: d["depth"] for d in discovered}
        self.assertEqual(depth_by_name["AGENTS.md"], 1)
        self.assertEqual(depth_by_name["standards.md"], 2)
        # Provenance: standards.md was reached via AGENTS.md
        std_entry = next(d for d in discovered if Path(d["path"]).name == "standards.md")
        self.assertEqual(std_entry["referenced_from"], "./AGENTS.md")

    def test_depth_cap_stops_traversal(self):
        """max_depth=1 stops after one hop."""
        a = self.root / "CLAUDE.md"
        b = self.root / "B.md"
        c = self.root / "C.md"
        a.write_text("@B.md\n")
        b.write_text("@C.md\n")
        c.write_text("# C\n")

        discovered = _follow_inclusion_graph(self._seed(a), self.root, max_depth=1)
        names = sorted(Path(d["path"]).name for d in discovered)
        self.assertEqual(names, ["B.md"])

    def test_depth_cap_zero_disables_traversal(self):
        """max_depth=0 means no traversal."""
        a = self.root / "CLAUDE.md"
        b = self.root / "B.md"
        a.write_text("@B.md\n")
        b.write_text("# B\n")

        discovered = _follow_inclusion_graph(self._seed(a), self.root, max_depth=0)
        self.assertEqual(discovered, [])

    def test_cycle_safe(self):
        """A → B → A does not infinitely loop, and each file is reported once."""
        a = self.root / "CLAUDE.md"
        b = self.root / "B.md"
        a.write_text("@B.md\n")
        b.write_text("@CLAUDE.md\n")

        discovered = _follow_inclusion_graph(self._seed(a), self.root)
        names = [Path(d["path"]).name for d in discovered]
        self.assertEqual(names, ["B.md"])  # CLAUDE.md is a seed, not reported

    def test_diamond_dedup(self):
        """A→B and A→C both linking to D yields D only once."""
        a = self.root / "CLAUDE.md"
        b = self.root / "B.md"
        c = self.root / "C.md"
        d = self.root / "D.md"
        a.write_text("@B.md\n@C.md\n")
        b.write_text("@D.md\n")
        c.write_text("@D.md\n")
        d.write_text("# D\n")

        discovered = _follow_inclusion_graph(self._seed(a), self.root)
        d_entries = [x for x in discovered if Path(x["path"]).name == "D.md"]
        self.assertEqual(len(d_entries), 1)

    def test_seed_skipped_when_referenced(self):
        """Following a reference back to an existing seed doesn't re-emit it."""
        claude = self.root / "CLAUDE.md"
        local = self.root / "CLAUDE.local.md"
        claude.write_text("@CLAUDE.local.md\n")
        local.write_text("# Local\n")

        seeds = [
            {"path": str(claude), "relative_path": "./CLAUDE.md", "type": "root"},
            {"path": str(local), "relative_path": "./CLAUDE.local.md", "type": "local"},
        ]
        discovered = _follow_inclusion_graph(seeds, self.root)
        self.assertEqual(discovered, [])

    @patch("lib.reflect_utils.get_claude_dir")
    def test_path_outside_allowlist_is_rejected(self, mock_claude_dir):
        """References escaping {project root, ~/.claude} are not surfaced.

        Defends against pasted-in [x](/etc/passwd.md) markdown turning the
        host filesystem into routing targets.
        """
        # Point fake claude dir well away from the test's outside_root so
        # the external file is in neither allowed root.
        fake_claude = self.root / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        outside_root = Path(tempfile.mkdtemp())
        try:
            external = outside_root / "external.md"
            external.write_text("# External\n")

            claude = self.root / "CLAUDE.md"
            claude.write_text(f"[X]({external})\n")

            discovered = _follow_inclusion_graph(self._seed(claude), self.root)
            self.assertEqual(discovered, [])
        finally:
            import shutil
            shutil.rmtree(outside_root, ignore_errors=True)

    @patch("lib.reflect_utils.get_claude_dir")
    def test_symlink_to_non_md_is_rejected(self, mock_claude_dir):
        """A `.md` symlink that resolves to a non-md file is rejected.

        Defends against `evil.md → /etc/passwd` patterns where the raw target
        passes the `.md` check but the resolved file is something else.
        """
        if os.name == "nt":
            self.skipTest("Symlink creation requires admin on Windows")

        fake_claude = self.root / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        # Real non-md file inside the project (so allowlist passes)
        real = self.root / "secrets.txt"
        real.write_text("sensitive\n")

        # Symlink with .md extension pointing at the non-md file
        evil = self.root / "evil.md"
        try:
            evil.symlink_to(real)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks not supported in this environment")

        claude = self.root / "CLAUDE.md"
        claude.write_text("@evil.md\n")

        discovered = _follow_inclusion_graph(self._seed(claude), self.root)
        self.assertEqual(discovered, [])

    def test_bfs_shortest_path_provenance(self):
        """When two paths reach the same file, depth + parent reflect the shorter one.

        Setup: A→C (1 hop) and A→B→C (2 hops). C must end up at depth 1
        with referenced_from=A, not depth 2 via B.
        """
        a = self.root / "CLAUDE.md"
        b = self.root / "B.md"
        c = self.root / "C.md"
        # A references both C (direct) and B (which also references C)
        a.write_text("@C.md\n@B.md\n")
        b.write_text("@C.md\n")
        c.write_text("# C\n")

        discovered = _follow_inclusion_graph(self._seed(a), self.root)
        c_entry = next(d for d in discovered if Path(d["path"]).name == "C.md")
        self.assertEqual(c_entry["depth"], 1)
        self.assertEqual(c_entry["referenced_from"], "./CLAUDE.md")

    def test_max_nodes_caps_traversal(self):
        """max_nodes bounds the total number of newly-discovered files."""
        a = self.root / "CLAUDE.md"
        a.write_text("\n".join(f"@f{i}.md" for i in range(10)) + "\n")
        for i in range(10):
            (self.root / f"f{i}.md").write_text("# x\n")

        discovered = _follow_inclusion_graph(
            self._seed(a), self.root, max_nodes=3,
        )
        self.assertEqual(len(discovered), 3)

    def test_multiple_links_one_line(self):
        """Multiple links on a single line are all captured."""
        a = self.root / "CLAUDE.md"
        for name in ("a.md", "b.md", "c.md"):
            (self.root / name).write_text("# x\n")
        a.write_text("See [A](a.md), [B](b.md), and [C](c.md).\n")

        discovered = _follow_inclusion_graph(self._seed(a), self.root)
        names = sorted(Path(d["path"]).name for d in discovered)
        self.assertEqual(names, ["a.md", "b.md", "c.md"])



class TestFindClaudeFilesWithInclusions(unittest.TestCase):
    """Tests for find_claude_files() with inclusion graph traversal."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("lib.reflect_utils.get_claude_dir")
    def test_referenced_files_appear_in_results(self, mock_claude_dir):
        """Files transitively referenced via @ and md-links are surfaced."""
        fake_claude = self.root / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        (self.root / "CLAUDE.md").write_text("@AGENTS.md\n[Std](standards.md)\n")
        (self.root / "AGENTS.md").write_text("- @architecture.md\n")
        (self.root / "architecture.md").write_text("# Arch\n")
        (self.root / "standards.md").write_text("# Std\n")

        files = find_claude_files(self.temp_dir)
        referenced = [f for f in files if f["type"] == "referenced"]
        names = sorted(Path(f["path"]).name for f in referenced)
        self.assertEqual(names, ["AGENTS.md", "architecture.md", "standards.md"])

    @patch("lib.reflect_utils.get_claude_dir")
    def test_follow_includes_can_be_disabled(self, mock_claude_dir):
        """follow_includes=False returns no referenced files."""
        fake_claude = self.root / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        (self.root / "CLAUDE.md").write_text("@AGENTS.md\n")
        (self.root / "AGENTS.md").write_text("# A\n")

        files = find_claude_files(self.temp_dir, follow_includes=False)
        referenced = [f for f in files if f["type"] == "referenced"]
        self.assertEqual(referenced, [])

    @patch("lib.reflect_utils.get_claude_dir")
    def test_referenced_does_not_double_with_seed(self, mock_claude_dir):
        """A subdirectory CLAUDE.md referenced by root CLAUDE.md isn't duplicated."""
        fake_claude = self.root / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        (self.root / "CLAUDE.md").write_text("@src/CLAUDE.md\n")
        sub = self.root / "src"
        sub.mkdir()
        (sub / "CLAUDE.md").write_text("# Src\n")

        files = find_claude_files(self.temp_dir)
        # src/CLAUDE.md should appear once as 'subdirectory', not 'referenced'
        sub_path = str((sub / "CLAUDE.md").resolve())
        matching = [f for f in files if Path(f["path"]).resolve() == Path(sub_path)]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["type"], "subdirectory")

    @patch("lib.reflect_utils.get_claude_dir")
    def test_auto_discovery_still_excludes_node_modules(self, mock_claude_dir):
        """Auto-discovery exclusion holds even when inclusion follow is on.

        Note: explicit references INTO an excluded dir are still followed
        (the user wrote them) — that's a deliberate trust boundary. This
        test only asserts that auto-walking doesn't pull in node_modules.
        """
        fake_claude = self.root / "fake_claude"
        fake_claude.mkdir()
        mock_claude_dir.return_value = fake_claude

        nm = self.root / "node_modules"
        nm.mkdir()
        (nm / "CLAUDE.md").write_text("# Should NOT be discovered\n")

        (self.root / "CLAUDE.md").write_text("# P\n")

        files = find_claude_files(self.temp_dir)
        self.assertFalse(any("node_modules" in f["path"] for f in files))

if __name__ == "__main__":
    unittest.main()
