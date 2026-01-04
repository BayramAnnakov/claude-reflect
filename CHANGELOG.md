# Changelog

All notable changes to claude-reflect will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-04

### Added
- **Windows Support** - Native Python scripts replace bash, no WSL required (#1)
- **Semantic AI Detection** - Multi-language support via `claude -p` (#2, #3)
- **UserPromptSubmit Hook** - Automatic capture now properly registered
- **GitHub Actions CI** - Automated testing on Windows, macOS, Linux (Python 3.8 & 3.11)
- **Comparison Tool** - `scripts/compare_detection.py` for testing detection accuracy
- **90 Unit Tests** - Comprehensive test coverage with mocked Claude CLI calls

### Changed
- Hooks now use Python scripts instead of bash for cross-platform compatibility
- `/reflect` command validates queue items with semantic AI before presenting
- Detection uses hybrid approach: regex patterns (fast, real-time) + semantic AI (accurate, during /reflect)
- Updated documentation (README.md, CLAUDE.md) with new architecture

### Deprecated
- Bash scripts moved to `scripts/legacy/` (still available for reference)

### Fixed
- Hooks failing on Windows due to bash dependency (#1)
- False positives from English-only regex patterns (#2)
- Multi-language corrections not being detected (#3)
- UserPromptSubmit hook not registered in hooks.json

## [1.4.1] - 2025-12-xx

### Fixed
- Critical jq filter bug in distribution files
- Historical scan now ensures matches are always presented to user
- Queue items being ignored during history scan

## [1.4.0] - 2025-12-xx

### Added
- Confidence scoring for learnings (0.60-0.90)
- Positive feedback pattern detection
- AGENTS.md sync support
- Semantic deduplication (`/reflect --dedupe`)

## [1.3.5] - 2025-12-xx

### Changed
- PreCompact hook now informs and backs up instead of blocking

## [1.3.4] - 2025-12-xx

### Fixed
- Restored UserPromptSubmit hook for automatic capture
