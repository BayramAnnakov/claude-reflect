#!/usr/bin/env python3
"""Semantic learning detection using Claude Code CLI.

Uses `claude -p` (print mode) to semantically analyze user messages
and determine if they contain reusable learnings.
"""
import json
import subprocess
import sys
from typing import Optional, Dict, Any

# Default timeout for Claude CLI calls (seconds)
DEFAULT_TIMEOUT = 30

# Semantic analysis prompt template
ANALYSIS_PROMPT = """Analyze this user message from a coding session. Determine if it contains
a reusable learning, correction, or preference that should be remembered for future sessions.

Message: "{text}"

Respond ONLY with valid JSON (no markdown, no explanation):
{{
  "is_learning": true or false,
  "type": "correction" or "positive" or "explicit" or null,
  "confidence": 0.0 to 1.0,
  "reasoning": "brief 1-sentence explanation",
  "extracted_learning": "concise actionable statement to add to CLAUDE.md, or null if not a learning"
}}

Guidelines:
- correction: User telling AI to do something differently ("use X not Y", "don't use Z")
- positive: User affirming good behavior ("perfect!", "exactly right", "great approach")
- explicit: User explicitly asking to remember ("remember: ...", "always do X")
- is_learning=true only if it's reusable across sessions (not one-time task instructions)
- confidence: How certain this is a genuine, reusable learning (0.6+ to be useful)
- extracted_learning: Should be actionable and concise (e.g., "Use gpt-5.1 for reasoning tasks")
- Works for ANY language - understand intent, not just English keywords
- Filter out: questions, greetings, one-time commands, context-specific requests"""


def semantic_analyze(
    text: str,
    timeout: int = DEFAULT_TIMEOUT,
    model: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Analyze text using Claude to determine if it's a learning.

    Args:
        text: The user message to analyze
        timeout: Timeout in seconds for the Claude CLI call
        model: Optional model override (e.g., "sonnet", "haiku")

    Returns:
        Dictionary with analysis results, or None on failure:
        {
            "is_learning": bool,
            "type": "correction" | "positive" | "explicit" | None,
            "confidence": float (0.0-1.0),
            "reasoning": str,
            "extracted_learning": str | None
        }
    """
    if not text or not text.strip():
        return None

    # Build the prompt
    prompt = ANALYSIS_PROMPT.format(text=text.replace('"', '\\"'))

    # Build command
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd.extend(["--model", model])

    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

        if result.returncode != 0:
            # Claude CLI failed
            return None

        # Parse the JSON output
        output = result.stdout.strip()
        if not output:
            return None

        # Claude -p --output-format json wraps the response
        # Try to extract the actual JSON from the response
        try:
            response = json.loads(output)
            # If it's wrapped in a "result" field, extract it
            if isinstance(response, dict) and "result" in response:
                content = response["result"]
            else:
                content = response
        except json.JSONDecodeError:
            # Try to find JSON in the output
            content = _extract_json_from_text(output)
            if content is None:
                return None

        # Validate and normalize the response
        return _validate_response(content)

    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        # Claude CLI not installed
        return None
    except Exception:
        return None


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract JSON from text that may have surrounding content."""
    # Find JSON object boundaries
    start = text.find('{')
    if start == -1:
        return None

    # Find matching closing brace
    depth = 0
    for i, char in enumerate(text[start:], start):
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def _validate_response(content: Any) -> Optional[Dict[str, Any]]:
    """Validate and normalize the response from Claude."""
    if not isinstance(content, dict):
        return None

    # Check required fields
    if "is_learning" not in content:
        return None

    # Normalize boolean
    is_learning = content.get("is_learning")
    if isinstance(is_learning, str):
        is_learning = is_learning.lower() in ("true", "yes", "1")
    else:
        is_learning = bool(is_learning)

    # Normalize type
    learning_type = content.get("type")
    if learning_type not in ("correction", "positive", "explicit", None):
        learning_type = None

    # Normalize confidence
    try:
        confidence = float(content.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
    except (TypeError, ValueError):
        confidence = 0.5 if is_learning else 0.0

    return {
        "is_learning": is_learning,
        "type": learning_type if is_learning else None,
        "confidence": confidence,
        "reasoning": str(content.get("reasoning", "")),
        "extracted_learning": content.get("extracted_learning") if is_learning else None,
    }


def validate_queue_items(
    items: list,
    timeout: int = DEFAULT_TIMEOUT,
    model: Optional[str] = None
) -> list:
    """
    Validate a list of queue items using semantic analysis.

    Items that fail semantic validation are filtered out.
    Items that pass have their confidence updated.

    Args:
        items: List of queue items from learnings-queue.json
        timeout: Timeout per item
        model: Optional model override

    Returns:
        Filtered and enhanced list of queue items
    """
    validated = []

    for item in items:
        message = item.get("message", "")
        if not message:
            continue

        # Run semantic analysis
        result = semantic_analyze(message, timeout=timeout, model=model)

        if result is None:
            # Fallback: keep original item if semantic fails
            validated.append(item)
            continue

        if not result.get("is_learning"):
            # Semantic says it's not a learning - filter out
            continue

        # Merge semantic analysis into item
        enhanced = {**item}
        enhanced["semantic_confidence"] = result["confidence"]
        enhanced["semantic_type"] = result["type"]
        enhanced["semantic_reasoning"] = result["reasoning"]

        if result.get("extracted_learning"):
            enhanced["extracted_learning"] = result["extracted_learning"]

        # Update confidence to be the higher of regex and semantic
        original_confidence = item.get("confidence", 0.6)
        enhanced["confidence"] = max(original_confidence, result["confidence"])

        validated.append(enhanced)

    return validated


if __name__ == "__main__":
    # Simple test when run directly
    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
    else:
        test_text = "no, use Python instead of JavaScript"

    print(f"Analyzing: {test_text!r}")
    result = semantic_analyze(test_text)
    if result:
        print(json.dumps(result, indent=2))
    else:
        print("Analysis failed or returned None")
