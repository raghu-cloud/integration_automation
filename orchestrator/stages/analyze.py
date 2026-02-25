"""
Stage 1 – Analyze Diff
=======================
Sends the raw comparison report to Claude Code CLI and asks it to return a
structured JSON summary of every meaningful change.

Returned dict shape
-------------------
{
  "summary": str,                      # plain-English paragraph
  "version_from": str,
  "version_to": str,
  "changes": [
    {
      "file": str,
      "change_type": str,              # new_constant | new_parameter | ...
      "description": str,
      "old_value": str | null,
      "new_value": str
    }
  ],
  "new_parameters": {
    "<param_name>": {
      "type": str,
      "default": any,
      "range": str,
      "description": str
    }
  },
  "integration_impact": str            # what each downstream client must change
}
"""

from __future__ import annotations

import json
import logging
import re

from ..utils.claude_cli import call_claude

logger = logging.getLogger(__name__)

_PROMPT_TEMPLATE = """\
You are a senior Python SDK analyst. Analyse the comparison report below and
extract every meaningful change as structured JSON.

COMPARISON REPORT
=================
{report}
=================

Return a single JSON object with these exact keys — no markdown fences,
no extra commentary, only the raw JSON:

{{
  "summary": "<one paragraph plain-English summary of what changed>",
  "version_from": "<old version string>",
  "version_to": "<new version string>",
  "changes": [
    {{
      "file": "<path/to/file.py>",
      "change_type": "<new_constant|new_parameter|function_signature_change|new_field|example_update|version_bump>",
      "description": "<what changed in one sentence>",
      "old_value": "<previous value or null>",
      "new_value": "<new value>"
    }}
  ],
  "new_parameters": {{
    "<param_name>": {{
      "type": "<Python type>",
      "default": "<default value>",
      "range": "<valid range or null>",
      "description": "<docstring-quality explanation>"
    }}
  }},
  "integration_impact": "<what crewai/langchain/llamaindex integrations must change>"
}}
"""


def analyze_diff(report_content: str) -> dict:
    """
    Call Claude CLI to parse the comparison report and return structured changes.

    Args:
        report_content: Raw text of comparison_report.txt.

    Returns:
        Parsed dict matching the schema above.

    Raises:
        ValueError: If Claude returns malformed JSON.
        RuntimeError: If the Claude CLI call fails.
    """
    prompt = _PROMPT_TEMPLATE.format(report=report_content)

    logger.info("[analyze] Sending report to Claude CLI for analysis …")
    raw = call_claude(prompt)

    # Strip accidental markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"[analyze] Claude returned invalid JSON: {exc}\n\nRaw output:\n{raw}"
        ) from exc

    changes_count = len(result.get("changes", []))
    params_count = len(result.get("new_parameters", {}))
    logger.info(
        "[analyze] Done. %d change(s), %d new parameter(s).",
        changes_count,
        params_count,
    )
    return result
