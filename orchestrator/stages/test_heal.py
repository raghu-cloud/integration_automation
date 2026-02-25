"""
Stage 3 – Test + Self-Heal
===========================
For each integration:
  1. Run pytest against the dedicated test file.
  2. If tests fail AND we still have heal rounds left, ask Claude CLI to fix
     the integration source, write the fix, and rerun.
  3. Repeat up to MAX_HEAL_ROUNDS times.
  4. Return the final pass/fail status with the full pytest output.

The self-heal loop stops as soon as all tests pass or rounds are exhausted.
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

from ..utils.claude_cli import call_claude

logger = logging.getLogger(__name__)

MAX_HEAL_ROUNDS = 3

_TEST_FILES: dict[str, str] = {
    "crewai": "tests/test_crewai_integration.py",
    "langchain": "tests/test_langchain_integration.py",
    "llamaindex": "tests/test_llamaindex_integration.py",
}

_INTEGRATION_FILES: dict[str, str] = {
    "crewai": "integrations/crewai_endee/tools.py",
    "langchain": "integrations/langchain_endee/vectorstore.py",
    "llamaindex": "integrations/llamaindex_endee/vector_store.py",
}

_HEAL_PROMPT = """\
The pytest test suite for the {client} endee integration has failing tests.
Your job is to fix the integration source code so that ALL tests pass.

Do NOT modify the test file.

─── FAILING TEST OUTPUT ─────────────────────────────────────────────────────
{test_output}
─────────────────────────────────────────────────────────────────────────────

─── CURRENT INTEGRATION SOURCE ──────────────────────────────────────────────
{current_code}
─────────────────────────────────────────────────────────────────────────────

Instructions:
- Read the test failures carefully to understand what is expected.
- Fix only the integration code (the source file shown above).
- Return the COMPLETE corrected file content.
- No markdown fences, no explanations — only raw Python source.
"""


def _run_pytest(test_file: str, base_dir: str = ".") -> tuple[bool, str]:
    """Run pytest on a single test file. Returns (passed, full_output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        cwd=base_dir,
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def _heal(client: str, test_output: str, base_dir: str) -> None:
    """Ask Claude to fix the integration source and write the fix to disk."""
    src_path = Path(base_dir) / _INTEGRATION_FILES[client]
    current_code = src_path.read_text(encoding="utf-8") if src_path.exists() else ""

    prompt = _HEAL_PROMPT.format(
        client=client,
        test_output=test_output,
        current_code=current_code,
    )

    logger.info("[test_heal][%s] Asking Claude to heal failing tests …", client)
    fixed_code = call_claude(prompt)

    # Strip markdown fences
    fixed_code = re.sub(r"^```(?:python)?\s*", "", fixed_code.strip())
    fixed_code = re.sub(r"\s*```$", "", fixed_code)

    src_path.parent.mkdir(parents=True, exist_ok=True)
    src_path.write_text(fixed_code, encoding="utf-8")
    logger.info("[test_heal][%s] Heal applied (%d chars).", client, len(fixed_code))


def _test_and_heal_one(client: str, base_dir: str = ".") -> dict:
    """Run test → heal loop for a single client."""
    test_file = _TEST_FILES[client]
    passed = False
    final_output = ""

    for round_num in range(MAX_HEAL_ROUNDS + 1):
        is_first = round_num == 0
        label = "Initial run" if is_first else f"Heal round {round_num}"
        logger.info("[test_heal][%s] %s …", client, label)

        passed, output = _run_pytest(test_file, base_dir)
        final_output = output

        if passed:
            logger.info("[test_heal][%s] All tests passed on %s.", client, label.lower())
            break

        logger.warning(
            "[test_heal][%s] Tests failed on %s (round %d/%d).",
            client, label.lower(), round_num + 1, MAX_HEAL_ROUNDS + 1,
        )

        if round_num < MAX_HEAL_ROUNDS:
            _heal(client, output, base_dir)
        else:
            logger.error(
                "[test_heal][%s] Exhausted %d heal round(s) — still failing.",
                client, MAX_HEAL_ROUNDS,
            )

    # Extract summary line from pytest output (e.g. "3 passed, 1 failed")
    summary_match = re.search(r"(\d+ (?:passed|failed)[^\n]*)", final_output)
    summary = summary_match.group(1) if summary_match else ("passed" if passed else "failed")

    return {
        "client": client,
        "passed": passed,
        "rounds_used": round_num + 1,
        "summary": summary,
        "output": final_output,
    }


def run_and_heal_all(
    scope: list[str] | None = None,
    base_dir: str = ".",
) -> dict[str, dict]:
    """
    Run tests + self-heal for all in-scope integrations.

    Args:
        scope:    Client names to test. None = all three.
        base_dir: Project root directory.

    Returns:
        Dict keyed by client name:
        {
          "crewai":    {"passed": bool, "rounds_used": int, "summary": str, "output": str},
          "langchain": { … },
          "llamaindex":{ … },
        }
    """
    targets = scope or list(_TEST_FILES.keys())
    results: dict[str, dict] = {}

    for client in targets:
        results[client] = _test_and_heal_one(client, base_dir)

    passed_list = [c for c, r in results.items() if r["passed"]]
    failed_list = [c for c, r in results.items() if not r["passed"]]
    logger.info(
        "[test_heal] Done. passed=%s  failed=%s", passed_list, failed_list
    )
    return results
