"""
Stage 3 – Test + Self-Heal
===========================
For each integration:
  1. Run pytest against the dedicated test file(s).
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

from ..integration_config import get_repo_root, get_source_dir, read_all_sources
from ..utils.claude_cli import call_claude

logger = logging.getLogger(__name__)

MAX_HEAL_ROUNDS = 3

# Project-level test files (fallback when repo has no tests/ dir)
_PROJECT_TEST_FILES: dict[str, str] = {
    "crewai": "tests/test_crewai_integration.py",
    "langchain": "tests/test_langchain_integration.py",
    "llamaindex": "tests/test_llamaindex_integration.py",
}


def _get_test_path(client: str, base_dir: str) -> tuple[str, str]:
    """
    Find the test file/dir and the working directory to run pytest in.

    Returns (test_target, cwd) where:
      - test_target: path to pass to pytest (file or directory)
      - cwd: working directory for the subprocess
    """
    repo_root = get_repo_root(client)
    repo_tests = repo_root / "tests"
    if repo_tests.exists():
        return str(repo_tests), str(repo_root)

    # Fall back to project-level test file
    return _PROJECT_TEST_FILES.get(client, f"tests/test_{client}_integration.py"), base_dir


# ── Multi-file heal prompt ─────────────────────────────────────────────────

_HEAL_PROMPT = """\
The pytest test suite for the {client} endee integration has failing tests.
Your job is to fix the integration source code so that ALL tests pass.

Do NOT modify the test file(s).

─── FAILING TEST OUTPUT ─────────────────────────────────────────────────────
{test_output}
─────────────────────────────────────────────────────────────────────────────

─── CURRENT INTEGRATION SOURCE FILES ────────────────────────────────────────
{all_files_block}
─────────────────────────────────────────────────────────────────────────────

Instructions:
- Read the test failures carefully to understand what is expected.
- Fix only the integration code (the source files shown above).
- Return ALL modified files using this EXACT format per file:

==== FILE: <relative_path> ====
<complete corrected Python source for that file>

Only include files you actually modified.
No markdown fences, no explanations — only the output in the format above.
"""


def _build_files_block(sources: dict[str, str]) -> str:
    parts = []
    for rel_path, content in sorted(sources.items()):
        parts.append(f"── FILE: {rel_path} ──\n{content}\n")
    return "\n".join(parts)


def _parse_multi_file_response(raw: str) -> dict[str, str]:
    """Parse Claude's multi-file response into {rel_path: code}."""
    raw = re.sub(r"^```(?:python)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    files: dict[str, str] = {}
    parts = re.split(r"={4,}\s*FILE:\s*(.+?)\s*={4,}", raw)

    i = 1
    while i < len(parts) - 1:
        rel_path = parts[i].strip()
        code = parts[i + 1].strip()
        code = re.sub(r"^```(?:python)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        if rel_path and code:
            files[rel_path] = code
        i += 2

    return files


def _run_pytest(test_target: str, cwd: str) -> tuple[bool, str]:
    """Run pytest on a test file or directory. Returns (passed, full_output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", test_target, "-v", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def _heal(client: str, test_output: str, base_dir: str) -> None:
    """Ask Claude to fix all integration source files and write fixes to disk."""
    sources = read_all_sources(client)
    repo_root = get_repo_root(client)

    prompt = _HEAL_PROMPT.format(
        client=client,
        test_output=test_output,
        all_files_block=(
            _build_files_block(sources) if sources
            else "(no source files found)"
        ),
    )

    logger.info("[test_heal][%s] Asking Claude to heal failing tests …", client)
    raw = call_claude(prompt)

    updated_files = _parse_multi_file_response(raw)

    if not updated_files and len(sources) == 1:
        # Fallback for single-file repos
        only_path = list(sources.keys())[0]
        code = re.sub(r"^```(?:python)?\s*", "", raw.strip())
        code = re.sub(r"\s*```$", "", code)
        updated_files = {only_path: code}

    for rel_path, code in updated_files.items():
        abs_path = repo_root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(code, encoding="utf-8")
        logger.info("[test_heal][%s] Heal applied to %s (%d chars).", client, rel_path, len(code))


def _test_and_heal_one(client: str, base_dir: str = ".") -> dict:
    """Run test → heal loop for a single client."""
    test_target, cwd = _get_test_path(client, base_dir)
    passed = False
    final_output = ""

    for round_num in range(MAX_HEAL_ROUNDS + 1):
        is_first = round_num == 0
        label = "Initial run" if is_first else f"Heal round {round_num}"
        logger.info("[test_heal][%s] %s …", client, label)

        passed, output = _run_pytest(test_target, cwd)
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
    from ..integration_config import all_clients

    targets = scope or all_clients()
    results: dict[str, dict] = {}

    for client in targets:
        results[client] = _test_and_heal_one(client, base_dir)

    passed_list = [c for c, r in results.items() if r["passed"]]
    failed_list = [c for c, r in results.items() if not r["passed"]]
    logger.info(
        "[test_heal] Done. passed=%s  failed=%s", passed_list, failed_list
    )
    return results
