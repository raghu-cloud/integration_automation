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

from ..integration_config import get_repo_root, get_source_dir, read_all_sources, read_all_test_files
from ..utils.claude_cli import call_claude, MODEL_SONNET

logger = logging.getLogger(__name__)

MAX_HEAL_ROUNDS = 3


def _get_venv_python(repo_root: Path) -> str:
    """Return the Python executable from the repo's own venv, or fall back to sys.executable."""
    for candidate in ("venv/bin/python3", "venv/bin/python", ".venv/bin/python3", ".venv/bin/python"):
        py = repo_root / candidate
        if py.exists():
            logger.info("[test_heal] Using repo venv Python: %s", py)
            return str(py)
    logger.info("[test_heal] No repo venv found at %s — using sys.executable", repo_root)
    return sys.executable


def _get_test_path(client: str, base_dir: str) -> tuple[str, str] | tuple[None, None]:
    """
    Find the test directory inside the framework's cloned repo.

    Only runs tests from the framework repos (crewai/tests/, langchain/tests/,
    llamaindex/tests/). Never falls back to integration_automation/tests/.

    Returns (test_target, cwd) or (None, None) if no tests dir found.
    """
    repo_root = get_repo_root(client)
    repo_tests = repo_root / "tests"
    if repo_tests.exists():
        return str(repo_tests), str(repo_root)

    logger.warning(
        "[test_heal][%s] No tests/ directory found in repo at %s — skipping.",
        client, repo_root,
    )
    return None, None


# ── Multi-file heal prompt ─────────────────────────────────────────────────

_HEAL_PROMPT = """\
The pytest test suite for the {client} endee integration has failing tests.
Your job is to fix the code so that ALL tests pass.

You may modify BOTH the source files AND the test files if needed.

─── FAILING TEST OUTPUT ─────────────────────────────────────────────────────
{test_output}
─────────────────────────────────────────────────────────────────────────────

─── CURRENT INTEGRATION SOURCE FILES ────────────────────────────────────────
{all_files_block}
─────────────────────────────────────────────────────────────────────────────

─── CURRENT TEST FILES ──────────────────────────────────────────────────────
{test_files_block}
─────────────────────────────────────────────────────────────────────────────

Instructions:
- Read the test failures carefully to understand what is expected.
- Fix either source code, test code, or both — whatever is needed.
- When fixing tests, ensure imports and mock targets match the actual source.
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
    """Parse Claude's multi-file response into {rel_path: code}.

    Falls back to alternative separators if the primary format isn't found.
    """
    stripped = re.sub(r"^```(?:python)?\s*", "", raw.strip())
    stripped = re.sub(r"\s*```$", "", stripped)

    # Strategy 1: ==== FILE: <path> ==== markers (primary)
    files = _try_parse_eq_markers(stripped)
    if files:
        return files

    # Strategy 2: ```python  # FILE: <path>  fenced blocks
    files = _try_parse_fenced_blocks(raw.strip())
    if files:
        return files

    # Strategy 3: --- <path> --- or ## <path> headers
    files = _try_parse_alt_separators(stripped)
    if files:
        return files

    logger.debug(
        "[parse] No file markers found. Raw response (first 500 chars):\n%s",
        raw[:500],
    )
    return {}


def _try_parse_eq_markers(raw: str) -> dict[str, str]:
    """Parse using ==== FILE: ... ==== markers."""
    parts = re.split(r"={4,}\s*FILE:\s*(.+?)\s*={4,}", raw)
    if len(parts) < 3:
        return {}
    files: dict[str, str] = {}
    i = 1
    while i < len(parts) - 1:
        rel_path = parts[i].strip()
        code = parts[i + 1].strip()
        code = re.sub(r"^```(?:python)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        if rel_path and code:
            files[rel_path] = _sanitize_code(code)
        i += 2
    return files


def _try_parse_fenced_blocks(raw: str) -> dict[str, str]:
    """Parse markdown fenced code blocks with a # FILE: <path> comment."""
    pattern = re.compile(
        r"```(?:python)?\s*\n"
        r"\s*#\s*FILE:\s*(.+?)\s*\n"
        r"(.*?)"
        r"\n\s*```",
        re.DOTALL,
    )
    files: dict[str, str] = {}
    for m in pattern.finditer(raw):
        rel_path = m.group(1).strip()
        code = m.group(2).strip()
        if rel_path and code:
            files[rel_path] = _sanitize_code(code)
    return files


def _try_parse_alt_separators(raw: str) -> dict[str, str]:
    """Parse using --- <path> --- or # FILE: <path> or ## <path> headers."""
    # Try --- path --- separators
    parts = re.split(r"-{3,}\s*(.+?\.py)\s*-{3,}", raw)
    if len(parts) >= 3:
        files: dict[str, str] = {}
        i = 1
        while i < len(parts) - 1:
            rel_path = parts[i].strip()
            code = parts[i + 1].strip()
            code = re.sub(r"^```(?:python)?\s*", "", code)
            code = re.sub(r"\s*```$", "", code)
            if rel_path and code:
                files[rel_path] = _sanitize_code(code)
            i += 2
        if files:
            return files

    # Try # FILE: path  or  ## path.py  headers
    parts = re.split(r"^#{1,2}\s*(?:FILE:\s*)?(.+?\.py)\s*$", raw, flags=re.MULTILINE)
    if len(parts) >= 3:
        files = {}
        i = 1
        while i < len(parts) - 1:
            rel_path = parts[i].strip()
            code = parts[i + 1].strip()
            code = re.sub(r"^```(?:python)?\s*", "", code)
            code = re.sub(r"\s*```$", "", code)
            if rel_path and code:
                files[rel_path] = _sanitize_code(code)
            i += 2
        if files:
            return files

    return {}


def _sanitize_code(code: str) -> str:
    """
    Strip leading/trailing prose lines that Claude sometimes adds.

    Removes lines before the first line that looks like Python code
    and after the last Python-looking line.
    """
    lines = code.split("\n")
    python_line = re.compile(
        r"^\s*("
        r"import |from |class |def |#|@|\"\"\"|\'\'\'"
        r"|if |elif |else:|try:|except |finally:|with "
        r"|return |raise |yield |assert "
        r"|[A-Z_][A-Z_0-9]*\s*=|[a-z_][a-z_0-9]*\s*="
        r"|__"
        r"|\)"
        r"|\]"
        r")"
    )

    first_code = 0
    for idx, line in enumerate(lines):
        if python_line.match(line):
            first_code = idx
            break

    last_code = len(lines) - 1
    for idx in range(len(lines) - 1, first_code - 1, -1):
        line = lines[idx]
        if line.strip() == "":
            continue
        if line.startswith("    ") or line.startswith("\t"):
            last_code = idx
            break
        if python_line.match(line):
            last_code = idx
            break
        last_code = idx - 1

    return "\n".join(lines[first_code : last_code + 1]).strip()


def _validate_python(code: str, filepath: str) -> bool:
    """Check if code is valid Python. Returns True if it compiles."""
    try:
        compile(code, filepath, "exec")
        return True
    except SyntaxError as e:
        logger.warning(
            "[validate] REJECTED %s — invalid Python (line %s): %s",
            filepath, e.lineno, e.msg,
        )
        return False


def _run_pytest(test_target: str, cwd: str, repo_root: Path | None = None) -> tuple[bool, str]:
    """Run pytest on a test file or directory. Returns (passed, full_output)."""
    python = _get_venv_python(repo_root) if repo_root else sys.executable
    result = subprocess.run(
        [python, "-m", "pytest", test_target, "-v", "--tb=short", "--no-header"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    output = result.stdout + result.stderr
    return result.returncode == 0, output


def _parse_test_counts(output: str) -> dict[str, int]:
    """Extract passed/failed/error counts from pytest output."""
    counts = {"passed": 0, "failed": 0, "errors": 0}
    m = re.search(r"(\d+) passed", output)
    if m:
        counts["passed"] = int(m.group(1))
    m = re.search(r"(\d+) failed", output)
    if m:
        counts["failed"] = int(m.group(1))
    m = re.search(r"(\d+) error", output)
    if m:
        counts["errors"] = int(m.group(1))
    return counts


def _is_env_error(test_output: str) -> bool:
    """Return True if all failures are environment/import errors (not worth healing)."""
    env_patterns = ["ModuleNotFoundError", "ImportError", "No module named"]
    # Check if there are real assertion failures (worth healing)
    has_assertion_fail = bool(
        re.search(r"(AssertionError|assert |FAILED|failed)", test_output, re.IGNORECASE)
    )
    has_env_error = any(p in test_output for p in env_patterns)
    # Skip heal only if we have env errors but NO assertion failures
    return has_env_error and not has_assertion_fail


def _extract_failing_test_files(test_output: str, test_sources: dict[str, str]) -> dict[str, str]:
    """Filter test_sources to only include files that appear in the pytest failure output."""
    if not test_sources:
        return test_sources
    failing = {}
    for rel_path, content in test_sources.items():
        # Check if this test file is mentioned in the pytest output
        basename = Path(rel_path).name
        if basename in test_output or rel_path in test_output:
            failing[rel_path] = content
    # If we couldn't match any, return all (safer fallback)
    return failing if failing else test_sources


def _heal(client: str, test_output: str, base_dir: str) -> None:
    """Ask Claude to fix source and/or test files and write fixes to disk."""
    # Skip heal for environment/import errors — these need manual fixes
    if _is_env_error(test_output):
        logger.info(
            "[test_heal][%s] Skipping heal — failures are environment/import errors.",
            client,
        )
        return

    sources = read_all_sources(client)
    test_sources = read_all_test_files(client)
    repo_root = get_repo_root(client)

    # Truncate test output to last 4K chars (failure info is at the end)
    _MAX_TEST_OUTPUT = 4_000
    if len(test_output) > _MAX_TEST_OUTPUT:
        test_output = (
            "... [TRUNCATED — showing last 4000 chars] ...\n"
            + test_output[-_MAX_TEST_OUTPUT:]
        )

    # Only include test files that actually failed to reduce token usage
    relevant_tests = _extract_failing_test_files(test_output, test_sources)

    all_files_block = _build_files_block(sources) if sources else "(no source files found)"
    test_files_block = _build_files_block(relevant_tests) if relevant_tests else "(no test files found)"

    # Cap total prompt size ~100K chars
    _MAX_PROMPT = 100_000
    total = len(test_output) + len(all_files_block) + len(test_files_block)
    if total > _MAX_PROMPT:
        logger.info(
            "[test_heal][%s] Heal prompt too large (%d chars); trimming.",
            client, total,
        )
        # Trim test files block — keep only first 2 files
        if len(relevant_tests) > 2:
            trimmed = dict(list(relevant_tests.items())[:2])
            test_files_block = (
                _build_files_block(trimmed)
                + f"\n... ({len(relevant_tests) - 2} more test file(s) omitted to fit context) ..."
            )
        # Trim source block — keep only first 2 files
        if len(sources) > 2:
            trimmed = dict(list(sources.items())[:2])
            all_files_block = (
                _build_files_block(trimmed)
                + f"\n... ({len(sources) - 2} more source file(s) omitted to fit context) ..."
            )

    prompt = _HEAL_PROMPT.format(
        client=client,
        test_output=test_output,
        all_files_block=all_files_block,
        test_files_block=test_files_block,
    )

    logger.info("[test_heal][%s] Asking Claude to heal failing tests (prompt=%d chars) …", client, len(prompt))
    raw = call_claude(prompt, model=MODEL_SONNET, timeout=180)

    updated_files = _parse_multi_file_response(raw)

    if not updated_files and len(sources) == 1:
        only_path = list(sources.keys())[0]
        code = re.sub(r"^```(?:python)?\s*", "", raw.strip())
        code = re.sub(r"\s*```$", "", code)
        code = _sanitize_code(code)
        updated_files = {only_path: code}

    for rel_path, code in updated_files.items():
        if not _validate_python(code, rel_path):
            logger.error(
                "[test_heal][%s] SKIPPED writing %s — Claude output is not valid Python.",
                client, rel_path,
            )
            continue
        abs_path = repo_root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(code, encoding="utf-8")
        logger.info("[test_heal][%s] Heal applied to %s (%d chars).", client, rel_path, len(code))


def _test_and_heal_one(client: str, base_dir: str = ".", notify=None) -> dict:
    """Run test → heal loop for a single client."""

    def _notify(msg: str) -> None:
        if notify:
            try:
                notify(msg)
            except Exception:
                pass

    test_target, cwd = _get_test_path(client, base_dir)

    # Skip if no tests/ directory exists in the framework repo
    if test_target is None:
        _notify(f"  ⏭️  `{client}` — skipped (no tests/ directory in repo)")
        return {
            "client": client,
            "passed": False,
            "rounds_used": 0,
            "summary": "skipped — no tests/ directory in repo",
            "passed_count": 0,
            "failed_count": 0,
            "error_count": 0,
            "output": "",
        }

    _notify(f"  🔄 `{client}` — running tests …")
    passed = False
    final_output = ""
    round_num = 0
    repo_root = get_repo_root(client)

    for round_num in range(MAX_HEAL_ROUNDS + 1):
        is_first = round_num == 0
        label = "Initial run" if is_first else f"Heal round {round_num}"
        logger.info("[test_heal][%s] %s …", client, label)

        passed, output = _run_pytest(test_target, cwd, repo_root=repo_root)
        final_output = output

        if passed:
            logger.info("[test_heal][%s] All tests passed on %s.", client, label.lower())
            # Extract summary for Slack
            summary_match = re.search(r"(\d+ (?:passed|failed)[^\n]*)", output)
            summary_text = summary_match.group(1) if summary_match else "all passed"
            _notify(
                f"  ✅ `{client}` — {summary_text} "
                f"({'first run' if round_num == 0 else f'{round_num + 1} rounds'})"
            )
            break

        logger.warning(
            "[test_heal][%s] Tests failed on %s (round %d/%d).",
            client, label.lower(), round_num + 1, MAX_HEAL_ROUNDS + 1,
        )

        if round_num < MAX_HEAL_ROUNDS:
            _notify(f"  🔧 `{client}` — tests failed, auto-healing (round {round_num + 1}/{MAX_HEAL_ROUNDS}) …")
            try:
                _heal(client, output, base_dir)
            except Exception as exc:
                logger.error(
                    "[test_heal][%s] Heal attempt failed: %s — skipping remaining rounds.",
                    client, exc,
                )
                _notify(f"  ❌ `{client}` — heal failed: {exc}")
                break
        else:
            logger.error(
                "[test_heal][%s] Exhausted %d heal round(s) — still failing.",
                client, MAX_HEAL_ROUNDS,
            )

    # If we exited without passing, send a failure notification
    if not passed:
        summary_match = re.search(r"(\d+ (?:passed|failed)[^\n]*)", final_output)
        summary_text = summary_match.group(1) if summary_match else "failed"
        _notify(
            f"  ❌ `{client}` — {summary_text} "
            f"(after {round_num + 1} round(s))"
        )

    # Extract summary line and structured counts from pytest output
    summary_match = re.search(r"(\d+ (?:passed|failed)[^\n]*)", final_output)
    summary = summary_match.group(1) if summary_match else ("passed" if passed else "failed")
    counts = _parse_test_counts(final_output)

    return {
        "client": client,
        "passed": passed,
        "rounds_used": round_num + 1,
        "summary": summary,
        "passed_count": counts["passed"],
        "failed_count": counts["failed"],
        "error_count": counts["errors"],
        "output": final_output,
    }


def run_and_heal_all(
    scope: list[str] | None = None,
    base_dir: str = ".",
    notify=None,
) -> dict[str, dict]:
    """
    Run tests + self-heal for all in-scope integrations.

    Args:
        scope:    Client names to test. None = all three.
        base_dir: Project root directory.
        notify:   Optional callback for real-time Slack notifications.

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
        try:
            results[client] = _test_and_heal_one(client, base_dir, notify=notify)
        except Exception as exc:
            logger.error(
                "[test_heal][%s] Unhandled error during test+heal: %s", client, exc
            )
            if notify:
                try:
                    notify(f"  ❌ `{client}` — error: {exc}")
                except Exception:
                    pass
            results[client] = {
                "client": client,
                "passed": False,
                "rounds_used": 0,
                "summary": f"error — {exc}",
                "passed_count": 0,
                "failed_count": 0,
                "error_count": 1,
                "output": str(exc),
            }

    passed_list = [c for c, r in results.items() if r["passed"]]
    failed_list = [c for c, r in results.items() if not r["passed"]]
    logger.info(
        "[test_heal] Done. passed=%s  failed=%s", passed_list, failed_list
    )
    return results
