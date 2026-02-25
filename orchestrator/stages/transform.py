"""
Stage 2 – Transform Integrations  (parallel × 3)
==================================================
For each integration in scope, reads **all** source files from the repo's
source directory, sends (current code + analysis) to Claude CLI, receives
updated code, and writes the changes back to disk.

All three Claude CLI calls are dispatched concurrently via ThreadPoolExecutor
so the total wall-clock time is roughly the slowest single call, not the sum.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
from pathlib import Path

from ..integration_config import get_repo_root, get_source_dir, read_all_sources, read_all_test_files
from ..utils.claude_cli import call_claude

logger = logging.getLogger(__name__)


_CONTEXT: dict[str, str] = {
    "crewai": (
        "CrewAI BaseTool — the _run() method calls index.query(). "
        "New params must appear in the Pydantic input schema (EndeeSearchInput) "
        "AND be forwarded to index.query() inside _run()."
    ),
    "langchain": (
        "LangChain VectorStore — similarity_search() and similarity_search_with_score() "
        "call index.query(). New params should be explicit keyword args with defaults "
        "so existing callers are unaffected."
    ),
    "llamaindex": (
        "LlamaIndex BasePydanticVectorStore — query() calls index.query(). "
        "New params should also be readable from query.query_kwargs so callers can "
        "pass them without changing the VectorStoreQuery API."
    ),
}

# ── Multi-file prompt ──────────────────────────────────────────────────────

_FILE_SEPARATOR = "═" * 72
_FILE_BLOCK = """\
── FILE: {rel_path} ─────────────────────────────────────────────────────
{content}
"""

_PROMPT_TEMPLATE = """\
You are an expert Python engineer updating a downstream SDK integration.

INTEGRATION: {client}
SOURCE DIRECTORY: {src_dir}
CONTEXT: {context}

─── CURRENT SOURCE FILES ────────────────────────────────────────────────
{all_files_block}
{separator}

─── ENDEE CLIENT CHANGES (from the comparison report) ───────────────────
{summary}

New parameters added to endee.Index.query():

{new_params_block}
─────────────────────────────────────────────────────────────────────────

REQUIREMENTS
1. Add both new parameters to every public query/search method across ALL files.
2. Forward them to index.query() with identical names and the same defaults.
3. Existing callers must not break — use default values that match the endee defaults.
4. Update docstrings to document the new parameters.
5. Keep the same code style, imports, and class structure.
6. Return ALL modified files. Use this EXACT format for each file:

==== FILE: <relative_path> ====
<complete updated Python source for that file>

Only include files that you actually modified. If a file needs no changes (e.g. __init__.py), omit it.
Do NOT wrap output in markdown fences.
"""

# ── Test-file transform prompt ─────────────────────────────────────────────

_TEST_TRANSFORM_PROMPT = """\
You are an expert Python test engineer.
The integration source code for the {client} endee integration has just been
updated. Your job is to update the **test scripts** so they stay in sync with
the new source code.

─── UPDATED SOURCE FILES ───────────────────────────────────────────────
{updated_source_block}
─────────────────────────────────────────────────────────────────────────

─── CURRENT TEST FILES ─────────────────────────────────────────────────
{test_files_block}
─────────────────────────────────────────────────────────────────────────

REQUIREMENTS
1. Update imports, mock targets, and assertions in the tests to match
   any renamed functions, new parameters, or changed signatures in the source.
2. Add new test cases for any newly added parameters or features.
3. Keep existing test coverage — do NOT remove tests unless the feature they
   tested has been entirely removed.
4. For unit tests that use mocks, make sure the mock targets match the
   actual import paths in the updated source code.
5. Return ALL modified files using this EXACT format per file:

==== FILE: <relative_path> ====
<complete updated Python source for that file>

Only include test files you actually modified. Do NOT modify source files.
No markdown fences, no explanations — only the output in the format above.
"""


def _format_new_params(new_parameters: dict) -> str:
    lines = []
    for name, meta in new_parameters.items():
        lines.append(
            f"  {name}: {meta.get('type', 'int')} "
            f"(default={meta.get('default')}, range={meta.get('range')}) "
            f"— {meta.get('description', '')}"
        )
    return "\n".join(lines) if lines else "  (see summary above)"


def _build_files_block(sources: dict[str, str]) -> str:
    """Concatenate all source files into a single labelled block."""
    parts = []
    for rel_path, content in sorted(sources.items()):
        parts.append(_FILE_BLOCK.format(rel_path=rel_path, content=content))
    return "\n".join(parts)


def _parse_multi_file_response(raw: str) -> dict[str, str]:
    """
    Parse Claude's multi-file response into {rel_path: code}.

    Expected format per file:
        ==== FILE: crewai_endee/utils.py ====
        <code>
    """
    # Strip accidental markdown fences from the whole response
    raw = re.sub(r"^```(?:python)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    files: dict[str, str] = {}
    # Split on the file header markers
    parts = re.split(r"={4,}\s*FILE:\s*(.+?)\s*={4,}", raw)

    # parts[0] is any preamble (should be empty), then alternating:
    #   parts[1] = filename, parts[2] = code
    #   parts[3] = filename, parts[4] = code, ...
    i = 1
    while i < len(parts) - 1:
        rel_path = parts[i].strip()
        code = parts[i + 1].strip()
        # Strip per-file markdown fences if Claude added them
        code = re.sub(r"^```(?:python)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
        if rel_path and code:
            # Sanitize: remove any leading prose before actual Python code
            code = _sanitize_code(code)
            files[rel_path] = code
        i += 2

    return files


def _sanitize_code(code: str) -> str:
    """
    Strip leading prose/summary lines that Claude sometimes adds before code.

    Removes lines before the first line that looks like Python (import, from,
    class, def, #, triple-quote, decorator, or assignment).
    Also strips trailing prose after the last Python-looking line.
    """
    lines = code.split("\n")

    # Pattern for lines that are clearly Python code
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

    # Find first Python-looking line
    first_code = 0
    for idx, line in enumerate(lines):
        if python_line.match(line):
            first_code = idx
            break

    # Find last Python-looking line (scan from end, skip blanks)
    last_code = len(lines) - 1
    for idx in range(len(lines) - 1, first_code - 1, -1):
        line = lines[idx]
        # Skip trailing blank lines
        if line.strip() == "":
            continue
        # Indented lines are always code (continuations, method bodies, etc.)
        if line.startswith("    ") or line.startswith("\t"):
            last_code = idx
            break
        # Unindented lines must match Python patterns, otherwise they're prose
        if python_line.match(line):
            last_code = idx
            break
        # This is an unindented non-Python line — prose.  Keep scanning backwards.
        last_code = idx - 1

    cleaned = "\n".join(lines[first_code : last_code + 1])
    return cleaned.strip()


def _validate_python(code: str, filepath: str) -> bool:
    """
    Check if code is valid Python by attempting to compile it.

    Returns True if the code compiles, False otherwise.
    Logs a warning on failure.
    """
    try:
        compile(code, filepath, "exec")
        return True
    except SyntaxError as e:
        logger.warning(
            "[validate] REJECTED %s — invalid Python (line %s): %s",
            filepath, e.lineno, e.msg,
        )
        return False


def _transform_one(client: str, analysis: dict, base_dir: str) -> dict:
    """Transform all source files for a single integration. Runs inside a thread."""
    sources = read_all_sources(client)
    src_dir = get_source_dir(client)
    repo_root = get_repo_root(client)

    if not sources:
        logger.warning(
            "[transform][%s] No source files found in %s — generating from scratch.",
            client,
            src_dir,
        )

    prompt = _PROMPT_TEMPLATE.format(
        client=client,
        src_dir=str(src_dir),
        context=_CONTEXT.get(client, ""),
        all_files_block=(
            _build_files_block(sources) if sources
            else "(no source files found — write them from scratch)"
        ),
        separator=_FILE_SEPARATOR,
        summary=analysis.get("summary", ""),
        new_params_block=_format_new_params(analysis.get("new_parameters", {})),
    )

    logger.info("[transform][%s] Calling Claude CLI …", client)
    raw = call_claude(prompt)

    # Parse per-file output
    updated_files = _parse_multi_file_response(raw)

    if not updated_files:
        # Fallback: if Claude returned a single block without file markers,
        # and there was only one source file, attribute it to that file.
        if len(sources) == 1:
            only_path = list(sources.keys())[0]
            code = re.sub(r"^```(?:python)?\s*", "", raw.strip())
            code = re.sub(r"\s*```$", "", code)
            code = _sanitize_code(code)
            updated_files = {only_path: code}
        else:
            logger.error(
                "[transform][%s] Could not parse multi-file response.", client
            )
            return {
                "client": client,
                "success": False,
                "error": "Failed to parse Claude multi-file response",
            }

    # Write each modified file to disk
    total_chars = 0
    files_written = []
    for rel_path, code in updated_files.items():
        if not _validate_python(code, rel_path):
            logger.error(
                "[transform][%s] SKIPPED writing %s — Claude output is not valid Python.",
                client, rel_path,
            )
            continue
        abs_path = repo_root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(code, encoding="utf-8")
        total_chars += len(code)
        files_written.append(rel_path)
        logger.info(
            "[transform][%s] Wrote %d chars to %s",
            client,
            len(code),
            rel_path,
        )

    # ── Phase 2: Update test scripts to match the new source ──────────────
    test_sources = read_all_test_files(client)
    test_files_written = []

    if test_sources:
        # Re-read the updated source files we just wrote
        updated_source_contents = read_all_sources(client)
        test_prompt = _TEST_TRANSFORM_PROMPT.format(
            client=client,
            updated_source_block=_build_files_block(updated_source_contents),
            test_files_block=_build_files_block(test_sources),
        )

        logger.info("[transform][%s] Calling Claude CLI to update tests …", client)
        raw_tests = call_claude(test_prompt)
        updated_test_files = _parse_multi_file_response(raw_tests)

        if not updated_test_files and len(test_sources) == 1:
            only_path = list(test_sources.keys())[0]
            code = re.sub(r"^```(?:python)?\s*", "", raw_tests.strip())
            code = re.sub(r"\s*```$", "", code)
            code = _sanitize_code(code)
            updated_test_files = {only_path: code}

        for rel_path, code in updated_test_files.items():
            if not _validate_python(code, rel_path):
                logger.error(
                    "[transform][%s] SKIPPED writing test %s — not valid Python.",
                    client, rel_path,
                )
                continue
            abs_path = repo_root / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(code, encoding="utf-8")
            test_files_written.append(rel_path)
            logger.info(
                "[transform][%s] Updated test file %s (%d chars)",
                client,
                rel_path,
                len(code),
            )
    else:
        logger.info("[transform][%s] No test files found — skipping test update.", client)

    return {
        "client": client,
        "files_written": files_written,
        "test_files_written": test_files_written,
        "chars_written": total_chars,
        "success": True,
    }


def transform_all(
    analysis: dict,
    scope: list[str] | None = None,
    base_dir: str = ".",
) -> list[dict]:
    """
    Update all in-scope integration files in parallel.

    Args:
        analysis:  Dict returned by analyze_diff().
        scope:     List of client names to update. None = all three.
        base_dir:  Project root directory.

    Returns:
        List of result dicts, one per client.
        Each has keys: client, files_written, success, [error].
    """
    from ..integration_config import all_clients

    targets = scope or all_clients()
    results: list[dict] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        future_to_client = {
            pool.submit(_transform_one, client, analysis, base_dir): client
            for client in targets
        }

        for future in concurrent.futures.as_completed(future_to_client):
            client = future_to_client[future]
            try:
                result = future.result(timeout=600)
            except Exception as exc:
                logger.error("[transform][%s] Failed: %s", client, exc)
                result = {"client": client, "success": False, "error": str(exc)}
            results.append(result)

    ok = [r["client"] for r in results if r.get("success")]
    fail = [r["client"] for r in results if not r.get("success")]
    logger.info("[transform] Done. ok=%s  fail=%s", ok, fail)
    return results
