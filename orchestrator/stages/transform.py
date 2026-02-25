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

from ..integration_config import get_repo_root, get_source_dir, read_all_sources
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
            files[rel_path] = code
        i += 2

    return files


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

    return {
        "client": client,
        "files_written": files_written,
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
