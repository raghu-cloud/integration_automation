"""
Stage 2 – Transform Integrations  (parallel × 3)
==================================================
For each integration in scope, reads the current source file from disk, sends
(current code + analysis) to Claude CLI, receives the updated code, and writes
it back to disk.

All three Claude CLI calls are dispatched concurrently via ThreadPoolExecutor
so the total wall-clock time is roughly the slowest single call, not the sum.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
from pathlib import Path

from ..utils.claude_cli import call_claude

logger = logging.getLogger(__name__)

# Map client name → integration file path (relative to project root)
_INTEGRATION_FILES: dict[str, str] = {
    "crewai": os.getenv("CREWAI_REPO_PATH", "integrations/crewai_endee") + "/tools.py",
    "langchain": os.getenv("LANGCHAIN_REPO_PATH", "integrations/langchain_endee") + "/vectorstore.py",
    "llamaindex": os.getenv("LLAMAINDEX_REPO_PATH", "integrations/llamaindex_endee") + "/vector_store.py",
}

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

_PROMPT_TEMPLATE = """\
You are an expert Python engineer updating a downstream SDK integration.

INTEGRATION: {client}
FILE: {file_path}
CONTEXT: {context}

─── CURRENT FILE CONTENT ────────────────────────────────────────────────────
{current_code}
─────────────────────────────────────────────────────────────────────────────

─── ENDEE CLIENT CHANGES (from the comparison report) ───────────────────────
{summary}

New parameters added to endee.Index.query():

{new_params_block}
─────────────────────────────────────────────────────────────────────────────

REQUIREMENTS
1. Add both new parameters to every public query/search method in this file.
2. Forward them to index.query() with identical names and the same defaults.
3. Existing callers must not break — use default values that match the endee defaults.
4. Update docstrings to document the new parameters.
5. Keep the same code style, imports, and class structure.
6. Return ONLY the complete, updated Python source — no markdown fences, no prose.
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


def _transform_one(client: str, analysis: dict, base_dir: str) -> dict:
    """Transform a single integration file. Runs inside a thread."""
    file_path = _INTEGRATION_FILES[client]
    abs_path = Path(base_dir) / file_path

    current_code = ""
    if abs_path.exists():
        current_code = abs_path.read_text(encoding="utf-8")
    else:
        logger.warning("[transform][%s] File not found — generating from scratch.", client)

    prompt = _PROMPT_TEMPLATE.format(
        client=client,
        file_path=file_path,
        context=_CONTEXT[client],
        current_code=current_code or "(file does not exist yet — write it from scratch)",
        summary=analysis.get("summary", ""),
        new_params_block=_format_new_params(analysis.get("new_parameters", {})),
    )

    logger.info("[transform][%s] Calling Claude CLI …", client)
    raw = call_claude(prompt)

    # Strip accidental markdown fences
    new_code = re.sub(r"^```(?:python)?\s*", "", raw.strip())
    new_code = re.sub(r"\s*```$", "", new_code)

    # Write to disk
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(new_code, encoding="utf-8")
    logger.info("[transform][%s] Wrote %d chars to %s.", client, len(new_code), file_path)

    return {
        "client": client,
        "file_path": file_path,
        "chars_written": len(new_code),
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
        Each has keys: client, file_path, success, [error].
    """
    targets = scope or list(_INTEGRATION_FILES.keys())
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
