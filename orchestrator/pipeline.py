"""
pipeline.py — Integration Automation Pipeline
=============================================

Master orchestrator using true Claude Code subagent architecture.
Instead of standalone Python stages, we define `AgentDefinition` subagents
and hand them to a single master agent via the `run_orchestrator` wrapper.

Master Agent:
  - Guided by the main prompt
  - Delegates to named subagents (analyzer, transformer, healer)
  - Uses the `Task` tool for delegation

Subagents:
  - analyzer:    Parses diffs into structured summaries
  - transformer: Rewrites integration source code using Read/Edit tools
  - healer:      Runs pytest and uses Read/Edit/Bash to fix failing tests
"""

from __future__ import annotations

import logging
from typing import Callable

from claude_agent_sdk import AgentDefinition

from .integration_config import all_clients, get_repo_root, get_source_dir
from .utils.claude_sdk import MODEL_OPUS, MODEL_SONNET, run_orchestrator

logger = logging.getLogger(__name__)


# ── Subagent Definitions ───────────────────────────────────────────────────

_AGENT_ANALYZER = AgentDefinition(
    description=(
        "Analyzes upstream SDK diffs (comparison reports) and identifies "
        "meaningful API changes, extracting new parameters, types, and defaults."
    ),
    prompt=(
        "You are a senior Python SDK analyst. Your job is to extract every "
        "meaningful change from the provided comparison report and return a "
        "concise summary of WHAT changed, WHICH parameters were added, "
        "and their default values/types. Do not write code."
    ),
    model=MODEL_SONNET,
)

_AGENT_TRANSFORMER = AgentDefinition(
    description=(
        "An expert Python engineer that updates integration source code "
        "folders to support new upstream parameters and API changes."
    ),
    prompt=(
        "You are an expert Python engineer updating downstream SDK integrations. "
        "Given an analysis of upstream changes: "
        "1. CD into the provided integration repository. "
        "2. Add new parameters to index.query() calls and public methods. "
        "3. Preserve existing defaults to remain backward compatible. "
        "4. Fix tests if there are test files. "
        "Use Read/Edit tools to explore and modify the codebase directly."
    ),
    tools=["Read", "Edit", "Glob", "Grep", "Bash"],
    model=MODEL_OPUS,
)

_AGENT_HEALER = AgentDefinition(
    description=(
        "An expert Python test engineer that runs pytest and fixes failing "
        "tests iteratively using Bash, Read, and Edit tools."
    ),
    prompt=(
        "You are an expert Python test engineer. "
        "Your task is to CD into an integration repository, run `pytest`, "
        "read the failure output, and fix the source or test code until ALL "
        "tests pass. Use Bash to run the tests. Use Read/Edit to fix the files. "
        "Keep trying until tests pass or you conclude it is impossible."
    ),
    tools=["Read", "Edit", "Bash", "Glob", "Grep"],
    model=MODEL_SONNET,
)

# ── Master Orchestrator Prompt ─────────────────────────────────────────────

_MASTER_PROMPT_TEMPLATE = """\
You are the Master Orchestrator for the Integration Update Pipeline.
Your goal is to update the downstream endee integrations ({scope_csv})
based on a new upstream Python SDK release.

─── COMPARISON REPORT (UPSTREAM CHANGES) ─────────────────────────────────
{report}
─────────────────────────────────────────────────────────────────────────

You have three specialized subagents available via the Task tool:
1. `analyzer`:    Extracts a concise summary and list of new parameters from the diff.
2. `transformer`: Edits the integration repos to support the new features.
3. `healer`:      Runs tests and fixes any breakage.

INSTRUCTIONS:
Step 1: Delegate the 'comparison report' to the `analyzer` subagent to get a
        clear summary of the new parameters and changes.
        
Step 2: For EACH integration in scope ({scope_csv}), note its Repo Path and Source Dir:
{client_contexts}

Step 3: For each integration, delegate a task to the `transformer` subagent:
        - Provide the analyzer's summary.
        - Give it the exact Repo Path so it knows where to cd.
        - Tell it to update the python files in the Source Dir.
        (Do them one by one).

Step 4: Once all integrations are transformed, delegate a task to the `healer`
        subagent for EACH integration to run `pytest` and fix any failures.
        Make sure the healer runs from the Repo Path.

Step 5: Summarize the final pass/fail test status for every integration.
"""


def _build_client_contexts(scope: list[str]) -> str:
    """Build a string describing the paths for each integration in scope."""
    lines = []
    for client in scope:
        repo = get_repo_root(client)
        src = get_source_dir(client)
        lines.append(
            f"  - Client: {client}\n"
            f"    Repo Path: {repo}\n"
            f"    Source Dir: {src.name}"
        )
    return "\n".join(lines)


async def run_pipeline(
    report_content: str,
    branch: str = "auto/endee-update",
    scope: str | list[str] | None = None,
    base_dir: str = ".",
    notify: Callable[[str], None] | None = None,
) -> dict:
    """
    Run the integration automation pipeline via Claude Code subagents.

    Args:
        report_content: Raw text from comparison_report.txt.
        branch:         Branch name (kept for identification/logging).
        scope:          Which integrations to touch ("all" or list).
        base_dir:       Project root (defaults to current directory).
        notify:         Callable invoked with a status string as the master
                        agent streams its thoughts.

    Returns:
        A dict with the final orchestrator summary.
    """

    def _notify(msg: str) -> None:
        logger.info(msg)
        if notify:
            try:
                # Add a prefix to distinguish master agent streams
                notify(f"🤖 [Orchestrator] {msg.strip()}")
            except Exception as exc:
                logger.warning("[pipeline] notify() raised: %s", exc)

    from .integration_config import all_clients

    # Normalize scope
    if scope is None or scope == "all":
        targets = all_clients()
    elif isinstance(scope, list):
        targets = [s.strip().lower() for s in scope if s.strip()]
    else:
        targets = [s.strip().lower() for s in str(scope).split(",") if s.strip()]

    results: dict = {
        "branch": branch,
        "scope": targets,
        "success": False,
        "errors": [],
        "summary": "",
    }

    if notify:
        # Initial greeting without the prefix
        try:
            notify(f"🚀 *Triggered subagent pipeline* for `{', '.join(targets)}`")
        except Exception:
            pass

    prompt = _MASTER_PROMPT_TEMPLATE.format(
        scope_csv=", ".join(targets),
        report=report_content,
        client_contexts=_build_client_contexts(targets),
    )

    agents = {
        "analyzer": _AGENT_ANALYZER,
        "transformer": _AGENT_TRANSFORMER,
        "healer": _AGENT_HEALER,
    }

    try:
        final_summary = await run_orchestrator(
            prompt=prompt,
            agents=agents,
            cwd=base_dir,
            max_turns=100,  # Master agent needs many turns to coordinate all subagents
            on_message=_notify,
        )
        results["summary"] = final_summary
        results["success"] = True

    except Exception as exc:
        msg = f"❌ Pipeline orchestrator crashed: {exc}"
        logger.exception("[pipeline] %s", msg)
        if notify:
            try:
                notify(msg)
            except Exception:
                pass
        results["errors"].append(msg)

    return results
