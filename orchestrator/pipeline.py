"""
pipeline.py — Integration Automation Pipeline
=============================================

Master orchestrator using true Claude Code subagent architecture.
Instead of standalone Python stages, we define framework-specific `AgentDefinition` 
subagents and hand them to a single master agent via the `run_orchestrator` wrapper.

Master Agent:
  - Guided by the main prompt to process a new SDK release
  - Analyzes the diff report
  - Delegates the complete update/testing flow to the appropriate framework subagent
  - Uses the `Task` tool for delegation

Subagents:
  - crewai_agent:      Updates CrewAI integrations & fixes their tests
  - langchain_agent:   Updates LangChain integrations & fixes their tests
  - llamaindex_agent:  Updates LlamaIndex integrations & fixes their tests
"""

from __future__ import annotations

import logging
from typing import Callable

from claude_agent_sdk import AgentDefinition

from .integration_config import all_clients, get_repo_root, get_source_dir
from .utils.claude_sdk import MODEL_OPUS, MODEL_SONNET, run_orchestrator

logger = logging.getLogger(__name__)


# ── Subagent Definitions ───────────────────────────────────────────────────

_COMMON_FRAMEWORK_PROMPT = """\
Role: Expert Python engineer maintaining this framework's endee integration.
Task: Auto-update the integration for new upstream API changes.

STRICT INSTRUCTIONS:
1. Navigate to the provided Repo Path. Review files in the Source Dir.
2. Update the source files. Add the new endee parameters to the public vector store API and forward them directly to `index.query()`.
3. Preserve full backward compatibility. Use defaults. Do not break existing API calls.
4. Strictly abide by the framework's native API patterns.
5. Update tests in tests/. Run `pytest` via Bash. Automatically fix files if tests fail until they all pass.
6. OUTPUT ONLY a concise summary of the files modified and final test pass/fail status. No conversational filler.
"""

_AGENT_CREWAI = AgentDefinition(
    description=(
        "An expert Python engineer that updates CrewAI integrations. "
        "It can read/edit files and run tests autonomously."
    ),
    prompt=(
        _COMMON_FRAMEWORK_PROMPT + 
        "\nIMPORTANT FRAMEWORK CONTEXT:\n"
        "CrewAI tools inherit from `BaseTool`. The new params must appear in the "
        "Pydantic input schema (e.g. `EndeeSearchInput`) AND be forwarded to "
        "`index.query()` inside the `_run()` method."
    ),
    tools=["Read", "Edit", "Glob", "Grep", "Bash"],
    model=MODEL_SONNET,
)

_AGENT_LANGCHAIN = AgentDefinition(
    description=(
        "An expert Python engineer that updates LangChain integrations. "
        "It can read/edit files and run tests autonomously."
    ),
    prompt=(
        _COMMON_FRAMEWORK_PROMPT + 
        "\nIMPORTANT FRAMEWORK CONTEXT:\n"
        "LangChain vector stores implement `similarity_search()` and `similarity_search_with_score()`. "
        "New params should be explicit keyword args with defaults so existing callers are unaffected."
    ),
    tools=["Read", "Edit", "Glob", "Grep", "Bash"],
    model=MODEL_SONNET,
)

_AGENT_LLAMAINDEX = AgentDefinition(
    description=(
        "An expert Python engineer that updates LlamaIndex integrations. "
        "It can read/edit files and run tests autonomously."
    ),
    prompt=(
        _COMMON_FRAMEWORK_PROMPT + 
        "\nIMPORTANT FRAMEWORK CONTEXT:\n"
        "LlamaIndex uses `BasePydanticVectorStore`. The `query()` method takes a `VectorStoreQuery` object. "
        "New params should also be readable from `query.query_kwargs` so callers can pass them "
        "without changing the base API."
    ),
    tools=["Read", "Edit", "Glob", "Grep", "Bash"],
    model=MODEL_SONNET,
)

# ── Master Orchestrator Prompt ─────────────────────────────────────────────

_MASTER_PROMPT_TEMPLATE = """\
Role: Master Orchestrator for Integration Update Pipeline.
Goal: Update downstream endee integrations ({scope_csv}) for a new upstream Python SDK release.

UPSTREAM CHANGES REPORT:
{report}

Available Task tools (subagents):
- `crewai_agent`
- `langchain_agent`
- `llamaindex_agent`

STRICT INSTRUCTIONS:
1. Extract new parameters, types, and defaults from the report concisely.
2. For EACH integration in scope ({scope_csv}), note its context:
{client_contexts}
3. Sequentially use the `Task` tool to call the corresponding subagent (e.g., `langchain_agent`). Pass it:
   - The concise API changes summary.
   - The exact Repo Path and Source Dir.
4. Wait for the subagent's test results before proceeding to the next.
5. End your response with ONLY a final, concise status report (pass/fail per integration). NO conversational filler and NO hallucinated steps.
"""


def _build_client_contexts(scope: list[str]) -> str:
    """Build a string describing the paths for each integration in scope."""
    lines = []
    for client in scope:
        repo = get_repo_root(client)
        src = get_source_dir(client)
        lines.append(
            f"  - Client: {client}\n"
            f"    Subagent: {client}_agent\n"
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
                notify(f"[Orchestrator] {msg.strip()}")
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
        "cost": 0.0,
        "usage": {},
    }

    if notify:
        # Initial greeting without the prefix
        try:
            notify(f"*Triggered framework-specific subagent pipeline* for `{', '.join(targets)}`")
        except Exception:
            pass

    prompt = _MASTER_PROMPT_TEMPLATE.format(
        scope_csv=", ".join(targets),
        report=report_content,
        client_contexts=_build_client_contexts(targets),
    )

    agents = {
        "crewai_agent": _AGENT_CREWAI,
        "langchain_agent": _AGENT_LANGCHAIN,
        "llamaindex_agent": _AGENT_LLAMAINDEX,
    }

    # Only provide the configured agents that match the targets in scope
    active_agents = {k: v for k, v in agents.items() if k.replace("_agent", "") in targets}

    try:
        final_summary, cost, usage = await run_orchestrator(
            prompt=prompt,
            agents=active_agents,
            cwd=base_dir,
            max_turns=100,  # Master agent needs many turns to coordinate all subagents
            on_message=_notify,
        )
        results["summary"] = final_summary
        results["cost"] = cost
        results["usage"] = usage
        results["success"] = True

    except Exception as exc:
        msg = f"Error: Pipeline orchestrator crashed: {exc}"
        logger.exception("[pipeline] %s", msg)
        if notify:
            try:
                notify(msg)
            except Exception:
                pass
        results["errors"].append(msg)

    return results
