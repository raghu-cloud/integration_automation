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
You are an expert Python engineer responsible for maintaining this framework's endee integration.
Given an assigned task summarizing upstream changes (new parameters, types, defaults, etc.):

1. Navigate to the provided repository path.
2. Read the source files in the specific source dir to understand the current API.
3. Update the source files to support the new endee parameters:
   - Make sure they are correctly added to the public API and forwarded to index.query().
   - Maintain full backward compatibility (use provided defaults).
   - Abide by the framework's specific API patterns.
4. Update the test files in tests/ and run `pytest` via Bash.
5. If tests fail, fix the files and rerun `pytest` until they all pass.
6. Return a concise summary of the files changed and the final test results.
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
    model=MODEL_OPUS,
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
    model=MODEL_OPUS,
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
    model=MODEL_OPUS,
)

# ── Master Orchestrator Prompt ─────────────────────────────────────────────

_MASTER_PROMPT_TEMPLATE = """\
You are the Master Orchestrator for the Integration Update Pipeline.
Your goal is to update the downstream endee integrations ({scope_csv})
based on a new upstream Python SDK release.

─── COMPARISON REPORT (UPSTREAM CHANGES) ─────────────────────────────────
{report}
─────────────────────────────────────────────────────────────────────────

You have dedicated subagents available via the Task tool for each framework:
- `crewai_agent`
- `langchain_agent`
- `llamaindex_agent`

INSTRUCTIONS:
Step 1: Read the comparison report above and deduce what the new parameters, types, 
        and defaults are. Prepare a concise summary of these changes.

Step 2: For EACH integration in scope ({scope_csv}), note its Repo Path and Source Dir:
{client_contexts}

Step 3: For each integration, use the `Task` tool to call the corresponding 
        subagent (e.g., call `langchain_agent` for the `langchain` integration):
        - Pass it the summary of upstream changes.
        - Give it the exact Repo Path and Source Dir so it knows where to work.
        - Let it autonomously modify the code and run the tests.
        (Do them one by one).

Step 4: Wait for the subagent to report the final pass/fail test status.

Step 5: Provide a final summary indicating which frameworks succeeded and their test results.
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
            notify(f"🚀 *Triggered framework-specific subagent pipeline* for `{', '.join(targets)}`")
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
        final_summary = await run_orchestrator(
            prompt=prompt,
            agents=active_agents,
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
