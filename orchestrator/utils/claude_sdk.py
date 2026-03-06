"""
claude_sdk.py
=============
Thin wrapper around the Claude Agent SDK for running the pipeline
orchestrator as a single master agent with subagent delegation.

Authentication is handled by the Claude Code CLI's own ``claude login``
— no separate API key is needed.

Usage
-----
    from orchestrator.utils.claude_sdk import run_orchestrator

    result = await run_orchestrator(
        prompt="Run the full pipeline...",
        agents={...},
        cwd="/path/to/project",
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

from claude_agent_sdk import (
    AgentDefinition,
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TaskNotificationMessage,
    TaskStartedMessage,
    query,
)

logger = logging.getLogger(__name__)

# ── Model constants ─────────────────────────────────────────────────────────
MODEL_OPUS = "opus"
MODEL_SONNET = "sonnet"
MODEL_HAIKU = "haiku"


async def run_orchestrator(
    prompt: str,
    *,
    agents: dict[str, AgentDefinition],
    cwd: str | None = None,
    max_turns: int | None = None,
    on_message: callable | None = None,
) -> tuple[str, float, dict]:
    """
    Run a master orchestrator agent that delegates to subagents via Task tool.

    Args:
        prompt:     The master orchestrator prompt with all dynamic context.
        agents:     Dict of named AgentDefinition subagents.
        cwd:        Working directory for the session.
        max_turns:  Max agentic loop iterations (default 60).
        on_message: Optional callback for streaming assistant text updates.

    Returns:
        A tuple of (result_text, total_cost_usd, usage_dict).
    """
    options = ClaudeAgentOptions(
        agents=agents,
        allowed_tools=["Task", "Read", "Bash", "Glob"],
        permission_mode="bypassPermissions",
        max_turns=max_turns or 60,
        cwd=cwd,
        model=MODEL_HAIKU,
    )

    logger.info(
        "[claude_sdk] Starting orchestrator (agents=%s, cwd=%s, prompt_len=%d)",
        list(agents.keys()),
        cwd or ".",
        len(prompt),
    )

    result_text = ""
    total_cost_usd = 0.0
    usage_dict = {"master_usage": {}, "subagent_usage": {}}
    
    # Map task_id -> agent_name (from task_type)
    task_agent_map = {}

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, TaskStartedMessage):
            if message.task_type:
                task_agent_map[message.task_id] = message.task_type

        if isinstance(message, TaskNotificationMessage):
            agent_name = task_agent_map.get(message.task_id, "unknown_agent")
            if agent_name not in usage_dict["subagent_usage"]:
                usage_dict["subagent_usage"][agent_name] = {"total_tokens": 0}
            if message.usage:
                usage_dict["subagent_usage"][agent_name]["total_tokens"] += message.usage.get("total_tokens", 0)

        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "text") and block.text:
                    logger.debug("[orchestrator] %s", block.text[:200])
                    if on_message:
                        try:
                            on_message(block.text)
                        except Exception:
                            pass
                elif hasattr(block, "name"):
                    logger.info("[orchestrator] Tool: %s", block.name)

        if isinstance(message, ResultMessage):
            if message.result:
                result_text = message.result
            if message.total_cost_usd is not None:
                total_cost_usd = message.total_cost_usd
            if message.usage:
                usage_dict["master_usage"] = message.usage
            if message.is_error:
                logger.error("[orchestrator] Finished with error.")
            logger.info(
                "[orchestrator] Done. turns=%d, cost=$%s",
                message.num_turns,
                message.total_cost_usd,
            )

    return result_text, total_cost_usd, usage_dict
