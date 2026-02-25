"""
Stage 4 – Create Pull Requests
================================
For each integration that passed its tests:
  1. Commit the generated code to the specified branch.
  2. Push to origin.
  3. Ask Claude CLI to draft the PR title + body (rich markdown).
  4. Use the `gh` CLI directly to open the PR and capture its URL.
  5. Return the URL so the Slack notifier can include it in the summary.

The actual git + gh commands are run as Python subprocesses (not via Claude)
for reliability. Claude is used only for the creative/descriptive part.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from pathlib import Path

from ..utils.claude_cli import call_claude
from ..utils.git_utils import commit_and_push

logger = logging.getLogger(__name__)

_INTEGRATION_FILES: dict[str, str] = {
    "crewai": os.getenv("CREWAI_REPO_PATH", "integrations/crewai_endee") + "/tools.py",
    "langchain": os.getenv("LANGCHAIN_REPO_PATH", "integrations/langchain_endee") + "/vectorstore.py",
    "llamaindex": os.getenv("LLAMAINDEX_REPO_PATH", "integrations/llamaindex_endee") + "/vector_store.py",
}

_REPO_ENV: dict[str, str] = {
    "crewai": "CREWAI_REPO",
    "langchain": "LANGCHAIN_REPO",
    "llamaindex": "LLAMAINDEX_REPO",
}

_PR_PROMPT = """\
Write a GitHub Pull Request for the {client} endee integration update.

Changes being propagated from the endee Python client:
{summary}

New parameters added to index.query():
{new_params}

Test status: {test_status}

Return a JSON object with exactly two keys:
{{
  "title": "<concise PR title, ≤ 72 chars>",
  "body":  "<markdown body with ## Summary, ## Changes, ## Test Results sections>"
}}
No markdown fences, no extra text — only the raw JSON.
"""


def _draft_pr_content(client: str, analysis: dict, test_result: dict) -> dict:
    """Ask Claude to generate the PR title and body."""
    params = "\n".join(
        f"  - {name}: {meta.get('type')} (default={meta.get('default')}) — {meta.get('description')}"
        for name, meta in analysis.get("new_parameters", {}).items()
    )
    test_status = (
        f"✅ All tests passed ({test_result.get('summary', '')})"
        if test_result.get("passed")
        else f"❌ Tests failed — {test_result.get('summary', '')}"
    )

    prompt = _PR_PROMPT.format(
        client=client,
        summary=analysis.get("summary", ""),
        new_params=params or "(see summary)",
        test_status=test_status,
    )

    import json

    raw = call_claude(prompt)
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)
    except Exception:
        # Fallback if Claude's response isn't valid JSON
        return {
            "title": f"chore: propagate endee {analysis.get('version_to', 'latest')} API changes to {client}",
            "body": f"## Summary\n\nPropagated endee client updates to the {client} integration.\n\n{analysis.get('summary', '')}",
        }


def _open_pr_with_gh(
    repo_path: str,
    branch: str,
    title: str,
    body: str,
    base_branch: str = "main",
) -> str:
    """Run `gh pr create` and return the PR URL."""
    result = subprocess.run(
        [
            "gh", "pr", "create",
            "--title", title,
            "--body", body,
            "--head", branch,
            "--base", base_branch,
        ],
        capture_output=True,
        text=True,
        cwd=repo_path,
    )

    if result.returncode != 0:
        raise RuntimeError(f"gh pr create failed: {result.stderr}")

    # `gh pr create` prints the PR URL on stdout
    url = result.stdout.strip()
    return url


def create_prs(
    branch: str,
    analysis: dict,
    test_results: dict[str, dict],
    scope: list[str] | None = None,
    base_dir: str = ".",
) -> dict[str, dict]:
    """
    Commit, push, and open a PR for every passing integration.

    Args:
        branch:       Branch name to push to and open the PR from.
        analysis:     Dict returned by analyze_diff().
        test_results: Dict returned by run_and_heal_all().
        scope:        Clients to process. None = all.
        base_dir:     Project root directory.

    Returns:
        Dict keyed by client name:
        {
          "crewai":    {"url": "https://github.com/…", "skipped": False},
          "langchain": {"url": None, "skipped": True, "reason": "Tests failed"},
          …
        }
    """
    targets = scope or list(_INTEGRATION_FILES.keys())
    results: dict[str, dict] = {}

    version_to = analysis.get("version_to", "latest")

    for client in targets:
        t = test_results.get(client, {})

        if not t.get("passed", False):
            logger.warning("[pr][%s] Skipping PR — tests did not pass.", client)
            results[client] = {"skipped": True, "reason": "Tests failed", "url": None}
            continue

        repo_path = str(Path(base_dir) / Path(_INTEGRATION_FILES[client]).parent)
        integration_file = _INTEGRATION_FILES[client]

        # 1. Commit & push
        commit_msg = (
            f"chore: propagate endee {version_to} API changes\n\n"
            f"Added prefilter_cardinality_threshold and filter_boost_percentage "
            f"parameters.\nAuto-committed by Integration Automation Orchestrator."
        )
        pushed = commit_and_push(
            local_path=base_dir,
            branch_name=branch,
            commit_message=commit_msg,
            files=[integration_file],
        )
        if not pushed:
            results[client] = {"skipped": False, "url": None, "error": "git push failed"}
            continue

        # 2. Draft PR content with Claude
        logger.info("[pr][%s] Drafting PR content …", client)
        pr_content = _draft_pr_content(client, analysis, t)

        # 3. Open PR with gh CLI
        try:
            url = _open_pr_with_gh(
                repo_path=base_dir,
                branch=branch,
                title=pr_content["title"],
                body=pr_content["body"],
            )
            logger.info("[pr][%s] PR created: %s", client, url)
            results[client] = {"skipped": False, "url": url}
        except RuntimeError as exc:
            logger.error("[pr][%s] %s", client, exc)
            results[client] = {"skipped": False, "url": None, "error": str(exc)}

    return results
