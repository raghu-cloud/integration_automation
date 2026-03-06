"""
pipeline.py — Integration Automation Pipeline
=============================================

Pure-Python orchestrator. No external framework — just four sequential stages
with a concise progress callback so the caller can stream status to Slack.

            ┌──────────────────────────────────────────────────────┐
            │                  PIPELINE FLOW                       │
            │                                                      │
            │  Stage 1: Analyze Diff                               │
            │    claude -p "parse the comparison report…"          │
            │                                                      │
            │  Stage 2: Transform  (×3 parallel threads)           │
            │    claude -p "update crewai code…"                   │
            │    claude -p "update langchain code…"                │
            │    claude -p "update llamaindex code…"               │
            │                                                      │
            │  Stage 3: Test + Self-Heal  (per integration)        │
            │    pytest → if fail → claude -p "fix it" → retry     │
            │                                                      │
            │  Stage 4: Create PRs  (for every passing client)     │
            │    claude -p "write PR title + body…"                │
            │    gh pr create …                                    │
            └──────────────────────────────────────────────────────┘

Usage
-----
    from orchestrator.pipeline import run_pipeline

    results = run_pipeline(
        report_content = open("comparison_report.txt").read(),
        branch         = "auto/endee-0.1.13",
        scope          = ["crewai", "langchain", "llamaindex"],  # or None for all
        notify         = lambda msg: slack_client.chat_postMessage(channel=ch, text=msg),
    )
"""

from __future__ import annotations

import logging
from typing import Callable

from .stages.analyze import analyze_diff
from .stages.pull_request import create_prs
from .stages.test_heal import run_and_heal_all
from .stages.transform import transform_all

logger = logging.getLogger(__name__)

_ALL_CLIENTS = ["llamaindex"]


def _build_test_report(test_results: dict) -> str:
    """Build a formatted Slack test-results report."""
    lines = ["📊 *Test Results Report*", ""]

    # Header
    lines.append(f"{'Framework':<14} {'Passed':>7} {'Failed':>7} {'Errors':>7} {'Rounds':>7}  Status")
    lines.append("─" * 68)

    total_passed = total_failed = total_errors = 0

    for client, tr in test_results.items():
        p = tr.get("passed_count", 0)
        f = tr.get("failed_count", 0)
        e = tr.get("error_count", 0)
        rounds = tr.get("rounds_used", 0)
        icon = "✅" if tr["passed"] else "❌"
        total_passed += p
        total_failed += f
        total_errors += e
        lines.append(f"{icon} {client:<12} {p:>7} {f:>7} {e:>7} {rounds:>7}")

    lines.append("─" * 68)
    lines.append(f"{'Total':<14} {total_passed:>7} {total_failed:>7} {total_errors:>7}")
    lines.append("")

    # Add failure details (truncated) for any failing frameworks
    failing = {c: tr for c, tr in test_results.items() if not tr["passed"]}
    if failing:
        lines.append("*Failure Details:*")
        for client, tr in failing.items():
            output = tr.get("output", "")
            # Extract just the FAILURES section if available
            failure_section = ""
            if "FAILURES" in output:
                start = output.index("FAILURES")
                failure_section = output[start:start + 800]
            elif "ERRORS" in output:
                start = output.index("ERRORS")
                failure_section = output[start:start + 800]
            else:
                # Last 400 chars as fallback
                failure_section = output[-400:]

            lines.append(f"\n`{client}` — {tr.get('summary', 'failed')}")
            lines.append(f"```{failure_section.strip()}```")

    return "\n".join(lines)


def _parse_scope(scope: str | list[str] | None) -> list[str]:
    """Normalise the scope argument into a list of client names."""
    if scope is None or scope == "all":
        return _ALL_CLIENTS
    if isinstance(scope, list):
        return [s.strip().lower() for s in scope if s.strip()]
    # Comma-separated string: "crewai,langchain"
    return [s.strip().lower() for s in str(scope).split(",") if s.strip()]


def run_pipeline(
    report_content: str,
    branch: str = "auto/endee-update",
    scope: str | list[str] | None = None,
    base_dir: str = ".",
    notify: Callable[[str], None] | None = None,
) -> dict:
    """
    Run the full four-stage integration automation pipeline.

    Args:
        report_content: Raw text from comparison_report.txt.
        branch:         Git branch name for commits and PRs.
        scope:          Which integrations to touch. Accepts:
                          - None / "all"              → all three clients
                          - "crewai"                  → single client
                          - "crewai,langchain"        → comma-separated list
                          - ["crewai", "llamaindex"]  → Python list
        base_dir:       Project root (defaults to current directory).
        notify:         Callable invoked with a status string after each stage.
                        Typically posts a message to Slack.

    Returns:
        A dict with keys: analysis, transform, tests, prs, success, errors.
    """

    def _notify(msg: str) -> None:
        logger.info(msg)
        if notify:
            try:
                notify(msg)
            except Exception as exc:
                logger.warning("[pipeline] notify() raised: %s", exc)

    targets = _parse_scope(scope)
    results: dict = {
        "branch": branch,
        "scope": targets,
        "analysis": None,
        "transform": [],
        "tests": {},
        "prs": {},
        "success": False,
        "errors": [],
    }

    # ── Stage 1: Analyze ─────────────────────────────────────────────────────
    _notify(f"🔍 *Stage 1/4 — Analysing diff* (branch: `{branch}`, scope: `{', '.join(targets)}`)")

    try:
        analysis = analyze_diff(report_content)
        results["analysis"] = analysis
        n_changes = len(analysis.get("changes", []))
        n_params = len(analysis.get("new_parameters", {}))
        _notify(
            f"✅ Analysis complete — {n_changes} change(s), {n_params} new parameter(s) detected."
        )
    except Exception as exc:
        msg = f"❌ Stage 1 (analyze) failed: {exc}"
        _notify(msg)
        results["errors"].append(msg)
        return results

    # ── Stage 2: Transform ───────────────────────────────────────────────────
    _notify(f"⚡ *Stage 2/4 — Updating integration code* ({len(targets)} client(s) in parallel) …")

    try:
        transform_results = transform_all(analysis, scope=targets, base_dir=base_dir)
        results["transform"] = transform_results
        ok = [r["client"] for r in transform_results if r.get("success")]
        fail = [r["client"] for r in transform_results if not r.get("success")]
        _notify(
            f"✅ Transform complete — {len(ok)}/{len(targets)} succeeded."
            + (f"  ⚠️ Failed: {', '.join(fail)}" if fail else "")
        )
        if fail:
            for r in transform_results:
                if not r.get("success"):
                    results["errors"].append(f"transform[{r['client']}]: {r.get('error')}")
    except Exception as exc:
        msg = f"❌ Stage 2 (transform) failed: {exc}"
        _notify(msg)
        results["errors"].append(msg)
        return results

    # ── Stage 3: Test + Self-Heal ────────────────────────────────────────────
    _notify("🧪 *Stage 3/4 — Running tests (auto-healing on failure)* …")

    try:
        test_results = run_and_heal_all(scope=targets, base_dir=base_dir, notify=_notify)
        results["tests"] = test_results
    except Exception as exc:
        msg = f"❌ Stage 3 (test) failed: {exc}"
        _notify(msg)
        results["errors"].append(msg)
        return results

    # ── Test Results Report ───────────────────────────────────────────────
    _notify(_build_test_report(test_results))

    passing = [c for c, tr in test_results.items() if tr["passed"]]
    failing = [c for c, tr in test_results.items() if not tr["passed"]]

    if failing:
        _notify(
            f"⚠️ Tests failed for: {', '.join(failing)}. "
            f"Creating PRs for passing clients only."
        )
        results["errors"].append(
            f"Tests still failing for: {', '.join(failing)}"
        )

    if not passing:
        _notify("❌ *Stage 4/4 — Skipped PR creation* (all tests failed)")
        return results

    _notify("🚀 *Stage 4/4 — Creating GitHub Pull Requests* …")

    try:
        pr_results = create_prs(
            branch=branch,
            analysis=analysis,
            test_results=test_results,
            scope=targets,
            base_dir=base_dir,
        )
        results["prs"] = pr_results

        for client, pr in pr_results.items():
            if pr.get("url"):
                _notify(f"  ✅ `{client}` PR → {pr['url']}")
            elif pr.get("skipped"):
                _notify(f"  ⏭️  `{client}` skipped — {pr.get('reason', '')}")
            else:
                _notify(f"  ❌ `{client}` PR failed — {pr.get('error', 'unknown error')}")
    except Exception as exc:
        msg = f"❌ Stage 4 (PRs) failed: {exc}"
        _notify(msg)
        results["errors"].append(msg)
        return results

    results["success"] = len(results["errors"]) == 0
    _notify(
        "🎉 *Pipeline complete!* "
        + ("All stages succeeded." if results["success"] else f"{len(results['errors'])} error(s) encountered.")
    )
    return results
