"""
claude_cli.py
=============
Thin wrapper around the Claude Code CLI (`claude -p`).

The CLI is invoked as a subprocess in print mode so that each orchestrator
stage is a self-contained call: you supply a prompt, Claude returns the
response, and the Python process continues.

Prerequisites
-------------
Install the Claude Code CLI once:
    npm install -g @anthropic-ai/claude-code
    claude login          # authenticate

Usage inside the pipeline
--------------------------
    from orchestrator.utils.claude_cli import call_claude

    result = call_claude("Explain what changed in this diff: ...")
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Model constants ─────────────────────────────────────────────────────────
# Use these in call_claude(model=...) to select per-stage models.
MODEL_OPUS = "opus"
MODEL_SONNET = "claude-sonnet-4-6"
MODEL_HAIKU = "claude-haiku-3-5-20241022"

# Maximum characters to pass as a CLI argument directly.
# Prompts longer than this threshold are written to a temp file and read via
# stdin to avoid OS argument-length limits (macOS: ~2 MB, Linux: ~128 KB).
_ARG_LIMIT = 60_000


def call_claude(
    prompt: str,
    *,
    model: str | None = None,
    cwd: str | None = None,
    timeout: int = 300,
    extra_flags: list[str] | None = None,
) -> str:
    """
    Invoke the Claude Code CLI in print mode and return the model's response.

    Args:
        prompt:       The full prompt to send to Claude.
        model:        Model name or alias to pass via --model (e.g. "opus",
                      "claude-sonnet-4-6"). None = CLI default.
        cwd:          Working directory for the subprocess (default: current dir).
        timeout:      Seconds before the subprocess is killed (default: 300).
        extra_flags:  Additional CLI flags, e.g. ["--allowedTools", "Bash"].

    Returns:
        The model's text response (stdout), stripped of leading/trailing whitespace.

    Raises:
        RuntimeError: If the `claude` binary is not found or exits with a
                      non-zero status code.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        raise RuntimeError(
            "`claude` CLI not found on PATH.\n"
            "Install it with:  npm install -g @anthropic-ai/claude-code\n"
            "Then authenticate: claude login"
        )

    base_flags = extra_flags or []

    # Inject --model flag when a specific model is requested
    if model:
        base_flags = ["--model", model] + base_flags

    tmp_path = None

    if len(prompt) <= _ARG_LIMIT:
        # Short prompt → pass directly as a positional argument
        cmd = [claude_bin, "-p", prompt] + base_flags
        stdin_data = None
    else:
        # Long prompt → write to a temp file and feed via stdin so we stay
        # well under OS argument limits
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        )
        tmp.write(prompt)
        tmp.flush()
        tmp_path = tmp.name
        tmp.close()

        cmd = [claude_bin, "--print"] + base_flags
        stdin_data = Path(tmp_path).read_text(encoding="utf-8")

    logger.info(
        "[claude_cli] Calling claude -p (model=%s, prompt_len=%d, cwd=%s)",
        model or "default", len(prompt), cwd or ".",
    )

    try:
        result = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Claude CLI timed out after {timeout}s.")
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass

    if result.returncode != 0:
        logger.error("[claude_cli] stderr:\n%s", result.stderr)
        raise RuntimeError(
            f"Claude CLI exited with code {result.returncode}:\n{result.stderr}"
        )

    response = result.stdout.strip()
    logger.info("[claude_cli] Response received (%d chars).", len(response))
    return response
