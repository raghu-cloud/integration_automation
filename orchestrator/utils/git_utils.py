"""
git_utils.py
============
Thin subprocess wrappers for the git operations the pipeline needs:
  - clone_or_pull  : ensure a local repo is present and up to date
  - create_branch  : checkout a new branch
  - commit_and_push: stage, commit, and push changed files
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _run(args: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    """Run a shell command and return (returncode, stdout, stderr)."""
    result = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def clone_or_pull(repo_url: str, local_path: str, branch: str | None = None) -> bool:
    """
    Clone the repo if it doesn't exist locally; otherwise pull the latest changes.

    Args:
        repo_url:   HTTPS or SSH URL of the remote repo.
        local_path: Where to clone / where the local copy lives.
        branch:     If cloning, checkout this branch. Ignored for pulls.

    Returns:
        True on success, False on failure.
    """
    path = Path(local_path)

    if path.exists() and (path / ".git").exists():
        logger.info("[git] Pulling latest in %s …", local_path)
        code, out, err = _run(["git", "pull", "--ff-only"], cwd=local_path)
    else:
        logger.info("[git] Cloning %s → %s …", repo_url, local_path)
        cmd = ["git", "clone", repo_url, local_path]
        if branch:
            cmd += ["-b", branch]
        code, out, err = _run(cmd)

    if code != 0:
        logger.error("[git] Error: %s", err)
        return False

    logger.info("[git] %s", out or "OK")
    return True


def create_branch(local_path: str, branch_name: str) -> bool:
    """
    Create and checkout a new local branch.

    If the branch already exists, checks it out instead.
    Returns True on success, False if git fails.
    """
    logger.info("[git] Creating branch '%s' in %s …", branch_name, local_path)
    code, out, err = _run(
        ["git", "checkout", "-b", branch_name], cwd=local_path
    )
    if code != 0:
        if "already exists" in err:
            logger.info("[git] Branch '%s' already exists — checking out.", branch_name)
            code, out, err = _run(
                ["git", "checkout", branch_name], cwd=local_path
            )
            if code != 0:
                logger.error("[git] Checkout failed: %s", err)
                return False
            return True
        logger.error("[git] Branch creation failed: %s", err)
        return False
    return True


def commit_and_push(
    local_path: str,
    branch_name: str,
    commit_message: str,
    files: list[str] | None = None,
) -> bool:
    """
    Stage files (or all changes), commit, and push to origin.

    Args:
        local_path:     Path to the local git repo.
        branch_name:    Remote branch to push to.
        commit_message: Commit message.
        files:          Specific files to stage. If None, stages all changes.

    Returns:
        True on success, False on any failure.
    """
    # Stage
    if files:
        for f in files:
            code, _, err = _run(["git", "add", f], cwd=local_path)
            if code != 0:
                logger.error("[git] git add %s failed: %s", f, err)
                return False
    else:
        code, _, err = _run(["git", "add", "-A"], cwd=local_path)
        if code != 0:
            logger.error("[git] git add -A failed: %s", err)
            return False

    # Commit
    code, out, err = _run(
        ["git", "commit", "-m", commit_message], cwd=local_path
    )
    if code != 0:
        if "nothing to commit" in out or "nothing to commit" in err:
            logger.info("[git] Nothing to commit in %s — skipping push.", local_path)
            return True
        logger.error("[git] git commit failed: %s", err)
        return False

    # Push
    code, _, err = _run(
        ["git", "push", "origin", branch_name], cwd=local_path
    )
    if code != 0:
        logger.error("[git] git push failed: %s", err)
        return False

    logger.info("[git] Committed and pushed branch '%s'.", branch_name)
    return True
