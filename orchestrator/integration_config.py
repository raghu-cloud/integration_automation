"""
integration_config.py
=====================
Centralised configuration for all downstream integration repos.

Instead of hardcoding a single file per integration, this module discovers
every .py file inside the actual source directory of each cloned repo —
so the pipeline works with the *real* codebase layout, not assumed filenames.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Per-client configuration ────────────────────────────────────────────────
#
# repo_path_env : env-var that holds the absolute path to the cloned repo
# fallback      : relative path used when the env-var is not set
#                 (points to the local stubs inside this project)
# src_subdir    : the package directory *inside* the repo that contains
#                 the actual integration source files
# ────────────────────────────────────────────────────────────────────────────

_CLIENT_CONFIG: dict[str, dict[str, str]] = {
    "crewai": {
        "repo_path_env": "CREWAI_REPO_PATH",
        "fallback": "integrations/crewai_endee",
        "src_subdir": "crewai_endee",
    },
    "langchain": {
        "repo_path_env": "LANGCHAIN_REPO_PATH",
        "fallback": "integrations/langchain_endee",
        "src_subdir": "langchain_endee",
    },
    "llamaindex": {
        "repo_path_env": "LLAMAINDEX_REPO_PATH",
        "fallback": "integrations/llamaindex_endee",
        "src_subdir": "llama_index_endee",
    },
}

# Clients whose tests live inside the *cloned repo* rather than this project
_TEST_DIR_INSIDE_REPO = "tests"

# ── Public helpers ──────────────────────────────────────────────────────────


def get_repo_root(client: str) -> Path:
    """
    Return the root directory of the cloned repo for *client*.

    When ``CREWAI_REPO_PATH`` (etc.) is set, that value is used as-is.
    Otherwise the fallback is resolved relative to *base_dir* (which
    defaults to ".").
    """
    cfg = _CLIENT_CONFIG[client]
    env_path = os.getenv(cfg["repo_path_env"])
    if env_path:
        return Path(env_path)
    return Path(cfg["fallback"])


def get_source_dir(client: str) -> Path:
    """
    Return the actual source-package directory inside the repo.

    Example:
        CREWAI_REPO_PATH=/Users/harish/Documents/GitHub/crewai
        → returns Path("/Users/harish/Documents/GitHub/crewai/crewai_endee")
    """
    cfg = _CLIENT_CONFIG[client]
    repo = get_repo_root(client)
    return repo / cfg["src_subdir"]


def get_all_source_files(client: str) -> list[Path]:
    """
    Discover every ``.py`` file inside the integration's source directory.

    Returns a sorted list of *absolute* paths.
    Missing directories log a warning and return an empty list.
    """
    src_dir = get_source_dir(client)
    if not src_dir.exists():
        logger.warning(
            "[config] Source dir for '%s' does not exist: %s", client, src_dir
        )
        return []
    files = sorted(src_dir.rglob("*.py"))
    logger.info(
        "[config] Found %d .py file(s) for '%s' in %s", len(files), client, src_dir
    )
    return files


def read_all_sources(client: str) -> dict[str, str]:
    """
    Read every ``.py`` file and return ``{relative_path: content}``.

    The keys are relative to the *repo root* so they look like:
        ``crewai_endee/utils.py``
    """
    repo_root = get_repo_root(client)
    files = get_all_source_files(client)
    result: dict[str, str] = {}
    for f in files:
        try:
            rel = str(f.relative_to(repo_root))
        except ValueError:
            rel = f.name
        result[rel] = f.read_text(encoding="utf-8", errors="ignore")
    return result


def get_test_dir(client: str) -> Path | None:
    """
    Return the test directory for a client.

    Looks first at ``<repo_root>/tests/``, then falls back to
    the project-level ``tests/`` directory.  Returns ``None`` if
    neither exists.
    """
    repo_root = get_repo_root(client)
    repo_tests = repo_root / _TEST_DIR_INSIDE_REPO
    if repo_tests.exists():
        return repo_tests
    return None


def get_all_test_files(client: str) -> list[Path]:
    """
    Discover every ``.py`` test file inside the repo's ``tests/`` directory.

    Returns a sorted list of *absolute* paths.
    Returns an empty list if the tests directory doesn't exist.
    """
    test_dir = get_test_dir(client)
    if test_dir is None or not test_dir.exists():
        logger.warning(
            "[config] Test dir for '%s' does not exist", client
        )
        return []
    files = sorted(test_dir.rglob("*.py"))
    logger.info(
        "[config] Found %d test file(s) for '%s' in %s", len(files), client, test_dir
    )
    return files


def read_all_test_files(client: str) -> dict[str, str]:
    """
    Read every ``.py`` file in the repo's ``tests/`` directory.

    Returns ``{relative_path: content}`` where keys are relative to repo root,
    e.g. ``tests/test_vector_store.py``.
    """
    repo_root = get_repo_root(client)
    files = get_all_test_files(client)
    result: dict[str, str] = {}
    for f in files:
        try:
            rel = str(f.relative_to(repo_root))
        except ValueError:
            rel = f.name
        result[rel] = f.read_text(encoding="utf-8", errors="ignore")
    return result


def all_clients() -> list[str]:
    """Return the list of all known client names."""
    return list(_CLIENT_CONFIG.keys())
