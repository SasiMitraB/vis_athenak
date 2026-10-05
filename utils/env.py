"""
Machine-specific root folders, from environment variables.

Variables set in the shell (or a job script) win; otherwise they are read
from a ``.env`` file in the current directory or the repo root (see
``.env.example``).  ``${VAR}`` in .env values is expanded, and ``~`` in paths.

    from utils.env import env_path
    root = env_path("ATHENAK_DIR")
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load .env files without overriding variables already set."""
    # Loaded first, so a .env in the current directory wins over the repo's.
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(REPO_ROOT / ".env")


def env_path(name: str) -> Path:
    """Folder given by environment variable ``name``."""
    if name not in os.environ:
        raise RuntimeError(f"{name} is not set; copy .env.example to .env "
                           "or export it in your shell")
    return Path(os.environ[name]).expanduser()


load_env()
