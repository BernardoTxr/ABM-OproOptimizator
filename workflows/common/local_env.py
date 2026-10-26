"""Load repository-local secrets without adding a dotenv dependency."""
from __future__ import annotations

from collections.abc import MutableMapping
import os
from pathlib import Path
import re
import shlex


_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def load_env_file(
    path: Path,
    *,
    environment: MutableMapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Load simple ``KEY=VALUE`` entries, preserving exported overrides."""
    target = os.environ if environment is None else environment
    if not path.is_file():
        return ()
    loaded: list[str] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not _KEY.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: invalid environment key")
        lexer = shlex.shlex(raw_value, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        tokens = list(lexer)
        if len(tokens) > 1:
            raise ValueError(
                f"{path}:{line_number}: quote values containing whitespace"
            )
        value = tokens[0] if tokens else ""
        if key not in target:
            target[key] = value
            loaded.append(key)
    return tuple(loaded)


def load_repository_local_env() -> tuple[str, ...]:
    """Load the ignored ``.env.local`` file from the repository root."""
    repository_root = Path(__file__).resolve().parents[2]
    return load_env_file(repository_root / ".env.local")
