"""Repository-anchored `.env` loading for the tools that need a Lichess token.

`run_bot.sh` sources `.env` before exec'ing, so `chessformer.lichess` has always
had a token. Nothing sourced it for `tools/challenge_ladder.py`, and that is not
a cosmetic gap: the ladder meant to grade the `main-d2-ep1` checkpoint died on
`KeyError: 'LICHESS_TOKEN'` the moment it started, so the seven-hour run went
unmeasured and the rating that got quoted against it came from the previous
checkpoint still online.

The path is anchored to the repository rather than to the working directory for
the same reason `tablebase.DEFAULT_PATH` is: every script here happens to `cd`
into the repository, which hides the bug until one does not.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"


def read_env_file(path: Path | str = ENV_PATH) -> dict[str, str]:
    """Parse the `KEY=value` lines of a shell-style env file.

    Deliberately not a shell: no interpolation, no command substitution. `.env`
    holds three opaque secrets and nothing that needs evaluating.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if not separator:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def env_value(name: str, path: Path | str = ENV_PATH) -> str:
    """The process environment first, then the repository `.env`, else empty.

    The process wins so that `NAME=... python tool.py` and a sourced `.env` keep
    overriding the file, which is what every existing launcher relies on.
    """
    from_process = os.environ.get(name, "")
    if from_process:
        return from_process
    return read_env_file(path).get(name, "")


def lichess_token(explicit: str = "", path: Path | str = ENV_PATH) -> str:
    """A Lichess token, or a readable failure instead of a KeyError traceback."""
    token = explicit or env_value("LICHESS_TOKEN", path)
    if not token:
        raise SystemExit(
            "No Lichess token. Pass --token, export LICHESS_TOKEN, or put "
            f"LICHESS_TOKEN=... in {Path(path)}"
        )
    return token
