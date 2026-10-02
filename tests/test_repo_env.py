"""The ladder that was supposed to grade main-d2-ep1 died on KeyError:
'LICHESS_TOKEN' because nothing sourced .env for it. These pin the fix."""

from __future__ import annotations

import pytest

from chessformer.repo_env import ENV_PATH, env_value, lichess_token, read_env_file


def test_env_path_is_anchored_to_the_repository() -> None:
    # Not the working directory: every launcher here happens to cd into the
    # repository, which hides a relative path right up until one does not.
    assert ENV_PATH.is_absolute()
    assert ENV_PATH.name == ".env"
    assert (ENV_PATH.parent / "pyproject.toml").is_file()


def test_reads_keys_ignoring_comments_blanks_quotes_and_export(tmp_path) -> None:
    path = tmp_path / ".env"
    path.write_text(
        "# a comment\n"
        "\n"
        "LICHESS_TOKEN=lip_plain\n"
        'export QUOTED="dq"\n'
        "SINGLE='sq'\n"
        "NOT_AN_ASSIGNMENT\n"
        "  SPACED = spaced  \n",
        encoding="utf-8",
    )
    assert read_env_file(path) == {
        "LICHESS_TOKEN": "lip_plain",
        "QUOTED": "dq",
        "SINGLE": "sq",
        "SPACED": "spaced",
    }


def test_missing_file_is_empty_not_an_error(tmp_path) -> None:
    assert read_env_file(tmp_path / "absent") == {}


def test_process_environment_wins_over_the_file(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text("LICHESS_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setenv("LICHESS_TOKEN", "from_process")
    assert env_value("LICHESS_TOKEN", path) == "from_process"
    monkeypatch.delenv("LICHESS_TOKEN")
    assert env_value("LICHESS_TOKEN", path) == "from_file"


def test_explicit_token_wins_over_both(tmp_path, monkeypatch) -> None:
    path = tmp_path / ".env"
    path.write_text("LICHESS_TOKEN=from_file\n", encoding="utf-8")
    monkeypatch.setenv("LICHESS_TOKEN", "from_process")
    assert lichess_token("explicit", path) == "explicit"


def test_absent_token_raises_a_readable_message_not_a_keyerror(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)
    with pytest.raises(SystemExit) as caught:
        lichess_token("", tmp_path / "absent")
    assert "LICHESS_TOKEN" in str(caught.value)
