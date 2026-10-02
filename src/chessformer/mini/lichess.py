"""Play on Lichess with the strict single-forward mini player.

    .venv/bin/python -m chessformer.mini.lichess \
        --checkpoint models/chessformer-924k-v1.pt

The Lichess plumbing (event stream, reconnection, game slots) is the one the
143M bot uses; only the move decision differs. No search, book or tablebase:
one forward pass per move, optionally sampled at a temperature.

Set LICHESS_ACCOUNT_MINI and LICHESS_TOKEN_MINI in the repository's .env.
The account name is checked before any challenge is accepted. LICHESS_TOKEN
is deliberately never used, to keep separate projects' accounts independent.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import chess
import torch

from ..lichess import LichessClient, run_bot, should_accept
from ..repo_env import ENV_PATH, env_value
from .player import MiniPlayer, load_mini

ACCOUNT = "Chessformer-924K"
ACCOUNT_VARIABLE = "LICHESS_ACCOUNT_MINI"
TOKEN_VARIABLE = "LICHESS_TOKEN_MINI"


@dataclass(frozen=True)
class MiniChoice:
    move: chess.Move


class MiniEngine:
    """The attributes `GameSession` reads, over a MiniPlayer.

    `mode` is "policy", so the session's panic fallback has nothing to switch;
    `forced_nodes` exists only for the clock leash to set and restore.
    """

    mode = "policy"

    def __init__(self, player: MiniPlayer) -> None:
        self.player = player
        self.forced_nodes = 0
        self.book = None
        self.tablebase = None

    def search(self, board: chess.Board) -> MiniChoice | None:
        move = self.player.choose_move(board)
        return None if move is None else MiniChoice(move)


def mini_should_accept(challenge: dict) -> tuple[bool, str]:
    """The 143M rules, except bullet: a mini move costs milliseconds, not seconds.

    Ultrabullet stays refused: the network round trip of each move, not the
    forward, is what would run out a 15-second clock.
    """
    accepted, reason = should_accept(challenge)
    if not accepted and reason == "tooFast" and challenge.get("speed") == "bullet":
        return True, "ok"
    return accepted, reason


def mini_account(explicit: str = "", path: Path | str = ENV_PATH) -> str:
    """CLI override, then environment/.env; retain the original account fallback."""
    return (explicit or env_value(ACCOUNT_VARIABLE, path) or ACCOUNT).strip()


def mini_token(explicit: str = "", path: Path | str = ENV_PATH) -> str:
    token = explicit or env_value(TOKEN_VARIABLE, path)
    if not token:
        raise SystemExit(f"No mini bot token. Pass --token or put {TOKEN_VARIABLE}=... in {Path(path)} "
                         "(LICHESS_TOKEN is deliberately not used)")
    return token


def check_account(account: dict, expected: str) -> None:
    username = account.get("username", "")
    if username.lower() != expected.lower():
        raise SystemExit(f"Token belongs to {username or 'an unknown account'}, not {expected}; refusing to play")
    if account.get("title") != "BOT":
        raise SystemExit(f"{username} is not a BOT account yet: upgrade it with "
                         "POST /api/bot/account/upgrade before it has played any game")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--account", default="",
                        help="Expected BOT username; defaults to LICHESS_ACCOUNT_MINI in environment/.env")
    parser.add_argument("--token", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-plies", type=int, default=None,
                        help="Sample only while the game is shorter than this many plies")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--avoid-repetition", type=float, default=None, metavar="VALUE",
                        help="Withhold a move that repeats a position a third time while the "
                             "value head (side to move, -1..1) is above VALUE")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-games", type=int, default=1)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("threads must be positive")
    torch.set_num_threads(args.threads)
    player = MiniPlayer(load_mini(args.checkpoint), temperature=args.temperature, seed=args.seed,
                        sample_plies=args.sample_plies, repetition_value=args.avoid_repetition)
    client = LichessClient(mini_token(args.token))
    account = client.get("/api/account")
    check_account(account, mini_account(args.account))
    print(f"account={account['username']} checkpoint={args.checkpoint} temperature={args.temperature} "
          f"avoid_repetition={args.avoid_repetition} parameters={player.model.num_parameters()}", flush=True)
    run_bot(client, MiniEngine(player), max_games=args.max_games, acceptance=mini_should_accept)


if __name__ == "__main__":
    main()
