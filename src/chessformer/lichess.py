"""Play on Lichess as a BOT account, using only the standard library.

    export LICHESS_TOKEN=...            # token with the bot:play scope
    python -m chessformer.lichess --checkpoint checkpoints/<run>/best.pt
    python -m chessformer.lichess --checkpoint ... --challenge philidor-142M \
        --clock-limit 180 --clock-increment 2

The account must already be upgraded to a BOT account, which Lichess only allows
on an account that has never played a game:

    curl -X POST https://lichess.org/api/bot/account/upgrade \
        -H "Authorization: Bearer $LICHESS_TOKEN"
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Iterator
import urllib.error
import urllib.parse
import urllib.request

import chess

from .book import DEFAULT_PATH as BOOK_DEFAULT_PATH, open_book
from .engine import MODES, ChessFormerEngine, checkpoint_frame, load_model
from .repo_env import lichess_token
from .tablebase import DEFAULT_PATH as TB_DEFAULT_PATH, open_tables

# How a Syzygy verdict reads in the game log, from our own point of view.
_WDL_NAMES = {2: "win", 1: "cursed win", 0: "draw", -1: "blessed loss", -2: "loss"}

BASE_URL = "https://lichess.org"
# Below this much time left, drop the safety net's mate scan and just move.
PANIC_MS = 15_000
# Above this, the forced search may spend its whole node budget; below it the
# budget is cut so a sharp position cannot eat a minute of a 5+3 game. safe4
# never flagged, but safe5's forced search costs several times more per move and
# the guard above it was a single binary switch at 15 s.
COMFORTABLE_MS = 60_000
SQUEEZED_DIVISOR = 4
# How long to wait before reopening the event stream, and the ceiling the delay
# backs off to when Lichess keeps refusing.
RECONNECT_DELAY_S = 5
RECONNECT_MAX_S = 60
# What we tell an opponent we turn away because a game is already running.
# Lichess renders it as "This bot is playing, try again later".
DECLINE_BUSY = "later"


def _log(message: str) -> None:
    """Timestamped and flushed: a redirected stdout would otherwise buffer the
    whole game and leave the operator blind while it is being played."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class LichessClient:
    def __init__(self, token: str, *, base_url: str = BASE_URL) -> None:
        if not token:
            raise ValueError("A Lichess API token with the bot:play scope is required")
        self.token = token
        self.base_url = base_url.rstrip("/")

    def _request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        accept: str | None = None,
    ) -> urllib.request.Request:
        body = urllib.parse.urlencode(data).encode() if data else None
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=body, method=method
        )
        request.add_header("Authorization", f"Bearer {self.token}")
        if accept:
            request.add_header("Accept", accept)
        return request

    def post(self, path: str, data: dict | None = None) -> dict:
        try:
            with urllib.request.urlopen(self._request("POST", path, data)) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")
            raise RuntimeError(f"POST {path} failed: {error.code} {detail}") from error
        return json.loads(payload) if payload.strip() else {}

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self._request("GET", path)) as response:
            return json.loads(response.read().decode("utf-8"))

    def stream(self, path: str, accept: str | None = None) -> Iterator[dict]:
        """Yield each object of an ndjson stream; blank keep-alive lines are skipped.

        The bot endpoints answer ndjson by default. The game export does not --
        it answers PGN unless asked otherwise, and the JSON decoder then fails on
        the first `[Event ...]` line. Callers of that endpoint pass
        `accept="application/x-ndjson"`.
        """
        with urllib.request.urlopen(self._request("GET", path, accept=accept)) as response:
            for raw in response:
                line = raw.decode("utf-8").strip()
                if line:
                    yield json.loads(line)


def moves_of(state: dict) -> list[str]:
    text = (state.get("moves") or "").strip()
    return text.split() if text else []


def board_from_moves(moves: list[str], initial_fen: str | None = None) -> chess.Board:
    board = chess.Board() if not initial_fen or initial_fen == "startpos" else chess.Board(initial_fen)
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board


def our_turn(board: chess.Board, playing_white: bool) -> bool:
    return board.turn == (chess.WHITE if playing_white else chess.BLACK)


def remaining_ms(state: dict, playing_white: bool) -> int:
    return int(state.get("wtime" if playing_white else "btime", 0))


def should_accept(challenge: dict) -> tuple[bool, str]:
    """Accept only standard chess the engine can actually play."""
    variant = (challenge.get("variant") or {}).get("key", "standard")
    if variant != "standard":
        return False, "variant"
    if challenge.get("rules") and "noAbort" in challenge["rules"]:
        pass
    speed = challenge.get("speed", "")
    if speed in {"ultraBullet", "bullet"}:
        # A 143M forward costs seconds on CPU; bullet cannot be honoured.
        return False, "tooFast"
    return True, "ok"


class GameSlots:
    """How many games the bot has open at once, and the gate on that number.

    Every game runs in its own thread, but all of them queue on the one engine
    lock: a second game is not played in parallel, it is played in the gaps of
    the first, on the same four cores, with both clocks running. safe6 at
    --quiet-replies 12 spends about 18s a move of the 300s a 5+3 game gives --
    survivable once, not twice, so the useful limit is one.

    A slot is taken when a challenge is *accepted*, not when the game starts:
    Lichess reuses the challenge id as the game id, so both reserve the same key
    and the window between accepting and `gameStart` cannot be used to slip a
    second game in. A game that has already started is never refused a slot --
    the limit gates accepting, and abandoning a live game would be far worse
    than being busy.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(0, limit)  # 0 disables the gate entirely
        self._lock = threading.Lock()
        self._open: set[str] = set()

    def reserve(self, game_id: str) -> bool:
        """Take a slot, or return False when the bot is already full."""
        with self._lock:
            if game_id in self._open:
                return True
            if self.limit and len(self._open) >= self.limit:
                return False
            self._open.add(game_id)
            return True

    def take(self, game_id: str) -> None:
        """Record a game that exists whether or not there was room for it."""
        with self._lock:
            self._open.add(game_id)

    def release(self, game_id: str) -> None:
        with self._lock:
            self._open.discard(game_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._open)


class GameSession:
    """Plays a single Lichess game to completion."""

    def __init__(
        self,
        client: LichessClient,
        engine: ChessFormerEngine,
        game_id: str,
        account_id: str,
        *,
        log: Callable[[str], None] = _log,
        engine_lock: threading.Lock | None = None,
    ) -> None:
        self.client = client
        self.engine = engine
        self.game_id = game_id
        self.account_id = account_id
        self.log = log
        # Every game runs in its own thread against one shared engine, and
        # `select_move` mutates `engine.mode` and `engine.forced_nodes` before
        # restoring them. Interleave two of those and one thread restores the
        # other's saved value: the engine can be left in `policy` mode for the
        # rest of its life, which `evaluate_checkpoint.sh` prices at -382 Elo.
        # Three concurrent windows appeared in the last forty rated games. The
        # `_value_cache` and the node counters are shared too. A lock is the
        # whole fix: moves are CPU-bound on four cores, so they were never going
        # to run in parallel usefully anyway.
        self.engine_lock = engine_lock or threading.Lock()
        self.playing_white = True
        self.initial_fen: str | None = None

    def run(self) -> None:
        for event in self.client.stream(f"/api/bot/game/stream/{self.game_id}"):
            kind = event.get("type")
            if kind == "gameFull":
                self.playing_white = (event.get("white") or {}).get("id") == self.account_id
                self.initial_fen = event.get("initialFen")
                self.handle_state(event.get("state") or {})
            elif kind == "gameState":
                self.handle_state(event)
            elif kind == "chatLine":
                continue

    def handle_state(self, state: dict) -> None:
        if state.get("status") not in (None, "started", "created"):
            self.log(f"[{self.game_id}] finished: {state.get('status')}")
            return
        board = board_from_moves(moves_of(state), self.initial_fen)
        if board.is_game_over() or not our_turn(board, self.playing_white):
            return

        left = remaining_ms(state, self.playing_white)
        started = time.perf_counter()
        move = self.select_move(board, left)
        spent = time.perf_counter() - started
        # Asked before the move is posted, while `board` is still the position
        # the engine was given.
        tag = self._book_tag(board) or self._tablebase_tag(board)
        try:
            self.client.post(f"/api/bot/game/{self.game_id}/move/{move.uci()}")
        except RuntimeError as error:
            # Losing a race against the opponent's move is normal; log and wait.
            self.log(f"[{self.game_id}] move {move.uci()} rejected: {error}")
            return
        self.log(
            f"[{self.game_id}] {'white' if self.playing_white else 'black'} "
            f"played {move.uci()}{tag} in {spent:.1f}s ({left / 1000:.0f}s left)"
        )

    def _book_tag(self, board: chess.Board) -> str:
        """` [book]` when the move came out of the opening book.

        Asked before the move is posted, like the tablebase tag, and for the
        same reason: without it the log cannot answer "is the book being used,
        and on which lines?", which is the first question anyone asks of it.
        """
        book = self.engine.book
        if book is None:
            return ""
        entry = book.probe(board)
        if entry is None:
            return ""
        replaced = f", was {entry.replaces.uci()}" if entry.replaces else ""
        return f" [book{replaced}]"

    def _tablebase_tag(self, board: chess.Board) -> str:
        """` [TB win]` and friends, when the move came out of the tables.

        Without this the log cannot answer "were the tablebases used, and did
        they help?" -- which is the first question anyone asks of them. Asking
        after the fact is a popcount against `max_men`, so it costs nothing on
        the positions where the answer is no, which is nearly all of them.
        """
        tables = self.engine.tablebase
        if tables is None or not tables.covers(board):
            return ""
        value = tables.wdl(board)
        if value is None:
            return ""
        return f" [TB {_WDL_NAMES[value]}]"

    def select_move(self, board: chess.Board, remaining: int) -> chess.Move:
        with self.engine_lock:
            return self._select_move_locked(board, remaining)

    def _select_move_locked(self, board: chess.Board, remaining: int) -> chess.Move:
        previous = self.engine.mode
        budget = self.engine.forced_nodes
        if remaining and remaining < PANIC_MS and self.engine.mode != "policy":
            self.engine.mode = "policy"
        elif remaining and remaining < COMFORTABLE_MS:
            # Still searching, but on a shorter leash: a cheaper search that
            # moves in time beats a thorough one that flags.
            self.engine.forced_nodes = max(1, budget // SQUEEZED_DIVISOR)
            # Cutting the forced budget is not a leash for safe6. Measured on
            # four midgame positions, safe5 spends 300-1,400 forced nodes of the
            # 30,000 it is allowed, so quartering that number changes nothing at
            # all; what safe6 actually spends is neural passes -- 78 against
            # safe5's 10 at --quiet-replies 8, about five seconds a move. The
            # only guard that bites is dropping the second quiet ply.
            if self.engine.mode == "safe6":
                self.engine.mode = "safe5"
        try:
            result = self.engine.search(board)
        finally:
            self.engine.mode = previous
            self.engine.forced_nodes = budget
        if result is None:
            raise RuntimeError("No legal move in a position we were asked to move in")
        return result.move


def handle_event(
    client: LichessClient,
    engine: ChessFormerEngine,
    event: dict,
    account_id: str,
    *,
    log: Callable[[str], None],
    engine_lock: threading.Lock,
    games: GameSlots | None = None,
    acceptance: Callable[[dict], tuple[bool, str]] = should_accept,
) -> None:
    games = games if games is not None else GameSlots(0)
    kind = event.get("type")
    if kind == "challenge":
        challenge = event["challenge"]
        if challenge.get("challenger", {}).get("id") == account_id:
            return
        accept, reason = acceptance(challenge)
        challenge_id = challenge["id"]
        busy = accept and not games.reserve(challenge_id)
        if busy:
            accept, reason = False, DECLINE_BUSY
        if accept:
            try:
                client.post(f"/api/challenge/{challenge_id}/accept")
            except Exception:
                # The slot was taken before the network call: give it back, or a
                # failed accept wedges the bot out of every later challenge.
                games.release(challenge_id)
                raise
            log(f"Accepted challenge {challenge_id} from {challenge['challenger']['name']}")
        else:
            client.post(f"/api/challenge/{challenge_id}/decline", {"reason": reason})
            crowd = f", {len(games)} game(s) in progress" if busy else ""
            log(f"Declined challenge {challenge_id} ({reason}{crowd})")
    elif kind == "gameStart":
        game_id = event["game"]["id"]
        games.take(game_id)
        log(f"Game {game_id} started ({len(games)} in progress)")
        session = GameSession(
            client, engine, game_id, account_id, log=log, engine_lock=engine_lock
        )

        def play() -> None:
            try:
                session.run()
            finally:
                # However the game ends -- mate, resignation, a dropped game
                # stream -- the slot has to come back, or the bot answers
                # "later" for the rest of its life.
                games.release(game_id)

        threading.Thread(target=play, daemon=False).start()


def run_bot(
    client: LichessClient,
    engine: ChessFormerEngine,
    *,
    log: Callable[[str], None] = _log,
    reconnect: bool = True,
    max_games: int = 1,
    acceptance: Callable[[dict], tuple[bool, str]] = should_accept,
) -> None:
    """Accept challenges and play, reopening the event stream as often as needed.

    Lichess ends this stream on its own -- a deploy on their side, a proxy
    timeout, a dropped connection. The first version treated that as the end of
    the session: the `for` loop finished, this function returned, `main`
    returned, and the process exited with status 0 **in the middle of a game**,
    leaving a log whose last line is an ordinary move and no error anywhere.
    That is what happened at 06:44 on 2026-09-06: the bot vanished, game
    4MzJ74XH was left to lose on time, and the ladder went on issuing challenges
    to an account that was no longer listening. Nine minutes of silence looked
    exactly like nine minutes of nobody challenging.

    So the stream is reopened instead, with a backoff, and every reconnection is
    logged -- a bot that silently stops is worse than one that crashes.
    """
    account = client.get("/api/account")
    account_id = account["id"]
    log(f"Connected as {account['username']} (id {account_id})")
    # One lock for the one engine every game thread shares. Created here rather
    # than inside GameSession so that concurrent games actually share it.
    engine_lock = threading.Lock()
    games = GameSlots(max_games)
    log(f"Accepting at most {max_games} game(s) at a time" if max_games
        else "Accepting any number of concurrent games")

    delay = RECONNECT_DELAY_S
    while True:
        try:
            for event in client.stream("/api/stream/event"):
                handle_event(
                    client,
                    engine,
                    event,
                    account_id,
                    log=log,
                    engine_lock=engine_lock,
                    games=games,
                    acceptance=acceptance,
                )
            reason = "closed by Lichess"
            delay = RECONNECT_DELAY_S
        except (OSError, http.client.HTTPException, json.JSONDecodeError) as error:
            # A truncated ndjson line and a reset socket are both the network,
            # not a bug in the caller; neither is a reason to stop playing.
            reason = f"dropped ({type(error).__name__}: {error})"
        if not reconnect:
            return
        log(f"Event stream {reason}; reopening in {delay}s")
        time.sleep(delay)
        delay = min(delay * 2, RECONNECT_MAX_S)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play on Lichess with a ChessFormer checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", default=None)
    parser.add_argument("--device", default="cpu")
    # Validate at launch, not after the token has been accepted and a game is
    # already streaming: a typo here would otherwise surface mid-challenge.
    parser.add_argument("--mode", choices=list(MODES), default="safe")
    parser.add_argument("--token", default="")
    parser.add_argument("--challenge", default=None, help="Username to challenge on startup.")
    parser.add_argument("--clock-limit", type=int, default=180)
    parser.add_argument("--clock-increment", type=int, default=2)
    parser.add_argument("--rated", action="store_true")
    parser.add_argument(
        "--max-games",
        type=int,
        default=1,
        help=(
            "concurrent games to accept; further challengers are declined with "
            "'later'. Games share one engine lock, so a second game is not played "
            "in parallel but in the gaps of the first, with both clocks running: "
            "at safe6 --quiet-replies 12 a move costs ~18s of a 300s clock, which "
            "one game survives and two do not. 0 removes the limit"
        ),
    )
    parser.add_argument("--forced-depth", type=int, default=4)
    parser.add_argument("--forced-nodes", type=int, default=30000)
    parser.add_argument(
        "--quiet-replies",
        type=int,
        default=4,
        help=(
            "opponent replies weighed per quiet candidate. Raising it spends clock "
            "on separating candidates the pool already contains -- measured over "
            "4,352 rated moves, the bot uses 1s of a 300s clock and has never once "
            "reached the 60s throttle, so the clock is there to be spent"
        ),
    )
    parser.add_argument(
        "--no-value-quiet",
        action="store_true",
        help="safe5: rank quiet moves by policy alone, not by the value head.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Policy softmax temperature. 0 (default) plays the arg-max deterministically; "
        "above 0 the move order is sampled from the tempered policy, which varies the "
        "openings without loosening any tactical veto.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the temperature sampler. Left unset, every session differs.",
    )
    parser.add_argument(
        "--tablebase",
        type=Path,
        default=None,
        help=f"Syzygy directory. Defaults to {TB_DEFAULT_PATH} when it exists.",
    )
    parser.add_argument(
        "--no-tablebase",
        action="store_true",
        help="Play without endgame tablebases even if they are installed.",
    )
    parser.add_argument(
        "--book",
        type=Path,
        default=None,
        help=f"Opening book built by tools/build_book.py. Defaults to {BOOK_DEFAULT_PATH} "
        "when it exists.",
    )
    parser.add_argument(
        "--no-book",
        action="store_true",
        help="Play without the opening book even if one is installed.",
    )
    args = parser.parse_args(argv)

    client = LichessClient(lichess_token(args.token))
    model, _config, payload = load_model(
        args.checkpoint, preset=args.preset, device=args.device
    )
    tablebase = None if args.no_tablebase else open_tables(args.tablebase)
    if tablebase is not None:
        print(f"Tablebases: {tablebase.path} (up to {tablebase.max_men} men)", file=sys.stderr)
    book = None if args.no_book else open_book(args.book)
    if book is not None:
        print(f"Opening book: {book.path} ({len(book)} correction(s))", file=sys.stderr)
    engine = ChessFormerEngine(
        model,
        device=args.device,
        mode=args.mode,
        forced_depth=args.forced_depth,
        forced_nodes=args.forced_nodes,
        quiet_replies=args.quiet_replies,
        value_quiet=not args.no_value_quiet,
        temperature=args.temperature,
        seed=args.seed,
        tablebase=tablebase,
        book=book,
        **checkpoint_frame(payload),
    )

    if args.challenge:
        response = client.post(
            f"/api/challenge/{args.challenge}",
            {
                "rated": "true" if args.rated else "false",
                "clock.limit": args.clock_limit,
                "clock.increment": args.clock_increment,
                "variant": "standard",
                "color": "random",
            },
        )
        print(f"Challenged {args.challenge}: {response.get('id', response)}", file=sys.stderr)

    run_bot(client, engine, max_games=args.max_games)


if __name__ == "__main__":
    main()
