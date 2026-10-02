"""Exact endgame results from Syzygy tablebases.

The network is the weakest exactly where the answer is already known. Three of
the bot's drawn games ended in K+N versus K or K+B versus K while the material
term still read +320, and `engine._reaches_dead_draw` patches only the narrowest
case of that: a position drawn *by rule* because neither side has mating
material. It cannot see that K+P versus K is drawn because the pawn is a rook
pawn with the wrong-coloured bishop, nor that a rook endgame two pawns down is
still held. A tablebase can, and it is never wrong.

Two probe sites, deliberately different:

* the root (`best_move`), which replaces the whole search once the position is
  solved -- exact play, and it skips the 0.3 s forward pass entirely;
* the leaves of the forced search (`score_cp`), which is what stops the engine
  *entering* a drawn or lost endgame. The root probe only ever reacts to a
  trade that has already happened.

Everything here degrades to `None` when no table covers the position, so the
caller keeps its existing behaviour untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chess
import chess.syzygy

# Centipawn magnitude of a tablebase win at the leaves of the forced search.
# It has to outrank any material count that search can produce -- a theoretical
# maximum near 10,400 cp with nine queens on the board -- while staying below
# `engine.FORCED_MATE_THRESHOLD`, so a solved win is never mistaken for a mate
# the search found itself.
TB_WIN_CP = 30_000

# The engine object outlives a game, so the probe cache needs the same ceiling
# treatment as the value cache: bounded, cleared wholesale when full.
_PROBE_CACHE_MAX = 200_000

# Syzygy's own limit for the "cursed win" band: a win needing more than 100
# plies to the next zeroing move is a draw under the fifty-move rule.
_FIFTY_MOVE_PLIES = 100


@dataclass(frozen=True)
class RootProbe:
    """A tablebase-optimal move, with the result it forces."""

    move: chess.Move
    wdl: int
    """2 win, 1 cursed win, 0 draw, -1 blessed loss, -2 loss -- our point of view."""
    plies_to_zero: int
    """Plies from *now* until the next capture or pawn move, this move included.

    Counted from now rather than as a counter value so that a move which zeroes
    the clock itself and one which does not are the same quantity and can be
    ranked against each other. See `_probe_move`.
    """

    @property
    def is_win(self) -> bool:
        return self.wdl == 2

    @property
    def is_loss(self) -> bool:
        return self.wdl == -2


class EndgameTablebase:
    """Syzygy probes, cached, with every failure mode folded into `None`."""

    def __init__(self, path: str | Path, *, max_men: int = 5) -> None:
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError(f"tablebase directory not found: {directory}")
        self.path = directory
        self.max_men = max_men
        self._tables = chess.syzygy.open_tablebase(str(directory))
        self._wdl_cache: dict[int, int | None] = {}
        self.probes = 0
        self.hits = 0

    def close(self) -> None:
        self._tables.close()

    def __enter__(self) -> EndgameTablebase:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def covers(self, board: chess.Board) -> bool:
        """Can this position be probed at all?

        Castling rights are not encoded in Syzygy -- a position that still has
        them is a different position as far as the tables are concerned -- so
        python-chess refuses to probe it. Checking here keeps the refusal off
        the hot path instead of discovering it per probe.
        """
        if board.castling_rights:
            return False
        return chess.popcount(board.occupied) <= self.max_men

    def wdl(self, board: chess.Board) -> int | None:
        """Win/draw/loss for the side to move, or None if no table covers it."""
        if not self.covers(board):
            return None
        key = board._transposition_key()
        cached = self._wdl_cache.get(key, _MISSING)
        if cached is not _MISSING:
            return cached  # type: ignore[return-value]
        self.probes += 1
        value = self._tables.get_wdl(board)
        if value is not None:
            self.hits += 1
        if len(self._wdl_cache) >= _PROBE_CACHE_MAX:
            self._wdl_cache.clear()
        self._wdl_cache[key] = value
        return value

    def score_cp(self, board: chess.Board, ply: int) -> int | None:
        """Leaf score in centipawns for the side to move, or None if unsolved.

        A cursed win and a blessed loss both score 0: the fifty-move rule makes
        them draws in a real game, and pricing a cursed win as a win is how an
        engine talks itself into a losing trade to reach one.

        The ply term makes a faster win outrank a slower one, matching how
        `engine._forced_value` already discounts mate scores by depth.
        """
        value = self.wdl(board)
        if value is None:
            return None
        if value == 2:
            return TB_WIN_CP - ply
        if value == -2:
            return -TB_WIN_CP + ply
        return 0

    def best_move(self, board: chess.Board) -> RootProbe | None:
        """Tablebase-optimal move for `board`, or None if it is not solved.

        Only called when the *root* is covered, so every child is covered too:
        a move can remove a piece but never add one. Probing a root that is one
        capture away from the tables would mix solved children with unsolved
        ones and rank them against each other, which is worse than not probing.

        Illegal positions are refused here rather than in `covers`, which sits
        on the per-node path of the forced search: inside that search every
        position was reached by a legal move, so the check would never fire and
        would only cost. At the root a hand-typed FEN can be anything, and an
        illegal one makes Syzygy's key nonsense rather than raising -- the same
        trap that made Stockfish segfault in `tools/blunder_replay.py`.

        Ranking, in order: the true result first; then, when winning, the
        shortest path to the next zeroing move -- which is what guarantees
        progress and beats the fifty-move counter -- and when losing, the
        longest, which is the best practical defence and the only way a lost
        position is ever saved by the counter running out.
        """
        if not self.covers(board) or not board.is_valid():
            return None

        best: RootProbe | None = None
        best_key: tuple[int, int] | None = None
        for move in board.legal_moves:
            probe = self._probe_move(board, move)
            if probe is None:
                # One unsolved child makes the whole ranking untrustworthy.
                return None
            if probe.wdl > 0:
                key = (probe.wdl, -probe.plies_to_zero)
            elif probe.wdl < 0:
                key = (probe.wdl, probe.plies_to_zero)
            else:
                key = (0, 0)
            if best_key is None or key > best_key:
                best, best_key = probe, key
        return best

    def _probe_move(self, board: chess.Board, move: chess.Move) -> RootProbe | None:
        """Result of `move` from our point of view, honouring the real clock."""
        zeroing = board.is_zeroing(move)
        clock_after = 0 if zeroing else board.halfmove_clock + 1
        board.push(move)
        try:
            if board.is_checkmate():
                return RootProbe(move=move, wdl=2, plies_to_zero=0)
            if board.is_stalemate() or board.is_insufficient_material():
                return RootProbe(move=move, wdl=0, plies_to_zero=0)
            self.probes += 1
            dtz_child = self._tables.get_dtz(board)
            if dtz_child is not None:
                self.hits += 1
        finally:
            board.pop()

        if dtz_child is None:
            return None
        if dtz_child == 0:
            return RootProbe(move=move, wdl=0, plies_to_zero=0)

        # `dtz_child` counts plies for the side to move *in the child*, which is
        # the opponent, so its sign is already the negation of our result.
        magnitude = abs(dtz_child)
        winning = dtz_child < 0
        # Syzygy measures DTZ from a zeroed counter. The real counter is what
        # decides the game, so a win that needs more plies than the counter has
        # left is only a cursed win however the table scores it.
        decisive = magnitude <= _FIFTY_MOVE_PLIES and clock_after + magnitude <= _FIFTY_MOVE_PLIES
        wdl = (2 if decisive else 1) if winning else (-2 if decisive else -1)
        # `magnitude` is the child's distance to the *next* zeroing move. After a
        # zeroing move that next one is a different, later event and the two
        # numbers do not measure the same thing: ranking them together made the
        # engine value a promotion at the cost of the whole conversion that
        # follows it, so it never promoted and shuffled to a threefold draw in
        # game 1sPImi1G. A zeroing move is one ply from zeroing -- itself.
        plies = 1 if zeroing else 1 + magnitude
        return RootProbe(move=move, wdl=wdl, plies_to_zero=plies)


class _Missing:
    """Sentinel so a cached `None` is not re-probed on every visit."""


_MISSING = _Missing()

# Where the 3-4-5 man tables live, anchored to the repository rather than to the
# working directory. A cwd-relative default looked fine because every shell
# script here starts with `cd /home/ubuntu/CHESSFORMER`, but a UCI GUI launches
# the engine from wherever it likes -- and the failure mode is silent. Running
# from /tmp simply played on without tables and reported a plausible score, with
# nothing anywhere to say the endgame knowledge had been dropped.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = _REPO_ROOT / "data" / "syzygy"


def open_tables(path: str | Path | None, *, max_men: int = 5) -> EndgameTablebase | None:
    """Open `path`, or the conventional location, or return None.

    An explicit path that does not exist is an error -- asking for tablebases
    and silently playing without them is how a measurement ends up comparing a
    mode against itself. A missing *default* is not: most checkouts have no
    tables, and every caller works without them.
    """
    if path is not None:
        return EndgameTablebase(path, max_men=max_men)
    for candidate in (DEFAULT_PATH, Path("data/syzygy")):
        if candidate.is_dir():
            return EndgameTablebase(candidate, max_men=max_men)
    return None
