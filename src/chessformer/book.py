"""An opening book built from the bot's own repeated mistakes.

The bot plays at temperature 0, so it is deterministic, so it replays whole
games -- 41 of 78 in the last measurement. That is usually reported as a variety
problem. It is also an opportunity: a mistake that repeats can be *fixed once*.
A book entry turns a line the bot has lost the same way five times into a line it
has never played, for the price of a dictionary lookup.

Deliberately limited to the opening. Two reasons, and both are about what the
data can support. A centipawn loss on move six is local and verifiable -- the
analysis says this move drops 120 cp, whoever plays it -- while a loss forty
moves later cannot be attributed to it by anything in a PGN. And an opening book
is a normal part of an engine, where a table of expert answers covering the whole
game would no longer be the network playing.

The file is JSON keyed by EPD, written by `tools/build_book.py`:

    {"schema": 1, "entries": {"<epd>": {"move": "g1f3", "value_cp": 31, ...}}}

`DEFAULT_PATH` is anchored to the repository rather than to the working
directory, for the reason `tablebase.DEFAULT_PATH` is: every script here happens
to `cd` into the repository, which hides the bug until one does not, and a book
that silently never matches is exactly the kind of quiet downgrade this project
has already paid for.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import chess

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "book.json"
SCHEMA = 1


@dataclass(frozen=True)
class BookEntry:
    """One correction: what to play here, and what it replaces."""

    move: chess.Move
    # Side-to-move centipawns from the analysis that chose the move, so the
    # engine can report a score for a move it did not search.
    value_cp: int
    # What the bot played here before, kept so the log and the review can say
    # what the book actually changed rather than only what it plays.
    replaces: chess.Move | None = None
    occurrences: int = 0
    depth: int = 0


class OpeningBook:
    """EPD-keyed corrections, probed at the root before anything else."""

    def __init__(self, entries: dict[str, BookEntry], *, path: Path | None = None) -> None:
        self.entries = entries
        self.path = path

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def key(board: chess.Board) -> str:
        """A position's identity in the book.

        `epd()` writes the en-passant square only when the capture is legal,
        which is what makes two positions the same position here. The engine's
        value cache needs the opposite convention (`en_passant="fen"`) because it
        keys what `encode_board` fed the network; these are different questions
        and the answers are deliberately different.
        """
        return board.epd()

    def probe(self, board: chess.Board) -> BookEntry | None:
        entry = self.entries.get(self.key(board))
        if entry is None:
            return None
        # The load-time check already refused an illegal move, so this can only
        # fire if a book outlives the file it was validated from. It costs
        # microseconds and it is the difference between "the book did nothing"
        # and "the bot forfeited on an illegal move".
        if entry.move not in board.legal_moves:
            return None
        return entry


def _entry_from(epd: str, payload: dict) -> BookEntry:
    board = chess.Board()
    board.set_epd(epd)
    move = chess.Move.from_uci(str(payload["move"]))
    if move not in board.legal_moves:
        raise ValueError(f"Book move {move.uci()} is not legal in {epd}")
    replaces = payload.get("replaces")
    return BookEntry(
        move=move,
        value_cp=int(payload.get("value_cp", 0)),
        replaces=chess.Move.from_uci(str(replaces)) if replaces else None,
        occurrences=int(payload.get("occurrences", 0)),
        depth=int(payload.get("depth", 0)),
    )


def load_book(path: str | Path) -> OpeningBook:
    """Read a book, refusing one that does not describe legal positions.

    An illegal entry means the file was written against a different convention,
    and the symptom of carrying on would be a book that simply never matches --
    a downgrade with no error message, which is worse than a crash at launch.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = int(payload.get("schema", 0))
    if schema != SCHEMA:
        raise ValueError(f"{path}: book schema {schema}, expected {SCHEMA}")
    entries = {epd: _entry_from(epd, value) for epd, value in (payload.get("entries") or {}).items()}
    return OpeningBook(entries, path=path)


def open_book(path: str | Path | None = None) -> OpeningBook | None:
    """The book at `path`, or the repository's when it exists, or None.

    Same contract as `tablebase.open_tables`: an explicit path that is missing is
    an error, because someone asked for it; the default being absent is not.
    """
    if path is not None:
        return load_book(path)
    if DEFAULT_PATH.is_file():
        return load_book(DEFAULT_PATH)
    return None


def write_book(path: str | Path, entries: dict[str, dict], *, metadata: dict) -> int:
    """Write a book and return how many entries it holds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, **metadata, "entries": entries}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Read it back through the loader: a book that cannot be loaded is a book
    # the bot will refuse to start with, and finding that out now costs nothing.
    return len(load_book(path))
