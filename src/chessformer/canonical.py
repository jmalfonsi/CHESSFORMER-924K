"""Put every position on the board from the side to move's point of view.

Chess is exactly colour-symmetric, and V1 does not exploit that: the board is
encoded in absolute coordinates and the value target is relative to White, so the
network has to learn every pattern twice, once per colour. Measured on 400 held
out positions with `main-d2-ep1`, it has not managed it -- shown the same position
mirrored, the net picks a different best move in 27% of cases, its policy moves
0.197 of its probability mass, and its value contradicts itself by 0.094, about
57 centipawns. Every bit of that is error, because the two positions are the same
position.

Canonicalising removes the error by construction rather than asking the optimiser
to discover it, and it doubles how much each training row teaches. The transform
is a vertical flip with the colours swapped -- python-chess spells it
`Board.mirror()` -- applied whenever Black is to move, together with the matching
permutation of the action vocabulary and a sign flip on the value.

Files survive a rank flip, so `ep_file` is untouched; castling rights swap sides;
`side` becomes a constant and stops carrying information, which is why a network
trained on canonical input can drop its side embedding.
"""

from __future__ import annotations

from functools import lru_cache

import chess
import numpy as np

from .moves import N_ACTIONS, action_uci, uci_to_index

# 0 empty, 1..6 white P/N/B/R/Q/K, 7..12 black P/N/B/R/Q/K.
_WHITE_TOKENS = range(1, 7)
_BLACK_TOKENS = range(7, 13)


@lru_cache(maxsize=1)
def square_permutation() -> np.ndarray:
    """`out[s] = in[mirror(s)]`: a vertical flip of the 64 board squares."""
    return np.asarray([chess.square_mirror(s) for s in chess.SQUARES], dtype=np.int64)


@lru_cache(maxsize=1)
def piece_permutation() -> np.ndarray:
    """Token relabelling that swaps the two colours and leaves empty alone."""
    table = np.arange(13, dtype=np.uint8)
    for white, black in zip(_WHITE_TOKENS, _BLACK_TOKENS, strict=True):
        table[white] = black
        table[black] = white
    return table


@lru_cache(maxsize=1)
def action_permutation() -> np.ndarray:
    """`perm[i]` is the action that `i` becomes under the mirror.

    Total and self-inverse: the queen/knight geometries are closed under a rank
    flip, and White's rank-7-to-8 promotions map exactly onto Black's rank-2-to-1
    ones, which is why the vocabulary lists both explicitly.
    """
    index_of = uci_to_index()
    perm = np.empty(N_ACTIONS, dtype=np.int64)
    for index, uci in enumerate(action_uci()):
        move = chess.Move.from_uci(uci)
        mirrored = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
        perm[index] = index_of[mirrored.uci()]
    return perm


def mirror_castling(castling: int) -> int:
    """Swap the White (bits 1,2) and Black (bits 4,8) castling rights."""
    return ((castling & 0b0011) << 2) | ((castling & 0b1100) >> 2)


def mirror_move(move: chess.Move) -> chess.Move:
    """A move on the mirrored board, and back again -- the transform is its own inverse."""
    return chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )


def canonical_board(board: chess.Board) -> tuple[chess.Board, bool]:
    """`board` seen by the side to move, and whether it had to be mirrored.

    The caller needs the flag: a move read off the canonical board has to be
    mirrored back before it is played or reported.
    """
    if board.turn == chess.WHITE:
        return board, False
    return board.mirror(), True
