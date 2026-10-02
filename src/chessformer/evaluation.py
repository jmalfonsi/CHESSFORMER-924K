"""A positional term for the leaves of the forced search.

Measured on 2,407 quiet moves from 80 real positions the bot had to judge: the
material-only leaf produces exactly **one** distinct value per position. Every
quiet move scores the same, so the forced search cannot separate them at any
depth -- which is why raising `forced_depth` from 4 to 6 fixed 2 blunders and
broke 5, and why depth 8 was worse and 48% slower. There is nothing further out
to find while the leaf counts only material.

The obvious evaluator is the value head, and it cannot be used: one 143M forward
costs 110 ms per position batched on this Xeon, so a five-second move buys 45
node evaluations against the ~1,600 the forced search already visits. Whatever
runs in this tree has to cost microseconds, which means hand-crafted for now.

Piece-square values are Michniewski's simplified evaluation, the standard public
starting point, with the king's table interpolated between its middlegame and
endgame forms by the remaining non-pawn material. That interpolation is the part
that matters here: a king on g1 is right with queens on and wrong in a pawn
endgame, and this engine loses games in exactly that transition.
"""

from __future__ import annotations

import chess

# Tables are written rank 8 down to rank 1, as they are usually printed, and
# flipped once at import so that index 0 is a1. Writing them in board order and
# indexing them in square order is the classic way to get a mirrored evaluation
# that still looks plausible.
def _table(rows: list[int]) -> list[int]:
    if len(rows) != 64:
        raise ValueError("a piece-square table needs 64 entries")
    flipped: list[int] = []
    for rank in range(7, -1, -1):
        flipped.extend(rows[rank * 8 : rank * 8 + 8])
    return flipped


_PAWN = _table([
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
])
_KNIGHT = _table([
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
])
_BISHOP = _table([
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
])
_ROOK = _table([
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
])
_QUEEN = _table([
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
])
_KING_MIDDLE = _table([
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
])
_KING_END = _table([
   -50,-40,-30,-20,-20,-30,-40,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
])

_TABLES = {
    chess.PAWN: _PAWN,
    chess.KNIGHT: _KNIGHT,
    chess.BISHOP: _BISHOP,
    chess.ROOK: _ROOK,
    chess.QUEEN: _QUEEN,
}

# Phase weights: how much non-pawn material each piece contributes to "still a
# middlegame". Both sides' full complement sums to 24.
_PHASE = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
_PHASE_MAX = 24

BISHOP_PAIR_CP = 30


def game_phase(board: chess.Board) -> float:
    """1.0 with all the pieces on, 0.0 once only kings and pawns remain."""
    total = 0
    for piece_type, weight in _PHASE.items():
        total += weight * (
            len(board.pieces(piece_type, chess.WHITE))
            + len(board.pieces(piece_type, chess.BLACK))
        )
    return min(total, _PHASE_MAX) / _PHASE_MAX


def positional_centipawns(board: chess.Board) -> int:
    """Positional balance in centipawns, from the side to move's point of view.

    Deliberately not a full evaluation: piece-square tables, a phase-blended king
    table and the bishop pair. Mobility and pawn structure are the obvious next
    terms and are left out until this much is measured, because a term that is
    never A/B'd is a term nobody can remove later.
    """
    phase = game_phase(board)
    score = 0
    for piece_type, table in _TABLES.items():
        for square in board.pieces(piece_type, chess.WHITE):
            score += table[square]
        for square in board.pieces(piece_type, chess.BLACK):
            score -= table[chess.square_mirror(square)]

    for colour, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        king = board.king(colour)
        if king is not None:
            square = king if colour == chess.WHITE else chess.square_mirror(king)
            blended = phase * _KING_MIDDLE[square] + (1.0 - phase) * _KING_END[square]
            score += sign * blended
        if len(board.pieces(chess.BISHOP, colour)) >= 2:
            score += sign * BISHOP_PAIR_CP

    return int(score if board.turn == chess.WHITE else -score)
