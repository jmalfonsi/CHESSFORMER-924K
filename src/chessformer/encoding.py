from __future__ import annotations

from dataclasses import dataclass

import chess
import numpy as np

from .moves import legal_action_mask, move_to_index, pack_legal_mask

# 0 empty; 1..6 white P/N/B/R/Q/K; 7..12 black P/N/B/R/Q/K.
PIECE_TO_TOKEN: dict[tuple[bool, int], int] = {
    (chess.WHITE, chess.PAWN): 1,
    (chess.WHITE, chess.KNIGHT): 2,
    (chess.WHITE, chess.BISHOP): 3,
    (chess.WHITE, chess.ROOK): 4,
    (chess.WHITE, chess.QUEEN): 5,
    (chess.WHITE, chess.KING): 6,
    (chess.BLACK, chess.PAWN): 7,
    (chess.BLACK, chess.KNIGHT): 8,
    (chess.BLACK, chess.BISHOP): 9,
    (chess.BLACK, chess.ROOK): 10,
    (chess.BLACK, chess.QUEEN): 11,
    (chess.BLACK, chess.KING): 12,
}


@dataclass(frozen=True)
class EncodedPosition:
    pieces: np.ndarray  # uint8[64], a1..h8
    side: np.uint8  # 0 white, 1 black
    castling: np.uint8  # K=1,Q=2,k=4,q=8
    ep_file: np.uint8  # 0 none, 1..8 a..h
    halfmove: np.uint8  # clipped 0..100


def encode_board(board: chess.Board) -> EncodedPosition:
    pieces = np.zeros(64, dtype=np.uint8)
    for square, piece in board.piece_map().items():
        pieces[square] = PIECE_TO_TOKEN[(piece.color, piece.piece_type)]

    castling = 0
    castling |= int(board.has_kingside_castling_rights(chess.WHITE)) * 1
    castling |= int(board.has_queenside_castling_rights(chess.WHITE)) * 2
    castling |= int(board.has_kingside_castling_rights(chess.BLACK)) * 4
    castling |= int(board.has_queenside_castling_rights(chess.BLACK)) * 8

    ep_file = 0 if board.ep_square is None else chess.square_file(board.ep_square) + 1

    return EncodedPosition(
        pieces=pieces,
        side=np.uint8(0 if board.turn == chess.WHITE else 1),
        castling=np.uint8(castling),
        ep_file=np.uint8(ep_file),
        halfmove=np.uint8(min(max(board.halfmove_clock, 0), 100)),
    )


def encode_fen(fen: str) -> EncodedPosition:
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"Invalid standard chess position: {fen}")
    return encode_board(board)


# Analysed moves stored per record. Matches lichess_eval.MULTIPV_WIDTH; kept
# here as a literal so encoding does not have to import the parser.
MULTIPV_WIDTH = 5


def make_training_record(
    board: chess.Board,
    target_move: chess.Move,
    value: float,
    analysed: "list[chess.Move] | tuple[chess.Move, ...]" = (),
    losses_cp: "list[float] | tuple[float, ...]" = (),
    action_values: "list[float] | tuple[float, ...]" = (),
) -> dict[str, np.ndarray | np.uint8 | np.uint16 | np.float16]:
    """One fixed-width training row.

    `analysed` and `losses_cp` are the multi-PV supervision: every move the
    export scored, best first, with what each drops against the best from the
    side to move's point of view. Empty keeps the V1 record exactly as it was,
    so a dataset prepared without them still loads and still trains.
    """
    if target_move not in board.legal_moves:
        raise ValueError(f"Target move {target_move.uci()} is not legal in {board.fen()}")
    if not -1.0 <= value <= 1.0:
        raise ValueError("Value target must be in [-1, 1]")

    encoded = encode_board(board)
    legal = legal_action_mask(board)
    target = move_to_index(target_move)
    if not legal[target]:
        raise AssertionError("Target action missing from legal mask")

    record = {
        "pieces": encoded.pieces,
        "side": encoded.side,
        "castling": encoded.castling,
        "ep_file": encoded.ep_file,
        "halfmove": encoded.halfmove,
        "policy": np.uint16(target),
        "value": np.float16(value),
        "legal": pack_legal_mask(legal),
    }
    if action_values and not analysed:
        raise ValueError("action_values require analysed candidate moves")
    if not analysed:
        return record

    if len(analysed) != len(losses_cp):
        raise ValueError("analysed and losses_cp must have the same length")
    if action_values and len(analysed) != len(action_values):
        raise ValueError("analysed and action_values must have the same length")
    if analysed[0] != target_move:
        raise ValueError("The first analysed move must be the policy target")
    kept = list(zip(analysed, losses_cp))[:MULTIPV_WIDTH]
    # Padding repeats the target rather than leaving a zero, so that a consumer
    # that forgets to honour policy_count gathers a legal action instead of
    # silently teaching the model that action 0 is good everywhere.
    indices = np.full(MULTIPV_WIDTH, target, dtype=np.uint16)
    penalties = np.zeros(MULTIPV_WIDTH, dtype=np.float16)
    q_values = np.zeros(MULTIPV_WIDTH, dtype=np.float16) if action_values else None
    for slot, (move, loss) in enumerate(kept):
        if move not in board.legal_moves:
            raise ValueError(f"Analysed move {move.uci()} is not legal in {board.fen()}")
        indices[slot] = np.uint16(move_to_index(move))
        penalties[slot] = np.float16(max(0.0, float(loss)))
        if q_values is not None:
            q_value = float(action_values[slot])
            if not -1.0 <= q_value <= 1.0:
                raise ValueError("Action-value targets must be in [-1, 1]")
            q_values[slot] = np.float16(q_value)
    record["policy_multi"] = indices
    record["policy_loss_cp"] = penalties
    record["policy_count"] = np.uint8(len(kept))
    if q_values is not None:
        record["action_value"] = q_values
    return record
