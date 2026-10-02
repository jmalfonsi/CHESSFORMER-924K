from __future__ import annotations

from functools import lru_cache

import chess
import numpy as np

N_ACTIONS = 1968
PROMOTIONS = ("q", "r", "b", "n")


def _square_name(square: int) -> str:
    return chess.square_name(square)


def _base_geometries() -> list[str]:
    """All directed queen-like and knight from->to geometries on an 8x8 board.

    Pawns, kings, rooks and bishops are subsets of these geometries. Castling is
    represented by the normal king UCI move (e1g1/e1c1/e8g8/e8c8).
    """
    actions: list[str] = []
    queen_dirs = ((1, 0), (-1, 0), (0, 1), (0, -1),
                  (1, 1), (1, -1), (-1, 1), (-1, -1))
    knight_dirs = ((1, 2), (2, 1), (-1, 2), (-2, 1),
                   (1, -2), (2, -1), (-1, -2), (-2, -1))

    for from_sq in chess.SQUARES:
        f = chess.square_file(from_sq)
        r = chess.square_rank(from_sq)
        destinations: set[int] = set()

        for df, dr in queen_dirs:
            nf, nr = f + df, r + dr
            while 0 <= nf < 8 and 0 <= nr < 8:
                destinations.add(chess.square(nf, nr))
                nf += df
                nr += dr

        for df, dr in knight_dirs:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                destinations.add(chess.square(nf, nr))

        for to_sq in sorted(destinations):
            actions.append(_square_name(from_sq) + _square_name(to_sq))

    return actions


def _promotion_actions() -> list[str]:
    """Explicit UCI promotion actions for both colors: 44 geometries * 4 pieces."""
    actions: list[str] = []
    # White: rank 7 -> rank 8. Black: rank 2 -> rank 1.
    for from_rank, to_rank in ((6, 7), (1, 0)):
        for from_file in range(8):
            from_sq = chess.square(from_file, from_rank)
            for df in (-1, 0, 1):
                to_file = from_file + df
                if not 0 <= to_file < 8:
                    continue
                to_sq = chess.square(to_file, to_rank)
                prefix = _square_name(from_sq) + _square_name(to_sq)
                for promo in PROMOTIONS:
                    actions.append(prefix + promo)
    return actions


@lru_cache(maxsize=1)
def action_uci() -> tuple[str, ...]:
    actions = _base_geometries() + _promotion_actions()
    if len(_base_geometries()) != 1792:
        raise AssertionError("Expected 1792 base geometries")
    if len(actions) != N_ACTIONS:
        raise AssertionError(f"Expected {N_ACTIONS} actions, got {len(actions)}")
    if len(set(actions)) != len(actions):
        raise AssertionError("Action vocabulary contains duplicates")
    return tuple(actions)


@lru_cache(maxsize=1)
def uci_to_index() -> dict[str, int]:
    return {uci: i for i, uci in enumerate(action_uci())}


def move_to_index(move: chess.Move) -> int:
    try:
        return uci_to_index()[move.uci()]
    except KeyError as exc:
        raise ValueError(f"Move is outside the standard V1 action vocabulary: {move.uci()}") from exc


def index_to_uci(index: int) -> str:
    if not 0 <= index < N_ACTIONS:
        raise IndexError(index)
    return action_uci()[index]


def index_to_move(index: int) -> chess.Move:
    return chess.Move.from_uci(index_to_uci(index))


def legal_action_indices(board: chess.Board) -> np.ndarray:
    indices = [move_to_index(move) for move in board.legal_moves]
    return np.asarray(indices, dtype=np.int32)


def legal_action_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(N_ACTIONS, dtype=np.bool_)
    indices = legal_action_indices(board)
    mask[indices] = True
    return mask


def pack_legal_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != (N_ACTIONS,):
        raise ValueError(f"Expected legal mask shape ({N_ACTIONS},), got {mask.shape}")
    packed = np.packbits(mask, bitorder="little")
    if packed.shape != (N_ACTIONS // 8,):
        raise AssertionError("1968 must pack to exactly 246 bytes")
    return packed


def unpack_legal_mask(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.shape[-1] != N_ACTIONS // 8:
        raise ValueError(f"Expected packed legal mask width 246, got {packed.shape[-1]}")
    return np.unpackbits(packed, axis=-1, count=N_ACTIONS, bitorder="little").astype(np.bool_)


# Promotion pieces in the order the action vocabulary lists them.
PROMOTION_IDS = {"q": 1, "r": 2, "b": 3, "n": 4}


@lru_cache(maxsize=1)
def action_squares() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Origin square, destination square and promotion id of every action.

    What an attention policy head needs: it produces a 64x64 matrix of from-to
    logits and has to scatter it into the 1968-action vector, which is a gather
    through these indices.
    """
    origins = np.empty(N_ACTIONS, dtype=np.int64)
    destinations = np.empty(N_ACTIONS, dtype=np.int64)
    promotions = np.zeros(N_ACTIONS, dtype=np.int64)
    for index, uci in enumerate(action_uci()):
        move = chess.Move.from_uci(uci)
        origins[index] = move.from_square
        destinations[index] = move.to_square
        if len(uci) == 5:
            promotions[index] = PROMOTION_IDS[uci[4]]
    return origins, destinations, promotions
