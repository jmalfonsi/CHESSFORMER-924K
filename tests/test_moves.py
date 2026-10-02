import chess
import numpy as np

from chessformer.moves import (
    N_ACTIONS,
    action_uci,
    legal_action_mask,
    move_to_index,
    pack_legal_mask,
    unpack_legal_mask,
)


def test_action_vocab_is_exactly_1968_unique_actions():
    actions = action_uci()
    assert len(actions) == N_ACTIONS == 1968
    assert len(set(actions)) == 1968


def test_starting_position_has_20_legal_actions():
    board = chess.Board()
    mask = legal_action_mask(board)
    assert mask.sum() == 20
    for uci in ("e2e4", "g1f3", "b1c3"):
        assert mask[move_to_index(chess.Move.from_uci(uci))]


def test_all_four_promotions_have_distinct_actions():
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    mask = legal_action_mask(board)
    indices = [move_to_index(chess.Move.from_uci(f"a7a8{p}")) for p in "qrbn"]
    assert len(set(indices)) == 4
    assert all(mask[i] for i in indices)


def test_castling_actions_are_representable_and_legal():
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    mask = legal_action_mask(board)
    assert mask[move_to_index(chess.Move.from_uci("e1g1"))]
    assert mask[move_to_index(chess.Move.from_uci("e1c1"))]


def test_en_passant_is_representable():
    board = chess.Board("7k/8/8/3pP3/8/8/8/7K w - d6 0 1")
    move = chess.Move.from_uci("e5d6")
    assert board.is_en_passant(move)
    assert legal_action_mask(board)[move_to_index(move)]


def test_pack_roundtrip_is_246_bytes():
    mask = legal_action_mask(chess.Board())
    packed = pack_legal_mask(mask)
    assert packed.shape == (246,)
    assert np.array_equal(unpack_legal_mask(packed), mask)
