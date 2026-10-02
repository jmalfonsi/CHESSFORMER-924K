import chess

from chessformer.encoding import encode_board, make_training_record


def test_starting_encoding():
    board = chess.Board()
    x = encode_board(board)
    assert x.pieces.shape == (64,)
    assert x.side == 0
    assert x.castling == 15
    assert x.ep_file == 0
    assert x.halfmove == 0
    assert x.pieces[chess.A1] == 4  # white rook
    assert x.pieces[chess.E1] == 6  # white king
    assert x.pieces[chess.E8] == 12  # black king


def test_ep_and_halfmove_encoding():
    board = chess.Board("7k/8/8/3pP3/8/8/8/7K w - d6 42 30")
    x = encode_board(board)
    assert x.ep_file == 4  # d-file encoded as 1..8
    assert x.halfmove == 42


def test_training_record_target_is_legal():
    board = chess.Board()
    record = make_training_record(board, chess.Move.from_uci("e2e4"), value=0.25)
    assert record["pieces"].shape == (64,)
    assert record["legal"].shape == (246,)
