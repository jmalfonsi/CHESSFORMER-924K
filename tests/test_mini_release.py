"""Reference decisions recorded from the current champion before exporting it."""
import hashlib
import json
from pathlib import Path

import chess
import pytest
import torch

from chessformer.mini.player import MiniPlayer, load_mini

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/chessformer-924k-v1.pt"


@pytest.fixture(scope="module")
def released_player():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield MiniPlayer(load_mini(CHECKPOINT), temperature=0, sample_plies=20, repetition_value=0)
    torch.set_num_threads(previous)


def test_released_weights_match_the_manifest_and_need_no_optimizer(released_player):
    manifest = json.loads((ROOT / "models/manifest.json").read_text())
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == manifest["sha256"]
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    assert set(payload) == {"format", "input_format", "config", "model"}
    assert payload["config"] == manifest["config"]
    assert released_player.model.num_parameters() == manifest["parameters"] == 924164


@pytest.mark.parametrize("fen,move", [
    (chess.STARTING_FEN, "d2d4"),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "d7d5"),
    ("r3k2r/ppp2ppp/2npbn2/3Np3/2B1P3/2N2Q2/PPP2PPP/R3K2R w KQkq - 0 1", "d5c7"),
    ("4k3/1P6/8/8/8/8/6p1/4K3 w - - 0 1", "b7b8q"),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
])
def test_released_checkpoint_keeps_current_decisions(released_player, fen, move):
    board = chess.Board(fen)
    chosen = released_player.choose_move(board)
    assert chosen in board.legal_moves
    assert chosen.uci() == move
