import chess
import pytest
import torch

from chessformer.lichess import GameSession, board_from_moves
from chessformer.lichess import handle_event
from chessformer.mini.lichess import ACCOUNT, MiniEngine, check_account, mini_account, mini_should_accept, mini_token
from chessformer.mini.model import CANDIDATES, MiniChess
from chessformer.mini.player import MiniPlayer
from chessformer.moves import move_to_index


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return MiniChess(CANDIDATES["geometric"]).eval()


class FakeClient:
    def __init__(self):
        self.posts = []

    def post(self, path, data=None):
        self.posts.append(path)
        return {}


def test_zero_temperature_is_the_strict_arg_max(model):
    board = chess.Board()
    assert MiniPlayer(model, temperature=0.0, seed=1).choose_move(board) == MiniPlayer(model).choose_move(board)


def test_temperature_samples_legal_moves_reproducibly(model):
    board = board_from_moves(["e2e4", "e7e5"])
    first = [MiniPlayer(model, temperature=5.0, seed=7).choose_move(board) for _ in range(3)]
    player = MiniPlayer(model, temperature=5.0, seed=7)
    draws = [player.choose_move(board) for _ in range(60)]
    assert all(move in board.legal_moves for move in draws)
    assert len(set(draws)) > 1
    assert len(set(first)) == 1
    with pytest.raises(ValueError):
        MiniPlayer(model, temperature=-0.1)


def test_sampling_can_be_confined_to_the_opening(model):
    opening = board_from_moves(["e2e4", "e7e5"])
    late = board_from_moves(["e2e4", "e7e5", "g1f3", "b8c6"])
    player = MiniPlayer(model, temperature=5.0, seed=3, sample_plies=4)
    assert len({player.choose_move(opening) for _ in range(40)}) > 1
    assert {player.choose_move(late) for _ in range(20)} == {MiniPlayer(model).choose_move(late)}
    with pytest.raises(ValueError):
        MiniPlayer(model, sample_plies=-1)


def test_mini_engine_plays_through_the_lichess_session(model):
    client = FakeClient()
    session = GameSession(client, MiniEngine(MiniPlayer(model)), "g1", "me", log=lambda _: None)
    session.playing_white = False
    session.handle_state({"moves": "e2e4", "wtime": 300_000, "btime": 5_000})
    uci = client.posts[0].rsplit("/", 1)[1]
    assert chess.Move.from_uci(uci) in board_from_moves(["e2e4"]).legal_moves
    assert session.engine.mode == "policy" and session.engine.forced_nodes == 0


def test_the_143m_token_is_never_used(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LICHESS_TOKEN=big-bot-token\n")
    monkeypatch.setenv("LICHESS_TOKEN", "big-bot-token")
    monkeypatch.delenv("LICHESS_TOKEN_MINI", raising=False)
    with pytest.raises(SystemExit, match="LICHESS_TOKEN_MINI"):
        mini_token(path=env)
    env.write_text("LICHESS_TOKEN=big-bot-token\nLICHESS_TOKEN_MINI=mini-token\n")
    assert mini_token(path=env) == "mini-token"


def test_custom_account_configuration_and_overrides(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("LICHESS_ACCOUNT_MINI=MyOwnBot\nLICHESS_TOKEN_MINI=my-own-token\n")
    monkeypatch.delenv("LICHESS_ACCOUNT_MINI", raising=False)
    monkeypatch.delenv("LICHESS_TOKEN_MINI", raising=False)
    assert mini_account(path=env) == "MyOwnBot"
    assert mini_token(path=env) == "my-own-token"
    monkeypatch.setenv("LICHESS_ACCOUNT_MINI", "ShellBot")
    monkeypatch.setenv("LICHESS_TOKEN_MINI", "shell-token")
    assert mini_account(path=env) == "ShellBot"
    assert mini_token(path=env) == "shell-token"
    assert mini_account("ExplicitBot", path=env) == "ExplicitBot"
    assert mini_token("explicit-token", path=env) == "explicit-token"


def test_original_account_configuration_remains_usable(tmp_path, monkeypatch):
    monkeypatch.delenv("LICHESS_ACCOUNT_MINI", raising=False)
    assert mini_account(path=tmp_path / "absent") == ACCOUNT


@pytest.mark.parametrize("username,title,allowed", [
    ("MyOwnBot", "BOT", True),
    ("myownbot", "BOT", True),
    ("SomeOtherBot", "BOT", False),
    ("MyOwnBot", None, False),
])
def test_bot_cli_validates_custom_identity_before_playing(monkeypatch, username, title, allowed):
    from chessformer.mini import lichess

    calls = []
    class Client:
        def __init__(self, token):
            assert token == "fake-own-token"

        def get(self, path):
            assert path == "/api/account"
            return {"username": username, "title": title}

    # Use the released weights and real MiniPlayer; only the API is simulated.
    monkeypatch.setenv("LICHESS_ACCOUNT_MINI", "MyOwnBot")
    monkeypatch.setenv("LICHESS_TOKEN_MINI", "fake-own-token")
    monkeypatch.setattr(lichess, "LichessClient", Client)
    monkeypatch.setattr(lichess, "run_bot", lambda *args, **kwargs: calls.append((args, kwargs)))
    from pathlib import Path
    checkpoint = Path(__file__).resolve().parents[1] / "models/chessformer-924k-v1.pt"
    argv = ["--checkpoint", str(checkpoint), "--temperature", "0", "--sample-plies", "20",
            "--avoid-repetition", "0", "--max-games", "5"]
    if not allowed:
        with pytest.raises(SystemExit, match="refusing to play|not a BOT"):
            lichess.main(argv)
        assert not calls
    else:
        lichess.main(argv)
        assert len(calls) == 1
        args, kwargs = calls[0]
        player = args[1].player
        assert player.model.num_parameters() == 924164
        assert player.temperature == 0 and player.repetition_value == 0
        assert kwargs["max_games"] == 5


@pytest.mark.parametrize("speed, variant, expected", [
    ("bullet", "standard", True), ("blitz", "standard", True), ("rapid", "standard", True),
    ("ultraBullet", "standard", False), ("bullet", "atomic", False)])
def test_the_mini_bot_accepts_bullet_but_not_ultrabullet(speed, variant, expected):
    assert mini_should_accept({"speed": speed, "variant": {"key": variant}})[0] is expected


def test_the_acceptance_rule_reaches_the_event_handler(model):
    client = FakeClient()
    event = {"type": "challenge", "challenge": {"id": "c1", "speed": "bullet", "variant": {"key": "standard"},
                                                "challenger": {"id": "other", "name": "Other"}}}
    handle_event(client, MiniEngine(MiniPlayer(model)), event, "me", log=lambda _: None,
                 engine_lock=None, acceptance=mini_should_accept)
    handle_event(client, MiniEngine(MiniPlayer(model)), dict(event, challenge=dict(event["challenge"], id="c2")),
                 "me", log=lambda _: None, engine_lock=None)
    assert client.posts == ["/api/challenge/c1/accept", "/api/challenge/c2/decline"]


def test_account_must_be_the_mini_bot():
    check_account({"username": "Chessformer-924K", "title": "BOT"}, "Chessformer-924K")
    with pytest.raises(SystemExit, match="not Chessformer-924K"):
        check_account({"username": "CHESSFORMER-143M", "title": "BOT"}, "Chessformer-924K")
    with pytest.raises(SystemExit, match="not a BOT"):
        check_account({"username": "Chessformer-924K"}, "Chessformer-924K")


class ScriptedModel(torch.nn.Module):
    """Ranks the given moves first and reports a fixed side-to-move value."""

    def __init__(self, preferences, value):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.register_buffer("value_support", torch.linspace(-1, 1, 64))
        self.preferences = [move_to_index(chess.Move.from_uci(uci)) for uci in preferences]
        self.value = value

    def forward(self, pieces, castling, ep_file):
        logits = torch.zeros(1, 1968)
        for rank, index in enumerate(self.preferences):
            logits[0, index] = 10.0 - rank
        value = torch.full((1, 64), -30.0)
        value[0, round((self.value + 1) / 2 * 63)] = 30.0
        return logits, value


def rook_shuffle():
    # White to move; b2a2 would bring back the position after the first Ra2 a
    # third time, while every position so far has occurred at most twice.
    board = chess.Board("7k/8/8/8/8/8/8/R5K1 w - - 0 1")
    for san in ["Ra2", "Kg8", "Rb2", "Kh8", "Ra2", "Kg8", "Rb2", "Kh8"]:
        board.push_san(san)
    return board


def test_a_third_repetition_is_withheld_only_while_the_value_head_is_ahead():
    board = rook_shuffle()
    history = list(board.move_stack)
    repeat, other = chess.Move.from_uci("b2a2"), chess.Move.from_uci("b2c2")
    ahead = MiniPlayer(ScriptedModel(["b2a2", "b2c2"], 0.8), repetition_value=0.0)
    assert ahead.choose_move(board) == other
    assert ahead.repetitions_withheld == 1
    assert list(board.move_stack) == history
    behind = MiniPlayer(ScriptedModel(["b2a2", "b2c2"], -0.8), repetition_value=0.0)
    assert behind.choose_move(board) == repeat
    assert behind.repetitions_withheld == 0
    strict = MiniPlayer(ScriptedModel(["b2a2", "b2c2"], 0.8))
    assert strict.choose_move(board) == repeat


def test_the_rule_does_not_count_a_repetition_the_network_did_not_want():
    player = MiniPlayer(ScriptedModel(["b2c2"], 0.8), repetition_value=0.0)
    assert player.choose_move(rook_shuffle()) == chess.Move.from_uci("b2c2")
    assert player.repetitions_withheld == 0
