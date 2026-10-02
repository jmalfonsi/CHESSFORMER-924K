"""The ladder must challenge from the same custom account as the mini bot."""
import pytest

from tools import challenge_ladder as ladder


@pytest.mark.parametrize("username,title,allowed", [
    ("MyOwnBot", "BOT", True),
    ("OtherBot", "BOT", False),
    ("MyOwnBot", None, False),
])
def test_ladder_validates_custom_identity_and_excludes_itself(monkeypatch, username, title, allowed):
    challenges = []
    class Client:
        def __init__(self, token):
            assert token == "fake-own-token"

        def get(self, path):
            assert path == "/api/account"
            return {"username": username, "title": title,
                    "perfs": {"blitz": {"rating": 1600, "games": 500}}}

        def stream(self, path):
            assert path.startswith("/api/bot/online?")
            return iter([
                {"username": "myownbot", "perfs": {"blitz": {"rating": 1600, "games": 500}}},
                {"username": "Opponent", "perfs": {"blitz": {"rating": 1700, "games": 500}}},
            ])

    monkeypatch.setenv("LICHESS_ACCOUNT_MINI", "MyOwnBot")
    monkeypatch.setenv("LICHESS_TOKEN_MINI", "fake-own-token")
    monkeypatch.setattr(ladder, "LichessClient", Client)
    def play(client, name, rating, colour, args):
        challenges.append((name, rating, colour, args.clock_limit, args.clock_increment))
        return "played"
    monkeypatch.setattr(ladder, "play", play)
    argv = ["--rounds", "1", "--size", "3", "--min-rating", "1400", "--max-rating", "2100",
            "--clock-limit", "180", "--clock-increment", "2", "--duel"]
    if allowed:
        ladder.main(argv)
        assert challenges == [("Opponent", 1700, "random", 180, 2)]
    else:
        with pytest.raises(SystemExit, match="refusing to play|not a BOT"):
            ladder.main(argv)
        assert not challenges
