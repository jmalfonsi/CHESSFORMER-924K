"""Issue rated challenges to a ladder of bots discovered live, one at a time.

The opponents used to be a hardcoded list with their published ratings written
in beside them. Two things killed that:

  * the ratings went stale. Measured on 2026-08-16, the constants in the source
    were off by -84 (maia1) to +56 (bernstein-4ply), and the file carried a
    comment claiming maia1's 429,000 rated games made its rating immovable. It
    had moved 84 points. A number copied into a source file has no way to say
    it is out of date;
  * the bot outgrew the list. At blitz 1522 every one of the eleven entries was
    below it, the strongest by 50 points. Beating an opponent 250 points down
    gains almost nothing and a single slip costs a great deal, so the ladder had
    stopped discriminating.

So the opponents are now discovered from `/api/bot/online` at the start of every
round, filtered to established accounts inside a rating window, and spread evenly
across it. Discovering per round also means the list reflects who is actually
online: challenging an offline bot only burns the accept timeout.

Only *established* accounts with a real game count are kept. Measuring against a
provisional opponent is a random walk -- both ratings chase each other and
neither ends up meaning anything.

    python tools/challenge_ladder.py --rounds 2 --min-rating 1500

Every round also opens with its *duels*: named opponents challenged whatever
discovery says, one game per round, colours alternating round by round so a
multi-round run is not colour-skewed (White is worth roughly 35 Elo in bot
games). RDTChessBot is the first of them -- another network trained from
scratch rather than a search engine, which is the comparison worth having, and
one discovery cannot reach: measured on 2026-09-05 its blitz rating is 1496,
below `--min-rating`, on 201 rated games, below `--min-games`. Both filters are
right for a rating anchor and wrong for a fixed matchup, so the duels bypass
them instead of loosening them. `--duel` with no names plays none.

A duel is a matchup, not a measurement: at 240 points down it moves our rating
by almost nothing whatever happens. The discovered window is what grades the
model; the duels say who we are actually better than.

Challenges go out one at a time and the next only once the previous game is
over, so concurrent games never share the 8 cores and halve each other's move
budget.
"""

from __future__ import annotations

import argparse
import time
import urllib.error

from chessformer.lichess import LichessClient, _log
from chessformer.mini.lichess import check_account, mini_account, mini_token

# Opponents played every round on top of whatever discovery finds.
DUELS = ("RDTChessBot","philidor-142M")


def _retry(call, *, attempts: int = 5, delay: int = 10, default=None):
    """Lichess returns a transient 502 often enough to kill an unguarded loop.

    One such 502 ended a ladder run mid-session. A read failing is never a
    reason to stop playing, so retry and fall back rather than propagate.
    """
    for attempt in range(attempts):
        try:
            return call()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            _log(f"  transient API error ({error}); retry {attempt + 1}/{attempts}")
            time.sleep(delay)
    return default


def account(client: LichessClient) -> dict:
    return _retry(lambda: client.get("/api/account"), default={}) or {}


def perf_of(user: dict, perf: str) -> dict:
    return (user.get("perfs") or {}).get(perf) or {}


def blitz_of(user: dict) -> dict:
    return perf_of(user, "blitz")


def provisional(blitz: dict) -> bool:
    """Lichess *omits* `prov` once a rating is established, it does not send false.

    Defaulting the missing key to True -- which the first version of this file
    did -- rejects every established opponent, so discovery returned an empty
    ladder, and logged our own 64-game rating as provisional besides.
    """
    return bool(blitz.get("prov", False))


def our_rating(client: LichessClient, perf: str = "blitz") -> tuple[int, bool, int]:
    stats = perf_of(account(client), perf)
    return stats.get("rating", 0), provisional(stats), stats.get("games", 0)


def playing_now(client: LichessClient) -> int:
    payload = _retry(lambda: client.get("/api/account/playing"))
    if payload is None:
        # Unknown is not "idle": claiming idle here would fire a second
        # challenge on top of a game that is still running.
        return 1
    return len(payload.get("nowPlaying", []))


def spread(candidates: list[tuple[int, str, int]], size: int) -> list[tuple[int, str, int]]:
    """Pick `size` opponents spaced evenly across a rating-sorted list.

    Taking the first `size` would cluster every game at the bottom of the
    window, which is the failure the old list ended in. Sampling by index keeps
    both ends and the middle, so one run brackets the whole range.
    """
    if size <= 0 or len(candidates) <= size:
        return candidates
    last = len(candidates) - 1
    picked: dict[str, tuple[int, str, int]] = {}
    for index in range(size):
        entry = candidates[round(index * last / (size - 1))] if size > 1 else candidates[0]
        picked[entry[1]] = entry
    return sorted(picked.values())


def discover(
    client: LichessClient,
    *,
    min_rating: int,
    max_rating: int,
    min_games: int,
    pool: int,
    size: int,
    exclude: set[str],
    perf: str = "blitz",
) -> list[tuple[int, str, int]]:
    """Online bots inside the window, established, spread across the range.

    The window, the game count and the provisional flag are all read in `perf`,
    the pool the games will be rated in: a bot's blitz rating says little about
    its bullet one.
    """
    lowered = {name.lower() for name in exclude}
    candidates: list[tuple[int, str, int]] = []
    stream = _retry(
        lambda: list(client.stream(f"/api/bot/online?nb={pool}")), default=[]
    )
    for user in stream or []:
        name = user.get("username", "")
        if not name or name.lower() in lowered:
            continue
        # A closed or sanctioned account still appears in the stream and will
        # refuse every challenge, so drop it before it costs an accept timeout.
        if user.get("disabled") or user.get("tosViolation"):
            continue
        stats = perf_of(user, perf)
        rating, games = stats.get("rating", 0), stats.get("games", 0)
        if provisional(stats) or games < min_games:
            continue
        if min_rating <= rating <= max_rating:
            candidates.append((rating, name, games))
    return spread(sorted(candidates), size)



def duel_entries(
    client: LichessClient, names: list[str], *, refused: set[str], perf: str = "blitz"
) -> list[tuple[int, str]]:
    """`(rating, name)` for each duel opponent that has not already said no.

    The rating is only for the log, so a failed lookup yields 0 and the duel is
    played anyway: a fixed matchup that skipped itself because `/api/user` gave
    a 502 would be the one part of the round that discovery cannot replace.
    """
    skip = {name.lower() for name in refused}
    entries: list[tuple[int, str]] = []
    for name in names:
        if not name or name.lower() in skip:
            continue
        user = _retry(lambda name=name: client.get(f"/api/user/{name}"), default={}) or {}
        entries.append((perf_of(user, perf).get("rating", 0), name))
    return entries


def duel_colour(round_index: int) -> str:
    """Alternate rather than draw at random.

    Over a handful of rounds a random draw hands out lopsided colours, and White
    is worth roughly 35 Elo in bot games -- more than a short series against one
    opponent can resolve at all. The discovered ladder keeps `random`: it is
    measured by the rating, which already accounts for colour.
    """
    return "white" if round_index % 2 == 0 else "black"


def schedule(
    duels: list[tuple[int, str]], ladder: list[tuple[int, str, int]], colour: str
) -> list[tuple[int, str, str]]:
    """The round's opponents in play order: duels first, then the window.

    Duels lead because they are the fixed part of the round -- a run stopped
    early, by a daily cap or by the operator, has then played the matchup it was
    asked to play.
    """
    return [(rating, name, colour) for rating, name in duels] + [
        (rating, name, "random") for rating, name, _ in ladder
    ]


def play(client: LichessClient, username: str, rating: int, colour: str, args) -> str:
    """Challenge one opponent and return once the game it started is over.

    "played", "refused" (they said no; a bot at its 100-games-a-day bot-vs-bot
    cap will keep saying no all day) or "missed" (never accepted in time).
    """
    while playing_now(client) > 0:
        time.sleep(args.settle_s)

    try:
        response = client.post(
            f"/api/challenge/{username}",
            {
                "rated": args.rated,
                "clock.limit": args.clock_limit,
                "clock.increment": args.clock_increment,
                "variant": "standard",
                "color": colour,
            },
        )
    except RuntimeError as error:
        _log(f"  {username} ({rating}) refused the challenge: {error}")
        return "refused"

    _log(f"  challenged {username} ({rating}) as {colour} -> {response.get('id', '?')}")

    # Wait for the game to *appear* before waiting for it to finish. The first
    # version tested `playing_now() > 0` immediately after issuing the
    # challenge, saw the zero of a game that had not started yet, concluded the
    # game was over and fired the next challenge on top of it. Nothing ever got
    # played.
    started = False
    for _ in range(max(1, args.accept_timeout_s // 5)):
        time.sleep(5)
        if playing_now(client) > 0:
            started = True
            break
    if not started:
        _log(f"    {username} never accepted within {args.accept_timeout_s}s, moving on")
        return "missed"

    while playing_now(client) > 0:
        time.sleep(args.settle_s)

    current, prov, games = our_rating(client, args.perf)
    _log(f"    now {args.perf} {current} ({'prov' if prov else 'established'}, {games} games)")
    return "played"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--clock-limit", type=int, default=300)
    parser.add_argument("--clock-increment", type=int, default=3)
    parser.add_argument(
        "--perf",
        choices=["bullet", "blitz", "rapid", "classical"],
        default="blitz",
        help="rating pool used to pick opponents and report ours; match it to the clock",
    )
    parser.add_argument("--rated", default="true")
    parser.add_argument("--settle-s", type=int, default=15)
    parser.add_argument("--accept-timeout-s", type=int, default=60)
    parser.add_argument("--min-rating", type=int, default=1500)
    parser.add_argument("--max-rating", type=int, default=1750)
    parser.add_argument(
        "--min-games",
        type=int,
        default=300,
        help="an opponent with fewer rated games is too loose an anchor to measure against",
    )
    parser.add_argument("--pool", type=int, default=300, help="online bots to consider")
    parser.add_argument(
        "--size", type=int, default=10, help="opponents per round; 0 plays the whole window"
    )
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument(
        "--duel",
        nargs="*",
        default=list(DUELS),
        metavar="USERNAME",
        help="opponents played every round whatever discovery says; --duel alone plays none",
    )
    parser.add_argument(
        "--account",
        default="",
        help="Expected BOT username; defaults to LICHESS_ACCOUNT_MINI in environment/.env",
    )
    parser.add_argument(
        "--token",
        default="",
        help="Lichess token; defaults to LICHESS_TOKEN_MINI, then this project's .env",
    )
    args = parser.parse_args(argv)

    client = LichessClient(mini_token(args.token))
    me = account(client)
    check_account(me, mini_account(args.account))
    stats = perf_of(me, args.perf)
    _log(
        f"account={me['username']} starting at {args.perf} {stats.get('rating', 0)} "
        f"({'provisional' if provisional(stats) else 'established'}, "
        f"{stats.get('games', 0)} rated games)"
    )

    # Our own account is in the online-bot stream while the bot is running, and
    # a self-challenge is a 400 every round. Exclude it up front.
    exclude = {me.get("username", ""), *args.exclude}
    # A bot that has hit its 100-games-per-day bot-vs-bot cap refuses with a 400
    # and will keep refusing all day; remembering it stops later rounds from
    # rediscovering it and paying for the same refusal again.
    refused: set[str] = set()

    for round_index in range(args.rounds):
        colour = duel_colour(round_index)
        duels = duel_entries(client, args.duel, refused=refused, perf=args.perf)
        ladder = discover(
            client,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
            min_games=args.min_games,
            pool=args.pool,
            size=args.size,
            perf=args.perf,
            # A duel opponent inside the window would otherwise be played twice
            # in one round, once at a fixed colour and once at random.
            exclude=exclude | refused | {name for _, name in duels},
        )
        if not ladder:
            _log(
                f"round {round_index + 1}: no established bot online between "
                f"{args.min_rating} and {args.max_rating}; widen the window"
            )
            # A duel does not need the window, so an empty one only ends the run
            # when there is nothing else left to play either.
            if not duels:
                break
        else:
            _log(
                f"round {round_index + 1}: {len(ladder)} opponents "
                f"({ladder[0][0]}-{ladder[-1][0]}): "
                + ", ".join(f"{name} {rating}" for rating, name, _ in ladder)
            )
        if duels:
            _log(
                f"round {round_index + 1}: duels as {colour}: "
                + ", ".join(f"{name} {rating}" for rating, name in duels)
            )

        for rating, username, seat in schedule(duels, ladder, colour):
            if play(client, username, rating, seat, args) == "refused":
                refused.add(username)

    current, prov, games = our_rating(client, args.perf)
    _log(f"final: {args.perf} {current} ({'provisional' if prov else 'established'}, {games} rated games)")


if __name__ == "__main__":
    main()
