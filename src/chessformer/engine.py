"""Move selection from a trained ChessFormer checkpoint.

The policy head alone is a move-ordering prior, not an engine. This module adds
legal masking and an optional one-ply value search so the network can actually be
played -- and therefore measured in games rather than by top-1 agreement.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

import chess
import numpy as np
import torch

from .canonical import action_permutation, canonical_board
from .config import PRESETS, ChessFormerConfig
from .encoding import encode_board
from .evaluation import positional_centipawns
from .losses import mask_policy_logits
from .model import ChessFormer
from .moves import legal_action_mask, move_to_index
from .book import OpeningBook
from .tablebase import TB_WIN_CP, EndgameTablebase

# Stage 2 mapped a Stockfish centipawn score to a value target with tanh(cp/600).
# Inverting it turns the bounded value head back into a UCI-reportable score.
VALUE_CP_SCALE = 600.0
MATE_VALUE = 1.0
# Floor on the uniform draw behind the Gumbel noise: log(0) is -inf and would
# make one move's sort key not-a-number.
_GUMBEL_EPS = 1e-20


@dataclass(frozen=True)
class SearchResult:
    move: chess.Move
    value: float
    """Position score for the side to move, in [-1, 1]."""
    nodes: int
    pv: list[chess.Move]

    def score_cp(self) -> int:
        clamped = min(max(self.value, -0.999999), 0.999999)
        return int(round(VALUE_CP_SCALE * math.atanh(clamped)))


def _infer_preset(payload: dict) -> str:
    args = payload.get("args") or {}
    preset = args.get("preset")
    if preset in PRESETS:
        return preset
    raise ValueError(
        "Checkpoint does not record a known preset; pass preset= explicitly."
    )


def load_model(
    checkpoint: str | Path,
    *,
    preset: str | None = None,
    device: torch.device | str = "cpu",
) -> tuple[ChessFormer, ChessFormerConfig, dict]:
    device = torch.device(device)
    payload = torch.load(Path(checkpoint), map_location=device, weights_only=False)
    name = preset or _infer_preset(payload)
    config = PRESETS[name]
    # A preset names the transformer, not the heads. `value_bins` changes the
    # shape of the last layer, so a checkpoint that recorded it has to be
    # rebuilt with it or load_state_dict fails on a size mismatch.
    args = payload.get("args") or {}
    value_bins = int(args.get("value_bins", 0)) if isinstance(args, dict) else 0
    if value_bins != config.value_bins:
        config = replace(config, value_bins=value_bins)
    # Same reasoning for the policy head: "attention" replaces the 1968-way
    # Linear with a query/key pair over the board, so a checkpoint trained with
    # it has different parameters under a preset name that says nothing about it.
    policy_head = str(args.get("policy_head", "linear")) if isinstance(args, dict) else "linear"
    if policy_head != config.policy_head:
        config = replace(config, policy_head=policy_head)
    action_value_bins = (
        int(args.get("action_value_bins", 0)) if isinstance(args, dict) else 0
    )
    if action_value_bins != config.action_value_bins:
        config = replace(config, action_value_bins=action_value_bins)
    model = ChessFormer(config)
    state = payload["model"]
    # Checkpoints are always saved unwrapped, but tolerate a compiled prefix.
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config, payload


def checkpoint_frame(payload: dict) -> dict[str, object]:
    """How a checkpoint wants its board encoded and its value read.

    Every checkpoint trained before canonicalisation records nothing here, and
    every one of them was trained on absolute coordinates against a White-relative
    label -- which is exactly what the defaults say. Guessing wrong is silent:
    reading a side-to-move value head as White-relative negates every evaluation
    Black makes, and the engine would go on printing plausible scores. This
    project has already paid for one silent degradation of that shape, when
    `tablebase.DEFAULT_PATH` was relative and a run from /tmp played without
    tables while announcing a normal score.
    """
    args = payload.get("args") or {}
    canonical = bool(args.get("canonical", False)) if isinstance(args, dict) else False
    return {
        "canonical": canonical,
        "value_pov": "side_to_move" if canonical else "white",
    }


def load_engine(
    checkpoint: str | Path,
    *,
    preset: str | None = None,
    device: torch.device | str = "cpu",
    **kwargs,
) -> "ChessFormerEngine":
    """Load a checkpoint and build an engine that reads it the way it was trained.

    The single place that pairs a checkpoint with its frame. Callers may still
    override either key, which is what an A/B that deliberately mismatches them
    needs, but they have to say so.
    """
    model, _config, payload = load_model(checkpoint, preset=preset, device=device)
    settings = checkpoint_frame(payload)
    settings.update(kwargs)
    return ChessFormerEngine(model, device=device, **settings)


def _terminal_value(board: chess.Board) -> float | None:
    """Value of `board` for its side to move, if the game is already over."""
    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        return None
    if outcome.winner is None:
        return 0.0
    # The side to move can only be the loser: it is checkmated.
    return -MATE_VALUE


MODES = (
    "policy",
    "safe",
    "safe2",
    "safe3",
    "safe4",
    "safe5",
    "safe6",
    "value1",
    "alphabeta",
    "search2",
)
# Which frame a checkpoint's value head predicts in. Every checkpoint trained on
# the Lichess eval export is "white"; see _to_side_to_move.
VALUE_POV = ("white", "side_to_move")
# The halfmove clock every training record carries. The Lichess eval export is a
# four-field EPD with no move counters, and set_epd() zeroes them, so rows 1..100
# of halfmove_embedding never received a gradient -- they are still their
# initialisation, shrunk by weight decay to a norm of ~0.20 against ~0.60 for the
# trained side embedding. Feeding the real clock at play time therefore adds an
# untrained random vector to the state token that both heads read.
#
# Measured on 128 positions from the bot's own games (checkpoint main-20m-ep4):
# zeroing the clock changes the top-1 policy move in 10.2% of them and moves the
# value head by a median of 0.024 (~15 cp), up to 0.091 (~55 cp).
TRAINED_HALFMOVE = 0
# Chunk size for batched evaluation, so a wide frontier cannot exhaust memory.
EVAL_CHUNK = 256
# The engine outlives a game, so the FEN-keyed value cache needs a ceiling.
_VALUE_CACHE_MAX = 200_000
# Forced-search scores are centipawns, so mate needs a magnitude no material
# balance can reach. Ply-adjusted, so a faster mate outranks a slower one and
# the search prefers the longest defence when it is losing.
FORCED_MATE_CP = 100_000
# Anything past this is a mate score rather than a material count.
FORCED_MATE_THRESHOLD = FORCED_MATE_CP - 1_000

# Centipawn values for the optional material term in the leaf evaluation.
_MATERIAL_CP = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}


def _least_valuable_attacker(board: chess.Board, square: int) -> chess.Move | None:
    """Cheapest legal capture of `square` by the side to move, if any."""
    best: tuple[int, chess.Move] | None = None
    for origin in board.attackers(board.turn, square):
        piece_type = board.piece_type_at(origin)
        if piece_type is None:
            continue
        move = chess.Move(origin, square)
        if piece_type == chess.PAWN and chess.square_rank(square) in (0, 7):
            move = chess.Move(origin, square, promotion=chess.QUEEN)
        # A pinned attacker is listed by `attackers` but cannot legally capture.
        if not board.is_legal(move):
            continue
        value = _MATERIAL_CP.get(piece_type, 0)
        if best is None or value < best[0]:
            best = (value, move)
    return None if best is None else best[1]


def _capture_gain(board: chess.Board, square: int) -> int:
    """Material the side to move can win on `square`, recursively.

    `max(0, ...)` encodes that a side may simply decline a losing exchange.
    """
    move = _least_valuable_attacker(board, square)
    if move is None:
        return 0
    victim = _MATERIAL_CP.get(board.piece_type_at(square) or 0, 0)
    board.push(move)
    try:
        return max(0, victim - _capture_gain(board, square))
    finally:
        board.pop()


def static_exchange_evaluation(board: chess.Board, move: chess.Move) -> int:
    """Net centipawns won by `move` after every recapture on its target square.

    A negative result means the move loses material -- it hangs a piece or walks
    into a losing exchange. This uses no neural network at all, which is what
    makes it affordable on CPU where a 143M forward pass costs seconds.
    """
    target = move.to_square
    if board.is_en_passant(move):
        captured = _MATERIAL_CP[chess.PAWN]
    else:
        piece_type = board.piece_type_at(target)
        captured = _MATERIAL_CP.get(piece_type, 0) if piece_type else 0

    promotion_gain = 0
    if move.promotion:
        promotion_gain = _MATERIAL_CP.get(move.promotion, 0) - _MATERIAL_CP[chess.PAWN]

    board.push(move)
    try:
        return captured + promotion_gain - _capture_gain(board, target)
    finally:
        board.pop()


def _to_side_to_move(value: float, board: chess.Board, value_pov: str) -> float:
    """Put a raw value-head output into the side-to-move frame the search expects.

    Lichess eval scores are stored relative to White, and the data pipeline never
    negated them for Black-to-move positions, so every checkpoint trained on that
    export predicts a White-relative value. Measured on 1M training labels: the
    correlation between White's material and the stored label is +0.53 for
    Black-to-move positions, where a side-to-move label would be strongly
    negative. Reading such a head as side-to-move inverts the sign on roughly
    half of all positions, which steers the search into losing lines rather than
    away from them.
    """
    if value_pov == "white" and board.turn == chess.BLACK:
        return -value
    return value


def _material_centipawns(board: chess.Board) -> int:
    """Material balance in centipawns, from the side to move's point of view."""
    total = 0
    for piece_type, value in _MATERIAL_CP.items():
        total += value * len(board.pieces(piece_type, board.turn))
        total -= value * len(board.pieces(piece_type, not board.turn))
    return total

# Most-valuable-victim ordering for the quiescence search only.
_MVV = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


class ChessFormerEngine:
    """Legal-masked move selection, from raw policy up to alpha-beta search."""

    def __init__(
        self,
        model: ChessFormer,
        *,
        device: torch.device | str = "cpu",
        mode: str = "value1",
        depth: int = 2,
        width: int = 8,
        quiescence_depth: int = 4,
        material_weight: float = 0.0,
        safety_candidates: int = 12,
        value_pov: str = "white",
        avoid_repetition: bool = True,
        forced_depth: int = 4,
        positional_weight: float = 0.0,
        forced_nodes: int = 30000,
        check_extension: int = 2,
        capture_extension: int = 8,
        value_quiet: bool = True,
        quiet_candidates: int = 12,
        quiet_replies: int = 4,
        eval_budget: int | None = None,
        policy_weight: float = 0.5,
        repetition_value_floor: float = -0.10,
        temperature: float = 0.0,
        seed: int | None = None,
        tablebase: EndgameTablebase | None = None,
        book: OpeningBook | None = None,
        match_trained_halfmove: bool = True,
        canonical: bool = False,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if value_pov not in VALUE_POV:
            raise ValueError(f"value_pov must be one of {VALUE_POV}")
        if depth < 1 or width < 1 or quiescence_depth < 0:
            raise ValueError("depth and width must be >= 1, quiescence_depth >= 0")
        if not 0.0 <= material_weight <= 1.0:
            raise ValueError("material_weight must be in [0, 1]")
        if forced_depth < 1 or forced_nodes < 1:
            raise ValueError("forced_depth and forced_nodes must be >= 1")
        if positional_weight < 0.0:
            raise ValueError("positional_weight must be >= 0")
        if check_extension < 0 or capture_extension < 0:
            raise ValueError("check_extension and capture_extension must be >= 0")
        if policy_weight < 0.0:
            raise ValueError("policy_weight must be >= 0")
        if eval_budget is not None and eval_budget < 1:
            raise ValueError("eval_budget must be >= 1 or None")
        if not math.isfinite(temperature) or temperature < 0.0:
            raise ValueError("temperature must be a finite value >= 0")
        self.model = model
        self.device = torch.device(device)
        self.mode = mode
        self.depth = depth
        self.width = width
        self.quiescence_depth = quiescence_depth
        self.material_weight = material_weight
        self.safety_candidates = max(1, safety_candidates)
        self.value_pov = value_pov
        self.avoid_repetition = avoid_repetition
        self.forced_depth = forced_depth
        # How much of the piece-square term the forced search sees at its leaves.
        # 0.0 is the material-only leaf every measurement in this repo was taken
        # with, and is bit-for-bit the old behaviour; the A/B needs that default
        # to stay put until the alternative is measured, not because it is right.
        self.positional_weight = positional_weight
        self.forced_nodes = forced_nodes
        self.check_extension = check_extension
        self.capture_extension = capture_extension
        self.value_quiet = value_quiet
        self.quiet_candidates = max(1, quiet_candidates)
        self.quiet_replies = max(1, quiet_replies)
        # Positions the quiet decision may evaluate for one move. None is
        # unbounded; the Lichess client sets it from the clock so that a move in a
        # 3+0 game and a move in a 10+5 game buy different amounts of search from
        # the same engine.
        self.eval_budget = eval_budget
        self.policy_weight = policy_weight
        self.repetition_value_floor = repetition_value_floor
        self.temperature = temperature
        self.seed = seed
        # A generator of our own rather than the global RNG: sampling a move must
        # not shift the stream that a match runner uses for openings and opponent
        # replies, or two engines would stop seeing the same games.
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
        self.tablebase = tablebase
        self.book = book
        self.match_trained_halfmove = match_trained_halfmove
        # A canonical checkpoint saw every position from the side to move's point
        # of view, so its value head is side-to-move relative by construction and
        # `value_pov` must say so or every Black evaluation comes back negated.
        self.canonical = canonical
        if canonical and value_pov != "side_to_move":
            raise ValueError("A canonical checkpoint has a side_to_move value head")
        self._action_permutation = torch.from_numpy(action_permutation())
        self._value_cache: dict[str, float] = {}
        self._nodes = 0
        self._forced_nodes = 0

    def _cache_key(self, board: chess.Board) -> str:
        """Exactly the inputs `encode_board` hands the model, and nothing else.

        `board.fen()` was wrong in both directions. It writes `-` for the en
        passant square whenever no en passant capture happens to be legal, while
        `encode_board` passes the raw square: measured on random games, 94% of the
        positions with an armed `ep_square` have a FEN that hides it, and a
        collision -- one key, two different `ep_file` inputs -- turns up within
        a hundred random games. A cached score was then served for a position the
        model scores differently. `en_passant="fen"` prints the raw square, which
        is what the encoder actually reads.

        In the other direction the FEN also carries the fullmove number, which is
        no input at all, and the halfmove clock, which is only an input when it is
        not pinned to its training value -- both split one score across many
        entries and guarantee misses on quiet moves.
        """
        key = board.epd(en_passant="fen")
        return key if self.match_trained_halfmove else f"{key} {board.halfmove_clock}"

    def _score_value(self, raw: float, board: chess.Board) -> float:
        """Put a raw head output in the side-to-move frame and blend in material."""
        result = _to_side_to_move(raw, board, self.value_pov)
        if self.material_weight > 0.0:
            # The value head was trained on tanh(cp/600), so a centipawn material
            # count maps onto the same scale and the two are directly blendable.
            # Measured need: an 11.5M head scores a queen-down position at +0.18.
            material = math.tanh(_material_centipawns(board) / VALUE_CP_SCALE)
            result = (1.0 - self.material_weight) * result + self.material_weight * material
        return result

    def _remember(self, board: chess.Board, value: float) -> float:
        """Cache a scored value, bounding the cache so a long session cannot leak.

        The engine object outlives a single game -- the Lichess bot builds it once
        and plays every game with it -- so an unbounded FEN-keyed dict grows for
        as long as the process is up.
        """
        if len(self._value_cache) >= _VALUE_CACHE_MAX:
            self._value_cache.clear()
        self._value_cache[self._cache_key(board)] = value
        return value

    @torch.no_grad()
    def _value_of(self, board: chess.Board) -> float:
        """Value head for `board`, from its own side-to-move point of view."""
        cached = self._value_cache.get(self._cache_key(board))
        if cached is not None:
            return cached
        _, value = self._forward([board])
        self._nodes += 1
        return self._remember(board, self._score_value(float(value[0]), board))

    def _ordered_moves(self, board: chess.Board, width: int) -> list[chess.Move]:
        """Top-`width` policy moves, plus every capture and check.

        The policy prior is far too weak to prune with on its own: a refutation
        such as winning a hung queen routinely falls outside its top moves. Any
        move that changes material or gives check is therefore always searched.
        """
        ordered = [move for move, _ in self.policy_ranking(board)[:width]]
        seen = set(ordered)
        forcing = [
            move
            for move in board.legal_moves
            if move not in seen and (board.is_capture(move) or board.gives_check(move))
        ]
        forcing.sort(key=lambda m: -_MVV.get(board.piece_type_at(m.to_square) or 0, 0))
        return ordered + forcing

    def _quiescence(self, board: chess.Board, alpha: float, beta: float, depth: int) -> float:
        """Resolve captures so the search never stops on a hanging piece."""
        terminal = _terminal_value(board)
        if terminal is not None:
            return terminal

        stand_pat = self._value_of(board)
        if depth <= 0 or stand_pat >= beta:
            return stand_pat
        alpha = max(alpha, stand_pat)

        captures = [move for move in board.legal_moves if board.is_capture(move)]
        # Search the most valuable victims first so alpha-beta prunes sooner.
        captures.sort(
            key=lambda m: -_MVV.get(
                (board.piece_type_at(m.to_square) or chess.PAWN)
                if not board.is_en_passant(m)
                else chess.PAWN,
                0,
            )
        )
        for move in captures:
            board.push(move)
            score = -self._quiescence(board, -beta, -alpha, depth - 1)
            board.pop()
            if score >= beta:
                return score
            alpha = max(alpha, score)
        return alpha

    def _negamax(self, board: chess.Board, depth: int, alpha: float, beta: float) -> float:
        terminal = _terminal_value(board)
        if terminal is not None:
            return terminal
        if depth <= 0:
            return self._quiescence(board, alpha, beta, self.quiescence_depth)

        best = -math.inf
        for move in self._ordered_moves(board, self.width):
            board.push(move)
            score = -self._negamax(board, depth - 1, -beta, -alpha)
            board.pop()
            if score > best:
                best = score
            if best >= beta:
                break
            alpha = max(alpha, best)
        return best

    @torch.no_grad()
    def _forward(self, boards: list[chess.Board]) -> tuple[torch.Tensor, torch.Tensor]:
        """Both heads for a batch of positions, in the caller's own frame.

        A canonical checkpoint has only ever seen the side to move at the bottom
        of the board, so Black-to-move positions are mirrored on the way in and
        their logits permuted back on the way out. Doing it here and nowhere else
        keeps the mirror invisible: every caller goes on indexing logits with
        `move_to_index` of a move on the real board.
        """
        if self.canonical:
            prepared = [canonical_board(board) for board in boards]
            boards = [prepared_board for prepared_board, _ in prepared]
            mirrored = [was_mirrored for _, was_mirrored in prepared]
        else:
            mirrored = [False] * len(boards)
        encoded = [encode_board(board) for board in boards]
        halfmove = [
            TRAINED_HALFMOVE if self.match_trained_halfmove else int(e.halfmove)
            for e in encoded
        ]
        batch = {
            "pieces": torch.from_numpy(
                np.stack([e.pieces for e in encoded]).astype(np.int64, copy=False)
            ),
            "side": torch.tensor([int(e.side) for e in encoded], dtype=torch.int64),
            "castling": torch.tensor([int(e.castling) for e in encoded], dtype=torch.int64),
            "ep_file": torch.tensor([int(e.ep_file) for e in encoded], dtype=torch.int64),
            "halfmove": torch.tensor(halfmove, dtype=torch.int64),
        }
        batch = {key: value.to(self.device) for key, value in batch.items()}
        logits, value = self.model(
            batch["pieces"],
            batch["side"],
            batch["castling"],
            batch["ep_file"],
            batch["halfmove"],
        )
        logits = logits.float().cpu()
        if any(mirrored):
            rows = torch.tensor(
                [index for index, flip in enumerate(mirrored) if flip], dtype=torch.long
            )
            # The permutation is an involution, so the same gather undoes it.
            logits[rows] = logits[rows][:, self._action_permutation]
        # A categorical head returns [B, value_bins] of logits, not a number.
        # `scalar_value` is the model's own single place for collapsing them, and
        # this is the engine's single call into the model -- so every caller
        # downstream (`_value_of`, `_values_of`, `policy_ranking`) goes on
        # holding one float per position and never learns which head it has.
        # Without this, a categorical checkpoint does not merely score badly: it
        # raises "only one element tensors can be converted to Python scalars"
        # on the first position of the first game.
        value = self._collapse_value(value)
        return logits, value.float().cpu()

    def _collapse_value(self, value: torch.Tensor) -> torch.Tensor:
        """One number per position, whichever head produced it.

        Test doubles and any future scalar head do not have to implement
        `scalar_value`; a distribution without a way to collapse it is an error
        rather than something to guess at, because guessing here would put a bin
        index where the search expects a score in [-1, 1].
        """
        collapse = getattr(self.model, "scalar_value", None)
        if collapse is not None:
            return collapse(value)
        if value.ndim > 1:
            raise TypeError(
                f"The value head returned {tuple(value.shape)} but the model offers "
                "no scalar_value() to collapse it."
            )
        return value

    def _rank(
        self, logits: torch.Tensor, legal: list[chess.Move]
    ) -> list[tuple[chess.Move, float]]:
        """Legal moves with their policy probabilities, best first.

        `temperature` is the softmax temperature of the policy head. 0 -- the
        default, and the setting under which every measurement in this repo was
        taken -- is the deterministic engine: probabilities read straight off the
        trained logits, arg-max first, ties broken by UCI so the order never
        depends on move generation.

        Above 0 the logits are divided by the temperature *and* the order is
        drawn from the resulting distribution, without replacement, by the Gumbel
        top-k trick (perturb each logit with -log(-log U) and sort). Scaling
        alone would be invisible: dividing every logit by the same constant
        cannot change which one is largest, so a temperature that only reshaped
        the printed probabilities would play exactly the same moves.

        Sampling the whole *order*, not just the first move, is what carries the
        temperature into the safe modes. They take this ranking as their
        candidate list and fall back on its order every time the forced search
        cannot separate two moves -- `pool[0]` in `_search_safe_policy`, the
        `-order[m]` tie-break in `_search_safe_tactical`. The tactical vetoes are
        untouched either way: a sampled move that hangs a piece is still refused,
        so temperature widens the choice among moves the search calls safe rather
        than licensing blunders.
        """
        if self.temperature == 0.0:
            probabilities = torch.softmax(logits, dim=-1)
            scored = [(move, float(probabilities[move_to_index(move)])) for move in legal]
            # Sort by probability, then by UCI so equal scores never depend on move order.
            scored.sort(key=lambda item: (-item[1], item[0].uci()))
            return scored

        scaled = logits / self.temperature
        probabilities = torch.softmax(scaled, dim=-1)
        indices = [move_to_index(move) for move in legal]
        # torch.rand is half-open on [0, 1), so only the 0 end needs a floor;
        # -log(-log(0)) is not a number and would sort as one.
        uniform = torch.rand(len(indices), generator=self._rng).clamp_min(_GUMBEL_EPS)
        keys = scaled[indices] + -torch.log(-torch.log(uniform))
        order = sorted(
            range(len(legal)), key=lambda i: (-float(keys[i]), legal[i].uci())
        )
        return [(legal[i], float(probabilities[indices[i]])) for i in order]

    @torch.no_grad()
    def policy_ranking(self, board: chess.Board) -> list[tuple[chess.Move, float]]:
        """Legal moves with their masked policy probabilities, best first."""
        legal = list(board.legal_moves)
        if not legal:
            return []
        logits, value = self._forward([board])
        # The forward returns both heads. Dropping the value here is what made
        # every safe mode pay a second forward pass for `_value_of(board)` at the
        # end of the search -- half the neural cost of a move, spent twice on the
        # same position. Caching it makes the value head free to consult.
        self._nodes += 1
        self._remember(board, self._score_value(float(value[0]), board))
        mask = torch.from_numpy(legal_action_mask(board).astype(np.bool_))[None, :]
        masked = mask_policy_logits(logits, mask)
        return self._rank(masked[0], legal)

    @torch.no_grad()
    def search(self, board: chess.Board) -> SearchResult | None:
        """Best move for `board`, or None if the game is already over."""
        legal = list(board.legal_moves)
        if not legal:
            return None

        # Before everything: a book entry is a correction already paid for in
        # Stockfish time, and probing it costs a dictionary lookup. It cannot
        # collide with the tablebase below -- one covers the opening, the other
        # five men -- and it is deliberately deterministic even under a non-zero
        # temperature: the point of the entry is that this line stops being
        # replayed the way it was lost.
        entry = self.book.probe(board) if self.book is not None else None
        if entry is not None:
            return SearchResult(
                move=entry.move,
                value=math.tanh(entry.value_cp / VALUE_CP_SCALE),
                nodes=0,
                pv=[entry.move],
            )

        # Before the network: once the position is solved there is nothing left
        # for a 0.3 s forward pass to contribute, and DTZ play is exact where
        # every mode above is a heuristic. This also removes the repetition
        # problem in endgames -- minmaxing DTZ makes progress by construction.
        solved = self._tablebase_result(board)
        if solved is not None:
            return solved

        ranking = self.policy_ranking(board)
        if self.mode == "policy":
            move, probability = ranking[0]
            # _value_of, not _forward: the reported value feeds the UCI `score cp`,
            # which would print with the wrong sign for Black from the raw head.
            return SearchResult(move=move, value=self._value_of(board), nodes=1, pv=[move])

        if self.mode == "safe":
            return self._search_safe_policy(board, ranking)

        if self.mode == "safe2":
            return self._search_safe_policy(board, ranking, count_replies="rank")

        if self.mode == "safe3":
            return self._search_safe_policy(board, ranking, count_replies="veto")

        if self.mode == "safe4":
            return self._search_safe_policy(board, ranking, count_replies="veto+check")

        if self.mode == "safe5":
            return self._search_safe_tactical(board, ranking)

        if self.mode == "safe6":
            return self._search_safe_tactical(board, ranking, quiet_plies=2)

        if self.mode == "alphabeta":
            return self._search_alphabeta(board, ranking)

        if self.mode == "search2":
            return self._search_two_ply(board, ranking)

        # One-ply search: score every child with the value head. A child's value is
        # from the opponent's point of view, so negate it to score our own move.
        priors = {move: probability for move, probability in ranking}
        children: list[chess.Board] = []
        pending: list[chess.Move] = []
        scores: dict[chess.Move, float] = {}
        for move in legal:
            board.push(move)
            terminal = _terminal_value(board)
            if terminal is None:
                children.append(board.copy(stack=False))
                pending.append(move)
            else:
                scores[move] = -terminal
            board.pop()

        if children:
            # Must go through _values_of, not _forward: the raw head is
            # White-relative, and only _values_of puts it in the child's own
            # side-to-move frame. Negating a White-relative value here scored
            # roughly half of all children backwards.
            for move, child_value in zip(pending, self._values_of(children), strict=True):
                scores[move] = -child_value

        # Break ties with the policy prior, then UCI, so play is fully deterministic.
        best = max(legal, key=lambda m: (scores[m], priors.get(m, 0.0), m.uci()))
        reply = self._best_reply(board, best)
        return SearchResult(
            move=best,
            value=scores[best],
            nodes=1 + len(legal),
            pv=[best] + reply,
        )

    def _tablebase_result(self, board: chess.Board) -> SearchResult | None:
        """Exact play from the tablebase, or None when the position is unsolved.

        A cursed win reports 0.0 rather than a win: the fifty-move rule makes it
        a draw, and reporting it as won would put a mate score on the UCI info
        line for a game that ends in a handshake.
        """
        if self.tablebase is None:
            return None
        probe = self.tablebase.best_move(board)
        if probe is None:
            return None
        if probe.wdl == 2:
            value = MATE_VALUE
        elif probe.wdl == -2:
            value = -MATE_VALUE
        else:
            value = 0.0
        return SearchResult(move=probe.move, value=value, nodes=0, pv=[probe.move])

    def _allows_mate_in_one(self, board: chess.Board, move: chess.Move) -> bool:
        board.push(move)
        try:
            for reply in board.legal_moves:
                board.push(reply)
                mate = board.is_checkmate()
                board.pop()
                if mate:
                    return True
            return False
        finally:
            board.pop()

    @torch.no_grad()
    def _best_capture_gain(self, board: chess.Board) -> int:
        """Most material the side to move can win with one capture, 0 if none."""
        best = 0
        for move in board.legal_moves:
            if board.is_capture(move):
                gain = static_exchange_evaluation(board, move)
                if gain > best:
                    best = gain
        return best

    def _concedes_to_check(self, board: chess.Board, move: chess.Move) -> int:
        """Material lost to a *forcing* reply, two plies out, by move generation.

        `_concedes` only prices the opponent's best immediate capture, so it is
        blind to a quiet move that wins material next. Rated game h5yt9RZS was
        lost exactly there: after 11...a6, White played the quiet 12.Nc7+ forking
        king and rook and took it on move 13. Nc7 captures nothing, so the veto
        scored a6 at zero and waved it through.

        Searching every reply two plies deep needs a real search. Restricting it
        to *checks* does not: our answer to a check is forced, so the threat
        actually executes, and there are only a handful of checks in a position.
        This is what makes the extra ply affordable without a single forward
        pass, on a CPU where two-ply minimax costs 36 s per move.
        """
        board.push(move)
        try:
            worst = 0
            for reply in board.legal_moves:
                if not board.gives_check(reply):
                    continue
                board.push(reply)
                try:
                    # We are in check; take whichever escape concedes least.
                    best_escape: int | None = None
                    for escape in board.legal_moves:
                        board.push(escape)
                        try:
                            conceded = self._best_capture_gain(board)
                        finally:
                            board.pop()
                        if best_escape is None or conceded < best_escape:
                            best_escape = conceded
                        if best_escape == 0:
                            break  # cannot do better than losing nothing
                    if best_escape is not None and best_escape > worst:
                        worst = best_escape
                finally:
                    board.pop()
            return worst
        finally:
            board.pop()

    def _forcing_candidates(
        self, board: chess.Board, *, quiet_checks: bool
    ) -> tuple[list[chess.Move], bool]:
        """Captures, promotions and checks: the moves that compel a reply.

        Ordered most-valuable-victim first so alpha-beta cuts sooner. `gives_check`
        pushes and pops internally, so it is only asked of moves that are not
        already forcing for a cheaper reason -- and only while `quiet_checks` is
        set, since past the nominal depth the search resolves exchanges only.

        Also reports whether the position had *any* legal move. The caller needs
        that to tell a quiet position from a stalemate, and this loop has already
        generated every legal move, so returning it costs nothing where asking
        `board.legal_moves` a second time would double the cost of the commonest
        leaf in the search.
        """
        captures: list[tuple[int, str, chess.Move]] = []
        others: list[chess.Move] = []
        any_legal = False
        for move in board.legal_moves:
            any_legal = True
            if board.is_capture(move):
                victim = (
                    chess.PAWN
                    if board.is_en_passant(move)
                    else (board.piece_type_at(move.to_square) or chess.PAWN)
                )
                captures.append((-_MVV.get(victim, 0), move.uci(), move))
            elif quiet_checks and (move.promotion or board.gives_check(move)):
                others.append(move)
        captures.sort()
        return [move for _, _, move in captures] + others, any_legal

    def _leaf_cp(self, board: chess.Board) -> int:
        """What the forced search sees at a leaf, from the side to move's view.

        Material alone cannot separate quiet moves: measured on 2,407 quiet moves
        from 80 real positions, the material leaf returns exactly one distinct
        value per position. So no amount of depth can rank them, and raising
        `forced_depth` past 4 has always been measured as a wash or a loss. This
        is where that changes or does not.
        """
        score = _material_centipawns(board)
        if self.positional_weight:
            score += int(self.positional_weight * positional_centipawns(board))
        return score

    def _forced_value(
        self, board: chess.Board, alpha: int, beta: int, depth: int, ply: int
    ) -> int:
        """Centipawns the side to move can force, searching only forcing moves.

        This is a quiescence search widened with checks and promotions, evaluated
        by material alone -- no forward pass anywhere in the tree. That is what
        makes 4 plies affordable on a CPU where one 143M forward costs 0.3 s and
        a 2-ply neural minimax costs 36 s.

        It subsumes both of safe4's vetoes. `_concedes` priced only the opponent's
        best immediate capture and `_concedes_to_check` only a check followed by
        one capture; neither could see a mate two moves out, which is how the mate
        after 8...fxe5 got through. Here a mate at any reachable ply comes back as
        a score past FORCED_MATE_THRESHOLD.

        Standing pat encodes that the side to move is never obliged to start a
        forcing sequence -- except when in check, where every reply is forced and
        the search must resolve the check before it may trust a material count.
        """
        if self._forced_nodes >= self.forced_nodes:
            return self._leaf_cp(board)
        self._forced_nodes += 1

        # A solved position ends the subtree: the exact result is known, so
        # searching on can only replace it with a guess. This is the site that
        # actually changes moves -- it prices the trade *into* a drawn or lost
        # endgame before it is played, where the root probe can only make the
        # best of one already on the board.
        if self.tablebase is not None:
            solved = self.tablebase.score_cp(board, ply)
            if solved is not None:
                return solved

        if board.is_check():
            evasions = list(board.legal_moves)
            if not evasions:
                return -FORCED_MATE_CP + ply
            # Check extensions run past `depth`, or a mate delivered at the
            # horizon would be scored as a quiet material count. The allowance is
            # bounded so a perpetual cannot run the search forever.
            if depth <= -self.check_extension:
                return self._leaf_cp(board)
            moves = evasions
            best = -FORCED_MATE_CP + ply  # no standing pat while in check
        else:
            best = self._leaf_cp(board)
            if best >= beta:
                return best
            # Captures run past the nominal depth until the position is quiet.
            # Stopping an exchange half way through scores it as if the recapture
            # never came, which is the classic horizon effect: Nxd5 looks like a
            # free piece right up to the ply where exd5 answers it. A capture
            # chain is self-limiting -- every capture removes a piece -- so only
            # a backstop bound is needed against pathological positions.
            if depth <= -self.capture_extension:
                return best
            alpha = max(alpha, best)
            moves, any_legal = self._forcing_candidates(board, quiet_checks=depth > 0)
            if not any_legal:
                # Stalemate. Returning the material count here would let the
                # search walk a won endgame into a draw while still scoring it as
                # a win -- the same class of error as trading into K+N versus K.
                return 0
            if not moves:
                return best

        for move in moves:
            board.push(move)
            try:
                score = -self._forced_value(board, -beta, -alpha, depth - 1, ply + 1)
            finally:
                board.pop()
            if score > best:
                best = score
            if best >= beta:
                break
            alpha = max(alpha, best)
        return best

    def _forced_net(self, board: chess.Board, move: chess.Move) -> int:
        """Centipawns `move` wins or loses once the opponent replies by force.

        Positive means the forcing sequence ends up better for us than the
        position we started from; a score past FORCED_MATE_THRESHOLD in either
        direction is a mate we give or allow.

        The baseline has to be measured on the same scale as the leaves. Leave it
        at pure material while the leaves also count the piece-square term and
        every quiet move acquires a constant offset -- the whole candidate set
        shifts, `net[move] >= 0` stops meaning "loses nothing", and the safety
        filter silently changes what it filters.
        """
        before = self._leaf_cp(board)
        board.push(move)
        try:
            reply = self._forced_value(
                board, -FORCED_MATE_CP, FORCED_MATE_CP, self.forced_depth - 1, 1
            )
        finally:
            board.pop()
        # `reply` is in the opponent's frame; negate it to get ours.
        ours = -reply
        if ours > FORCED_MATE_THRESHOLD or ours < -FORCED_MATE_THRESHOLD:
            return ours
        return ours - before

    def _reaches_dead_draw(self, board: chess.Board, move: chess.Move) -> bool:
        """Does `move` reach a position drawn by insufficient material?

        Material alone cannot see this. Three of the bot's drawn games ended in
        K+N versus K or K+B versus K, where the material term still read +320 for
        a position that is drawn by rule. The last capture that empties the board
        of pawns keeps scoring as a winning edge right up to the moment the
        arbiter calls it a draw.

        The question is whether *neither* side can mate, not whether we can. A
        side that is merely unable to mate itself may still be losing, and
        pricing that as a draw would be an error in the other direction.
        """
        board.push(move)
        try:
            return board.is_insufficient_material()
        finally:
            board.pop()

    def _repeats(self, board: chess.Board, move: chess.Move) -> bool:
        """Does `move` return to a position already seen in this game?

        Observed in rated play: with a bishop and two pawns against four pawns,
        the engine played Bh6 Bf8 Bh6 Bf8 until the game was drawn. A policy with
        no search has no plan in a quiet position, so its arg-max can oscillate
        between two moves that undo each other. Twofold rather than threefold,
        because the point is to never start the shuffle.
        """
        board.push(move)
        try:
            return board.is_repetition(2)
        finally:
            board.pop()

    def _concedes(self, board: chess.Board, move: chess.Move) -> int:
        """Material the opponent wins in reply to `move`, by static exchange.

        `static_exchange_evaluation(board, move)` only settles the exchange on
        the square `move` lands on. It cannot see that the move left something
        hanging elsewhere -- walking a defender away from a piece, or simply
        moving while a queen stands en prise -- because neither is an exchange
        on the destination square. Those moves score 0 and are waved through as
        "safe". This prices the opponent's best single capture in reply.
        """
        board.push(move)
        try:
            return self._best_capture_gain(board)
        finally:
            board.pop()

    def _search_safe_policy(
        self,
        board: chess.Board,
        ranking: list[tuple[chess.Move, float]],
        *,
        count_replies: str | None = None,
    ) -> SearchResult:
        """Policy arg-max behind a tactical safety net that costs no forward pass.

        On a weak CPU a 143M forward costs seconds, so any search that expands
        the tree is unaffordable. This keeps the single policy pass and rejects
        its blunders with pure move generation: take a mate when one exists,
        never hang material, and never allow mate in one.
        """
        # 1. Take a forced mate rather than whatever the policy prefers.
        for move in (candidate for candidate, _ in ranking):
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return SearchResult(move=move, value=MATE_VALUE, nodes=0, pv=[move])

        # 2. Candidates are the policy's best moves plus every capture and check.
        # A weak policy ranks free material outside its top moves, so restricting
        # the set to the prior alone would walk straight past a hanging queen.
        candidates = [move for move, _ in ranking][: self.safety_candidates]
        seen = set(candidates)
        candidates += [
            move
            for move in board.legal_moves
            if move not in seen and (board.is_capture(move) or board.gives_check(move))
        ]

        exchanges = {move: static_exchange_evaluation(board, move) for move in candidates}
        # Net material: what the move wins on its own square, minus what the
        # opponent wins in reply anywhere on the board.
        # The two vetoes are complementary, not alternatives. `_concedes` misses a
        # quiet fork; `_concedes_to_check` misses a piece left hanging to a plain
        # capture. In the position that lost h5yt9RZS, Qxb5 scores 0 on the check
        # veto while dropping the queen, and a6 scores 0 on the capture veto while
        # dropping the exchange. Only the max of the two rejects both.
        if count_replies == "veto+check":
            net = {
                move: gain
                - max(self._concedes(board, move), self._concedes_to_check(board, move))
                for move, gain in exchanges.items()
            }
        elif count_replies:
            net = {
                move: gain - self._concedes(board, move)
                for move, gain in exchanges.items()
            }
        else:
            net = exchanges
        # "rank" lets the net value pick the move; "veto" only uses it to reject.
        # Measured over 100 games per pairing, seed 7, on the Small 6-epoch
        # checkpoint:
        #
        #   mode    vs material1        vs stockfish1320
        #   safe    88-12-0  (0.940)    23-12-65  (0.290)
        #   safe2   94- 6-0  (0.970)    24-15-61  (0.315)
        #
        # "rank" is better on the deterministic material opponent (paired sign
        # test p = 0.016) and indistinguishable on Stockfish (p ~ 0.68). A
        # 30-game read of the same pairing had suggested "rank" was much worse
        # against Stockfish; that sample was unrepresentative, and its score of
        # 0.433 for `safe` falls outside the 100-game interval [0.207, 0.373].
        ordering = net if count_replies == "rank" else exchanges
        safe = [
            move
            for move in candidates
            if net[move] >= 0 and not self._allows_mate_in_one(board, move)
        ]

        if safe:
            # Take free material when it is on offer; otherwise trust the policy,
            # whose order `ranking` already encodes.
            winning = [move for move in safe if ordering[move] > 0]
            if winning:
                best = max(winning, key=lambda m: (ordering[m], -candidates.index(m)))
            else:
                # Repeating is only ever a mistake when we are the side with
                # something to convert, so the rule is deliberately narrow: it
                # applies when we are materially ahead, and only when a safe
                # non-repeating move exists. A losing side may want the draw.
                pool = safe
                if self.avoid_repetition and _material_centipawns(board) > 0:
                    fresh = [m for m in safe if not self._repeats(board, m)]
                    pool = fresh or safe
                best = pool[0]
        else:
            # Everything loses something: lose as little as possible, and still
            # avoid an immediate mate if any candidate does. `net` is the honest
            # cost here even under "veto", since the reply is what we are paying.
            survivable = [m for m in candidates if not self._allows_mate_in_one(board, m)]
            pool = survivable or candidates
            best = max(pool, key=lambda m: (net[m], m.uci()))

        return SearchResult(
            move=best, value=self._value_of(board), nodes=len(candidates), pv=[best]
        )

    def _repetition_is_bad(self, board: chess.Board) -> bool:
        """Is a repetition worth avoiding here, or is the draw the better result?

        safe4 asked only whether we were materially ahead, so it repeated freely
        in every materially equal position -- which is most of a game against a
        peer, and is where the observed threefolds came from. Material is the
        wrong question: the right one is whether we are better, and the value
        head answers that for free now that the policy pass caches it.
        """
        material = _material_centipawns(board)
        if material > 0:
            return True
        if material < -100:
            # Down real material: a draw by repetition is a good result.
            return False
        return self._value_of(board) >= self.repetition_value_floor

    @torch.no_grad()
    def _search_safe_tactical(
        self,
        board: chess.Board,
        ranking: list[tuple[chess.Move, float]],
        *,
        quiet_plies: int = 1,
    ) -> SearchResult:
        """safe4's candidate set, refereed by a real forced search and the value head.

        Three changes from safe4, each aimed at an observed loss mode:
          * the veto is a forced search over checks, captures and promotions, so
            it sees mates and material swings several plies out rather than one;
          * quiet moves that the search cannot separate are ranked by the value
            head instead of by the policy alone -- one batched forward for the
            whole candidate set, not one per move;
          * repetition is avoided whenever we are not worse, not only when we are
            materially ahead.
        """
        self._forced_nodes = 0

        # 1. A mate in one is still worth taking before anything else is searched.
        for move in (candidate for candidate, _ in ranking):
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return SearchResult(move=move, value=MATE_VALUE, nodes=0, pv=[move])

        # 2. Same candidate set as safe4: the policy's best, plus everything
        # forcing, because a weak policy ranks free material outside its top moves.
        candidates = [move for move, _ in ranking][: self.safety_candidates]
        seen = set(candidates)
        candidates += [
            move
            for move in board.legal_moves
            if move not in seen and (board.is_capture(move) or board.gives_check(move))
        ]

        # 3. Price every candidate by what the opponent can force in reply.
        net = {move: self._forced_net(board, move) for move in candidates}
        order = {move: index for index, move in enumerate(candidates)}

        # A trade that leaves neither side able to mate is a draw however the
        # material count reads, so price it as one: the whole edge when we are
        # ahead, and a rescue worth exactly the deficit when we are behind.
        standing = _material_centipawns(board)
        if standing:
            for move in candidates:
                # Never overwrite a mate score. No position drawn by insufficient
                # material can contain one, so this guard should be unreachable;
                # it is here so that a future change to either rule cannot make a
                # mate look like a draw.
                if abs(net[move]) > FORCED_MATE_THRESHOLD:
                    continue
                if self._reaches_dead_draw(board, move):
                    net[move] = min(net[move], -standing) if standing > 0 else max(
                        net[move], -standing
                    )

        forced_win = [m for m in candidates if net[m] > FORCED_MATE_THRESHOLD]
        if forced_win:
            # A forced mate: take the shortest, which is the highest score.
            best = max(forced_win, key=lambda m: (net[m], -order[m]))
            return SearchResult(
                move=best, value=MATE_VALUE, nodes=self._forced_nodes, pv=[best]
            )

        safe = [move for move in candidates if net[move] >= 0]
        if not safe:
            # Everything loses something: lose as little as possible, preferring
            # the policy's order among equally costly moves.
            best = max(candidates, key=lambda m: (net[m], -order[m]))
            return SearchResult(
                move=best, value=self._value_of(board), nodes=self._forced_nodes, pv=[best]
            )

        winning = [move for move in safe if net[move] > 0]
        if winning:
            best = max(winning, key=lambda m: (net[m], -order[m]))
            return SearchResult(
                move=best, value=self._value_of(board), nodes=self._forced_nodes, pv=[best]
            )

        # 4. Nothing wins material by force: this is the quiet-move decision that
        # safe4 left entirely to the policy. Drop repetitions first, then let the
        # value head rank what is left.
        pool = safe
        if self.avoid_repetition and self._repetition_is_bad(board):
            fresh = [m for m in pool if not self._repeats(board, m)]
            pool = fresh or pool

        if not self.value_quiet or len(pool) == 1:
            return SearchResult(
                move=pool[0], value=self._value_of(board), nodes=self._forced_nodes, pv=[pool[0]]
            )

        limited = pool[: self.quiet_candidates]
        if quiet_plies >= 2 and len(limited) > 1:
            scores, replies, evaluated = self._quiet_two_ply(board, limited)
        else:
            scores, replies, evaluated = self._quiet_one_ply(board, limited)

        # Blend rather than replace: the policy is trained on Stockfish's best
        # move and the value head is not, so the head steers the choice without
        # being allowed to override a strong prior on its own.
        priors = dict(ranking)
        blended = {
            move: scores[move] + self.policy_weight * priors.get(move, 0.0)
            for move in limited
        }
        best = max(limited, key=lambda m: (blended[m], -order[m]))
        principal = [best] + ([replies[best]] if best in replies else [])
        return SearchResult(
            move=best,
            value=scores[best],
            nodes=self._forced_nodes + evaluated,
            pv=principal,
        )

    @torch.no_grad()
    def _quiet_one_ply(
        self, board: chess.Board, moves: list[chess.Move]
    ) -> tuple[dict[chess.Move, float], dict[chess.Move, chess.Move], int]:
        """Score each quiet move by the value head after it.

        One forward pass for the whole pool, not one per move: batching is what
        makes consulting the value head cost the same as a single position.
        """
        children = []
        for move in moves:
            board.push(move)
            children.append(board.copy(stack=False))
            board.pop()
        values = self._values_of(children)
        # A child's value is from the opponent's point of view, so negate it.
        return (
            {move: -value for move, value in zip(moves, values, strict=True)},
            {},
            len(moves),
        )

    @torch.no_grad()
    def _quiet_two_ply(
        self, board: chess.Board, moves: list[chess.Move]
    ) -> tuple[dict[chess.Move, float], dict[chess.Move, chess.Move], int]:
        """Score each quiet move by what the opponent's best reply leaves us.

        Measured on the 70 quiet positions the bot actually had to judge, the move
        Stockfish wants is inside this pool 98.6% of the time and is the policy's
        first choice only 51% of the time. So the clock buys nothing by widening
        the pool and everything by separating the moves already in it, which is
        what a second ply does and a static evaluation of the child cannot.

        The whole frontier is scored in two batched passes -- one for the children
        and one for the grandchildren -- so the cost is set by the number of
        positions, not by the shape of the tree. `_search_two_ply` does the same
        minimax over *every* legal move and every capture and check, which costs
        36 s a move with the 143M model; here the root is the handful of moves the
        forced search already cleared of material loss, and the replies are the
        opponent's policy favourites, because the tactical refutations were priced
        by `_forced_net` before this function is ever reached.
        """
        # Terminal children are exact and must not reach the network: a mate we
        # deliver is not a position with a value.
        open_moves: list[chess.Move] = []
        open_children: list[chess.Board] = []
        scores: dict[chess.Move, float] = {}
        for move in moves:
            board.push(move)
            terminal = _terminal_value(board)
            if terminal is None:
                open_moves.append(move)
                open_children.append(board.copy(stack=False))
            else:
                # Terminal after our move, so from the opponent's point of view.
                scores[move] = -terminal
            board.pop()

        if not open_moves:
            return scores, {}, 0

        # Split the budget: one evaluation per child for its reply ordering, then
        # `replies` grandchildren each. A budget too small for even one reply per
        # child falls back to the one-ply decision rather than truncating the pool,
        # which would silently hide the very move the pool exists to consider.
        per_child = self.quiet_replies
        if self.eval_budget is not None:
            per_child = min(per_child, self.eval_budget // max(len(open_moves), 1) - 1)
        if per_child < 1:
            return self._quiet_one_ply(board, moves)

        child_rankings = self._policy_rankings(open_children)
        frontier: list[chess.Board] = []
        tags: list[tuple[chess.Move, chess.Move]] = []
        exact: dict[chess.Move, list[tuple[chess.Move, float]]] = {}
        for move, child_ranking in zip(open_moves, child_rankings, strict=True):
            board.push(move)
            exact[move] = []
            for reply, _ in child_ranking[:per_child]:
                board.push(reply)
                terminal = _terminal_value(board)
                if terminal is None:
                    frontier.append(board.copy(stack=False))
                    tags.append((move, reply))
                else:
                    # Side to move after the reply is us, so this is already ours.
                    exact[move].append((reply, terminal))
                board.pop()
            board.pop()

        values = self._values_of(frontier)
        scored = {move: list(replies) for move, replies in exact.items()}
        for (move, reply), value in zip(tags, values, strict=True):
            scored[move].append((reply, value))

        best_reply: dict[chess.Move, chess.Move] = {}
        for move, replies in scored.items():
            # The opponent picks the reply that minimises our value.
            reply, value = min(replies, key=lambda item: (item[1], item[0].uci()))
            scores[move] = value
            best_reply[move] = reply
        return scores, best_reply, len(open_moves) + len(frontier)

    @torch.no_grad()
    def _policy_rankings(
        self, boards: list[chess.Board]
    ) -> list[list[tuple[chess.Move, float]]]:
        """`policy_ranking` for many positions, batched into whole forward passes."""
        rankings: list[list[tuple[chess.Move, float]]] = []
        for start in range(0, len(boards), EVAL_CHUNK):
            chunk = boards[start : start + EVAL_CHUNK]
            logits, _ = self._forward(chunk)
            self._nodes += len(chunk)
            for index, board in enumerate(chunk):
                mask = torch.from_numpy(legal_action_mask(board).astype(np.bool_))[None, :]
                masked = mask_policy_logits(logits[index : index + 1], mask)
                rankings.append(self._rank(masked[0], list(board.legal_moves)))
        return rankings

    def _forcing_moves(
        self, board: chess.Board, ranking: list[tuple[chess.Move, float]], width: int
    ) -> list[chess.Move]:
        """Top-`width` policy moves plus every capture and check, from a ranking."""
        ordered = [move for move, _ in ranking[:width]]
        seen = set(ordered)
        forcing = [
            move
            for move in board.legal_moves
            if move not in seen and (board.is_capture(move) or board.gives_check(move))
        ]
        forcing.sort(key=lambda m: -_MVV.get(board.piece_type_at(m.to_square) or 0, 0))
        return ordered + forcing

    @torch.no_grad()
    def _values_of(self, boards: list[chess.Board]) -> list[float]:
        """Value head for many positions, batched, each from its own side to move."""
        results: list[float] = []
        for start in range(0, len(boards), EVAL_CHUNK):
            chunk = boards[start : start + EVAL_CHUNK]
            _, values = self._forward(chunk)
            self._nodes += len(chunk)
            for board, value in zip(chunk, values.tolist(), strict=True):
                score = _to_side_to_move(value, board, self.value_pov)
                if self.material_weight > 0.0:
                    material = math.tanh(_material_centipawns(board) / VALUE_CP_SCALE)
                    score = (
                        1.0 - self.material_weight
                    ) * score + self.material_weight * material
                results.append(score)
        return results

    @torch.no_grad()
    def _search_two_ply(
        self, board: chess.Board, ranking: list[tuple[chess.Move, float]]
    ) -> SearchResult:
        """Exact 2-ply minimax whose entire frontier is evaluated in one batch.

        Node-at-a-time alpha-beta spends a separate forward pass per node, which
        costs tens of seconds per move even for the 11.5M model. Here every leaf
        of the two-ply tree is collected first and scored in a single batch, so
        the search costs two forward passes regardless of tree width.
        """
        self._nodes = 0
        priors = dict(ranking)
        root_moves = [move for move, _ in ranking]

        root_scores: dict[chess.Move, float] = {}
        best_reply: dict[chess.Move, chess.Move] = {}
        # Frontier positions, tagged with the root move and reply that produced them.
        frontier: list[chess.Board] = []
        tags: list[tuple[chess.Move, chess.Move]] = []
        # Replies whose outcome is already exact and needs no evaluation.
        exact: dict[chess.Move, list[tuple[chess.Move, float]]] = {}

        # Rank every root child in one batch; ranking them one at a time would cost
        # a forward pass per root move and dominate the whole search.
        open_moves: list[chess.Move] = []
        open_children: list[chess.Board] = []
        for move in root_moves:
            board.push(move)
            terminal = _terminal_value(board)
            if terminal is not None:
                # Terminal after our move: negate to our point of view.
                root_scores[move] = -terminal
            else:
                open_moves.append(move)
                open_children.append(board.copy(stack=False))
            board.pop()

        child_rankings = self._policy_rankings(open_children)

        for move, child, ranking_child in zip(
            open_moves, open_children, child_rankings, strict=True
        ):
            board.push(move)
            exact[move] = []
            for reply in self._forcing_moves(child, ranking_child, self.width):
                board.push(reply)
                reply_terminal = _terminal_value(board)
                if reply_terminal is not None:
                    # Side to move after the reply is us, so this value is already ours.
                    exact[move].append((reply, reply_terminal))
                else:
                    frontier.append(board.copy(stack=False))
                    tags.append((move, reply))
                board.pop()
            board.pop()

        values = self._values_of(frontier)
        scored: dict[chess.Move, list[tuple[chess.Move, float]]] = {
            move: list(replies) for move, replies in exact.items()
        }
        for (move, reply), value in zip(tags, values, strict=True):
            scored[move].append((reply, value))

        for move, replies in scored.items():
            if not replies:
                continue
            # The opponent picks the reply that minimises our value.
            reply, value = min(replies, key=lambda item: (item[1], item[0].uci()))
            root_scores[move] = value
            best_reply[move] = reply

        best = max(
            root_moves,
            key=lambda m: (root_scores.get(m, -math.inf), priors.get(m, 0.0), m.uci()),
        )
        principal = [best]
        if best in best_reply:
            principal.append(best_reply[best])
        return SearchResult(
            move=best,
            value=root_scores.get(best, 0.0),
            nodes=self._nodes,
            pv=principal,
        )

    @torch.no_grad()
    def _search_alphabeta(
        self, board: chess.Board, ranking: list[tuple[chess.Move, float]]
    ) -> SearchResult:
        self._nodes = 0
        self._value_cache.clear()
        priors = dict(ranking)
        # Widen the root so a good move is never pruned before it is examined.
        root_moves = [move for move, _ in ranking][: max(self.width * 2, self.width)]

        best_move = root_moves[0]
        best_score = -math.inf
        alpha = -math.inf
        for move in root_moves:
            board.push(move)
            score = -self._negamax(board, self.depth - 1, -math.inf, -alpha)
            board.pop()
            # Ties fall back to the policy prior, then UCI, so play stays deterministic.
            if (score, priors.get(move, 0.0), move.uci()) > (
                best_score,
                priors.get(best_move, 0.0),
                best_move.uci(),
            ):
                best_score, best_move = score, move
            alpha = max(alpha, score)

        return SearchResult(
            move=best_move,
            value=best_score,
            nodes=self._nodes,
            pv=[best_move] + self._best_reply(board, best_move),
        )

    @torch.no_grad()
    def _best_reply(self, board: chess.Board, move: chess.Move) -> list[chess.Move]:
        board.push(move)
        try:
            if _terminal_value(board) is not None:
                return []
            ranking = self.policy_ranking(board)
            return [ranking[0][0]] if ranking else []
        finally:
            board.pop()
