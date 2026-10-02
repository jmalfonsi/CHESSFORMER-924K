"""Strict policy runtime: encode -> ONE forward -> legal mask -> argmax.

No imports from the existing engine, book, tablebase or training modules. One
opt-in check sits outside the network: `repetition_value` withholds a move that
would repeat a position a third time (see `MiniPlayer`).
"""
from __future__ import annotations

import secrets
from pathlib import Path

import chess
import torch

from ..canonical import canonical_board, mirror_move
from ..encoding import encode_board
from ..moves import index_to_move, legal_action_mask
from .model import INPUT_FORMAT, MiniChess, MiniConfig


def load_mini(checkpoint: str | Path, device: str = "cpu") -> MiniChess:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "chessformer-mini-v1" or payload.get("input_format") != INPUT_FORMAT:
        raise ValueError("Expected a chessformer-mini-v1 checkpoint with matching input format")
    model = MiniChess(MiniConfig(**payload["config"]))
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval()


class MiniPlayer:
    """Arg-max by default; a temperature samples from the same single forward.

    Two deterministic bots that meet again replay the same game move for move,
    so an online bot needs a little sampling. It never adds a forward pass.

    The board encoding carries no history, so the network cannot see that Lichess
    ends a game on a third repetition: on its first Lichess night it drew six
    games that way from +350 to +990 cp. With `repetition_value` set, while the
    value head rates the position above it for the side to move, a move that
    would make a position occur a third time is withheld -- unless every legal
    move would. This is the workaround of Ruoss et al. (2024, section 4), which
    scores such a move as a draw. It reads the game's own move stack and adds no
    forward pass.
    """

    def __init__(self, model: MiniChess, temperature: float = 0.0, seed: int | None = None,
                 sample_plies: int | None = None, repetition_value: float | None = None) -> None:
        if not temperature >= 0:
            raise ValueError("temperature must be non-negative")
        if sample_plies is not None and sample_plies < 0:
            raise ValueError("sample_plies must be non-negative")
        if repetition_value is not None and not -1.0 <= repetition_value <= 1.0:
            raise ValueError("repetition_value must lie in [-1, 1]")
        self.model = model.eval()
        self.temperature = temperature
        # Replays start in the opening; sampling only there keeps the arg-max,
        # and its strength, for the rest of the game. None samples throughout.
        self.sample_plies = sample_plies
        self.repetition_value = repetition_value
        # Positions where the network's arg-max was a third repetition and was withheld.
        self.repetitions_withheld = 0
        self.generator = torch.Generator()
        self.generator.manual_seed(seed if seed is not None else secrets.randbits(63))

    @torch.inference_mode()
    def choose_move(self, board: chess.Board) -> chess.Move | None:
        if board.chess960:
            raise ValueError("The action vocabulary supports standard chess only")
        canonical, mirrored = canonical_board(board)
        legal = legal_action_mask(canonical)
        if not legal.any():
            return None
        encoded = encode_board(canonical)
        device = next(self.model.parameters()).device
        pieces = torch.as_tensor(encoded.pieces, device=device).long()[None]
        castling = torch.tensor([int(encoded.castling)], device=device)
        ep_file = torch.tensor([int(encoded.ep_file)], device=device)
        logits, value_logits = self.model(pieces, castling, ep_file)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite neural policy; refusing to substitute a scripted move")
        mask = torch.as_tensor(legal, device=device)
        masked = logits[0].masked_fill(~mask, -torch.inf)
        if self.repetition_value is not None:
            masked = self._withhold_third_repetitions(board, masked, mirrored, value_logits[0])
        if self.temperature > 0 and (self.sample_plies is None or board.ply() < self.sample_plies):
            probabilities = torch.softmax(masked.float().cpu() / self.temperature, dim=-1)
            action = torch.multinomial(probabilities, 1, generator=self.generator).item()
        else:
            action = masked.argmax().item()
        move = index_to_move(action)
        return mirror_move(move) if mirrored else move

    def _withhold_third_repetitions(self, board: chess.Board, masked: torch.Tensor, mirrored: bool,
                                    value_logits: torch.Tensor) -> torch.Tensor:
        support = self.model.value_support.to(device=value_logits.device, dtype=torch.float32)
        value = float((torch.softmax(value_logits.float(), dim=-1) * support).sum())
        if value <= self.repetition_value:
            return masked
        legal = torch.isfinite(masked)
        repeating = torch.zeros_like(legal)
        for index in legal.nonzero().flatten().tolist():
            move = index_to_move(index)
            board.push(mirror_move(move) if mirrored else move)
            repeating[index] = board.is_repetition(3)
            board.pop()
        if not repeating.any() or not (legal & ~repeating).any():
            return masked
        if repeating[masked.argmax()]:
            self.repetitions_withheld += 1
        return masked.masked_fill(repeating, -torch.inf)
