from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChessFormerConfig:
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    n_actions: int = 1968
    n_piece_tokens: int = 13
    n_squares: int = 64
    n_castling_states: int = 16
    n_ep_files: int = 9  # 0 = none, 1..8 = a..h
    n_halfmove_buckets: int = 101  # clipped 0..100
    dropout: float = 0.0
    # 0 keeps the V1 scalar head: tanh, trained by Huber regression. Any value
    # >= 2 makes the head categorical over that many bins spanning [-1, 1],
    # trained by cross-entropy against a Gaussian smeared around the target.
    #
    # Why it matters here rather than in general: 15.3% of the labels in
    # eval-d2-sampled are exactly +/-1 (mate), which a tanh cannot reach at all,
    # and another 26% sit inside [0, +0.1). The target is trimodal, so its
    # conditional mean -- the only thing a regression can fit -- describes almost
    # none of it. A distribution can hold that shape; a point estimate cannot.
    value_bins: int = 0
    # "linear" is V1: one Linear(d_model, 1968) reading the single state token,
    # so all 64 square representations the transformer computed are discarded and
    # the whole move distribution has to squeeze through one 768-vector.
    # "attention" scores a move as the dot product of a query on its origin square
    # with a key on its destination -- the same shape as the board it describes,
    # for fewer parameters and a fraction of a percent of the model's compute.
    policy_head: str = "linear"
    policy_head_dim: int = 256
    # Optional categorical action-value head. It scores only the candidate action
    # ids supplied to forward(), while reusing the one board encoding already
    # computed for policy and V. Zero preserves every pre-Q checkpoint exactly.
    action_value_bins: int = 0
    action_value_head_dim: int = 128

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_actions != 1968:
            raise ValueError("V1 action vocabulary is fixed at 1968")
        if self.value_bins != 0 and self.value_bins < 2:
            raise ValueError("value_bins must be 0 (scalar head) or >= 2")
        if self.policy_head not in ("linear", "attention"):
            raise ValueError("policy_head must be 'linear' or 'attention'")
        if self.policy_head_dim < 1:
            raise ValueError("policy_head_dim must be >= 1")
        if self.action_value_bins != 0 and self.action_value_bins < 2:
            raise ValueError("action_value_bins must be 0 (disabled) or >= 2")
        if self.action_value_head_dim < 1:
            raise ValueError("action_value_head_dim must be >= 1")


# Cheap CPU/debug model. Not part of the $30 training plan.
MICRO_CONFIG = ChessFormerConfig(
    n_layers=2,
    d_model=128,
    n_heads=4,
    d_ff=384,
)

# Pipeline validation model: ~11.5M parameters.
SMALL_CONFIG = ChessFormerConfig(
    n_layers=6,
    d_model=384,
    n_heads=6,
    d_ff=1024,
)

# Main V1: ~143M parameters, safely below 150M.
MAIN_CONFIG = ChessFormerConfig(
    n_layers=20,
    d_model=768,
    n_heads=12,
    d_ff=2048,
)

PRESETS = {
    "micro": MICRO_CONFIG,
    "small": SMALL_CONFIG,
    "main": MAIN_CONFIG,
}
