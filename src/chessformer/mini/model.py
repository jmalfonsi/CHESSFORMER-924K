from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..model import RMSNorm
from ..moves import action_squares

PARAMETER_LIMIT = 1_000_000
INPUT_FORMAT = "canonical_pieces_castling_ep_v1"


@dataclass(frozen=True)
class MiniConfig:
    architecture: str = "geometric"
    width: int = 96
    layers: int = 8
    heads: int = 4
    ff_width: int = 256
    policy_dim: int = 64
    value_bins: int = 64
    # Passes through the block stack. The blocks are shared: every further pass adds
    # a full stack of compute but only a `width`-sized marker of which pass it is.
    loops: int = 1

    def __post_init__(self) -> None:
        if self.architecture not in ("geometric", "hybrid", "resnet"):
            raise ValueError(f"Unknown architecture: {self.architecture}")
        if min(self.width, self.layers, self.heads, self.ff_width, self.policy_dim, self.loops) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads or self.value_bins < 2:
            raise ValueError("width must be divisible by heads; value_bins must be >= 2")

    def to_dict(self) -> dict:
        values = asdict(self)
        # A single pass is what every checkpoint written before `loops` describes;
        # leaving the key out keeps their configurations equal for resume and --init.
        if values["loops"] == 1:
            del values["loops"]
        return values


CANDIDATES = {
    "geometric": MiniConfig(),
    "hybrid": MiniConfig(architecture="hybrid", width=112, layers=6, ff_width=288),
    "resnet": MiniConfig(architecture="resnet", width=80, layers=8),
    "geometric-r2": MiniConfig(loops=2),
    "geometric-r3": MiniConfig(loops=3),
}


def as_grid(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], 8, 8)


def as_squares(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2)


class GeometricAttention(nn.Module):
    """Global attention with learned biases indexed only by square displacement."""

    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        self.relative_bias = nn.Parameter(torch.zeros(config.heads, 225))
        squares = torch.arange(64)
        ranks, files = squares // 8, squares % 8
        index = (ranks[:, None] - ranks[None, :] + 7) * 15
        index = index + files[:, None] - files[None, :] + 7
        self.register_buffer("relative_index", index, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).unbind(2)
        bias = self.relative_bias[:, self.relative_index].to(q.dtype)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=bias[None], dropout_p=0.0,
        )
        return self.out(y.transpose(1, 2).reshape(b, n, d))


class AttentionBlock(nn.Module):
    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        d = config.width
        self.local_norm = RMSNorm(d) if config.architecture == "hybrid" else None
        self.local = (
            nn.Conv2d(d, d, 3, padding=1, groups=d, bias=False)
            if self.local_norm is not None else None
        )
        self.attn_norm = RMSNorm(d)
        self.attn = GeometricAttention(config)
        self.ff_norm = RMSNorm(d)
        self.gate = nn.Linear(d, config.ff_width, bias=False)
        self.up = nn.Linear(d, config.ff_width, bias=False)
        self.down = nn.Linear(config.ff_width, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.local is not None:
            x = x + as_squares(self.local(as_grid(self.local_norm(x))))
        x = x + self.attn(self.attn_norm(x))
        y = self.ff_norm(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class ConvBlock(nn.Module):
    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        d = config.width
        self.norm1 = RMSNorm(d)
        self.conv1 = nn.Conv2d(d, d, 3, padding=1, bias=False)
        self.norm2 = RMSNorm(d)
        self.conv2 = nn.Conv2d(d, d, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = as_squares(self.conv1(as_grid(F.silu(self.norm1(x)))))
        return x + as_squares(self.conv2(as_grid(F.silu(self.norm2(y)))))


class MoveHead(nn.Module):
    """All 1,968 logits at once, including context-dependent underpromotions."""

    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        self.scale = config.policy_dim ** -0.5
        self.query = nn.Linear(config.width, config.policy_dim, bias=False)
        self.key = nn.Linear(config.width, config.policy_dim, bias=False)
        self.promotion_origin = nn.Linear(config.width, 4)
        self.promotion_destination = nn.Linear(config.width, 4, bias=False)
        origins, destinations, promotions = action_squares()
        for name, values in zip(("origins", "destinations", "promotions"),
                                (origins, destinations, promotions), strict=True):
            self.register_buffer(name, torch.from_numpy(values.copy()), persistent=False)

    def forward(self, squares: torch.Tensor) -> torch.Tensor:
        scores = (self.query(squares) @ self.key(squares).transpose(1, 2)) * self.scale
        logits = scores[:, self.origins, self.destinations]
        promotion = (self.promotion_origin(squares)[:, self.origins]
                     + self.promotion_destination(squares)[:, self.destinations])
        index = (self.promotions - 1).clamp_min(0)[None, :, None]
        extra = promotion.gather(2, index.expand(squares.shape[0], -1, 1)).squeeze(2)
        return logits + extra * (self.promotions != 0)


class MiniChess(nn.Module):
    """One board encoding produces a complete policy and an auxiliary value.

    Inputs are canonical (side to move is White). Counters/history are excluded
    explicitly because the initial Lichess evaluation corpus does not have them.
    No legal mask or candidate list enters the neural network.
    """

    def __init__(self, config: MiniConfig) -> None:
        super().__init__()
        self.config = config
        d = config.width
        self.pieces = nn.Embedding(13, d)
        self.squares = nn.Embedding(64, d)
        self.castling = nn.Embedding(16, d)
        self.ep_file = nn.Embedding(9, d)
        block = ConvBlock if config.architecture == "resnet" else AttentionBlock
        self.blocks = nn.ModuleList(block(config) for _ in range(config.layers))
        if config.loops > 1:
            self.loop_embedding = nn.Parameter(torch.zeros(config.loops, d))
        self.norm = RMSNorm(d)
        self.policy = MoveHead(config)
        self.value = nn.Sequential(
            nn.Linear(d, d // 2, bias=False), nn.SiLU(),
            nn.Linear(d // 2, config.value_bins),
        )
        self.register_buffer("square_ids", torch.arange(64), persistent=False)
        self.register_buffer("value_support", torch.linspace(-1, 1, config.value_bins),
                             persistent=False)
        if self.num_parameters() > PARAMETER_LIMIT:
            raise ValueError(f"Model has {self.num_parameters():,} parameters; limit is 1,000,000")
        self.apply(self._init)
        for layer in self.blocks:
            projections = (layer.conv2,) if isinstance(layer, ConvBlock) else (layer.attn.out, layer.down)
            for projection in projections:
                with torch.no_grad():
                    projection.weight.mul_(1 / math.sqrt(2 * config.layers * config.loops))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, pieces: torch.Tensor, castling: torch.Tensor,
                ep_file: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pieces.ndim != 2 or pieces.shape[1] != 64:
            raise ValueError("pieces must have shape [B,64]")
        x = self.pieces(pieces.long()) + self.squares(self.square_ids)[None]
        x = x + self.castling(castling.long())[:, None] + self.ep_file(ep_file.long())[:, None]
        for loop in range(self.config.loops):
            if self.config.loops > 1:
                x = x + self.loop_embedding[loop]
            for block in self.blocks:
                x = block(x)
        x = self.norm(x)
        return self.policy(x), self.value(x.mean(dim=1))
