from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ChessFormerConfig
from .moves import action_squares


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate the variance in fp32 for numerical stability, then cast back.
        dtype = x.dtype
        x_fp32 = x.float()
        x_norm = x_fp32 * torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_norm.to(dtype) * self.weight)


class SelfAttention(nn.Module):
    def __init__(self, config: ChessFormerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x).view(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out(y)


class SwiGLU(nn.Module):
    def __init__(self, config: ChessFormerConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, config: ChessFormerConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = SelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class AttentionPolicyHead(nn.Module):
    """Move logits as a query on the origin square against a key on the target.

    The V1 head is `Linear(d_model, 1968)` applied to the state token alone: the
    64 square representations the transformer just computed are thrown away, and
    a single 768-vector has to carry the ordering of every move on the board. Here
    a move's score is read off the two squares it actually connects, which is the
    inductive bias the board already has, for fewer parameters than the linear
    head and about a tenth of a percent of the model's compute.

    Promotions carry a learned scalar per piece on top of their underlying
    geometry. That is deliberately crude -- it says "a queen is usually right"
    and nothing about when a knight is -- but underpromotion is decided by the
    forced search, which generates every promotion regardless of what the policy
    thinks of it.
    """

    def __init__(self, config: ChessFormerConfig) -> None:
        super().__init__()
        self.head_dim = config.policy_head_dim
        self.query = nn.Linear(config.d_model, self.head_dim, bias=False)
        self.key = nn.Linear(config.d_model, self.head_dim, bias=False)
        # Index 0 is "no promotion" and stays at zero: a softmax is only blind to
        # an offset added to *every* logit, so a learnable constant on the 1792
        # ordinary moves would silently be a promotion penalty.
        self.promotion_bias = nn.Parameter(torch.zeros(5))
        origins, destinations, promotions = action_squares()
        self.register_buffer(
            "action_index",
            torch.from_numpy(origins * 64 + destinations),
            persistent=False,
        )
        self.register_buffer(
            "promotion_index", torch.from_numpy(promotions), persistent=False
        )

    def forward(self, squares: torch.Tensor) -> torch.Tensor:
        """[B, 64, d_model] square representations -> [B, 1968] action logits."""
        query = self.query(squares)
        key = self.key(squares)
        scores = query @ key.transpose(1, 2) / math.sqrt(self.head_dim)
        flat = scores.reshape(scores.shape[0], 64 * 64)
        bias = torch.cat(
            (torch.zeros(1, device=self.promotion_bias.device, dtype=self.promotion_bias.dtype),
             self.promotion_bias[1:])
        )
        return flat[:, self.action_index] + bias[self.promotion_index]


class ActionValueHead(nn.Module):
    """Categorical Q(s,a) for a supplied set of candidate actions.

    The board Transformer runs once. Each candidate then reads the encoded
    origin square, destination square and global state, plus its promotion kind.
    This keeps the action-value cost linear in a small head instead of encoding
    every child board independently.
    """

    def __init__(self, config: ChessFormerConfig) -> None:
        super().__init__()
        hidden = config.action_value_head_dim
        self.origin = nn.Linear(config.d_model, hidden, bias=False)
        self.destination = nn.Linear(config.d_model, hidden, bias=False)
        self.state = nn.Linear(config.d_model, hidden, bias=False)
        self.promotion = nn.Embedding(5, hidden)
        self.output = nn.Linear(hidden, config.action_value_bins, bias=True)

        origins, destinations, promotions = action_squares()
        self.register_buffer("action_origins", torch.from_numpy(origins), persistent=False)
        self.register_buffer(
            "action_destinations", torch.from_numpy(destinations), persistent=False
        )
        self.register_buffer(
            "action_promotions", torch.from_numpy(promotions), persistent=False
        )

    def forward(
        self,
        state: torch.Tensor,
        squares: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return `[B,K,bins]` logits for action ids shaped `[B,K]`."""
        if actions.ndim != 2 or actions.shape[0] != squares.shape[0]:
            raise ValueError(
                "actions must have shape [B,K] with the same batch size as the board"
            )
        actions = actions.long()

        origins = self.action_origins[actions]
        destinations = self.action_destinations[actions]
        promotions = self.action_promotions[actions]
        gather_shape = (*origins.shape, squares.shape[-1])
        origin_state = torch.gather(
            squares, 1, origins[..., None].expand(gather_shape)
        )
        destination_state = torch.gather(
            squares, 1, destinations[..., None].expand(gather_shape)
        )
        hidden = (
            self.origin(origin_state)
            + self.destination(destination_state)
            + self.state(state)[:, None, :]
            + self.promotion(promotions)
        )
        return self.output(F.silu(hidden))


class ChessFormer(nn.Module):
    """Bidirectional board Transformer with policy and value heads."""

    def __init__(self, config: ChessFormerConfig) -> None:
        super().__init__()
        self.config = config

        self.piece_embedding = nn.Embedding(config.n_piece_tokens, config.d_model)
        self.square_embedding = nn.Embedding(config.n_squares, config.d_model)

        self.state_token = nn.Parameter(torch.zeros(config.d_model))
        self.side_embedding = nn.Embedding(2, config.d_model)
        self.castling_embedding = nn.Embedding(config.n_castling_states, config.d_model)
        self.ep_embedding = nn.Embedding(config.n_ep_files, config.d_model)
        self.halfmove_embedding = nn.Embedding(config.n_halfmove_buckets, config.d_model)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)

        self.policy_head: nn.Module
        if config.policy_head == "attention":
            self.policy_head = AttentionPolicyHead(config)
        else:
            self.policy_head = nn.Linear(config.d_model, config.n_actions, bias=True)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2, bias=False),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, max(1, config.value_bins), bias=True),
        )
        self.action_value_head = (
            ActionValueHead(config) if config.action_value_bins else None
        )
        if config.value_bins:
            # Bin centres, evenly spaced and inclusive of both ends, so that +/-1
            # is a bin the head can actually predict rather than an asymptote it
            # can only approach.
            self.register_buffer(
                "value_support",
                torch.linspace(-1.0, 1.0, config.value_bins),
                persistent=False,
            )
        if config.action_value_bins:
            self.register_buffer(
                "action_value_support",
                torch.linspace(-1.0, 1.0, config.action_value_bins),
                persistent=False,
            )

        self.register_buffer("square_ids", torch.arange(64), persistent=False)
        self.apply(self._init_weights)
        nn.init.normal_(self.state_token, mean=0.0, std=0.02)
        self._scale_residual_projections()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        # GPT-style residual scaling keeps deep residual streams stable at init.
        std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out.weight, mean=0.0, std=std)
            nn.init.normal_(block.ffn.down.weight, mean=0.0, std=std)

    def forward(
        self,
        pieces: torch.Tensor,
        side: torch.Tensor,
        castling: torch.Tensor,
        ep_file: torch.Tensor,
        halfmove: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if pieces.ndim != 2 or pieces.shape[1] != 64:
            raise ValueError(f"pieces must have shape [B,64], got {tuple(pieces.shape)}")

        square_ids = self.square_ids.to(pieces.device)
        board_tokens = self.piece_embedding(pieces.long()) + self.square_embedding(square_ids)[None, :, :]

        state = self.state_token[None, :]
        state = state + self.side_embedding(side.long())
        state = state + self.castling_embedding(castling.long())
        state = state + self.ep_embedding(ep_file.long())
        state = state + self.halfmove_embedding(halfmove.long())

        x = torch.cat((state[:, None, :], board_tokens), dim=1)  # [B,65,D]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        state_out = x[:, 0]
        # The attention head reads the board; the linear head reads the summary.
        policy_logits = self.policy_head(
            x[:, 1:] if self.config.policy_head == "attention" else state_out
        )
        value = self.value_head(state_out)
        if not self.config.value_bins:
            value = torch.tanh(value.squeeze(-1))
        # Categorical heads return [B, value_bins] logits. `scalar_value` is the
        # single place that collapses them, so nothing downstream has to know
        # which head it is holding.
        if actions is None:
            return policy_logits, value
        if self.action_value_head is None:
            raise ValueError("candidate actions require an enabled action-value head")
        action_values = self.action_value_head(state_out, x[:, 1:], actions)
        return policy_logits, value, action_values

    def scalar_value(self, value: torch.Tensor) -> torch.Tensor:
        """The head's output as one number per position, in [-1, 1]."""
        if not self.config.value_bins:
            return value
        probabilities = torch.softmax(value.float(), dim=-1)
        return probabilities @ self.value_support.to(probabilities.dtype)

    def scalar_action_values(self, action_values: torch.Tensor) -> torch.Tensor:
        """Collapse categorical Q logits to one side-to-move value per candidate."""
        if self.action_value_head is None:
            raise ValueError("the action-value head is disabled")
        probabilities = torch.softmax(action_values.float(), dim=-1)
        return probabilities @ self.action_value_support.to(probabilities.dtype)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
