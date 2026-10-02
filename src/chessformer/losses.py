from __future__ import annotations

import math

import torch
from torch.nn import functional as F

# Gaussian width as a multiple of the bin spacing. Too narrow and the head is
# back to predicting one bin, which is the classification problem the smoothing
# exists to avoid; too wide and neighbouring values stop being distinguishable.
DEFAULT_SIGMA_RATIO = 0.75


def mask_policy_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != legal_mask.shape:
        raise ValueError(f"logits and legal_mask shapes differ: {logits.shape} vs {legal_mask.shape}")
    if legal_mask.dtype != torch.bool:
        legal_mask = legal_mask.bool()
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def _weighted_mean(values: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Mean over weighted rows; padded rows carry weight 0 and drop out exactly."""
    weight = weight.to(values.dtype)
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def masked_policy_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    legal_mask: torch.Tensor,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy over legal moves, against a hard index or a distribution.

    A `[B]` target is the single best move, which is all V1 ever had. A
    `[B, n_actions]` one is the multi-PV distribution: same loss, but a move three
    centipawns behind is no longer scored as wrong as a move a rook behind.
    """
    masked = mask_policy_logits(logits, legal_mask)
    if target.ndim == 2:
        per_sample = -(target.float() * F.log_softmax(masked.float(), dim=-1)).sum(dim=-1)
    else:
        per_sample = F.cross_entropy(masked, target.long(), reduction="none")
    if weight is None:
        return per_sample.mean()
    return _weighted_mean(per_sample, weight)


def value_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.2,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if weight is None:
        return F.huber_loss(prediction.float(), target.float(), delta=delta)
    per_sample = F.huber_loss(prediction.float(), target.float(), delta=delta, reduction="none")
    return _weighted_mean(per_sample, weight)


def hl_gauss_targets(
    target: torch.Tensor, support: torch.Tensor, sigma: float
) -> torch.Tensor:
    """Spread each scalar target over the bins as a Gaussian, and normalise.

    The bin a target lands in gets most of the mass and its neighbours get the
    rest, so a head that predicts an adjacent bin is scored as nearly right --
    which plain classification over bins does not do, and which is the whole
    reason to smear rather than one-hot.

    The outermost edges run to infinity so that a target sitting exactly on
    +/-1, which 15.3% of eval-d2-sampled does, keeps all of its mass instead of
    losing the half that falls off the end. The expectation of the resulting
    distribution is then a shade inside +/-1 rather than on it -- about 11 cp at
    64 bins, in the region where the forced search decides the move anyway.
    """
    centres = support.to(torch.float32)
    infinity = torch.full((1,), float("inf"), device=centres.device)
    edges = torch.cat([-infinity, (centres[:-1] + centres[1:]) / 2, infinity])
    scaled = (edges[None, :] - target.to(torch.float32)[:, None]) / (sigma * math.sqrt(2.0))
    cdf = 0.5 * (1.0 + torch.erf(scaled))
    probabilities = cdf[:, 1:] - cdf[:, :-1]
    return probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def value_categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    support: torch.Tensor,
    sigma: float | None = None,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL divergence of the value head from the smeared target.

    A divergence rather than a raw cross-entropy, so that a perfect prediction
    scores 0. The entropy of the smeared target is a constant of the smearing --
    about 3.4 nats at 64 bins -- and carries no gradient, but leaving it in makes
    `--value-weight` mean something different from one head to the other, and a
    run whose value term silently swamped its policy term would look like a
    training loss that simply refused to fall.
    """
    if sigma is None:
        spacing = float(support[1] - support[0]) if support.numel() > 1 else 1.0
        sigma = DEFAULT_SIGMA_RATIO * spacing
    soft = hl_gauss_targets(target, support, sigma)
    log_soft = torch.log(soft.clamp_min(1e-12))
    per_sample = (soft * (log_soft - F.log_softmax(logits.float(), dim=-1))).sum(dim=-1)
    if weight is None:
        return per_sample.mean()
    return _weighted_mean(per_sample, weight)


def action_value_categorical_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    support: torch.Tensor,
    sigma: float | None = None,
    row_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """HL-Gauss loss for candidate Q values, normalised per board.

    `mask` distinguishes analysed candidates from fixed-width padding. A padded
    board row also drops out through `row_weight`, using the same semantics as
    policy and V without teaching Q that its zero-filled padding is a draw.
    """
    if logits.ndim != 3:
        raise ValueError("action-value logits must have shape [B,K,bins]")
    if target.shape != logits.shape[:2] or mask.shape != logits.shape[:2]:
        raise ValueError("action-value targets and mask must match logits [B,K]")
    if sigma is None:
        spacing = float(support[1] - support[0]) if support.numel() > 1 else 1.0
        sigma = DEFAULT_SIGMA_RATIO * spacing
    soft = hl_gauss_targets(target.reshape(-1), support, sigma).reshape_as(logits)
    log_soft = torch.log(soft.clamp_min(1e-12))
    per_candidate = (
        soft * (log_soft - F.log_softmax(logits.float(), dim=-1))
    ).sum(dim=-1)
    candidate_weight = mask.to(per_candidate.dtype)
    per_row = (per_candidate * candidate_weight).sum(dim=-1) / candidate_weight.sum(
        dim=-1
    ).clamp_min(1.0)
    if row_weight is None:
        return per_row.mean()
    return _weighted_mean(per_row, row_weight)


def combined_loss(
    policy_logits: torch.Tensor,
    value_prediction: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    legal_mask: torch.Tensor,
    value_weight: float,
    weight: torch.Tensor | None = None,
    value_support: torch.Tensor | None = None,
    value_sigma: float | None = None,
    action_value_prediction: torch.Tensor | None = None,
    action_value_target: torch.Tensor | None = None,
    action_value_mask: torch.Tensor | None = None,
    action_value_weight: float = 0.0,
    action_value_support: torch.Tensor | None = None,
    action_value_sigma: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Policy cross-entropy plus the value term the head's shape calls for.

    A `[B]` prediction is the V1 scalar head and is regressed; a `[B, bins]` one
    is the categorical head and is trained by cross-entropy. The shape decides,
    so a head and a loss can never be paired wrongly by a forgotten flag.
    """
    p_loss = masked_policy_cross_entropy(policy_logits, policy_target, legal_mask, weight=weight)
    if value_prediction.ndim == 2:
        if value_support is None:
            raise ValueError("A categorical value head needs its support")
        v_loss = value_categorical_loss(
            value_prediction, value_target, value_support, value_sigma, weight=weight
        )
    else:
        v_loss = value_huber_loss(value_prediction, value_target, weight=weight)
    if action_value_prediction is None:
        if action_value_weight:
            raise ValueError("action_value_weight requires action-value predictions")
        q_loss = torch.zeros((), device=p_loss.device, dtype=p_loss.dtype)
    else:
        if action_value_target is None or action_value_mask is None:
            raise ValueError("action-value predictions require targets and a mask")
        if action_value_support is None:
            raise ValueError("A categorical action-value head needs its support")
        q_loss = action_value_categorical_loss(
            action_value_prediction,
            action_value_target,
            action_value_mask,
            action_value_support,
            action_value_sigma,
            row_weight=weight,
        )
    total = p_loss + value_weight * v_loss + action_value_weight * q_loss
    return total, {
        "loss": total.detach(),
        "policy_loss": p_loss.detach(),
        "value_loss": v_loss.detach(),
        "action_value_loss": q_loss.detach(),
    }
