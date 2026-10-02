import torch

from chessformer.losses import masked_policy_cross_entropy


def test_illegal_huge_logit_is_ignored():
    logits = torch.zeros(1, 4)
    logits[0, 3] = 1000.0  # illegal action tries to dominate
    legal = torch.tensor([[True, True, False, False]])
    target = torch.tensor([0])
    loss = masked_policy_cross_entropy(logits, target, legal)
    # Two equally scored legal actions -> CE = ln(2), not ~1000.
    assert torch.allclose(loss, torch.tensor(0.6931472), atol=1e-5)
