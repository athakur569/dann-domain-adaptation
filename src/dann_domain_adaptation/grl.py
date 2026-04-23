"""Gradient Reversal Layer for domain-adversarial training."""

from __future__ import annotations

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradients with scale factor lambda_ during backward pass.

    Forward: identity
    Backward: multiply gradient by -lambda_

    Reference: Ganin et al., "Domain-Adversarial Training of Neural Networks,"
    JMLR 2016.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.save_for_backward(torch.tensor(lambda_))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (lambda_,) = ctx.saved_tensors
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction as a standard nn.Module.

    Args:
        lambda_: Reversal scale. Typically ramped from 0→1 via a schedule.
    """

    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)

    @staticmethod
    def lambda_schedule(p: float, gamma: float = 10.0) -> float:
        """Progressive lambda schedule: 2/(1+exp(-gamma*p)) - 1.

        Args:
            p:     Training progress in [0, 1].
            gamma: Steepness of the sigmoid ramp.

        Returns:
            lambda_ in [0, 1].
        """
        import math
        return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0
