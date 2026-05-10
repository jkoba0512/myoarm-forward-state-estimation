"""Residual MLP forward model: ``[x_t, u_t] -> Δx_t``.

Two hidden layers with LayerNorm + ReLU, output layer predicts the
state delta directly. Caller computes ``x_{t+1} = x_t + Δx_t``. This is
the standard residual parameterization used in recent forward dynamics
papers (e.g. Chua et al. 2018, PETS); it tends to be easier to fit
than predicting ``x_{t+1}`` directly because most coordinates change
little per control step.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ForwardMLP(nn.Module):
    """Residual MLP that predicts the per-step state delta.

    Architecture (defaults are the Phase 0 baseline):
    ``Linear(state+action, h0) → LayerNorm → ReLU → … → Linear(h_last, state)``.
    The output is ``Δx``; ``forward(x, u)`` returns it without adding ``x``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        if isinstance(state_dim, bool) or not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError(f"state_dim must be a positive int, got {state_dim!r}")
        if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
            raise ValueError(f"action_dim must be a positive int, got {action_dim!r}")
        if not hidden_dims:
            raise ValueError("hidden_dims must be non-empty")
        for h in hidden_dims:
            if isinstance(h, bool) or not isinstance(h, int) or h <= 0:
                raise ValueError(
                    f"each hidden_dim must be a positive int, got {h!r}"
                )

        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.hidden_dims: tuple[int, ...] = tuple(int(h) for h in hidden_dims)

        layers: list[nn.Module] = []
        prev = state_dim + action_dim
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, state_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Return ``Δx`` of shape ``(..., state_dim)``."""
        if x.shape[-1] != self.state_dim:
            raise ValueError(
                f"x.shape[-1] must be {self.state_dim}, got {x.shape[-1]}"
            )
        if u.shape[-1] != self.action_dim:
            raise ValueError(
                f"u.shape[-1] must be {self.action_dim}, got {u.shape[-1]}"
            )
        if x.shape[:-1] != u.shape[:-1]:
            raise ValueError(
                f"x and u leading dims must match, got x.shape={tuple(x.shape)}, "
                f"u.shape={tuple(u.shape)}"
            )
        z = torch.cat([x, u], dim=-1)
        return self.net(z)

    def predict_next(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Convenience: ``x + Δx`` (closed-loop step)."""
        return x + self.forward(x, u)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"ForwardMLP(state_dim={self.state_dim}, "
            f"action_dim={self.action_dim}, "
            f"hidden_dims={self.hidden_dims})"
        )
