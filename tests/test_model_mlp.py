"""Tests for myoarm_fse.models.mlp.ForwardMLP (PyTorch, no MyoSuite)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from myoarm_fse.models import ForwardMLP


# --- constructor validation ---


class TestConstructor:
    def test_valid_default(self) -> None:
        m = ForwardMLP(state_dim=83, action_dim=34)
        assert m.state_dim == 83
        assert m.action_dim == 34
        assert m.hidden_dims == (256, 256)

    def test_custom_hidden(self) -> None:
        m = ForwardMLP(state_dim=10, action_dim=4, hidden_dims=(64, 32, 16))
        assert m.hidden_dims == (64, 32, 16)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_state_dim(self, bad: int) -> None:
        with pytest.raises(ValueError):
            ForwardMLP(state_dim=bad, action_dim=4)

    def test_bool_state_dim(self) -> None:
        with pytest.raises(ValueError):
            ForwardMLP(state_dim=True, action_dim=4)  # type: ignore[arg-type]

    def test_empty_hidden_dims(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ForwardMLP(state_dim=10, action_dim=4, hidden_dims=())

    def test_zero_hidden_dim(self) -> None:
        with pytest.raises(ValueError):
            ForwardMLP(state_dim=10, action_dim=4, hidden_dims=(64, 0))

    def test_bool_hidden_dim(self) -> None:
        with pytest.raises(ValueError):
            ForwardMLP(state_dim=10, action_dim=4, hidden_dims=(64, True))  # type: ignore[arg-type]


# --- forward pass ---


class TestForward:
    def test_output_shape_batch(self) -> None:
        m = ForwardMLP(state_dim=10, action_dim=4, hidden_dims=(32, 32))
        x = torch.zeros((5, 10))
        u = torch.zeros((5, 4))
        out = m(x, u)
        assert out.shape == (5, 10)

    def test_output_shape_single(self) -> None:
        m = ForwardMLP(state_dim=10, action_dim=4, hidden_dims=(32, 32))
        x = torch.zeros(10)
        u = torch.zeros(4)
        out = m(x, u)
        assert out.shape == (10,)

    def test_predict_next_adds_x(self) -> None:
        m = ForwardMLP(state_dim=4, action_dim=2, hidden_dims=(8,))
        x = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        u = torch.tensor([[0.5, 0.5]])
        dx = m(x, u)
        nxt = m.predict_next(x, u)
        torch.testing.assert_close(nxt, x + dx)

    def test_state_dim_mismatch(self) -> None:
        m = ForwardMLP(state_dim=10, action_dim=4)
        with pytest.raises(ValueError, match="x.shape"):
            m(torch.zeros((3, 9)), torch.zeros((3, 4)))

    def test_action_dim_mismatch(self) -> None:
        m = ForwardMLP(state_dim=10, action_dim=4)
        with pytest.raises(ValueError, match="u.shape"):
            m(torch.zeros((3, 10)), torch.zeros((3, 5)))

    def test_leading_dims_mismatch(self) -> None:
        m = ForwardMLP(state_dim=10, action_dim=4)
        with pytest.raises(ValueError, match="leading dims"):
            m(torch.zeros((3, 10)), torch.zeros((4, 4)))


# --- gradient flow ---


def test_gradient_flows_to_all_parameters() -> None:
    m = ForwardMLP(state_dim=8, action_dim=3, hidden_dims=(16, 16))
    x = torch.randn((4, 8))
    u = torch.randn((4, 3))
    target = torch.randn((4, 8))
    pred = m(x, u)
    loss = ((pred - target) ** 2).mean()
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"


# --- residual semantics: untrained model can still produce sane Δx ---


def test_zero_input_produces_finite_output() -> None:
    m = ForwardMLP(state_dim=8, action_dim=3, hidden_dims=(16, 16))
    out = m(torch.zeros(8), torch.zeros(3))
    assert torch.isfinite(out).all()


def test_num_parameters() -> None:
    m = ForwardMLP(state_dim=8, action_dim=3, hidden_dims=(16, 16))
    n = m.num_parameters()
    # Rough sanity: at least 8 (linear bias only) and at most 1e5.
    assert 0 < n < 100_000


# --- determinism under seed ---


def test_init_deterministic_with_torch_manual_seed() -> None:
    torch.manual_seed(0)
    a = ForwardMLP(state_dim=4, action_dim=2, hidden_dims=(8,))
    torch.manual_seed(0)
    b = ForwardMLP(state_dim=4, action_dim=2, hidden_dims=(8,))
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(pa, pb)
