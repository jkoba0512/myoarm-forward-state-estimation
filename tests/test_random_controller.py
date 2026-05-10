"""Tests for myoarm_fse.controllers.random.RandomController."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.controllers import Controller, RandomController
from myoarm_fse.envs.state import MyoArmState


def _obs(action_dim: int = 4) -> MyoArmState:
    """Dummy observation; the random controller ignores its content."""
    return MyoArmState.from_arrays(
        qpos=np.zeros(2, dtype=np.float32),
        qvel=np.zeros(3, dtype=np.float32),
        act=np.zeros(action_dim, dtype=np.float32),
        tip_pos=np.zeros(3, dtype=np.float32),
        target_pos=np.zeros(3, dtype=np.float32),
        reach_err=np.zeros(3, dtype=np.float32),
    )


# --- constructor validation ---


class TestConstructor:
    def test_valid(self) -> None:
        c = RandomController(action_dim=34)
        assert c.action_dim == 34
        assert c.mean == 0.5
        assert c.sigma == 0.2

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_action_dim(self, bad: int) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=bad)

    def test_bool_action_dim(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=True)  # type: ignore[arg-type]

    def test_float_action_dim(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=10.0)  # type: ignore[arg-type]

    def test_bool_mean(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=4, mean=True)  # type: ignore[arg-type]

    def test_nan_mean(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=4, mean=float("nan"))

    def test_inf_mean(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=4, mean=float("inf"))

    def test_str_mean(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=4, mean="0.5")  # type: ignore[arg-type]

    def test_negative_sigma(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=4, sigma=-0.1)

    def test_bool_sigma(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=4, sigma=True)  # type: ignore[arg-type]

    def test_nan_sigma(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=4, sigma=float("nan"))

    def test_inf_sigma(self) -> None:
        with pytest.raises(ValueError):
            RandomController(action_dim=4, sigma=float("inf"))

    def test_bool_rng(self) -> None:
        with pytest.raises(TypeError):
            RandomController(action_dim=4, rng=True)  # type: ignore[arg-type]

    def test_random_state_rejected(self) -> None:
        with pytest.raises(TypeError):
            RandomController(action_dim=4, rng=np.random.RandomState(0))  # type: ignore[arg-type]


# --- output ---


class TestActOutput:
    def test_shape_and_dtype(self) -> None:
        c = RandomController(action_dim=34, rng=0)
        out = c.act(_obs(34))
        assert out.shape == (34,)
        assert out.dtype == np.float32

    def test_in_range(self) -> None:
        c = RandomController(action_dim=4, rng=0)
        for _ in range(50):
            out = c.act(_obs(4))
            assert (out >= 0.0).all()
            assert (out <= 1.0).all()

    def test_large_sigma_clips(self) -> None:
        # sigma=10 forces most samples outside [0,1]; clipping must keep them in.
        c = RandomController(action_dim=4, sigma=10.0, rng=0)
        for _ in range(50):
            out = c.act(_obs(4))
            assert (out >= 0.0).all()
            assert (out <= 1.0).all()

    def test_mean_is_centered(self) -> None:
        # Large sample, expect empirical mean ≈ configured mean.
        c = RandomController(action_dim=10000, mean=0.4, sigma=0.05, rng=0)
        out = c.act(_obs(10000))
        # Sample mean in [0.39, 0.41] is a generous tolerance.
        assert 0.39 <= out.mean() <= 0.41


# --- reproducibility ---


class TestReproducibility:
    def test_same_seed_same_output(self) -> None:
        a = RandomController(action_dim=4, rng=42)
        b = RandomController(action_dim=4, rng=42)
        np.testing.assert_array_equal(a.act(_obs(4)), b.act(_obs(4)))

    def test_different_seed_different_output(self) -> None:
        a = RandomController(action_dim=4, rng=1)
        b = RandomController(action_dim=4, rng=2)
        assert not np.array_equal(a.act(_obs(4)), b.act(_obs(4)))

    def test_reset_seed_restores_sequence(self) -> None:
        c = RandomController(action_dim=4, rng=42)
        first = c.act(_obs(4)).copy()
        c.act(_obs(4))
        c.reset(seed=42)
        again = c.act(_obs(4))
        np.testing.assert_array_equal(first, again)

    def test_reset_none_changes_sequence(self) -> None:
        c = RandomController(action_dim=4, rng=42)
        c.reset(seed=None)
        a = c.act(_obs(4)).copy()
        c.reset(seed=None)
        b = c.act(_obs(4))
        assert not np.array_equal(a, b)

    def test_reset_bool_seed_raises(self) -> None:
        c = RandomController(action_dim=4)
        with pytest.raises(TypeError):
            c.reset(seed=True)  # type: ignore[arg-type]


# --- input validation ---


class TestInputValidation:
    def test_act_rejects_non_state(self) -> None:
        c = RandomController(action_dim=4)
        with pytest.raises(ValueError):
            c.act([0, 0, 0])  # type: ignore[arg-type]


# --- protocol conformance ---


def test_implements_controller_protocol() -> None:
    c = RandomController(action_dim=4)
    assert isinstance(c, Controller)
