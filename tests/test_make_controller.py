"""Tests for the controllers.make_controller factory."""

from __future__ import annotations

import pytest

from myoarm_fse.controllers import (
    HoldController,
    RandomController,
    make_controller,
)


# --- random ---


class TestRandomDispatch:
    def test_basic(self) -> None:
        c = make_controller({"name": "random"}, action_dim=4, seed=0)
        assert isinstance(c, RandomController)
        assert c.action_dim == 4

    def test_with_kwargs(self) -> None:
        c = make_controller(
            {"name": "random", "mean": 0.4, "sigma": 0.05},
            action_dim=4,
            seed=42,
        )
        assert isinstance(c, RandomController)
        assert c.mean == pytest.approx(0.4)
        assert c.sigma == pytest.approx(0.05)


# --- hold ---


class TestHoldDispatch:
    def test_basic(self) -> None:
        c = make_controller({"name": "hold"}, action_dim=4, seed=0)
        assert isinstance(c, HoldController)
        assert c.action_dim == 4
        assert c.value == 0.0

    def test_with_value(self) -> None:
        c = make_controller(
            {"name": "hold", "value": 0.3}, action_dim=4, seed=0,
        )
        assert isinstance(c, HoldController)
        assert c.value == pytest.approx(0.3)

    def test_seed_ignored_for_deterministic(self) -> None:
        # Hold is deterministic — different seeds must yield identical behavior.
        a = make_controller({"name": "hold", "value": 0.5}, action_dim=4, seed=0)
        b = make_controller({"name": "hold", "value": 0.5}, action_dim=4, seed=999)
        assert a.value == b.value
        assert a.action_dim == b.action_dim


# --- error cases ---


class TestErrors:
    def test_non_dict_spec(self) -> None:
        with pytest.raises(ValueError, match="dict"):
            make_controller("random", action_dim=4, seed=0)  # type: ignore[arg-type]

    def test_missing_name(self) -> None:
        with pytest.raises(ValueError, match="name"):
            make_controller({"sigma": 0.1}, action_dim=4, seed=0)

    def test_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="unknown controller"):
            make_controller({"name": "banana"}, action_dim=4, seed=0)

    def test_propagates_constructor_errors(self) -> None:
        # Hold rejects value > 1 — factory should surface that ValueError.
        with pytest.raises(ValueError):
            make_controller(
                {"name": "hold", "value": 2.0}, action_dim=4, seed=0,
            )
