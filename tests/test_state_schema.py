"""Tests for myoarm_fse.envs.state.StateSpec and MyoArmState."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.envs import MyoArmState, StateSpec


# --- StateSpec ---


class TestStateSpecConstructor:
    def test_valid_dims(self) -> None:
        spec = StateSpec(qpos_dim=7, qvel_dim=7, act_dim=39)
        assert spec.qpos_dim == 7
        assert spec.qvel_dim == 7
        assert spec.act_dim == 39

    def test_dim_property_sums_fields(self) -> None:
        spec = StateSpec(qpos_dim=7, qvel_dim=7, act_dim=39)
        # qpos + qvel + act + tip_pos(3) + target_pos(3) + reach_err(3)
        assert spec.dim == 7 + 7 + 39 + 3 + 3 + 3

    @pytest.mark.parametrize("bad", [0, -1, -100])
    def test_non_positive_qpos_dim_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            StateSpec(qpos_dim=bad, qvel_dim=7, act_dim=39)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_qvel_dim_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            StateSpec(qpos_dim=7, qvel_dim=bad, act_dim=39)

    @pytest.mark.parametrize("bad", [0, -1])
    def test_non_positive_act_dim_raises(self, bad: int) -> None:
        with pytest.raises(ValueError):
            StateSpec(qpos_dim=7, qvel_dim=7, act_dim=bad)

    def test_bool_qpos_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            StateSpec(qpos_dim=True, qvel_dim=7, act_dim=39)  # type: ignore[arg-type]

    def test_bool_qvel_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            StateSpec(qpos_dim=7, qvel_dim=True, act_dim=39)  # type: ignore[arg-type]

    def test_bool_act_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            StateSpec(qpos_dim=7, qvel_dim=7, act_dim=True)  # type: ignore[arg-type]

    def test_float_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            StateSpec(qpos_dim=7.0, qvel_dim=7, act_dim=39)  # type: ignore[arg-type]

    def test_str_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            StateSpec(qpos_dim="7", qvel_dim=7, act_dim=39)  # type: ignore[arg-type]


class TestStateSpecLayout:
    def test_layout_keys_in_field_order(self) -> None:
        spec = StateSpec(qpos_dim=2, qvel_dim=3, act_dim=5)
        layout = spec.layout()
        assert list(layout.keys()) == [
            "qpos",
            "qvel",
            "act",
            "tip_pos",
            "target_pos",
            "reach_err",
        ]

    def test_layout_slices_are_contiguous_and_match_dims(self) -> None:
        spec = StateSpec(qpos_dim=2, qvel_dim=3, act_dim=5)
        layout = spec.layout()
        assert layout["qpos"] == slice(0, 2)
        assert layout["qvel"] == slice(2, 5)
        assert layout["act"] == slice(5, 10)
        assert layout["tip_pos"] == slice(10, 13)
        assert layout["target_pos"] == slice(13, 16)
        assert layout["reach_err"] == slice(16, 19)

    def test_layout_total_matches_dim(self) -> None:
        spec = StateSpec(qpos_dim=4, qvel_dim=4, act_dim=11)
        layout = spec.layout()
        assert layout["reach_err"].stop == spec.dim


# --- MyoArmState construction ---


def _valid_arrays(qpos_dim: int = 2, qvel_dim: int = 3, act_dim: int = 5) -> dict:
    return dict(
        qpos=np.zeros(qpos_dim, dtype=np.float32),
        qvel=np.zeros(qvel_dim, dtype=np.float32),
        act=np.zeros(act_dim, dtype=np.float32),
        tip_pos=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        target_pos=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        reach_err=np.array([-0.3, -0.3, -0.3], dtype=np.float32),
    )


class TestFromArrays:
    def test_basic_construction(self) -> None:
        state = MyoArmState.from_arrays(**_valid_arrays())
        assert isinstance(state, MyoArmState)
        assert state.qpos.shape == (2,)
        assert state.qvel.shape == (3,)
        assert state.act.shape == (5,)
        assert state.tip_pos.shape == (3,)
        assert state.target_pos.shape == (3,)
        assert state.reach_err.shape == (3,)

    def test_dtype_is_float32(self) -> None:
        # Pass float64 inputs; from_arrays should coerce.
        state = MyoArmState.from_arrays(
            qpos=np.zeros(2, dtype=np.float64),
            qvel=np.zeros(3, dtype=np.float64),
            act=np.zeros(5, dtype=np.float64),
            tip_pos=[0.1, 0.2, 0.3],
            target_pos=(0.4, 0.5, 0.6),
            reach_err=[-0.3, -0.3, -0.3],
        )
        for name in ("qpos", "qvel", "act", "tip_pos", "target_pos", "reach_err"):
            assert getattr(state, name).dtype == np.float32

    def test_accepts_list_and_tuple_inputs(self) -> None:
        state = MyoArmState.from_arrays(
            qpos=[0.0, 0.0],
            qvel=(0.0, 0.0, 0.0),
            act=[0.0] * 5,
            tip_pos=[0.0, 0.0, 0.0],
            target_pos=(0.0, 0.0, 0.0),
            reach_err=[0.0, 0.0, 0.0],
        )
        assert state.qpos.shape == (2,)
        assert state.qvel.shape == (3,)


class TestDirectConstructorValidation:
    def test_rejects_non_ndarray(self) -> None:
        kwargs = _valid_arrays()
        kwargs["qpos"] = [0.0, 0.0]  # list, not ndarray
        with pytest.raises(ValueError, match="np.ndarray"):
            MyoArmState(**kwargs)

    def test_rejects_wrong_dtype(self) -> None:
        kwargs = _valid_arrays()
        kwargs["qpos"] = np.zeros(2, dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            MyoArmState(**kwargs)


# --- shape validation ---


@pytest.mark.parametrize("name", ["qpos", "qvel", "act"])
def test_2d_input_for_dynamic_field_raises(name: str) -> None:
    arrays = _valid_arrays()
    arrays[name] = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="1-D"):
        MyoArmState(**arrays)


@pytest.mark.parametrize("name", ["tip_pos", "target_pos", "reach_err"])
@pytest.mark.parametrize("bad_shape", [(2,), (4,), (3, 1), ()])
def test_wrong_shape_for_cart_field_raises(
    name: str, bad_shape: tuple
) -> None:
    arrays = _valid_arrays()
    arrays[name] = np.zeros(bad_shape, dtype=np.float32)
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        MyoArmState(**arrays)


# --- finite validation ---


@pytest.mark.parametrize(
    "name", ["qpos", "qvel", "act", "tip_pos", "target_pos", "reach_err"]
)
def test_nan_in_field_raises(name: str) -> None:
    arrays = _valid_arrays()
    arrays[name] = arrays[name].copy()
    arrays[name][0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        MyoArmState(**arrays)


@pytest.mark.parametrize(
    "name", ["qpos", "qvel", "act", "tip_pos", "target_pos", "reach_err"]
)
def test_inf_in_field_raises(name: str) -> None:
    arrays = _valid_arrays()
    arrays[name] = arrays[name].copy()
    arrays[name][0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        MyoArmState(**arrays)


# --- spec / flatten / unflatten ---


class TestSpec:
    def test_spec_matches_field_dims(self) -> None:
        state = MyoArmState.from_arrays(**_valid_arrays(qpos_dim=4, qvel_dim=6, act_dim=11))
        spec = state.spec()
        assert spec.qpos_dim == 4
        assert spec.qvel_dim == 6
        assert spec.act_dim == 11
        assert spec.dim == 4 + 6 + 11 + 9


class TestFlatten:
    def test_shape_matches_spec_dim(self) -> None:
        state = MyoArmState.from_arrays(**_valid_arrays())
        flat = state.flatten()
        assert flat.shape == (state.spec().dim,)

    def test_dtype_is_float32(self) -> None:
        state = MyoArmState.from_arrays(**_valid_arrays())
        assert state.flatten().dtype == np.float32

    def test_order_matches_layout(self) -> None:
        state = MyoArmState.from_arrays(
            qpos=np.array([1.0, 2.0], dtype=np.float32),
            qvel=np.array([3.0, 4.0, 5.0], dtype=np.float32),
            act=np.array([6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float32),
            tip_pos=np.array([11.0, 12.0, 13.0], dtype=np.float32),
            target_pos=np.array([14.0, 15.0, 16.0], dtype=np.float32),
            reach_err=np.array([-3.0, -3.0, -3.0], dtype=np.float32),
        )
        flat = state.flatten()
        layout = state.spec().layout()
        np.testing.assert_array_equal(flat[layout["qpos"]], [1.0, 2.0])
        np.testing.assert_array_equal(flat[layout["qvel"]], [3.0, 4.0, 5.0])
        np.testing.assert_array_equal(
            flat[layout["act"]], [6.0, 7.0, 8.0, 9.0, 10.0]
        )
        np.testing.assert_array_equal(flat[layout["tip_pos"]], [11.0, 12.0, 13.0])
        np.testing.assert_array_equal(
            flat[layout["target_pos"]], [14.0, 15.0, 16.0]
        )
        np.testing.assert_array_equal(
            flat[layout["reach_err"]], [-3.0, -3.0, -3.0]
        )


class TestUnflatten:
    def test_round_trip(self) -> None:
        original = MyoArmState.from_arrays(
            qpos=np.array([0.1, 0.2], dtype=np.float32),
            qvel=np.array([0.3, 0.4, 0.5], dtype=np.float32),
            act=np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            tip_pos=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            target_pos=np.array([4.0, 5.0, 6.0], dtype=np.float32),
            reach_err=np.array([-3.0, -3.0, -3.0], dtype=np.float32),
        )
        spec = original.spec()
        recovered = MyoArmState.unflatten(original.flatten(), spec)
        for name in (
            "qpos",
            "qvel",
            "act",
            "tip_pos",
            "target_pos",
            "reach_err",
        ):
            np.testing.assert_array_equal(
                getattr(recovered, name), getattr(original, name)
            )

    def test_round_trip_dtype_preserved(self) -> None:
        state = MyoArmState.from_arrays(**_valid_arrays())
        recovered = MyoArmState.unflatten(state.flatten(), state.spec())
        for name in (
            "qpos",
            "qvel",
            "act",
            "tip_pos",
            "target_pos",
            "reach_err",
        ):
            assert getattr(recovered, name).dtype == np.float32

    def test_wrong_length_vec_raises(self) -> None:
        spec = StateSpec(qpos_dim=2, qvel_dim=3, act_dim=5)
        bad = np.zeros(spec.dim - 1, dtype=np.float32)
        with pytest.raises(ValueError, match="does not match spec.dim"):
            MyoArmState.unflatten(bad, spec)

    def test_2d_vec_raises(self) -> None:
        spec = StateSpec(qpos_dim=2, qvel_dim=3, act_dim=5)
        bad = np.zeros((spec.dim, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            MyoArmState.unflatten(bad, spec)

    def test_non_spec_raises(self) -> None:
        state = MyoArmState.from_arrays(**_valid_arrays())
        with pytest.raises(TypeError, match="StateSpec"):
            MyoArmState.unflatten(state.flatten(), spec="not a spec")  # type: ignore[arg-type]
