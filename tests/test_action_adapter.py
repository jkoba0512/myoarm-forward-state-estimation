"""Tests for myoarm_fse.envs.actions.ActionAdapter and detect_action_dim."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.envs import ActionAdapter, detect_action_dim


# --- constructor validation ---


class TestActionAdapterConstructor:
    def test_valid_action_dim(self) -> None:
        adapter = ActionAdapter(action_dim=10)
        assert adapter.action_dim == 10

    def test_zero_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            ActionAdapter(action_dim=0)

    def test_negative_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            ActionAdapter(action_dim=-5)

    def test_bool_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            ActionAdapter(action_dim=True)  # type: ignore[arg-type]

    def test_float_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            ActionAdapter(action_dim=10.0)  # type: ignore[arg-type]

    def test_str_dim_raises(self) -> None:
        with pytest.raises(ValueError):
            ActionAdapter(action_dim="10")  # type: ignore[arg-type]


# --- conversion correctness ---


def test_excitation_to_api_action_in_range() -> None:
    adapter = ActionAdapter(action_dim=3)
    api = adapter.excitation_to_api_action(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(api, [-1.0, 0.0, 1.0])


def test_api_action_to_excitation_in_range() -> None:
    adapter = ActionAdapter(action_dim=3)
    exc = adapter.api_action_to_excitation(np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_allclose(exc, [0.0, 0.5, 1.0])


def test_round_trip_excitation_in_range() -> None:
    adapter = ActionAdapter(action_dim=4)
    e = np.array([0.1, 0.3, 0.7, 0.9], dtype=np.float32)
    e_round = adapter.api_action_to_excitation(adapter.excitation_to_api_action(e))
    np.testing.assert_allclose(e, e_round, atol=1e-6)


def test_round_trip_api_action_in_range() -> None:
    adapter = ActionAdapter(action_dim=4)
    a = np.array([-0.8, -0.2, 0.4, 0.95], dtype=np.float32)
    a_round = adapter.excitation_to_api_action(adapter.api_action_to_excitation(a))
    np.testing.assert_allclose(a, a_round, atol=1e-6)


# --- clipping behavior ---


def test_excitation_to_api_action_clips_out_of_range() -> None:
    adapter = ActionAdapter(action_dim=3)
    api = adapter.excitation_to_api_action(np.array([-0.5, 0.5, 1.5]))
    np.testing.assert_allclose(api, [-1.0, 0.0, 1.0])


def test_api_action_to_excitation_clips_out_of_range() -> None:
    adapter = ActionAdapter(action_dim=3)
    exc = adapter.api_action_to_excitation(np.array([-2.0, 0.0, 2.0]))
    np.testing.assert_allclose(exc, [0.0, 0.5, 1.0])


def test_clip_excitation_method() -> None:
    adapter = ActionAdapter(action_dim=3)
    out = adapter.clip_excitation(np.array([-0.5, 0.5, 1.5]))
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])
    assert out.dtype == np.float32


def test_clip_api_action_method() -> None:
    adapter = ActionAdapter(action_dim=3)
    out = adapter.clip_api_action(np.array([-2.0, 0.0, 2.0]))
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])
    assert out.dtype == np.float32


# --- shape validation ---


def test_shape_mismatch_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.excitation_to_api_action(np.zeros(4))


def test_2d_batch_input_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.excitation_to_api_action(np.zeros((2, 3)))


def test_0d_scalar_input_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.excitation_to_api_action(np.float32(0.5))


def test_clip_excitation_shape_mismatch_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.clip_excitation(np.zeros(2))


def test_clip_api_action_shape_mismatch_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.clip_api_action(np.zeros((1, 3)))


# --- non-finite validation ---


def test_nan_in_excitation_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.excitation_to_api_action(np.array([0.5, np.nan, 0.5]))


def test_inf_in_excitation_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.excitation_to_api_action(np.array([0.5, np.inf, 0.5]))


def test_neg_inf_in_excitation_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.excitation_to_api_action(np.array([0.5, -np.inf, 0.5]))


def test_nan_in_api_action_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.api_action_to_excitation(np.array([0.0, np.nan, 0.0]))


def test_nan_in_clip_excitation_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.clip_excitation(np.array([0.5, np.nan, 0.5]))


def test_nan_in_clip_api_action_raises() -> None:
    adapter = ActionAdapter(action_dim=3)
    with pytest.raises(ValueError):
        adapter.clip_api_action(np.array([0.0, np.nan, 0.0]))


# --- dtype handling ---


def test_output_dtype_is_float32() -> None:
    adapter = ActionAdapter(action_dim=2)
    out = adapter.excitation_to_api_action(np.array([0.1, 0.9], dtype=np.float64))
    assert out.dtype == np.float32


def test_api_to_excitation_output_dtype_is_float32() -> None:
    adapter = ActionAdapter(action_dim=2)
    out = adapter.api_action_to_excitation(np.array([-0.5, 0.5], dtype=np.float64))
    assert out.dtype == np.float32


def test_output_shape_matches_action_dim() -> None:
    adapter = ActionAdapter(action_dim=5)
    out = adapter.excitation_to_api_action(np.zeros(5))
    assert out.shape == (5,)


# --- array-like input ---


def test_accepts_list_input() -> None:
    adapter = ActionAdapter(action_dim=3)
    out = adapter.excitation_to_api_action([0.0, 0.5, 1.0])
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])
    assert out.dtype == np.float32


def test_accepts_tuple_input() -> None:
    adapter = ActionAdapter(action_dim=3)
    out = adapter.excitation_to_api_action((0.0, 0.5, 1.0))
    np.testing.assert_allclose(out, [-1.0, 0.0, 1.0])


# --- detect_action_dim ---


class _StubBox:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


class _StubEnv:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.action_space = _StubBox(shape)


class _StubBoxNoShape:
    pass


class _StubEnvNoActionSpace:
    pass


def test_detect_action_dim_basic() -> None:
    env = _StubEnv((63,))
    assert detect_action_dim(env) == 63


def test_detect_action_dim_returns_int() -> None:
    env = _StubEnv((10,))
    dim = detect_action_dim(env)
    assert isinstance(dim, int)


def test_detect_action_dim_no_action_space_raises() -> None:
    env = _StubEnvNoActionSpace()
    with pytest.raises(TypeError):
        detect_action_dim(env)


def test_detect_action_dim_no_shape_raises() -> None:
    class _Env:
        action_space = _StubBoxNoShape()

    with pytest.raises(ValueError):
        detect_action_dim(_Env())


def test_detect_action_dim_shape_none_raises() -> None:
    class _Box:
        shape = None

    class _Env:
        action_space = _Box()

    with pytest.raises(ValueError):
        detect_action_dim(_Env())


def test_detect_action_dim_2d_shape_raises() -> None:
    env = _StubEnv((4, 5))
    with pytest.raises(ValueError):
        detect_action_dim(env)


def test_detect_action_dim_0d_shape_raises() -> None:
    env = _StubEnv(())
    with pytest.raises(ValueError):
        detect_action_dim(env)


def test_detect_action_dim_zero_size_raises() -> None:
    env = _StubEnv((0,))
    with pytest.raises(ValueError):
        detect_action_dim(env)


# --- ActionAdapter + detect_action_dim integration ---


def test_adapter_construction_via_detect_action_dim() -> None:
    env = _StubEnv((39,))
    adapter = ActionAdapter(action_dim=detect_action_dim(env))
    assert adapter.action_dim == 39
    out = adapter.excitation_to_api_action(np.zeros(39))
    assert out.shape == (39,)
    assert out.dtype == np.float32
