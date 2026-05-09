"""Tests for myoarm_fse.envs.noise.SignalDependentMotorNoise."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.envs import ActionAdapter, SignalDependentMotorNoise


# --- constructor: action_dim ---


class TestActionDimValidation:
    def test_valid(self) -> None:
        sdn = SignalDependentMotorNoise(action_dim=10, sigma=0.0)
        assert sdn.action_dim == 10

    def test_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalDependentMotorNoise(action_dim=0, sigma=0.0)

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalDependentMotorNoise(action_dim=-3, sigma=0.0)

    def test_bool_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalDependentMotorNoise(action_dim=True, sigma=0.0)  # type: ignore[arg-type]

    def test_float_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalDependentMotorNoise(action_dim=3.0, sigma=0.0)  # type: ignore[arg-type]


# --- constructor: sigma ---


class TestSigmaValidation:
    def test_int_sigma_accepted(self) -> None:
        sdn = SignalDependentMotorNoise(action_dim=3, sigma=1)
        assert sdn.sigma == 1.0
        assert isinstance(sdn.sigma, float)

    def test_float_sigma_accepted(self) -> None:
        sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.25)
        assert sdn.sigma == pytest.approx(0.25)

    def test_np_float32_sigma_accepted(self) -> None:
        sdn = SignalDependentMotorNoise(action_dim=3, sigma=np.float32(0.5))
        assert sdn.sigma == pytest.approx(0.5)
        assert isinstance(sdn.sigma, float)

    def test_np_float64_sigma_accepted(self) -> None:
        sdn = SignalDependentMotorNoise(action_dim=3, sigma=np.float64(0.1))
        assert sdn.sigma == pytest.approx(0.1)

    def test_zero_sigma_accepted(self) -> None:
        sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.0)
        assert sdn.sigma == 0.0

    def test_bool_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalDependentMotorNoise(action_dim=3, sigma=True)  # type: ignore[arg-type]

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalDependentMotorNoise(action_dim=3, sigma=-0.1)

    def test_nan_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalDependentMotorNoise(action_dim=3, sigma=float("nan"))

    def test_inf_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalDependentMotorNoise(action_dim=3, sigma=float("inf"))

    def test_str_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            SignalDependentMotorNoise(action_dim=3, sigma="0.5")  # type: ignore[arg-type]


# --- constructor: rng ---


class TestRngValidation:
    def test_none_rng_accepted(self) -> None:
        sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=None)
        assert isinstance(sdn.rng, np.random.Generator)

    def test_int_rng_accepted(self) -> None:
        sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=42)
        assert isinstance(sdn.rng, np.random.Generator)

    def test_generator_rng_accepted_as_is(self) -> None:
        gen = np.random.default_rng(123)
        sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=gen)
        assert sdn.rng is gen

    def test_bool_rng_raises(self) -> None:
        with pytest.raises(TypeError):
            SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=True)  # type: ignore[arg-type]

    def test_float_rng_raises(self) -> None:
        with pytest.raises(TypeError):
            SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=1.5)  # type: ignore[arg-type]

    def test_random_state_rejected(self) -> None:
        legacy = np.random.RandomState(0)
        with pytest.raises(TypeError):
            SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=legacy)  # type: ignore[arg-type]


# --- sigma=0 behavior: validation + clip only ---


def test_sigma_zero_applies_clip_only_in_range() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.0, rng=0)
    u = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = sdn(u)
    np.testing.assert_array_equal(out, u)


def test_sigma_zero_applies_clip_only_out_of_range() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.0, rng=0)
    u = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
    out = sdn(u)
    np.testing.assert_array_equal(out, np.array([0.0, 0.5, 1.0], dtype=np.float32))


def test_sigma_zero_does_not_consume_rng() -> None:
    """sigma=0 path should bypass the rng entirely."""
    rng = np.random.default_rng(7)
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.0, rng=rng)
    sdn(np.array([0.5, 0.5, 0.5]))
    sdn(np.array([0.5, 0.5, 0.5]))
    expected = np.random.default_rng(7).standard_normal(size=3, dtype=np.float32)
    actual = rng.standard_normal(size=3, dtype=np.float32)
    np.testing.assert_array_equal(actual, expected)


# --- sigma>0 behavior ---


def test_sigma_positive_perturbs_input() -> None:
    sdn = SignalDependentMotorNoise(action_dim=4, sigma=0.5, rng=0)
    u = np.full(4, 0.5, dtype=np.float32)
    out = sdn(u)
    assert not np.array_equal(out, u)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)


def test_noise_scales_with_abs_u_zero_components_unchanged() -> None:
    sdn = SignalDependentMotorNoise(action_dim=5, sigma=1.0, rng=0)
    u = np.array([0.0, 0.5, 0.0, 0.8, 0.0], dtype=np.float32)
    out = sdn(u)
    assert out[0] == 0.0
    assert out[2] == 0.0
    assert out[4] == 0.0


def test_clip_enforced_with_large_sigma() -> None:
    sdn = SignalDependentMotorNoise(action_dim=64, sigma=10.0, rng=0)
    u = np.full(64, 0.5, dtype=np.float32)
    out = sdn(u)
    assert np.all(out >= 0.0)
    assert np.all(out <= 1.0)
    assert (out == 0.0).any()
    assert (out == 1.0).any()


def test_rng_advances_between_calls() -> None:
    sdn = SignalDependentMotorNoise(action_dim=4, sigma=0.5, rng=42)
    u = np.full(4, 0.5, dtype=np.float32)
    out1 = sdn(u)
    out2 = sdn(u)
    assert not np.array_equal(out1, out2)


# --- reproducibility ---


def test_same_seed_yields_same_output() -> None:
    u = np.full(8, 0.5, dtype=np.float32)
    a = SignalDependentMotorNoise(action_dim=8, sigma=0.3, rng=42)
    b = SignalDependentMotorNoise(action_dim=8, sigma=0.3, rng=42)
    np.testing.assert_array_equal(a(u), b(u))
    np.testing.assert_array_equal(a(u), b(u))


def test_different_seeds_yield_different_output() -> None:
    u = np.full(16, 0.5, dtype=np.float32)
    a = SignalDependentMotorNoise(action_dim=16, sigma=0.3, rng=1)
    b = SignalDependentMotorNoise(action_dim=16, sigma=0.3, rng=2)
    assert not np.array_equal(a(u), b(u))


def test_reset_resyncs_to_seed() -> None:
    u = np.full(4, 0.5, dtype=np.float32)
    sdn = SignalDependentMotorNoise(action_dim=4, sigma=0.5, rng=42)
    out_before = sdn(u)
    sdn.reset(42)
    out_after = sdn(u)
    np.testing.assert_array_equal(out_before, out_after)


def test_reset_with_none_draws_independent_rng() -> None:
    sdn = SignalDependentMotorNoise(action_dim=4, sigma=0.5, rng=42)
    rng_before = sdn.rng
    sdn.reset(None)
    assert sdn.rng is not rng_before


def test_reset_rejects_bool_seed() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=0)
    with pytest.raises(TypeError):
        sdn.reset(True)  # type: ignore[arg-type]


def test_reset_rejects_non_int_seed() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=0)
    with pytest.raises(TypeError):
        sdn.reset(1.5)  # type: ignore[arg-type]


# --- shape / finite validation ---


def test_shape_mismatch_raises() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=0)
    with pytest.raises(ValueError):
        sdn(np.zeros(4))


def test_2d_batch_raises() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=0)
    with pytest.raises(ValueError):
        sdn(np.zeros((2, 3)))


def test_0d_scalar_raises() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=0)
    with pytest.raises(ValueError):
        sdn(np.float32(0.5))


def test_nan_raises() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=0)
    with pytest.raises(ValueError):
        sdn(np.array([0.5, np.nan, 0.5]))


def test_inf_raises() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=0)
    with pytest.raises(ValueError):
        sdn(np.array([0.5, np.inf, 0.5]))


# --- dtype / array-like ---


def test_output_dtype_is_float32() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.1, rng=0)
    out = sdn(np.array([0.1, 0.5, 0.9], dtype=np.float64))
    assert out.dtype == np.float32


def test_output_shape_matches_action_dim() -> None:
    sdn = SignalDependentMotorNoise(action_dim=7, sigma=0.1, rng=0)
    out = sdn(np.zeros(7))
    assert out.shape == (7,)


def test_accepts_list_input() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.0, rng=0)
    out = sdn([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(out, [0.0, 0.5, 1.0])
    assert out.dtype == np.float32


def test_accepts_tuple_input() -> None:
    sdn = SignalDependentMotorNoise(action_dim=3, sigma=0.0, rng=0)
    out = sdn((0.0, 0.5, 1.0))
    np.testing.assert_array_equal(out, [0.0, 0.5, 1.0])


# --- __call__ == apply ---


def test_call_matches_apply() -> None:
    u = np.full(4, 0.4, dtype=np.float32)
    a = SignalDependentMotorNoise(action_dim=4, sigma=0.5, rng=42)
    b = SignalDependentMotorNoise(action_dim=4, sigma=0.5, rng=42)
    np.testing.assert_array_equal(a(u), b.apply(u))


# --- integration with ActionAdapter ---


def test_sdn_then_action_adapter_chain() -> None:
    """Layer order: excitation_command -> SDN -> excitation -> ActionAdapter -> api_action."""
    sdn = SignalDependentMotorNoise(action_dim=5, sigma=0.2, rng=0)
    adapter = ActionAdapter(action_dim=5)

    excitation_command = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
    excitation = sdn(excitation_command)
    api_action = adapter.excitation_to_api_action(excitation)

    assert excitation.dtype == np.float32
    assert api_action.dtype == np.float32
    assert np.all(excitation >= 0.0) and np.all(excitation <= 1.0)
    assert np.all(api_action >= -1.0) and np.all(api_action <= 1.0)
