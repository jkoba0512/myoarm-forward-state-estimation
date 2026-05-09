"""Tests for myoarm_fse.envs.wrappers (Delayed / NoisyObservationWrapper)."""

from __future__ import annotations

import numpy as np
import pytest

from myoarm_fse.envs import (
    DelayedObservationWrapper,
    MyoArmState,
    NoisyObservationWrapper,
    StateSpec,
)


# --- helpers ---


_SPEC = StateSpec(qpos_dim=2, qvel_dim=3, act_dim=5)


def _state(value: float, spec: StateSpec = _SPEC) -> MyoArmState:
    """Build a MyoArmState whose flat representation is ``value`` everywhere."""
    return MyoArmState.from_arrays(
        qpos=np.full(spec.qpos_dim, value, dtype=np.float32),
        qvel=np.full(spec.qvel_dim, value, dtype=np.float32),
        act=np.full(spec.act_dim, value, dtype=np.float32),
        tip_pos=np.full(3, value, dtype=np.float32),
        target_pos=np.full(3, value, dtype=np.float32),
        reach_err=np.full(3, value, dtype=np.float32),
    )


def _flat_value(state: MyoArmState) -> float:
    """Return the constant value of a state built by ``_state``."""
    return float(state.qpos[0])


# =============================================================================
# DelayedObservationWrapper
# =============================================================================


class TestDelayedConstructor:
    def test_valid(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=3)
        assert w.delay_steps == 3
        assert w.spec is _SPEC

    def test_zero_delay_allowed(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=0)
        assert w.delay_steps == 0

    def test_negative_delay_raises(self) -> None:
        with pytest.raises(ValueError):
            DelayedObservationWrapper(_SPEC, delay_steps=-1)

    def test_bool_delay_raises(self) -> None:
        with pytest.raises(ValueError):
            DelayedObservationWrapper(_SPEC, delay_steps=True)  # type: ignore[arg-type]

    def test_float_delay_raises(self) -> None:
        with pytest.raises(ValueError):
            DelayedObservationWrapper(_SPEC, delay_steps=1.0)  # type: ignore[arg-type]

    def test_str_delay_raises(self) -> None:
        with pytest.raises(ValueError):
            DelayedObservationWrapper(_SPEC, delay_steps="1")  # type: ignore[arg-type]

    def test_non_spec_raises(self) -> None:
        with pytest.raises(TypeError):
            DelayedObservationWrapper("not a spec", delay_steps=1)  # type: ignore[arg-type]


class TestDelayedIdentity:
    def test_zero_delay_returns_same_values(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=0)
        s = _state(1.5)
        out = w.observe(s)
        np.testing.assert_array_equal(out.flatten(), s.flatten())

    def test_zero_delay_works_without_reset(self) -> None:
        # No reset call. Identity path must not raise.
        w = DelayedObservationWrapper(_SPEC, delay_steps=0)
        out = w.observe(_state(2.0))
        assert _flat_value(out) == 2.0

    def test_call_equals_observe(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=0)
        s = _state(1.0)
        np.testing.assert_array_equal(w(s).flatten(), w.observe(s).flatten())


class TestDelayedTiming:
    def test_delay_one_step(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=1)
        w.reset(_state(0.0))
        assert _flat_value(w.observe(_state(1.0))) == 0.0
        assert _flat_value(w.observe(_state(2.0))) == 1.0
        assert _flat_value(w.observe(_state(3.0))) == 2.0

    def test_delay_two_steps(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=2)
        w.reset(_state(0.0))
        assert _flat_value(w.observe(_state(1.0))) == 0.0
        assert _flat_value(w.observe(_state(2.0))) == 0.0
        assert _flat_value(w.observe(_state(3.0))) == 1.0
        assert _flat_value(w.observe(_state(4.0))) == 2.0

    def test_delay_three_steps(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=3)
        w.reset(_state(0.0))
        for _ in range(3):
            assert _flat_value(w.observe(_state(99.0))) == 0.0
        assert _flat_value(w.observe(_state(99.0))) == 99.0

    def test_reset_reinitializes_buffer(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=2)
        w.reset(_state(0.0))
        w.observe(_state(1.0))
        w.observe(_state(2.0))
        # Re-reset: subsequent observe should return the new initial state.
        w.reset(_state(7.0))
        assert _flat_value(w.observe(_state(8.0))) == 7.0
        assert _flat_value(w.observe(_state(9.0))) == 7.0
        assert _flat_value(w.observe(_state(10.0))) == 8.0


class TestDelayedRuntimeErrors:
    def test_observe_before_reset_raises_when_delay_positive(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=2)
        with pytest.raises(RuntimeError):
            w.observe(_state(1.0))

    def test_observe_after_reset_does_not_raise(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=2)
        w.reset(_state(0.0))
        w.observe(_state(1.0))


class TestDelayedSpecMismatch:
    def test_observe_with_wrong_spec_raises(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=1)
        w.reset(_state(0.0))
        wrong = MyoArmState.from_arrays(
            qpos=np.zeros(99, dtype=np.float32),
            qvel=np.zeros(3, dtype=np.float32),
            act=np.zeros(5, dtype=np.float32),
            tip_pos=np.zeros(3, dtype=np.float32),
            target_pos=np.zeros(3, dtype=np.float32),
            reach_err=np.zeros(3, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="does not match wrapper spec"):
            w.observe(wrong)

    def test_reset_with_wrong_spec_raises(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=1)
        wrong = MyoArmState.from_arrays(
            qpos=np.zeros(99, dtype=np.float32),
            qvel=np.zeros(3, dtype=np.float32),
            act=np.zeros(5, dtype=np.float32),
            tip_pos=np.zeros(3, dtype=np.float32),
            target_pos=np.zeros(3, dtype=np.float32),
            reach_err=np.zeros(3, dtype=np.float32),
        )
        with pytest.raises(ValueError):
            w.reset(wrong)

    def test_observe_with_non_state_raises(self) -> None:
        w = DelayedObservationWrapper(_SPEC, delay_steps=0)
        with pytest.raises(ValueError):
            w.observe("not a state")  # type: ignore[arg-type]


# =============================================================================
# NoisyObservationWrapper
# =============================================================================


class TestNoisyConstructor:
    def test_empty_sigma(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={})
        assert w.sigma == {}

    def test_partial_sigma(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.1})
        assert w.sigma == {"qpos": 0.1}

    def test_unknown_field_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown sigma field"):
            NoisyObservationWrapper(_SPEC, sigma={"banana": 0.1})

    def test_negative_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            NoisyObservationWrapper(_SPEC, sigma={"qpos": -0.1})

    def test_nan_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            NoisyObservationWrapper(_SPEC, sigma={"qpos": float("nan")})

    def test_inf_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            NoisyObservationWrapper(_SPEC, sigma={"qpos": float("inf")})

    def test_bool_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            NoisyObservationWrapper(_SPEC, sigma={"qpos": True})  # type: ignore[dict-item]

    def test_str_sigma_raises(self) -> None:
        with pytest.raises(ValueError):
            NoisyObservationWrapper(_SPEC, sigma={"qpos": "0.1"})  # type: ignore[dict-item]

    def test_non_mapping_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="mapping"):
            NoisyObservationWrapper(_SPEC, sigma=[("qpos", 0.1)])  # type: ignore[arg-type]

    def test_non_spec_raises(self) -> None:
        with pytest.raises(TypeError):
            NoisyObservationWrapper("not a spec", sigma={})  # type: ignore[arg-type]

    def test_sigma_property_returns_copy(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.1})
        prop = w.sigma
        prop["qpos"] = 999.0
        assert w.sigma == {"qpos": 0.1}


class TestNoisyIdentity:
    def test_empty_sigma_is_identity(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={}, rng=0)
        s = _state(1.5)
        out = w.observe(s)
        np.testing.assert_array_equal(out.flatten(), s.flatten())

    def test_all_zero_sigma_is_identity(self) -> None:
        w = NoisyObservationWrapper(
            _SPEC, sigma={"qpos": 0.0, "tip_pos": 0.0}, rng=0
        )
        s = _state(2.5)
        np.testing.assert_array_equal(w.observe(s).flatten(), s.flatten())

    def test_empty_sigma_does_not_consume_rng(self) -> None:
        rng = np.random.default_rng(0)
        baseline = rng.standard_normal(5)
        rng2 = np.random.default_rng(0)
        w = NoisyObservationWrapper(_SPEC, sigma={}, rng=rng2)
        w.observe(_state(0.0))
        # rng2 should be untouched: drawing 5 samples must equal baseline.
        np.testing.assert_array_equal(rng2.standard_normal(5), baseline)

    def test_all_zero_sigma_does_not_consume_rng(self) -> None:
        rng = np.random.default_rng(0)
        baseline = rng.standard_normal(5)
        rng2 = np.random.default_rng(0)
        w = NoisyObservationWrapper(
            _SPEC, sigma={f: 0.0 for f in ("qpos", "qvel", "act")}, rng=rng2
        )
        w.observe(_state(0.0))
        np.testing.assert_array_equal(rng2.standard_normal(5), baseline)

    def test_call_equals_observe(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.5}, rng=42)
        s = _state(1.0)
        out_call = w(s)
        w.reset(seed=42)
        out_obs = w.observe(s)
        np.testing.assert_array_equal(out_call.flatten(), out_obs.flatten())


class TestNoisyApplication:
    def test_noise_applied_only_to_specified_fields(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 1.0}, rng=0)
        s = _state(0.0)
        out = w.observe(s)
        # qpos should differ; all other fields should be unchanged.
        assert not np.array_equal(out.qpos, s.qpos)
        np.testing.assert_array_equal(out.qvel, s.qvel)
        np.testing.assert_array_equal(out.act, s.act)
        np.testing.assert_array_equal(out.tip_pos, s.tip_pos)
        np.testing.assert_array_equal(out.target_pos, s.target_pos)
        np.testing.assert_array_equal(out.reach_err, s.reach_err)

    def test_partial_sigma_leaves_other_fields_clean(self) -> None:
        w = NoisyObservationWrapper(
            _SPEC, sigma={"tip_pos": 0.5, "target_pos": 0.5}, rng=1
        )
        s = _state(7.0)
        out = w.observe(s)
        np.testing.assert_array_equal(out.qpos, s.qpos)
        np.testing.assert_array_equal(out.qvel, s.qvel)
        np.testing.assert_array_equal(out.act, s.act)
        np.testing.assert_array_equal(out.reach_err, s.reach_err)
        assert not np.array_equal(out.tip_pos, s.tip_pos)
        assert not np.array_equal(out.target_pos, s.target_pos)

    def test_no_clipping_with_large_sigma(self) -> None:
        # Large sigma at zero baseline should be able to push values well
        # outside [0, 1] / [-1, 1] ranges; verify by sampling many times.
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 100.0}, rng=0)
        seen_extreme = False
        for _ in range(50):
            out = w.observe(_state(0.0))
            if np.any(np.abs(out.qpos) > 5.0):
                seen_extreme = True
                break
        assert seen_extreme, "noise was unexpectedly bounded"

    def test_dtype_preserved(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.1}, rng=0)
        out = w.observe(_state(0.0))
        for name in ("qpos", "qvel", "act", "tip_pos", "target_pos", "reach_err"):
            assert getattr(out, name).dtype == np.float32


class TestNoisyReproducibility:
    def test_same_seed_same_output(self) -> None:
        s = _state(0.0)
        a = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.5}, rng=123)
        b = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.5}, rng=123)
        np.testing.assert_array_equal(
            a.observe(s).flatten(), b.observe(s).flatten()
        )

    def test_different_seeds_different_output(self) -> None:
        s = _state(0.0)
        a = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.5}, rng=1)
        b = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.5}, rng=2)
        assert not np.array_equal(
            a.observe(s).flatten(), b.observe(s).flatten()
        )

    def test_reset_seed_restores_sequence(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.5}, rng=42)
        first = w.observe(_state(0.0)).flatten().copy()
        w.observe(_state(0.0))
        w.reset(seed=42)
        again = w.observe(_state(0.0)).flatten()
        np.testing.assert_array_equal(first, again)

    def test_reset_with_none_changes_sequence(self) -> None:
        # Different fresh rngs should (almost certainly) produce different draws.
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 1.0}, rng=42)
        w.reset(seed=None)
        a = w.observe(_state(0.0)).flatten().copy()
        w.reset(seed=None)
        b = w.observe(_state(0.0)).flatten()
        assert not np.array_equal(a, b)

    def test_reset_bool_seed_raises(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.1})
        with pytest.raises(TypeError):
            w.reset(seed=True)  # type: ignore[arg-type]

    def test_reset_str_seed_raises(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.1})
        with pytest.raises(TypeError):
            w.reset(seed="42")  # type: ignore[arg-type]


class TestNoisySpecMismatch:
    def test_observe_with_wrong_spec_raises(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.1}, rng=0)
        wrong = MyoArmState.from_arrays(
            qpos=np.zeros(99, dtype=np.float32),
            qvel=np.zeros(3, dtype=np.float32),
            act=np.zeros(5, dtype=np.float32),
            tip_pos=np.zeros(3, dtype=np.float32),
            target_pos=np.zeros(3, dtype=np.float32),
            reach_err=np.zeros(3, dtype=np.float32),
        )
        with pytest.raises(ValueError, match="does not match wrapper spec"):
            w.observe(wrong)

    def test_observe_with_non_state_raises(self) -> None:
        w = NoisyObservationWrapper(_SPEC, sigma={}, rng=0)
        with pytest.raises(ValueError):
            w.observe([1, 2, 3])  # type: ignore[arg-type]


# =============================================================================
# Composition (smoke)
# =============================================================================


class TestComposition:
    def test_noisy_then_delayed(self) -> None:
        delayed = DelayedObservationWrapper(_SPEC, delay_steps=1)
        noisy = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.5}, rng=0)
        delayed.reset(_state(0.0))
        # delayed(noisy(s_t)) — first observe returns the (clean) reset state.
        out = delayed.observe(noisy.observe(_state(1.0)))
        np.testing.assert_array_equal(out.flatten(), _state(0.0).flatten())

    def test_delayed_then_noisy(self) -> None:
        delayed = DelayedObservationWrapper(_SPEC, delay_steps=1)
        noisy = NoisyObservationWrapper(_SPEC, sigma={"qpos": 0.5}, rng=0)
        delayed.reset(_state(0.0))
        # noisy(delayed(s_t)): delayed returns s_0 (clean), noisy adds noise to it.
        out = noisy.observe(delayed.observe(_state(1.0)))
        # qpos should differ from s_0 due to noise.
        assert not np.array_equal(out.qpos, _state(0.0).qpos)
        # but other fields are clean and equal to s_0.
        np.testing.assert_array_equal(out.qvel, _state(0.0).qvel)
