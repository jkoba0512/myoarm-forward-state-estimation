"""Unit tests for scripts/evaluate_estimator.py helpers.

The script is exercised end-to-end via the optional MyoSuite smoke
sweep; these tests cover the pure helpers (config resolution, CSV
emission) that don't need MyoSuite or a real model.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "scripts" / "evaluate_estimator.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_estimator_cli", _SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script_module()


# --- _resolve_noise_conditions ---


class TestResolveNoiseConditions:
    def test_legacy_obs_noise_sigma(self, script) -> None:
        cfg = {"obs_noise_sigma": {"qpos": 0.01, "qvel": 0.05}}
        out = script._resolve_noise_conditions(cfg)
        assert out == [("default", {"qpos": 0.01, "qvel": 0.05})]

    def test_legacy_missing_returns_empty_default(self, script) -> None:
        cfg: dict = {}
        out = script._resolve_noise_conditions(cfg)
        assert out == [("default", {})]

    def test_robustness_schema(self, script) -> None:
        cfg = {
            "noise_conditions": {
                "none": {"qpos": 0.0},
                "high": {"qpos": 0.02},
            }
        }
        out = script._resolve_noise_conditions(cfg)
        assert [name for name, _ in out] == ["none", "high"]
        assert out[1][1] == {"qpos": 0.02}

    def test_legacy_takes_priority_when_no_noise_conditions(self, script) -> None:
        # noise_conditions absent -> falls back to obs_noise_sigma.
        cfg = {"obs_noise_sigma": {"qpos": 0.01}}
        out = script._resolve_noise_conditions(cfg)
        assert out == [("default", {"qpos": 0.01})]

    def test_robustness_takes_priority_when_both_set(self, script) -> None:
        cfg = {
            "obs_noise_sigma": {"qpos": 0.01},
            "noise_conditions": {"high": {"qpos": 0.02}},
        }
        out = script._resolve_noise_conditions(cfg)
        # Robustness mode wins; legacy obs_noise_sigma is ignored.
        assert out == [("high", {"qpos": 0.02})]

    def test_invalid_noise_conditions_type(self, script) -> None:
        with pytest.raises(ValueError, match="mapping"):
            script._resolve_noise_conditions(
                {"noise_conditions": [("none", {})]}
            )

    def test_empty_noise_conditions(self, script) -> None:
        with pytest.raises(ValueError, match="empty"):
            script._resolve_noise_conditions({"noise_conditions": {}})

    def test_invalid_noise_condition_value(self, script) -> None:
        with pytest.raises(ValueError, match="mapping"):
            script._resolve_noise_conditions(
                {"noise_conditions": {"none": "not a dict"}}
            )


# --- _row_for_csv ---


def test_row_for_csv_basic(script) -> None:
    metrics = {
        "n": 3,
        "tip_estimation_error_mean": 0.012,
        "tip_estimation_error_final": 0.020,
        "tip_estimation_error_std": 0.003,
        "state_mse_mean": 0.0001,
        "mse_qpos_mean": 0.00005,
        "mse_qvel_mean": 0.00006,
        "mse_act_mean": 0.00004,
        "mse_tip_pos_mean": 0.0002,
        "mse_target_pos_mean": 0.0,
        "mse_reach_err_mean": 0.0001,
    }
    row = script._row_for_csv(
        controller="random",
        noise_condition="medium",
        delay_steps=2,
        gain=0.5,
        metrics=metrics,
        model_run_id="model-A",
    )
    assert row["controller"] == "random"
    assert row["noise_condition"] == "medium"
    assert row["delay_steps"] == 2
    assert row["gain"] == 0.5
    assert row["tip_estimation_error_mean"] == pytest.approx(0.012)
    assert row["tip_estimation_error_final"] == pytest.approx(0.020)
    assert row["state_mse_mean"] == pytest.approx(0.0001)
    assert row["n_episodes"] == 3
    assert row["model_run_id"] == "model-A"


def test_row_for_csv_missing_metrics_yield_nan(script) -> None:
    import math

    row = script._row_for_csv(
        controller="hold",
        noise_condition="none",
        delay_steps=0,
        gain=0.0,
        metrics={"n": 0},  # all metric keys missing
        model_run_id="m",
    )
    assert math.isnan(row["tip_estimation_error_mean"])
    assert math.isnan(row["state_mse_mean"])
    assert row["n_episodes"] == 0


# --- _write_metrics_csv / _write_best_by_condition_csv ---


def _row(controller, noise, delay, gain, tip_err) -> dict:
    return {
        "controller": controller,
        "noise_condition": noise,
        "delay_steps": delay,
        "gain": gain,
        "tip_estimation_error_mean": tip_err,
        "tip_estimation_error_final": tip_err,
        "tip_estimation_error_std": 0.0,
        "state_mse_mean": tip_err * tip_err,
        "mse_qpos_mean": 0.0,
        "mse_qvel_mean": 0.0,
        "mse_act_mean": 0.0,
        "mse_tip_pos_mean": 0.0,
        "mse_target_pos_mean": 0.0,
        "mse_reach_err_mean": 0.0,
        "n_episodes": 2,
        "model_run_id": "test-model",
    }


def test_write_metrics_csv_columns(tmp_path: Path, script) -> None:
    rows = [
        _row("random", "none", 0, 0.5, 0.05),
        _row("hold", "high", 6, 1.0, 0.30),
    ]
    path = tmp_path / "metrics.csv"
    script._write_metrics_csv(rows, path)
    with open(path) as f:
        reader = csv.DictReader(f)
        loaded = list(reader)
    assert reader.fieldnames == list(script._METRICS_CSV_COLUMNS)
    assert len(loaded) == 2
    assert loaded[0]["controller"] == "random"
    assert loaded[0]["noise_condition"] == "none"
    assert float(loaded[0]["tip_estimation_error_mean"]) == pytest.approx(0.05)
    assert loaded[1]["controller"] == "hold"


def test_best_by_condition_picks_min_tip_err(tmp_path: Path, script) -> None:
    # Two rows for (random, none, 0): gain 0.5 wins over gain 1.0.
    rows = [
        _row("random", "none", 0, 0.0, 5.00),
        _row("random", "none", 0, 0.5, 0.05),
        _row("random", "none", 0, 1.0, 0.10),
        _row("random", "none", 6, 1.0, 0.30),
        _row("hold", "high", 6, 1.0, 0.40),
        _row("hold", "high", 6, 0.75, 0.20),
    ]
    path = tmp_path / "best.csv"
    script._write_best_by_condition_csv(rows, path)
    with open(path) as f:
        loaded = list(csv.DictReader(f))
    # 3 unique groups: (random, none, 0), (random, none, 6), (hold, high, 6)
    assert len(loaded) == 3
    by_key = {
        (r["controller"], r["noise_condition"], int(r["delay_steps"])): r
        for r in loaded
    }
    assert float(by_key[("random", "none", 0)]["gain"]) == pytest.approx(0.5)
    assert float(by_key[("random", "none", 0)]["tip_estimation_error_mean"]) == pytest.approx(0.05)
    assert float(by_key[("hold", "high", 6)]["gain"]) == pytest.approx(0.75)


def test_best_by_condition_empty(tmp_path: Path, script) -> None:
    path = tmp_path / "best.csv"
    script._write_best_by_condition_csv([], path)
    with open(path) as f:
        loaded = list(csv.DictReader(f))
    assert loaded == []
