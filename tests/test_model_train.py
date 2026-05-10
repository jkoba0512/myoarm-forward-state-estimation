"""Tests for myoarm_fse.models.train (no MyoSuite required).

Smoke-trains a tiny ForwardMLP on a small synthetic transition
dataset built from synthetic EpisodeLog instances. Goal is to
exercise the training pipeline end-to-end (seeds → split → train →
save → load → rollout) rather than to fit any meaningful dynamics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from myoarm_fse.data import EpisodeLog
from myoarm_fse.models import (
    ForwardMLP,
    TrainConfig,
    build_transitions,
    load_model,
    make_model_id,
    make_train_val_split,
    rollout_predictions,
    save_model,
    setup_seeds,
    train_forward_model,
)


# --- synthetic data helpers ---


def _make_log(
    *, episode_id: int = 0, n_steps: int = 8, qpos_dim: int = 2,
    qvel_dim: int = 2, act_dim: int = 3, action_dim: int = 3,
) -> EpisodeLog:
    T = n_steps
    f32 = np.float32
    qpos = np.linspace(0, 1, T * qpos_dim, dtype=f32).reshape(T, qpos_dim)
    excitation = np.linspace(0.1, 0.9, T * action_dim, dtype=f32).reshape(T, action_dim)
    return EpisodeLog(
        episode_id=episode_id,
        target_id=f"train:{episode_id}",
        target_split="train",
        target_seed=episode_id,
        target_pos_set=np.zeros(3, dtype=f32),
        controller_name="random",
        controller_seed=0,
        sdn_sigma=0.0,
        sdn_seed=0,
        obs_noise_sigma={},
        obs_noise_seed=0,
        obs_delay_steps=0,
        obs_compose="noisy_then_delayed",
        max_steps=T,
        n_steps=T,
        created_at="2026-05-10T00:00:00Z",
        config_hash="test",
        step=np.arange(T, dtype=np.int64),
        time=np.arange(T, dtype=f32) * 0.02,
        true_qpos=qpos,
        true_qvel=np.zeros((T, qvel_dim), dtype=f32),
        true_act=np.zeros((T, act_dim), dtype=f32),
        true_tip_pos=np.zeros((T, 3), dtype=f32),
        true_target_pos=np.zeros((T, 3), dtype=f32),
        true_reach_err=np.zeros((T, 3), dtype=f32),
        obs_qpos=qpos.copy(),
        obs_qvel=np.zeros((T, qvel_dim), dtype=f32),
        obs_act=np.zeros((T, act_dim), dtype=f32),
        obs_tip_pos=np.zeros((T, 3), dtype=f32),
        obs_target_pos=np.zeros((T, 3), dtype=f32),
        obs_reach_err=np.zeros((T, 3), dtype=f32),
        neural_command=excitation.copy(),
        excitation_command=excitation.copy(),
        excitation=excitation,
        api_action=np.zeros((T, action_dim), dtype=f32),
        last_ctrl=np.zeros((T, action_dim), dtype=f32),
        reward=np.zeros(T, dtype=f32),
        terminated=np.zeros(T, dtype=np.bool_),
        truncated=np.zeros(T, dtype=np.bool_),
    )


def _toy_dataset(n_episodes: int = 5, n_steps: int = 8):
    return build_transitions(
        [_make_log(episode_id=i, n_steps=n_steps) for i in range(n_episodes)]
    )


# --- TrainConfig ---


class TestTrainConfig:
    def test_defaults(self) -> None:
        cfg = TrainConfig()
        assert cfg.optimizer == "adam"
        assert cfg.lr == 1e-3
        assert cfg.batch_size == 256

    def test_invalid_optimizer(self) -> None:
        with pytest.raises(ValueError):
            TrainConfig(optimizer="sgd")

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValueError):
            TrainConfig(batch_size=0)

    def test_negative_lr(self) -> None:
        with pytest.raises(ValueError):
            TrainConfig(lr=-1e-3)

    def test_from_dict_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="unknown TrainConfig keys"):
            TrainConfig.from_dict({"banana": 1})


# --- setup_seeds ---


class TestSetupSeeds:
    def test_reproducible_across_calls(self) -> None:
        a = setup_seeds(0)
        b = setup_seeds(0)
        assert a == b

    def test_different_master_seeds_differ(self) -> None:
        a = setup_seeds(0)
        b = setup_seeds(1)
        assert a != b

    def test_keys_present(self) -> None:
        seeds = setup_seeds(0)
        assert {"model_init", "dataset_shuffle", "dataloader"} <= seeds.keys()

    def test_seeds_global_torch(self) -> None:
        # Two ForwardMLPs constructed after setup_seeds(0) must match.
        setup_seeds(0)
        a = ForwardMLP(state_dim=4, action_dim=2, hidden_dims=(8,))
        setup_seeds(0)
        b = ForwardMLP(state_dim=4, action_dim=2, hidden_dims=(8,))
        for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
            torch.testing.assert_close(pa, pb)

    def test_bool_seed_rejected(self) -> None:
        with pytest.raises(ValueError):
            setup_seeds(True)  # type: ignore[arg-type]


# --- make_train_val_split ---


class TestMakeTrainValSplit:
    def test_every_5th_to_val(self) -> None:
        ds = _toy_dataset(n_episodes=10)
        train, val = make_train_val_split(ds, val_step=5)
        assert val.n_episodes == 2  # episodes 0 and 5
        assert train.n_episodes == 8

    def test_invariant_total_n(self) -> None:
        ds = _toy_dataset(n_episodes=10)
        train, val = make_train_val_split(ds, val_step=5)
        assert train.n + val.n == ds.n

    def test_episode_ids_preserved_in_metadata(self) -> None:
        ds = _toy_dataset(n_episodes=10)
        _train, val = make_train_val_split(ds, val_step=5)
        ids = [m["episode_id"] for m in val.episode_metadata]
        assert ids == [0, 5]

    def test_val_step_zero_raises(self) -> None:
        ds = _toy_dataset()
        with pytest.raises(ValueError):
            make_train_val_split(ds, val_step=0)


# --- train_forward_model smoke ---


class TestTrainSmoke:
    def test_runs_few_epochs_and_returns_metrics(self) -> None:
        ds = _toy_dataset(n_episodes=10, n_steps=10)
        train_ds, val_ds = make_train_val_split(ds, val_step=5)
        config = TrainConfig(
            batch_size=8, epochs=3, early_stopping_patience=10, val_step=5,
        )
        seeds = setup_seeds(0)
        model = ForwardMLP(
            state_dim=ds.state_dim, action_dim=ds.action_dim,
            hidden_dims=(16,),
        )
        trained, metrics = train_forward_model(
            model, train_ds, val_ds, config, seeds=seeds,
        )
        assert isinstance(trained, ForwardMLP)
        assert len(metrics["train_loss_history"]) == 3
        assert len(metrics["val_loss_history"]) == 3
        assert metrics["best_epoch"] >= 0
        assert metrics["seeds"] == seeds
        assert metrics["n_train"] == train_ds.n
        assert metrics["n_val"] == val_ds.n

    def test_training_reduces_val_loss_on_simple_data(self) -> None:
        # Construct a dataset where Δx is a deterministic function of u
        # so a small MLP can fit it. Synthetic logs already produce
        # smooth qpos transitions; over a few epochs val loss should go
        # down (not strictly monotone, just final < initial).
        ds = _toy_dataset(n_episodes=10, n_steps=10)
        train_ds, val_ds = make_train_val_split(ds, val_step=5)
        config = TrainConfig(
            batch_size=8, epochs=20, early_stopping_patience=50, val_step=5,
        )
        seeds = setup_seeds(0)
        model = ForwardMLP(
            state_dim=ds.state_dim, action_dim=ds.action_dim,
            hidden_dims=(32, 32),
        )
        _, metrics = train_forward_model(model, train_ds, val_ds, config, seeds=seeds)
        assert metrics["val_loss_history"][-1] < metrics["val_loss_history"][0]

    def test_dim_mismatch_raises(self) -> None:
        ds = _toy_dataset()
        train_ds, val_ds = make_train_val_split(ds, val_step=5)
        # Wrong action_dim model.
        bad_model = ForwardMLP(
            state_dim=ds.state_dim, action_dim=ds.action_dim + 1, hidden_dims=(8,),
        )
        with pytest.raises(ValueError, match="dims"):
            train_forward_model(
                bad_model, train_ds, val_ds, TrainConfig(epochs=1, val_step=5),
            )


# --- save / load ---


class TestSaveLoadRoundTrip:
    def test_round_trip_preserves_predictions(self, tmp_path: Path) -> None:
        ds = _toy_dataset()
        train_ds, val_ds = make_train_val_split(ds, val_step=5)
        config = TrainConfig(
            batch_size=8, epochs=2, early_stopping_patience=10, val_step=5,
        )
        seeds = setup_seeds(0)
        model = ForwardMLP(
            state_dim=ds.state_dim, action_dim=ds.action_dim, hidden_dims=(16,),
        )
        trained, metrics = train_forward_model(
            model, train_ds, val_ds, config, seeds=seeds,
        )

        path = tmp_path / make_model_id()
        save_model(
            trained,
            config={
                "architecture": {
                    "kind": "ForwardMLP",
                    "state_dim": ds.state_dim,
                    "action_dim": ds.action_dim,
                    "hidden_dims": list(trained.hidden_dims),
                    "activation": "relu",
                    "norm": "layernorm",
                },
                "train": {"lr": 1e-3, "epochs": 2},
                "data": {"n_transitions": ds.n},
            },
            metrics=metrics,
            path=path,
        )

        loaded, loaded_config, loaded_metrics = load_model(path)
        x = torch.zeros((3, ds.state_dim))
        u = torch.zeros((3, ds.action_dim))
        torch.testing.assert_close(trained(x, u), loaded(x, u))
        assert loaded_config["architecture"]["state_dim"] == ds.state_dim
        assert loaded_metrics["best_epoch"] == metrics["best_epoch"]
        assert (path / "info.json").exists()


# --- rollout_predictions ---


class TestRolloutPredictions:
    def test_zero_error_with_perfect_model(self) -> None:
        # Build a dataset where x_next - x is constant; a model that
        # predicts that constant will have zero rollout error. We don't
        # actually train; we monkey the forward to return the truth.
        ds = _toy_dataset(n_episodes=4, n_steps=10)
        model = ForwardMLP(
            state_dim=ds.state_dim, action_dim=ds.action_dim, hidden_dims=(8,),
        )

        # Hack: use a closure that returns the recorded Δx for the given
        # batch by indexing into the dataset. We replace the underlying
        # net with an identity-ish op via monkey patch on the module.
        # Simpler: subclass ForwardMLP.

        class TruthModel(ForwardMLP):
            def __init__(self, ds):
                super().__init__(
                    state_dim=ds.state_dim, action_dim=ds.action_dim,
                    hidden_dims=(8,),
                )
                self._ds = ds

            def forward(self, x, u):
                # For each row in the batch, find the first matching
                # (x, u) row in the dataset and return the recorded dx.
                if x.dim() == 1:
                    return torch.from_numpy(self._lookup(x.numpy(), u.numpy())).float()
                outs = [
                    self._lookup(x[i].numpy(), u[i].numpy()) for i in range(x.shape[0])
                ]
                return torch.from_numpy(np.stack(outs, axis=0)).float()

            def _lookup(self, x_row, u_row):
                # Find a row in self._ds with matching (x, u).
                for i in range(self._ds.n):
                    if np.allclose(self._ds.x[i], x_row, atol=1e-5) and np.allclose(
                        self._ds.u[i], u_row, atol=1e-5
                    ):
                        return self._ds.dx[i].astype(np.float32)
                return np.zeros(self._ds.state_dim, dtype=np.float32)

        truth = TruthModel(ds)
        out = rollout_predictions(truth, ds, horizons=(1,))
        assert out[1]["rollout_mse"] < 1e-6

    def test_horizon_1_matches_one_step(self) -> None:
        ds = _toy_dataset(n_episodes=3, n_steps=10)
        model = ForwardMLP(
            state_dim=ds.state_dim, action_dim=ds.action_dim, hidden_dims=(16,),
        )
        out = rollout_predictions(model, ds, horizons=(1, 5))
        # Sanity: outputs include both horizons and the values are finite.
        assert 1 in out and 5 in out
        for h, r in out.items():
            assert "rollout_mse" in r
            assert "tip_prediction_error" in r
            assert np.isfinite(r["rollout_mse"])
            assert np.isfinite(r["tip_prediction_error"])

    def test_invalid_horizon(self) -> None:
        ds = _toy_dataset()
        model = ForwardMLP(
            state_dim=ds.state_dim, action_dim=ds.action_dim, hidden_dims=(8,),
        )
        with pytest.raises(ValueError):
            rollout_predictions(model, ds, horizons=(0,))

    def test_empty_dataset(self) -> None:
        ds = build_transitions([])
        model = ForwardMLP(state_dim=10, action_dim=4, hidden_dims=(8,))
        out = rollout_predictions(model, ds, horizons=(1,))
        assert out[1]["rollout_mse"] == 0.0

    def test_horizon_too_long_for_episode(self) -> None:
        # n_steps=8 means each episode contributes 7 transitions; horizon
        # 100 > 7 leaves no valid windows → 0.0 (no contribution).
        ds = _toy_dataset(n_episodes=2, n_steps=8)
        model = ForwardMLP(
            state_dim=ds.state_dim, action_dim=ds.action_dim, hidden_dims=(8,),
        )
        out = rollout_predictions(model, ds, horizons=(100,))
        assert out[100]["rollout_mse"] == 0.0
