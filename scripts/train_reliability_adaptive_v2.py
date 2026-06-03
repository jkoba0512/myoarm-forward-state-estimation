"""Across-trial β adaptation for the reliability-adaptive observer (v2).

This script implements outcome-driven adaptation of the meta-parameters
β₀, β₁ in :class:`ReliabilityAdaptiveObserver` using SPSA (Simultaneous
Perturbation Stochastic Approximation, Spall 1992). The agent does not
access the true state, the oracle gain, or any offline label; only the
per-episode task outcome (``min_tip_error``) is fed back to update β.

The within-trial reliability adaptation (Sec. 3.X of the paper) stays
identical to v1; the only new ingredient is the outer trial loop:

```
init β
for iteration n = 1..N:
    sample perturbation Δ ∈ {-1, +1}^{2F}     # F = 5 fields, 2 = β₀+β₁
    c_n = c / (n + 1)^γ
    a_n = a / (n + 1 + A)^α
    outcome_+ = mean over sample_cells: run_episode(β + c_n · Δ)
    outcome_- = mean over sample_cells: run_episode(β − c_n · Δ)
    g_n[i] = (outcome_+ − outcome_−) / (2 · c_n · Δ[i])
    β ← β − a_n · g_n
```

A "sample" within an iteration is the closed-loop outcome on either a
single cell (PoC mode) or the average across the focused grid (global
mode), at a single target index drawn fresh each iteration.

Usage::

    uv run python scripts/train_reliability_adaptive_v2.py \
        --config configs/train/reliability_adaptive_v2_poc.yaml

The script writes the β / outcome trajectory to
``runs/reliability_adaptive_v2/<eval_id>/`` along with the final config.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import yaml

from myoarm_fse.controllers import (
    EndpointErrorFeedbackController,
    JointSpacePDController,
    StabilizedEndpointController,
    make_controller,
)
from myoarm_fse.envs.extractors import extract_state
from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim
from myoarm_fse.envs.factory import make_env
from myoarm_fse.envs.ik import (
    actuator_moment_dense,
    solve_ik,
    tip_jacobian_dense,
)
from myoarm_fse.envs.state import MyoArmState, StateSpec
from myoarm_fse.envs.targets import TargetSet
from myoarm_fse.envs.noise import SignalDependentMotorNoise
from myoarm_fse.envs.wrappers import (
    DelayedObservationWrapper,
    NoisyObservationWrapper,
)
from myoarm_fse.estimators import (
    ReliabilityAdaptiveConfig,
    ReliabilityAdaptiveObserver,
)
from myoarm_fse.evaluation.closed_loop import run_closed_loop_episode
from myoarm_fse.metrics.reaching import minimum_tip_error
from myoarm_fse.models import load_model

_ADAPTIVE_FIELDS: tuple[str, ...] = (
    "qpos", "qvel", "act", "tip_pos", "reach_err",
)
_BETA_KEYS: tuple[str, ...] = tuple(
    f"{name}_{field}" for name in ("beta0", "beta1") for field in _ADAPTIVE_FIELDS
)  # 10 params, ordered (beta0_qpos, beta0_qvel, ..., beta1_qpos, beta1_qvel, ...)


@dataclass
class SPSAState:
    a: float
    c: float
    alpha: float
    gamma: float
    A: float
    iteration: int = 0

    def a_n(self) -> float:
        return self.a / (self.iteration + 1 + self.A) ** self.alpha

    def c_n(self) -> float:
        return self.c / (self.iteration + 1) ** self.gamma


def _beta_to_vec(cfg: ReliabilityAdaptiveConfig) -> np.ndarray:
    """Pack (β₀_qpos, ..., β₀_reach_err, β₁_qpos, ..., β₁_reach_err) into a
    10-vector in the order of ``_BETA_KEYS``."""
    vec = np.zeros(len(_BETA_KEYS), dtype=np.float64)
    for i, field in enumerate(_ADAPTIVE_FIELDS):
        vec[i] = cfg.beta0[field]
        vec[i + len(_ADAPTIVE_FIELDS)] = cfg.beta1[field]
    return vec


def _vec_to_beta(vec: np.ndarray, base_cfg: ReliabilityAdaptiveConfig) -> ReliabilityAdaptiveConfig:
    """Unpack a 10-vector into a fresh ReliabilityAdaptiveConfig."""
    beta0 = {}
    beta1 = {}
    F = len(_ADAPTIVE_FIELDS)
    for i, field in enumerate(_ADAPTIVE_FIELDS):
        beta0[field] = float(vec[i])
        beta1[field] = float(vec[i + F])
    return ReliabilityAdaptiveConfig(
        alpha=base_cfg.alpha,
        epsilon=base_cfg.epsilon,
        var_init=base_cfg.var_init,
        beta0=beta0,
        beta1=beta1,
        target_pos_gain=base_cfg.target_pos_gain,
    )


def _build_controller(
    name: str,
    controller_spec: dict[str, Any],
    env: Any,
    target_pos: np.ndarray,
    action_dim: int,
    seed: int,
) -> Any:
    """Build a controller, including IK pre-solve / Jacobian capture as needed."""
    if name == "joint_pd":
        env.reset()
        mujoco.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
        target_qpos, _ = solve_ik(
            env, target_pos,
            max_iter=int(controller_spec.get("ik_max_iter", 200)),
            tol=float(controller_spec.get("ik_tol", 0.01)),
            damping=float(controller_spec.get("ik_damping", 0.1)),
        )
        moment_arm = actuator_moment_dense(env)
        c = JointSpacePDController(
            action_dim=action_dim,
            target_qpos=target_qpos,
            moment_arm=moment_arm,
            Kp=float(controller_spec.get("Kp", 30.0)),
            Kd=float(controller_spec.get("Kd", 3.0)),
            action_scale=float(controller_spec.get("action_scale", 5.0)),
        )
        c.reset(seed=seed)
        return c
    if name == "endpoint_feedback":
        env.reset()
        mujoco.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
        jacobian = tip_jacobian_dense(env)
        moment_arm = actuator_moment_dense(env)
        c = EndpointErrorFeedbackController(
            action_dim=action_dim,
            target_pos=target_pos,
            jacobian=jacobian,
            moment_arm=moment_arm,
            Kp=float(controller_spec.get("Kp", 30.0)),
            Kd=float(controller_spec.get("Kd", 3.0)),
            action_scale=float(controller_spec.get("action_scale", 5.0)),
        )
        c.reset(seed=seed)
        return c
    if name in ("stabilized_endpoint", "a_plus_b", "nnls_ramp"):
        # NNLS muscle routing + virtual target ramp (Stage B pivot).
        env.reset()
        mujoco.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
        init_tip = np.asarray(extract_state(env).tip_pos, dtype=np.float32)
        jacobian = tip_jacobian_dense(env)
        moment_arm = actuator_moment_dense(env)
        T_ramp_raw = controller_spec.get("T_ramp", 300)
        T_ramp_arg = None if T_ramp_raw is None else int(T_ramp_raw)
        c = StabilizedEndpointController(
            action_dim=action_dim,
            init_tip=init_tip,
            target_pos=target_pos,
            jacobian=jacobian,
            moment_arm=moment_arm,
            Kp=float(controller_spec.get("Kp", 30.0)),
            Kd=float(controller_spec.get("Kd", 3.0)),
            action_scale=float(controller_spec.get("action_scale", 5.0)),
            T_ramp=T_ramp_arg,
            record_history=bool(
                controller_spec.get("record_history", False)
            ),
        )
        c.reset(seed=seed)
        return c
    c = make_controller(controller_spec, action_dim=action_dim, seed=seed)
    c.reset(seed=seed)
    return c


def _episode_outcome(
    env: Any,
    state_spec: StateSpec,
    action_adapter: ActionAdapter,
    forward_model: Any,
    beta_cfg: ReliabilityAdaptiveConfig,
    target_pos: np.ndarray,
    sigma_dict: dict[str, float],
    delay: int,
    controller_name: str,
    controller_spec: dict[str, Any],
    action_dim: int,
    seed: int,
    max_steps: int,
) -> float:
    """Run one episode under the given β; return min_tip_error."""
    obs_noise = (
        NoisyObservationWrapper(
            spec=state_spec, sigma=sigma_dict, rng=seed,
        )
        if any(v != 0.0 for v in sigma_dict.values())
        else None
    )
    obs_delay = (
        DelayedObservationWrapper(spec=state_spec, delay_steps=delay)
        if delay > 0 else None
    )
    sdn = None  # No SDN by default for training, mirror config
    estimator = ReliabilityAdaptiveObserver(
        forward_model=forward_model,
        state_spec=state_spec,
        delay_steps=delay,
        config=beta_cfg,
    )
    controller = _build_controller(
        controller_name, controller_spec, env, target_pos, action_dim, seed,
    )
    result = run_closed_loop_episode(
        env, controller, estimator, target_pos,
        state_spec=state_spec,
        action_adapter=action_adapter,
        sdn=sdn,
        obs_noise=obs_noise,
        obs_delay=obs_delay,
        obs_compose="noisy_then_delayed",
        max_steps=max_steps,
    )
    return float(minimum_tip_error(result.log))


def _sample_cells_and_targets(
    cells: list[dict[str, Any]],
    target_set: TargetSet,
    rng: np.random.Generator,
    n_samples: int,
) -> list[tuple[dict[str, Any], np.ndarray, int]]:
    """Pick (cell, target_pos, ep_seed) tuples for one outcome evaluation."""
    out = []
    for _ in range(n_samples):
        cell = cells[int(rng.integers(0, len(cells)))]
        ep_idx = int(rng.integers(0, target_set.n))
        target_pos = target_set.target_pos[ep_idx].astype(np.float32, copy=True)
        seed = int(rng.integers(0, 2**31 - 1))
        out.append((cell, target_pos, seed))
    return out


def _aggregate_outcome(
    samples: list[tuple[dict[str, Any], np.ndarray, int]],
    *,
    env: Any,
    state_spec: StateSpec,
    action_adapter: ActionAdapter,
    forward_model: Any,
    beta_cfg: ReliabilityAdaptiveConfig,
    controller_name: str,
    controller_spec: dict[str, Any],
    action_dim: int,
    max_steps: int,
) -> float:
    outcomes = []
    for cell, target_pos, seed in samples:
        outcome = _episode_outcome(
            env, state_spec, action_adapter, forward_model, beta_cfg,
            target_pos, cell["sigma"], int(cell["delay"]),
            controller_name, controller_spec,
            action_dim, seed, max_steps,
        )
        outcomes.append(outcome)
    return float(np.mean(outcomes))


def main(argv: list[str] | None = None) -> Path:
    p = argparse.ArgumentParser(description="SPSA β adaptation for v2 observer.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--max-iter", type=int, default=None,
                   help="Override config.spsa.max_iter")
    p.add_argument("--samples-per-side", type=int, default=None,
                   help="Override config.spsa.samples_per_side")
    args = p.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed_master = int(cfg.get("seed", 0))
    rng = np.random.default_rng(seed_master)

    env_id = str(cfg["env_id"])
    horizon = int(cfg["horizon"])
    forward_model_path = str(cfg["forward_model"])
    target_set_path = str(cfg["target_set"])
    max_steps = int(cfg.get("max_steps", horizon))
    controller_name = str(cfg["controller"]["name"])
    controller_spec = dict(cfg["controller"])

    # Cells to sample: list of {sigma: dict, delay: int}
    cells: list[dict[str, Any]] = []
    for noise_name, sigma in cfg["noise_conditions"].items():
        for d in cfg["delay_grid"]:
            cells.append({
                "name": noise_name,
                "sigma": dict(sigma),
                "delay": int(d),
            })
    print(f"Sampling {len(cells)} cells: "
          f"{[(c['name'], c['delay']) for c in cells]}")

    # SPSA hyperparams
    spsa_cfg = cfg.get("spsa", {})
    max_iter = int(args.max_iter or spsa_cfg.get("max_iter", 100))
    samples_per_side = int(args.samples_per_side or spsa_cfg.get("samples_per_side", 1))
    spsa = SPSAState(
        a=float(spsa_cfg.get("a", 0.1)),
        c=float(spsa_cfg.get("c", 0.05)),
        alpha=float(spsa_cfg.get("alpha", 0.602)),
        gamma=float(spsa_cfg.get("gamma", 0.101)),
        A=float(spsa_cfg.get("A", max(10.0, max_iter * 0.1))),
    )

    init_beta_cfg = ReliabilityAdaptiveConfig(
        alpha=float(cfg.get("observer", {}).get("alpha", 0.05)),
        epsilon=float(cfg.get("observer", {}).get("epsilon", 1e-6)),
        var_init=float(cfg.get("observer", {}).get("var_init", 1.0)),
        beta0={
            f: float(cfg.get("observer", {}).get("init_beta0", {}).get(f, 0.0))
            for f in _ADAPTIVE_FIELDS
        },
        beta1={
            f: float(cfg.get("observer", {}).get("init_beta1", {}).get(f, 0.5))
            for f in _ADAPTIVE_FIELDS
        },
        target_pos_gain=float(cfg.get("observer", {}).get("target_pos_gain", 1.0)),
    )

    eval_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = Path(cfg.get("output_root", "runs/reliability_adaptive_v2")) / eval_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps({"eval_id": eval_id, "config": cfg}, indent=2)
    )

    # Build env and shared state
    forward_model, model_config, _ = load_model(forward_model_path)
    forward_model.eval()
    state_dim_check = int(model_config["architecture"]["state_dim"])
    if state_dim_check != 83:
        raise NotImplementedError(
            f"state_dim={state_dim_check} layout not auto-derivable"
        )
    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)
    target_set = TargetSet.load(target_set_path)
    env = make_env(env_id, horizon=horizon)
    try:
        action_dim = detect_action_dim(env)
        action_adapter = ActionAdapter(action_dim=action_dim)

        beta_vec = _beta_to_vec(init_beta_cfg)
        D = len(beta_vec)
        history: list[dict[str, Any]] = []
        print(f"SPSA D={D}, a={spsa.a}, c={spsa.c}, alpha={spsa.alpha}, "
              f"gamma={spsa.gamma}, A={spsa.A}, max_iter={max_iter}, "
              f"samples_per_side={samples_per_side}")
        print(f"Init β: {beta_vec}")

        for it in range(max_iter):
            t0 = time.time()
            spsa.iteration = it
            a_n = spsa.a_n()
            c_n = spsa.c_n()

            # Perturbation Δ in {-1, +1}^D
            delta = rng.choice([-1.0, +1.0], size=D)
            beta_plus = beta_vec + c_n * delta
            beta_minus = beta_vec - c_n * delta

            # Pair-matched samples (same target+seed for + and -):
            samples = _sample_cells_and_targets(cells, target_set, rng,
                                                samples_per_side)
            cfg_plus = _vec_to_beta(beta_plus, init_beta_cfg)
            cfg_minus = _vec_to_beta(beta_minus, init_beta_cfg)
            outcome_plus = _aggregate_outcome(
                samples, env=env, state_spec=state_spec,
                action_adapter=action_adapter, forward_model=forward_model,
                beta_cfg=cfg_plus, controller_name=controller_name,
                controller_spec=controller_spec, action_dim=action_dim,
                max_steps=max_steps,
            )
            outcome_minus = _aggregate_outcome(
                samples, env=env, state_spec=state_spec,
                action_adapter=action_adapter, forward_model=forward_model,
                beta_cfg=cfg_minus, controller_name=controller_name,
                controller_spec=controller_spec, action_dim=action_dim,
                max_steps=max_steps,
            )
            # SPSA gradient estimate: g = (out+ − out−) / (2 c) * (1 / Δ_i)
            # Since Δ_i ∈ {-1, +1}, 1/Δ_i = Δ_i, so g = (out+ − out−)/(2c) * Δ
            grad = (outcome_plus - outcome_minus) / (2.0 * c_n) * delta
            beta_vec = beta_vec - a_n * grad

            elapsed = time.time() - t0
            entry = {
                "iter": it,
                "a_n": a_n,
                "c_n": c_n,
                "outcome_plus": outcome_plus,
                "outcome_minus": outcome_minus,
                "outcome_mean": 0.5 * (outcome_plus + outcome_minus),
                "beta": beta_vec.tolist(),
                "grad_norm": float(np.linalg.norm(grad)),
                "delta": delta.tolist(),
                "elapsed_sec": elapsed,
            }
            history.append(entry)
            if it % 5 == 0 or it == max_iter - 1:
                print(f"  iter {it:3d}: outcome+ {outcome_plus:.4f}  "
                      f"outcome- {outcome_minus:.4f}  "
                      f"|grad|={entry['grad_norm']:.3f}  "
                      f"β[:3]={beta_vec[:3].round(3).tolist()}  "
                      f"({elapsed:.1f}s)")
    finally:
        env.close()

    # Save history
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    final_cfg = _vec_to_beta(beta_vec, init_beta_cfg)
    (out_dir / "final_beta.json").write_text(json.dumps({
        "beta0": final_cfg.beta0,
        "beta1": final_cfg.beta1,
        "alpha": final_cfg.alpha,
        "epsilon": final_cfg.epsilon,
        "var_init": final_cfg.var_init,
        "target_pos_gain": final_cfg.target_pos_gain,
    }, indent=2))
    print(f"Saved {out_dir}/history.json + final_beta.json")
    return out_dir


if __name__ == "__main__":
    main()
