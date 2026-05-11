"""Generate Phase 1-4 paper figures from existing run artifacts.

Reads CSV / JSON outputs under ``runs/`` and writes ``figures/F2..F6.{png,pdf}``
plus tidied summary CSVs under ``figures/data/``. F1 (system overview block
diagram) is a separate ``F1.png/pdf`` drawn with matplotlib primitives.

Inputs (locked in ``docs/03_PaperOutline.md``):

```
runs/estimators/2026-05-10T11-01-23Z/best_by_condition.csv
runs/learned_gain_evals/2026-05-10T12-59-43Z/comparison.csv
runs/closed_loop/2026-05-11T07-08-39Z/metrics.csv          # D MVP
runs/closed_loop/2026-05-11T06-15-10Z/metrics.csv          # E
runs/closed_loop/2026-05-11T08-16-43Z/metrics.csv          # BC full
runs/closed_loop/2026-05-11T08-32-19Z/metrics.csv          # BC v1
runs/closed_loop/2026-05-11T08-41-57Z/metrics.csv          # BC v2
```

Usage::

    uv run python scripts/make_paper_figures.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

FIG_DIR = Path("figures")
DATA_DIR = FIG_DIR / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Locked input paths.
STRESS_ORACLE_OLD = "runs/estimators/2026-05-10T11-01-23Z/best_by_condition.csv"
STRESS_ORACLE_NEW = "runs/estimators/2026-05-11T09-53-59Z/best_by_condition.csv"
STRESS_ORACLE_K8 = "runs/estimators/2026-05-11T11-48-05Z/best_by_condition.csv"
STRESS_ORACLE = STRESS_ORACLE_OLD  # back-compat alias used in some helpers
STRESS_EVAL = "runs/learned_gain_evals/2026-05-10T12-59-43Z/comparison.csv"
D_MVP = "runs/closed_loop/2026-05-11T07-08-39Z/metrics.csv"            # OLD baseline D MVP
D_MVP_PHASE_B = "runs/closed_loop/2026-05-11T10-43-45Z/metrics.csv"    # K=4 D MVP
D_MVP_PHASE_BPRIME = "runs/closed_loop/2026-05-11T12-37-26Z/metrics.csv"  # K=8 D MVP
E_MVP = "runs/closed_loop/2026-05-11T06-15-10Z/metrics.csv"
BC_FULL = "runs/closed_loop/2026-05-11T08-16-43Z/metrics.csv"
BC_V1 = "runs/closed_loop/2026-05-11T08-32-19Z/metrics.csv"
BC_V2 = "runs/closed_loop/2026-05-11T08-41-57Z/metrics.csv"

# Color-blind safe qualitative palette (ColorBrewer-inspired).
COLORS = {
    "K=0.0": "#d95f02",        # orange
    "K=1.0": "#1f78b4",        # blue
    "learned": "#33a02c",      # green
    "oracle": "#000000",       # black
    "best_per_delay": "#7570b3",
    "best_per_noise": "#e7298a",
    "global_best": "#a6761d",
}
NOISE_ORDER = ["none", "low", "medium", "high", "vhigh", "xhigh"]
DELAY_ORDER = [0, 6, 18, 36]


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("png", "pdf"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  saved {path}")
    plt.close(fig)


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 100,
        "savefig.dpi": 300,
    })


def _fbool(s: object) -> float:
    return 1.0 if str(s) in ("True", "true", "1") else 0.0


# ---------------- F1: System overview block diagram ----------------


def fig_F1_system_overview() -> None:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6.5); ax.axis("off")

    def box(x, y, w, h, label, color="#f0f0f0"):
        ax.add_patch(patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="black", linewidth=1.0,
        ))
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=9)

    def arrow(x1, y1, x2, y2, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.0, color="#444"))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label,
                    ha="center", fontsize=7, style="italic")

    # Boxes
    box(0.2, 4.2, 2.5, 1.0, "MuJoCo / MyoSuite\n(true state)", "#dddddd")
    box(3.4, 4.2, 2.5, 1.0, "Observation\nwrappers\n(noise + delay)")
    box(6.6, 4.2, 2.5, 1.0, "Kalman estimator\nx_pred + K·(y - x̂)", "#cfeacf")
    box(9.8, 4.2, 2.5, 1.0, "Controller\n(joint-PD / BC)", "#c8d8ee")
    # Forward model bypass
    box(6.6, 2.0, 2.5, 1.0, "Forward model\n(residual MLP)", "#fde6c4")
    # Action loop
    arrow(2.7, 4.7, 3.4, 4.7, "true state")
    arrow(5.9, 4.7, 6.6, 4.7, "y_obs")
    arrow(9.1, 4.7, 9.8, 4.7, "x_est")
    arrow(11.0, 4.2, 11.0, 1.0)
    arrow(11.0, 1.0, 1.4, 1.0)
    arrow(1.4, 1.0, 1.4, 4.2, "u (excitation)")
    arrow(7.8, 3.0, 7.8, 4.2)
    arrow(7.8, 4.2, 7.8, 3.0)  # double-headed visual

    ax.text(7.0, 6.0, "Closed loop: estimator output drives controller; "
            "true state is oracle only",
            fontsize=8, style="italic", ha="left")
    _save(fig, "F1_system_overview")


# ---------------- F2: Stress oracle K heatmap ----------------


def _oracle_pivot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["delay_steps"] = df["delay_steps"].astype(int)
    df["gain"] = df["gain"].astype(float)
    return (
        df.groupby(["delay_steps", "noise_condition"])["gain"]
        .mean()
        .unstack("noise_condition")
        .reindex(index=DELAY_ORDER, columns=NOISE_ORDER)
    )


def fig_F2_stress_oracle() -> None:
    """F2: 3-panel oracle K heatmap (single-step / K=4 / K=8)."""
    old = _oracle_pivot(STRESS_ORACLE_OLD)
    k4 = _oracle_pivot(STRESS_ORACLE_NEW)
    k8 = _oracle_pivot(STRESS_ORACLE_K8)
    old.to_csv(DATA_DIR / "F2_stress_oracle_K_old.csv")
    k4.to_csv(DATA_DIR / "F2_stress_oracle_K_new.csv")
    k8.to_csv(DATA_DIR / "F2_stress_oracle_K_k8.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2), sharey=True)
    for ax, pivot, title in (
        (axes[0], old, "Single-step"),
        (axes[1], k4, "K=4 multi-step"),
        (axes[2], k8, "K=8 multi-step"),
    ):
        im = ax.imshow(pivot.values, cmap="viridis_r", vmin=0.0, vmax=1.0,
                       aspect="auto")
        ax.set_xticks(range(len(NOISE_ORDER)))
        ax.set_xticklabels(NOISE_ORDER, rotation=20, ha="right")
        ax.set_yticks(range(len(DELAY_ORDER)))
        ax.set_yticklabels(DELAY_ORDER)
        ax.set_xlabel("Observation noise level")
        ax.set_title(title)
        for i, _d in enumerate(DELAY_ORDER):
            for j, _n in enumerate(NOISE_ORDER):
                v = pivot.iloc[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.55 else "black", fontsize=8)
    axes[0].set_ylabel("Observation delay (steps)")
    fig.colorbar(im, ax=axes, label="Oracle K*", shrink=0.85, pad=0.02)
    fig.suptitle("Oracle Kalman gain across the stress grid by forward-model "
                 "supervision",
                 y=1.04, fontsize=10)
    _save(fig, "F2_stress_oracle_K")


# ---------------- F3: 7-strategy stress eval comparison ----------------


def fig_F3_stress_eval() -> None:
    df = pd.read_csv(STRESS_EVAL)
    df["tip_estimation_error_mean"] = df["tip_estimation_error_mean"].astype(float)

    # Aggregate per strategy.
    strategies = ["K=0.0", "K=1.0", "global_best", "best_per_delay",
                  "best_per_noise", "learned", "oracle"]
    agg = (
        df[df["strategy"].isin(strategies)]
        .groupby("strategy")["tip_estimation_error_mean"]
        .agg(["mean", "max"])
        .reindex(strategies)
    )
    agg.to_csv(DATA_DIR / "F3_stress_eval_summary.csv")

    # Compute delta vs oracle.
    by = df.set_index(["strategy", "controller", "noise_condition", "delay_steps"])
    oracle_err = (
        df[df["strategy"] == "oracle"]
        .set_index(["controller", "noise_condition", "delay_steps"])
        ["tip_estimation_error_mean"]
    )
    deltas = defaultdict(list)
    for _, row in df.iterrows():
        key = (row["controller"], row["noise_condition"], int(row["delay_steps"]))
        if key in oracle_err.index:
            d = float(row["tip_estimation_error_mean"]) - float(oracle_err.loc[key])
            deltas[row["strategy"]].append(d)
    mean_d = {s: float(np.mean(deltas[s])) for s in strategies}
    max_d = {s: float(np.max(deltas[s])) for s in strategies}

    # Save delta summary
    pd.DataFrame({
        "strategy": strategies,
        "mean_tip_err": [agg.loc[s, "mean"] for s in strategies],
        "max_tip_err": [agg.loc[s, "max"] for s in strategies],
        "mean_delta_vs_oracle": [mean_d[s] for s in strategies],
        "max_delta_vs_oracle": [max_d[s] for s in strategies],
    }).to_csv(DATA_DIR / "F3_stress_eval_strategy_deltas.csv", index=False)

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True)
    x = np.arange(len(strategies))

    # Top: mean tip_estimation_error per strategy
    colors = [COLORS.get(s.replace("=", "="), "#888888") for s in strategies]
    # K=0 will dominate; show on log scale to keep small values readable.
    ax_top.bar(x, [agg.loc[s, "mean"] for s in strategies], color=colors)
    ax_top.set_yscale("log")
    ax_top.set_ylabel("mean tip est. error (m, log)")
    ax_top.set_title("Stress eval (3 ctrl × 6 noise × 4 delay = 72 conds)")
    ax_top.grid(axis="y", alpha=0.3, which="both")

    # Bottom: delta vs oracle (linear)
    ax_bot.bar(x, [mean_d[s] for s in strategies], color=colors,
               label="mean Δ", alpha=0.85)
    ax_bot.bar(x, [max_d[s] for s in strategies], color=colors,
               edgecolor="black", facecolor="none", linewidth=1.2,
               label="max Δ")
    ax_bot.set_ylabel("Δ vs oracle (m)")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(strategies, rotation=25, ha="right")
    ax_bot.legend(loc="upper right", frameon=False)
    ax_bot.set_yscale("symlog", linthresh=0.001)
    ax_bot.grid(axis="y", alpha=0.3, which="both")
    ax_bot.axhline(0, color="black", linewidth=0.5)
    _save(fig, "F3_stress_eval_comparison")


# ---------------- F4: Phase 2 D estimator differentiation heatmap ----------------


def fig_F4_phase2_d_delta() -> None:
    df = pd.read_csv(D_MVP)
    df["delay_steps"] = df["delay_steps"].astype(int)
    df["final_tip_error"] = df["final_tip_error"].astype(float)
    df["min_tip_error"] = df["min_tip_error"].astype(float)

    # Group means over 10 episodes per cell.
    g = (
        df.groupby(["estimator", "noise_condition", "delay_steps"])
        [["final_tip_error", "min_tip_error"]]
        .mean()
    )

    noise_axis = ["none", "high", "xhigh"]
    delay_axis = [0, 18]

    def matrix(metric, est):
        m = np.zeros((len(delay_axis), len(noise_axis)))
        for i, d in enumerate(delay_axis):
            for j, n in enumerate(noise_axis):
                m[i, j] = float(g.loc[(est, n, d), metric])
        return m

    k1_final = matrix("final_tip_error", "K=1.0")
    learned_final = matrix("final_tip_error", "learned_stage_a_retrained")
    delta_final = learned_final - k1_final

    k1_min = matrix("min_tip_error", "K=1.0")
    learned_min = matrix("min_tip_error", "learned_stage_a_retrained")
    delta_min = learned_min - k1_min

    pd.DataFrame(delta_final, index=delay_axis, columns=noise_axis).to_csv(
        DATA_DIR / "F4_phase2d_delta_final.csv"
    )
    pd.DataFrame(delta_min, index=delay_axis, columns=noise_axis).to_csv(
        DATA_DIR / "F4_phase2d_delta_min.csv"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 2.8))
    vmin = min(delta_final.min(), delta_min.min())
    vmax = max(delta_final.max(), delta_min.max())
    vbound = max(abs(vmin), abs(vmax))
    for ax, m, title in [
        (ax1, delta_final, "Δ final_tip_error (learned − K=1)"),
        (ax2, delta_min, "Δ min_tip_error (learned − K=1)"),
    ]:
        im = ax.imshow(m, cmap="RdBu_r", vmin=-vbound, vmax=vbound, aspect="auto")
        ax.set_xticks(range(len(noise_axis)))
        ax.set_xticklabels(noise_axis)
        ax.set_yticks(range(len(delay_axis)))
        ax.set_yticklabels(delay_axis)
        ax.set_xlabel("Noise")
        ax.set_ylabel("Delay (steps)")
        ax.set_title(title)
        for i in range(len(delay_axis)):
            for j in range(len(noise_axis)):
                ax.text(j, i, f"{m[i, j]:+.3f}", ha="center", va="center",
                        fontsize=8,
                        color="black" if abs(m[i, j]) < vbound * 0.5 else "white")
        fig.colorbar(im, ax=ax, label="m")
    fig.suptitle("Phase 2 D (joint-PD + IK): estimator effect on reaching",
                 fontsize=10, y=1.02)
    _save(fig, "F4_phase2d_delta_heatmap")


# ---------------- F5: Phase 2 D representative trajectories ----------------


def fig_F5_phase2_d_trajectories() -> None:
    """F5 needs per-step trajectories that the standard eval doesn't save.

    We re-run a single representative cell here (noise=none, delay=18,
    3 estimators on the same target) so the trajectory plot is reproducible
    without relying on saved tensors.
    """
    import mujoco
    import yaml

    from myoarm_fse.controllers import JointSpacePDController
    from myoarm_fse.data.rollout import EpisodeSpec
    from myoarm_fse.envs.actions import ActionAdapter, detect_action_dim
    from myoarm_fse.envs.factory import make_env
    from myoarm_fse.envs.ik import actuator_moment_dense, solve_ik
    from myoarm_fse.envs.state import StateSpec
    from myoarm_fse.envs.targets import TargetSet
    from myoarm_fse.envs.wrappers import DelayedObservationWrapper
    from myoarm_fse.estimators import (
        FixedGainKalmanEstimator,
        LearnedGainKalmanEstimator,
        load_learned_gain_model,
    )
    from myoarm_fse.evaluation import run_closed_loop_episode
    from myoarm_fse.models import load_model

    with open("configs/closed_loop/joint_pd_mvp.yaml") as f:
        cfg = yaml.safe_load(f)

    target_set = TargetSet.load(cfg["target_set"])
    forward_model, model_config, _ = load_model(cfg["forward_model"])
    forward_model.eval()
    state_spec = StateSpec(qpos_dim=20, qvel_dim=20, act_dim=34)

    learned_path = "runs/learned_gain_models/2026-05-10T11-51-09Z"
    predictor, learned_cfg, _ = load_learned_gain_model(learned_path)
    enc = learned_cfg.get("input_encoding", {})
    controller_names = tuple(enc.get("controller_names",
                                     ("random", "lowamp", "hold")))
    sigma_field_order = tuple(enc.get("sigma_field_order",
                                      ("qpos", "qvel", "tip_pos", "reach_err")))
    delay_max = int(enc.get("delay_max", 36))

    env = make_env(cfg["env_id"], horizon=int(cfg["horizon"]))
    action_dim = detect_action_dim(env)
    adapter = ActionAdapter(action_dim=action_dim)

    noise_sigma = cfg["noise_conditions"]["none"]
    delay = 18
    n_eps = int(cfg.get("episodes_per_cell", 10))

    estimators_specs = [
        ("K=0.0", "fixed", 0.0),
        ("K=1.0", "fixed", 1.0),
        ("learned", "learned", None),
    ]
    # series[name] shape (n_eps, T)
    all_series: dict[str, list[np.ndarray]] = {n: [] for n, _, _ in estimators_specs}

    try:
        for ep_idx in range(n_eps):
            target = target_set.target_pos[ep_idx]
            # Pre-solve IK + moment arm at the env's reset pose for this target.
            env.reset()
            target_sid = int(env.unwrapped.target_sids[0])
            env.unwrapped.mj_model.site_pos[target_sid] = target.astype(np.float64)
            mujoco.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
            target_qpos, _ = solve_ik(env, target, max_iter=200, tol=0.01)
            M = actuator_moment_dense(env)

            for name, kind, gain in estimators_specs:
                ctrl = JointSpacePDController(
                    action_dim=action_dim,
                    target_qpos=target_qpos,
                    moment_arm=M,
                    Kp=30.0, Kd=3.0, action_scale=5.0,
                )
                ctrl.reset()
                if kind == "fixed":
                    est = FixedGainKalmanEstimator(
                        forward_model=forward_model, gain=gain,
                        state_spec=state_spec, delay_steps=delay,
                    )
                else:
                    est = LearnedGainKalmanEstimator(
                        forward_model=forward_model,
                        gain_predictor=predictor, state_spec=state_spec,
                        delay_steps=delay, controller_name="random",
                        noise_sigma=noise_sigma,
                        controller_names=controller_names,
                        sigma_field_order=sigma_field_order, delay_max=delay_max,
                    )
                obs_delay = DelayedObservationWrapper(spec=state_spec,
                                                       delay_steps=delay)
                spec_obj = EpisodeSpec(
                    episode_id=ep_idx,
                    target_id=str(int(target_set.seeds[ep_idx])),
                    target_split=target_set.split,
                    target_seed=int(target_set.seeds[ep_idx]),
                    controller_name="joint_pd", controller_seed=ep_idx,
                )
                result = run_closed_loop_episode(
                    env, ctrl, est, target,
                    state_spec=state_spec, action_adapter=adapter,
                    sdn=None, obs_noise=None, obs_delay=obs_delay,
                    obs_compose="noisy_then_delayed",
                    max_steps=int(cfg["horizon"]), spec=spec_obj,
                )
                tip_err = np.linalg.norm(result.log.true_reach_err, axis=1)
                all_series[name].append(tip_err)
            print(f"  ep {ep_idx}: K=0 fin {all_series['K=0.0'][-1][-1]:.3f}, "
                  f"K=1 fin {all_series['K=1.0'][-1][-1]:.3f}, "
                  f"learned fin {all_series['learned'][-1][-1]:.3f}")
    finally:
        env.close()

    # Episodes may terminate early — pad each trace with NaN to the max length.
    max_T = max(arr.shape[0] for v in all_series.values() for arr in v)

    def _pad(arr: np.ndarray) -> np.ndarray:
        out = np.full(max_T, np.nan, dtype=np.float64)
        out[: arr.shape[0]] = arr
        return out

    series_arr = {
        n: np.stack([_pad(arr) for arr in v]) for n, v in all_series.items()
    }
    T = max_T
    t_axis = np.arange(T) * 0.02

    # Save raw series (nanmean / nanstd to keep CSV small).
    table = {"time_s": t_axis}
    for name, arr in series_arr.items():
        table[f"{name}_mean"] = np.nanmean(arr, axis=0)
        table[f"{name}_std"] = np.nanstd(arr, axis=0)
    pd.DataFrame(table).to_csv(
        DATA_DIR / "F5_phase2d_traj_noise-none_d-18.csv", index=False,
    )

    # Plot mean +/- std across n_eps episodes.
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    for name in ("K=0.0", "K=1.0", "learned"):
        arr = series_arr[name]
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        ax.fill_between(t_axis, mu - sd, mu + sd, color=COLORS[name], alpha=0.18)
        ax.plot(t_axis, mu, label=f"{name}", color=COLORS[name], linewidth=1.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Tip-to-target error (m, mean ± std over 10 eps)")
    ax.set_title(f"Phase 2 D trajectories: noise=none, delay=18 steps "
                 f"(n={n_eps})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False)
    _save(fig, "F5_phase2d_trajectory")


# ---------------- F7: Phase B closed-loop paradigm shift ----------------


def fig_F7_paradigm_shift() -> None:
    """F7: 2-panel bar chart per cell (3 estimators × min_tip) for OLD vs NEW.

    Highlights the closed-loop paradigm shift: with single-step forward
    model learned ≈ K=1 and K=0 diverges; with K=4 multi-step supervision
    K=0 becomes the best estimator across all 6 cells.
    """
    cells = [(n, d) for n in ("none", "high", "xhigh") for d in (0, 18)]
    cell_labels = [f"{n}\nd={d}" for n, d in cells]

    def _by_cell(path: str) -> dict[tuple[str, int], dict[str, tuple[float, float]]]:
        df = pd.read_csv(path)
        df["delay_steps"] = df["delay_steps"].astype(int)
        df["min_tip_error"] = df["min_tip_error"].astype(float)
        df["estimator_norm"] = df["estimator"].apply(
            lambda s: "learned" if s.startswith("learned") else s
        )
        out: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
        for (est, n, d), grp in df.groupby(
            ["estimator_norm", "noise_condition", "delay_steps"]
        ):
            vals = grp["min_tip_error"].to_numpy()
            out.setdefault((n, d), {})[est] = (float(vals.mean()),
                                                float(vals.std()))
        return out

    old = _by_cell(D_MVP)
    k4 = _by_cell(D_MVP_PHASE_B)
    k8 = _by_cell(D_MVP_PHASE_BPRIME)

    # Save tidy summary.
    rows = []
    for (n, d) in cells:
        for source, data in (("OLD_single-step", old),
                             ("K4_multi-step", k4),
                             ("K8_multi-step", k8)):
            for est in ("K=0.0", "K=1.0", "learned"):
                mu, sd = data[(n, d)].get(est, (float("nan"), float("nan")))
                rows.append({"forward_model": source, "noise": n,
                             "delay": d, "estimator": est,
                             "min_tip_mean": mu, "min_tip_std": sd})
    pd.DataFrame(rows).to_csv(
        DATA_DIR / "F7_paradigm_shift_min_tip.csv", index=False,
    )

    estimator_order = ("K=0.0", "K=1.0", "learned")
    palette = {"K=0.0": COLORS["K=0.0"], "K=1.0": COLORS["K=1.0"],
               "learned": COLORS["learned"]}

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 7.4), sharex=True, sharey=True)
    width = 0.26
    x = np.arange(len(cells))
    for ax, data, title in (
        (axes[0], old, "Single-step forward model (Phase 3.1 / 3.3-min)"),
        (axes[1], k4, "K=4 multi-step forward model (Phase B)"),
        (axes[2], k8, "K=8 multi-step forward model (Phase B')"),
    ):
        for j, est in enumerate(estimator_order):
            means = [data[c].get(est, (np.nan, np.nan))[0] for c in cells]
            stds = [data[c].get(est, (np.nan, np.nan))[1] for c in cells]
            ax.bar(x + (j - 1) * width, means, width, yerr=stds,
                   capsize=2, label=est, color=palette[est],
                   edgecolor="black", linewidth=0.4)
        ax.set_ylabel("min tip-to-target error (m)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, 1.0)
        ax.axhline(0.05, color="red", linestyle="--", linewidth=0.6,
                   alpha=0.4, label="_success_thr_5cm")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(cell_labels)
    axes[-1].set_xlabel("noise condition × delay")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Closed-loop reaching: forward-model supervision changes "
                 "the optimal estimator",
                 fontsize=10, y=1.0)
    _save(fig, "F7_paradigm_shift")


# ---------------- F6: Phase 4 BC trade-off scatter ----------------


def fig_F6_tradeoff() -> None:
    sources = {
        "Heuristic (E)": E_MVP,
        "Joint PD + IK (D)": D_MVP,
        "BC full": BC_FULL,
        "BC v1 (-target)": BC_V1,
        "BC v2 (-target, -reach_err)": BC_V2,
    }
    rows = []
    for name, path in sources.items():
        df = pd.read_csv(path)
        df["delay_steps"] = df["delay_steps"].astype(int)
        df["min_tip_error"] = df["min_tip_error"].astype(float)
        df["final_tip_error"] = df["final_tip_error"].astype(float)
        df["success_010_b"] = df["success_010"].map(_fbool)

        g = df.groupby(["estimator", "noise_condition", "delay_steps"])
        # K=1.0 and learned baseline per cell
        k1 = g["min_tip_error"].mean().xs("K=1.0", level="estimator")
        learned = g["min_tip_error"].mean().xs("learned_stage_a_retrained",
                                                level="estimator")
        delta = (learned - k1).abs().max()  # worst-case |Δ| over 6 cells
        delta_final = (
            g["final_tip_error"].mean().xs("learned_stage_a_retrained",
                                            level="estimator")
            - g["final_tip_error"].mean().xs("K=1.0", level="estimator")
        ).abs().max()
        # Reaching success: average over K=1.0 estimator cells of S010.
        success = (
            df[df["estimator"] == "K=1.0"]
            .groupby(["noise_condition", "delay_steps"])["success_010_b"]
            .mean()
        ).mean()
        rows.append({
            "controller": name,
            "success_010_K1": success,
            "max_abs_delta_min": delta,
            "max_abs_delta_final": delta_final,
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(DATA_DIR / "F6_tradeoff.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    markers = {"Heuristic (E)": "o", "Joint PD + IK (D)": "s", "BC full": "D",
               "BC v1 (-target)": "v", "BC v2 (-target, -reach_err)": "^"}
    palette = {"Heuristic (E)": "#888888", "Joint PD + IK (D)": "#33a02c",
               "BC full": "#1f78b4", "BC v1 (-target)": "#a6cee3",
               "BC v2 (-target, -reach_err)": "#6a3d9a"}
    for r in rows:
        ax.scatter(r["success_010_K1"], r["max_abs_delta_min"],
                   s=160, marker=markers[r["controller"]],
                   color=palette[r["controller"]], edgecolor="black",
                   linewidth=0.8, zorder=3, label=r["controller"])
    # Annotate D's outlying point.
    for r in rows:
        if r["controller"] == "Joint PD + IK (D)":
            ax.annotate(f"+0.086 m at\n(none, d=18)",
                        xy=(r["success_010_K1"], r["max_abs_delta_min"]),
                        xytext=(0.13, 0.07),
                        fontsize=8,
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set_xlabel("Reaching success_010 rate (K=1.0 baseline, avg over 6 cells)")
    ax.set_ylabel("max |Δ(learned − K=1)| min_tip (m)")
    ax.set_title("Controller trade-off: reaching vs. estimator differentiation")
    ax.set_yscale("symlog", linthresh=0.001)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    _save(fig, "F6_controller_tradeoff")


# ---------------- main ----------------


def main() -> None:
    _setup_style()
    print("F1 system overview ...")
    fig_F1_system_overview()
    print("F2 stress oracle K heatmap ...")
    fig_F2_stress_oracle()
    print("F3 stress eval 7-strategy comparison ...")
    fig_F3_stress_eval()
    print("F4 Phase 2 D delta heatmap ...")
    fig_F4_phase2_d_delta()
    print("F5 Phase 2 D trajectory ...")
    fig_F5_phase2_d_trajectories()
    print("F6 controller trade-off scatter ...")
    fig_F6_tradeoff()
    print("F7 Phase B paradigm shift ...")
    fig_F7_paradigm_shift()
    print("\nAll figures saved to figures/ and tidied CSVs to figures/data/")


if __name__ == "__main__":
    main()
