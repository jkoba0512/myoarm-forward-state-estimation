"""Reaching metrics computed from a single ``EpisodeLog``.

All metrics here read the ``true_*`` (oracle) fields. ``obs_*`` is
intentionally not consulted — observation noise / delay handicap the
controller, but the evaluator judges against ground truth (see Step 7
design decisions Q10).

``effort_norm`` reads ``excitation`` (post-SDN, the canonical
research-side input the muscles actually receive); see Q4 for why
``excitation_command`` and ``last_ctrl`` were rejected.
"""

from __future__ import annotations

import numpy as np

from myoarm_fse.data.schema import EpisodeLog


def minimum_tip_error(log: EpisodeLog) -> float:
    """Smallest ``||tip - target||`` reached anywhere in the episode.

    Returns ``inf`` for an empty episode (no recorded steps).
    """
    if log.n_steps == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(log.true_reach_err, axis=1)))


def final_tip_error(log: EpisodeLog) -> float:
    """``||tip - target||`` at the final recorded step.

    Returns ``inf`` for an empty episode.
    """
    if log.n_steps == 0:
        return float("inf")
    return float(np.linalg.norm(log.true_reach_err[-1]))


def success(
    log: EpisodeLog,
    *,
    threshold: float = 0.05,
    duration: int = 10,
) -> bool:
    """``True`` iff the tip stays within ``threshold`` for ≥ ``duration`` steps.

    Sustained-window definition: there exists a contiguous window of
    length ``duration`` such that ``||true_reach_err|| < threshold`` at
    every step in the window.

    Defaults: ``threshold=0.05 m`` (5 cm), ``duration=10 step`` (≈ 0.2 s
    at the myoArm control rate of ``dt=0.02 s``).
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating)):
        raise ValueError(
            f"threshold must be numeric, got {type(threshold).__name__}"
        )
    threshold_f = float(threshold)
    if not np.isfinite(threshold_f) or threshold_f < 0.0:
        raise ValueError(f"threshold must be finite and >= 0, got {threshold_f}")
    if isinstance(duration, bool) or not isinstance(duration, int):
        raise ValueError(
            f"duration must be a positive int, got {type(duration).__name__}"
        )
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration}")

    if log.n_steps < duration:
        return False
    err_norms = np.linalg.norm(log.true_reach_err, axis=1)
    within = err_norms < threshold_f
    # Sliding-window all-True via cumulative sum.
    cs = np.concatenate([[0], np.cumsum(within.astype(np.int64))])
    counts = cs[duration:] - cs[:-duration]
    return bool((counts == duration).any())


def effort_norm(log: EpisodeLog) -> float:
    """Mean-of-squares L2 effort over the episode (post-SDN excitation)::

        (1/T) * Σ_t ||excitation_t||₂²

    Returns ``0.0`` for an empty episode (no muscle activation occurred).
    """
    if log.n_steps == 0:
        return 0.0
    return float(np.mean(np.sum(log.excitation ** 2, axis=1)))
