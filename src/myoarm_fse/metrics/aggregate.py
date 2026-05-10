"""Run-level aggregation of reaching metrics across episodes."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from myoarm_fse.data.schema import EpisodeLog
from myoarm_fse.metrics.reaching import (
    effort_norm,
    final_tip_error,
    minimum_tip_error,
    success,
)


def aggregate_reaching(
    logs: Iterable[EpisodeLog],
    *,
    threshold: float = 0.05,
    duration: int = 10,
) -> dict[str, float]:
    """Per-run summary of reaching metrics across many episodes.

    Returns a JSON-serializable ``dict[str, float]`` so the result drops
    straight into ``index.json`` siblings or CLI ``json.dumps`` output.
    Empty input returns ``{"n": 0}``.

    ``threshold`` and ``duration`` are forwarded to :func:`success`.
    """
    log_list = list(logs)
    if not log_list:
        return {"n": 0}

    min_te = np.array([minimum_tip_error(l) for l in log_list], dtype=np.float64)
    final_te = np.array([final_tip_error(l) for l in log_list], dtype=np.float64)
    successes = np.array(
        [success(l, threshold=threshold, duration=duration) for l in log_list],
        dtype=np.float64,
    )
    efforts = np.array([effort_norm(l) for l in log_list], dtype=np.float64)

    return {
        "n": len(log_list),
        "minimum_tip_error_mean": float(min_te.mean()),
        "minimum_tip_error_median": float(np.median(min_te)),
        "minimum_tip_error_std": float(min_te.std()),
        "final_tip_error_mean": float(final_te.mean()),
        "final_tip_error_median": float(np.median(final_te)),
        "final_tip_error_std": float(final_te.std()),
        "success_rate": float(successes.mean()),
        "effort_mean": float(efforts.mean()),
        "effort_std": float(efforts.std()),
        "threshold": float(threshold),
        "duration": int(duration),
    }
