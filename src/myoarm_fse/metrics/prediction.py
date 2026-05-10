"""Forward-model prediction metrics (pure ``np.ndarray`` interface).

Computed from predicted vs ground-truth trajectories — both supplied by
the caller (Step 8 forward-model evaluation will produce the ``pred_*``
arrays). These functions do not consult ``EpisodeLog``; they live at a
lower layer so that dataset loaders and training pipelines can call
them without a logger dependency.

Shape conventions
-----------------

```text
true_next, pred_next : (T, state_dim)   one-step prediction targets
true_traj, pred_traj : (T, state_dim)   rollout prediction targets
true_tip,  pred_tip  : (T, 3)           tip-only target
```

All inputs must be finite; non-finite values raise ``ValueError`` so
silent NaN propagation through MSE is avoided.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _validate_pair(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    name_a: str,
    name_b: str,
) -> tuple[np.ndarray, np.ndarray]:
    arr_a = np.asarray(a)
    arr_b = np.asarray(b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"{name_a}.shape={arr_a.shape} does not match "
            f"{name_b}.shape={arr_b.shape}"
        )
    if not np.isfinite(arr_a).all():
        raise ValueError(f"{name_a} contains non-finite values (NaN or Inf)")
    if not np.isfinite(arr_b).all():
        raise ValueError(f"{name_b} contains non-finite values (NaN or Inf)")
    return arr_a, arr_b


def one_step_prediction_mse(
    true_next: npt.ArrayLike,
    pred_next: npt.ArrayLike,
) -> float:
    """Mean squared error between one-step predictions and ground truth.

    Both arrays must have shape ``(T, state_dim)`` and identical shapes.
    Returns the scalar MSE averaged over both axes.
    """
    a, b = _validate_pair(true_next, pred_next, "true_next", "pred_next")
    if a.ndim != 2:
        raise ValueError(
            f"true_next/pred_next must be 2-D (T, state_dim), got ndim={a.ndim}"
        )
    if a.size == 0:
        return 0.0
    return float(np.mean((a - b) ** 2))


def rollout_mse(
    true_traj: npt.ArrayLike,
    pred_traj: npt.ArrayLike,
) -> float:
    """MSE over a multi-step rollout. Same shape conventions as one-step."""
    a, b = _validate_pair(true_traj, pred_traj, "true_traj", "pred_traj")
    if a.ndim != 2:
        raise ValueError(
            f"true_traj/pred_traj must be 2-D (T, state_dim), got ndim={a.ndim}"
        )
    if a.size == 0:
        return 0.0
    return float(np.mean((a - b) ** 2))


def tip_prediction_error(
    true_tip: npt.ArrayLike,
    pred_tip: npt.ArrayLike,
) -> float:
    """Mean Euclidean ``||true_tip - pred_tip||`` over time.

    Both arrays must have shape ``(T, 3)``. Reports a distance (not MSE)
    because tip error is most readable in the same units as
    ``final_tip_error``.
    """
    a, b = _validate_pair(true_tip, pred_tip, "true_tip", "pred_tip")
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(
            f"true_tip/pred_tip must be (T, 3), got shape {a.shape}"
        )
    if a.shape[0] == 0:
        return 0.0
    return float(np.mean(np.linalg.norm(a - b, axis=1)))
