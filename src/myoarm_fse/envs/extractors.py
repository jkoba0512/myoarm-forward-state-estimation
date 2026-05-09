"""Extract ``MyoArmState`` and post-step ctrl from a MyoSuite myoArm env.

Sources (verified against MyoSuite ``ReachEnvV0`` for ``myoArmReachFixed-v0``):

- ``qpos`` : ``env.unwrapped.mj_data.qpos`` (raw joint positions, shape ``(20,)``)
- ``qvel`` : ``env.unwrapped.mj_data.qvel`` (raw joint velocities, shape
  ``(20,)``).

  Note: ``env.unwrapped.obs_dict['qvel']`` returns ``mj_data.qvel * dt`` (a
  per-step displacement), NOT the raw velocity. This extractor reads from
  ``mj_data.qvel`` directly to keep the schema's ``qvel`` semantics
  consistent with standard MuJoCo conventions.
- ``act`` : ``env.unwrapped.mj_data.act`` (muscle activation state, shape
  ``(34,)``)
- ``tip_pos`` : ``env.unwrapped.mj_data.site_xpos[tip_sids[0]]`` — the
  ``IFtip`` site's Cartesian position, shape ``(3,)``
- ``target_pos`` : ``env.unwrapped.mj_data.site_xpos[target_sids[0]]`` —
  shape ``(3,)``
- ``reach_err`` : computed as ``tip_pos - target_pos`` (this project's
  convention).

  Note: MyoSuite's ``obs_dict['reach_err']`` is ``target_pos - tip_pos``
  (opposite sign). We do not pass that field through.

ctrl source:

- ``mj_data.ctrl`` is observed to be reset to zero between control steps
  in MyoSuite, so it does not reflect the muscle excitation that was
  actually applied.
- ``env.unwrapped.last_ctrl`` is the post-sigmoid muscle ctrl returned by
  ``robot.step(...)``. With ``normalize_act=True``, MyoSuite maps the
  ``[-1, 1]`` ``api_action`` through a steep sigmoid before passing it to
  the muscle dynamics; ``last_ctrl`` is that post-sigmoid value. Use this
  as the canonical "MuJoCo actuator control" for logging.

Both helpers expect the env produced by ``factory.make_env(...)``. Only a
single tip site and single target site are supported (myoArm reach has
exactly one of each); a clear error is raised otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from myoarm_fse.envs.state import MyoArmState

_DTYPE: np.dtype = np.dtype(np.float32)


def _site_xpos(env: Any, sids: list[int], name: str) -> np.ndarray:
    if len(sids) != 1:
        raise ValueError(
            f"expected exactly one {name} site, got sids={sids!r}; "
            "multi-site envs are not yet supported"
        )
    return env.unwrapped.mj_data.site_xpos[sids[0]].copy()


def extract_state(env: Any) -> MyoArmState:
    """Build a ``MyoArmState`` from the env's current internal state.

    Reads from ``env.unwrapped.mj_data`` and the env's site index lists.
    Does not consume the obs / info returned by ``reset()`` or ``step()``;
    callers may discard those. Both reset-time and post-step states are
    handled identically.
    """
    uw = env.unwrapped
    tip_pos = _site_xpos(env, uw.tip_sids, "tip")
    target_pos = _site_xpos(env, uw.target_sids, "target")
    return MyoArmState.from_arrays(
        qpos=uw.mj_data.qpos.copy(),
        qvel=uw.mj_data.qvel.copy(),
        act=uw.mj_data.act.copy(),
        tip_pos=tip_pos,
        target_pos=target_pos,
        reach_err=tip_pos - target_pos,
    )


def extract_ctrl(env: Any) -> np.ndarray:
    """Return the post-step muscle ctrl actually applied by MyoSuite.

    Returns a fresh ``np.float32`` copy of ``env.unwrapped.last_ctrl``
    (MuJoCo's data buffers are reused across steps; copying avoids
    aliasing into them).
    """
    return np.asarray(env.unwrapped.last_ctrl, dtype=_DTYPE).copy()
