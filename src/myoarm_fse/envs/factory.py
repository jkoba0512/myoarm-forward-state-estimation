"""MyoSuite myoArm env factory.

Builds a Gymnasium env for myoArm reaching with episode horizon and
``normalize_act`` pinned. This is the only entry point in the package that
imports MyoSuite at module load time (registers env ids as a side effect).

Horizon override caveat
-----------------------
``env.unwrapped.horizon`` is a read-only property that resolves to
``gym.spec(env_id).max_episode_steps`` at access time. Setting only the
``max_episode_steps`` kwarg of ``gym.make`` updates ``env.spec`` but does
NOT update the global registry, so ``unwrapped.horizon`` would still report
the original (often 150) value and MyoSuite's internal truncation logic
would fire early. To make both views consistent, this factory updates the
global ``gym.spec(env_id).max_episode_steps`` *before* calling
``gym.make``. This is a process-global side effect that persists for the
lifetime of the interpreter.
"""

from __future__ import annotations

import gymnasium as gym
import myosuite  # noqa: F401  -- side-effect import: registers myo* env ids


def make_env(
    env_id: str,
    horizon: int = 600,
    normalize_act: bool = True,
) -> gym.Env:
    """Construct a myoArm Gym env with horizon and ``normalize_act`` pinned.

    Parameters
    ----------
    env_id : str
        Gym registry id, e.g. ``"myoArmReachFixed-v0"``.
    horizon : int
        Episode length in control steps (``dt = 0.02 s`` for myoArm, so
        ``horizon=600`` corresponds to a 12 s episode). Must be a positive
        ``int``; ``bool`` is rejected. Pinned on both the global registry
        spec and the returned env's ``spec.max_episode_steps``; the
        consistency between ``env.spec.max_episode_steps`` and
        ``env.unwrapped.horizon`` is asserted before return.
    normalize_act : bool
        Forwarded to MyoSuite. ``True`` (default) maps actuator inputs to
        ``[-1, 1]`` so ``ActionAdapter`` works without further conversion.

    Returns
    -------
    gym.Env
        The constructed env. The caller is responsible for ``env.close()``.
    """
    if isinstance(horizon, bool):
        raise ValueError(f"horizon must be a positive int, got bool: {horizon!r}")
    if not isinstance(horizon, int):
        raise ValueError(
            "horizon must be a positive int, "
            f"got {type(horizon).__name__}: {horizon!r}"
        )
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if not isinstance(normalize_act, bool):
        raise TypeError(
            f"normalize_act must be bool, got {type(normalize_act).__name__}"
        )

    # Override the registry entry FIRST so that env.unwrapped.horizon
    # (which reads from the registry) reports the requested value.
    gym.spec(env_id).max_episode_steps = horizon

    env = gym.make(
        env_id,
        max_episode_steps=horizon,
        normalize_act=normalize_act,
    )

    # Defensive: confirm the two horizon views are consistent. If this
    # assertion fails, MyoSuite's internal truncation would fire at the
    # registry's value rather than the env.spec value, which is the bug
    # the old project hit.
    if env.spec.max_episode_steps != env.unwrapped.horizon:
        env.close()
        raise RuntimeError(
            f"horizon inconsistency: env.spec.max_episode_steps="
            f"{env.spec.max_episode_steps}, env.unwrapped.horizon="
            f"{env.unwrapped.horizon}; expected both to equal {horizon}"
        )
    return env
