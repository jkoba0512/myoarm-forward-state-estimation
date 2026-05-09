# myoarm-forward-state-estimation

Project 1 for the new myoArm research line: forward dynamics prediction and Kalman-like state estimation for delayed/noisy MyoSuite myoArm reaching.

## Aim

Test whether forward prediction and Kalman-like state estimation improve myoArm reaching under sensory delay, observation noise, and signal-dependent motor noise.

## Initial Scope

1. Build a reproducible myoArm episode logger.
2. Define fixed train/validation/test target sets.
3. Separate `neural_command`, `excitation`, `api_action`, and `activation`.
4. Add explicit signal-dependent motor noise on `neural_command` / `excitation`.
5. Add delayed/noisy observation wrappers.
6. Train forward models for `x_t, u_t -> x_{t+1}`.
7. Compare delayed-feedback-only, prediction-only, fixed-gain update, and adaptive/learned Kalman-like estimators.

## Layout

```text
docs/      Research plans and primers.
src/       Python package source.
scripts/   Executable experiment scripts.
configs/   Experiment configs.
runs/      Local outputs; keep large results out of git when versioning.
tests/     Unit and smoke tests.
```

## Key Docs

- `docs/00_全体研究計画.md`
- `docs/01_Project1_ForwardStateEstimation研究計画.md`
- `docs/myoArm_Reaching_Primer.md`
- `docs/myoArm_MyoSuite_horizonとmax_steps混同メモ_2026-05-09.md`

## Development

Use `uv`.

```bash
uv sync
uv run python -c "import myoarm_fse; print(myoarm_fse.__version__)"
```
