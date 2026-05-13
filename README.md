# myoarm-forward-state-estimation

Forward-model prediction and Kalman-like state estimation for the MyoSuite
myoArm reaching task under sensorimotor delay, observation noise, and
signal-dependent motor noise.

This repository accompanies the manuscript

> **When Should a Kalman Filter Trust Its Forward Model?
> Closed-Loop Evidence from a Muscle-Driven Reaching Task**
> Jun Kobayashi, Kyushu Institute of Technology.

Compiled manuscript: [`paper/main.pdf`](paper/main.pdf) (11 pages, IEEE TNNLS draft).

## What the paper claims

A learned condition-level Kalman gain predictor for the myoArm reaching
task behaves very differently depending on which *oracle* it is trained
against:

- **C1**: the open-loop oracle $K^{\star}_{\mathrm{ol}}$ (minimises
  estimation error $\|\hat x - x\|^2$) collapses to $K{=}1$ at long
  delay because forward-model rollout error compounds faster than
  observation noise.
- **C2**: at a single (low-noise, large-delay) cell the predictor
  trained on $K^{\star}_{\mathrm{ol}}$ outputs $K{\approx}0.99$ and
  edges past $K{=}1$ by 8.6 cm in final-tip error (the per-cell paired
  $t$-test gives $p{=}0.21$; we report this as a marginal cell-level
  effect, not a robust improvement).
- **C3**: when the forward model is retrained with $H$-step rollout
  supervision ($H{\in}\{4,8\}$), the closed-loop optimum flips to
  prediction-only ($K{=}0$), which beats $K{=}1$ by 12–24 cm on the
  focused joint-PD grid under the min-tip objective. The predictor
  still outputs $K{\approx}1$ because the open-loop oracle never picks
  $K{=}0$.
- **C4**: replacing the predictor's training oracle with the
  closed-loop oracle $K^{\star}_{\mathrm{cl}}$ (per-cell argmin of
  closed-loop task error) and retraining the same MLP unchanged moves
  the output to $K{\approx}0.02$ and recovers 40–60\,% of the gap;
  Appendix E demonstrates the transfer to a variable-target variant.

The take-away: **what you train the gain predictor against matters at
least as much as the predictor architecture**. Learned-Kalman-gain
papers should specify the oracle target explicitly and report
closed-loop, not just open-loop, task error.

## Quick start

```bash
# 1. Install (uv is required; tested with Python 3.11)
uv sync

# 2. Verify the package imports
uv run python -c "import myoarm_fse; print(myoarm_fse.__version__)"

# 3. Run a single-cell smoke closed-loop episode
uv run python scripts/evaluate_closed_loop.py \
    --config configs/closed_loop/joint_pd_mvp.yaml --smoke
```

## Reproduce the headline numbers

The full pipeline is config-driven. To reproduce the ReachFixed C3/C4
result (the central finding in §4.5–4.6 of the paper):

```bash
# 1. Collect transition data (≈10 min)
uv run python scripts/generate_targets.py --config configs/targets/default.yaml
for ctrl in default lowamp_random hold; do
    uv run python scripts/collect_episodes.py \
        --config configs/episodes/$ctrl.yaml
done
# Concatenate into runs/datasets/expanded.npz
# (see configs/models/mlp_expanded.yaml comment for the build step)

# 2. Train the H=8 multi-step forward model (≈15 min CPU)
uv run python scripts/train_forward_model.py \
    --config configs/models/mlp_expanded_multistep8.yaml

# 3. Open-loop stress eval on the H=8 model (≈30 min)
uv run python scripts/evaluate_estimator.py \
    --config configs/estimators/fixed_kalman_stress.yaml \
    --forward-model runs/models/<H8_id>

# 4. Stage A predictor trained on the open-loop oracle (≈5 min)
uv run python scripts/train_learned_gain.py \
    --config configs/estimators/learned_gain_stress.yaml \
    --oracle-table runs/estimators/<stress_id>/best_by_condition.csv

# 5. Closed-loop K-sweep to obtain K*_cl per cell (≈30 min)
uv run python scripts/evaluate_closed_loop.py \
    --config configs/closed_loop/joint_pd_mvp_phaseBprime.yaml
uv run python scripts/evaluate_closed_loop.py \
    --config configs/closed_loop/closed_loop_oracle_sweep.yaml

# 6. Stage A predictor trained on the closed-loop oracle (≈5 min)
#    (build the K*_cl labels CSV from step 5 outputs, then:)
uv run python scripts/train_learned_gain.py \
    --config configs/estimators/learned_gain_stress.yaml \
    --oracle-table runs/estimators/<kcl_labels>.csv

# 7. Final closed-loop comparison (≈10 min)
uv run python scripts/evaluate_closed_loop.py \
    --config configs/closed_loop/joint_pd_mvp_phaseC.yaml
```

The ReachRandom transfer test (Appendix E) follows the same flow with
`configs/episodes/random_arm_*_smol.yaml` and
`configs/estimators/random_arm_stress_smol.yaml`.

## Paper figures and tables

```bash
# Regenerate F2-F7 and the tidied per-figure CSVs
uv run python scripts/make_paper_figures.py
```

Outputs land in `figures/` (PDF + PNG) and `figures/data/` (CSVs).

## Layout

```text
paper/             Manuscript + cover letter + submission checklist
  main.tex / .pdf  the 11-page TNNLS draft
  refs.bib         15 references, verified
  cover_letter.tex Cover letter for TNNLS submission
  SUBMISSION.md    Pre-submission checklist
docs/              Background docs (implementation plan, paper outline, primer)
src/myoarm_fse/    Python package
  envs/            Env factory, state schema, observation wrappers
  data/            Episode logger, rollout helpers
  models/          Forward-model architecture, training, dataset
  estimators/      Kalman filter, learned-gain predictor (Stage A / B)
  controllers/     Heuristic, joint-PD+IK, behaviour-cloning
  evaluation/      Open-loop and closed-loop evaluators
  metrics/         Reaching metrics
scripts/           CLI entry points (one per pipeline step)
configs/           YAML configs (episodes/, models/, estimators/, closed_loop/)
figures/           F1 (TikZ inside paper) + F2-F7 (PDF/PNG outputs)
tests/             Pytest suite (631 tests; `uv run pytest`)
runs/              Local outputs (gitignored)
```

## Key design decisions

These are documented in `docs/02_InitialImplementationPlan.md`:

- `api_action` (Gym/MyoSuite `[-1, 1]^n`) ≠ `neural_command` ≠ `excitation`
  (canonical `[0, 1]^n`) ≠ `activation` (muscle internal state) ≠ `ctrl`
  (MyoSuite internal final). Mixing these silently breaks everything.
- `true_state` is never passed to the controller in closed loop. Only
  estimator output flows downstream; the true state is logged for
  evaluation only.
- The fixed-lag Kalman estimator uses a length-$(d{+}1)$ ring buffer and
  re-rolls past estimates forward via the learned forward model — it is
  the standard buffered RTS smoother specialised to a scalar gain.
- The state vector is the 83-dim flat schema $[q, \dot q, a, p_{\mathrm{tip}},
  p_{\mathrm{tgt}}, e]$; the controllers see this state through the
  estimator only.

## Reproducibility

- Python 3.11, PyTorch (CPU), MuJoCo 2.3, MyoSuite 2.12.2,
  Gymnasium 0.29.
- All training is single-process on a 32-core workstation; no GPU is
  required. A full reproduction (episode collection + 3 forward
  models + 2 stress evals + Stage A × 2 + closed-loop eval) takes
  3–5 hours.
- Seeds: `seed: 0` is hard-coded in every config; per-episode child
  seeds are derived from `numpy.random.SeedSequence(seed_master).spawn`.

## Citation

```bibtex
@unpublished{kobayashi2026myoarmfse,
  title  = {When Should a Kalman Filter Trust Its Forward Model?
            Closed-Loop Evidence from a Muscle-Driven Reaching Task},
  author = {Kobayashi, Jun},
  year   = {2026},
  note   = {Manuscript under review at IEEE Transactions on Neural
            Networks and Learning Systems.},
}
```

## Contact

Jun Kobayashi — `kobayashi.jun184@m.kyutech.ac.jp`
Kyushu Institute of Technology, Iizuka, Fukuoka, Japan.
