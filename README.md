# myoarm-forward-state-estimation

Reproduction code for forward-model-based predictive state observation in
the MyoSuite myoArm reaching task under sensory delay and observation noise.
The observer combines learned forward prediction with sensory
prediction-error correction and can adapt field-wise correction gains from
innovation history and closed-loop reaching outcomes.

This repository accompanies:

> **How a Predictive State Observer Can Self-Adapt Its Sensory
> Prediction-Error Correction Gain: Closed-Loop Evidence from a
> Muscle-Driven Reaching Task**
> Jun Kobayashi, Kyushu Institute of Technology.

The manuscript was submitted to bioRxiv on June 3, 2026
(`BIORXIV/2026/729790`; DOI pending) and is prepared for submission to
*Biological Cybernetics*. The compiled 19-page manuscript is available at
[`paper/main.pdf`](paper/main.pdf).

## Main findings

- The best fixed correction gain depends on delay: intermediate gains
  (`K = 0.25-0.50`) are best without delay, while `K = 1.0` is best at an
  18-step delay.
- Prediction-only deployment (`K = 0`) is a diagnostic failure mode rather
  than the oracle. It is 1.9-6.1 cm worse than the best fixed gain and
  exhibits large controller residuals caused by autoregressive model drift.
- Outcome-trained reliability adaptation improves delayed reaches by
  1.9-2.5 cm over the default reliability rule while remaining neutral in
  no-delay cells.
- A feature-conditioned adapter nearly matches a per-cell trained diagnostic
  in five of six cells, but both remain 1.4-1.8 cm behind the fixed-gain
  oracle at the 18-step delay.

## Quick start

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --locked
uv run python -c "import myoarm_fse; print(myoarm_fse.__version__)"
uv run python scripts/evaluate_closed_loop.py \
  --config configs/closed_loop/smoke_stabilized_endpoint.yaml \
  --smoke
```

Run the default test suite:

```bash
uv run pytest -q
```

Tests marked `myosuite` require a working MyoSuite and MuJoCo runtime and are
excluded by default.

## R3 reproduction path

The current paper uses the IK-filtered reachable target set, an H=8 residual
MLP forward model, and the stabilized endpoint probe controller.

```bash
# 1. Generate the training and below-shoulder evaluation target sets.
uv run python scripts/generate_reachable_targets.py \
  --env-id myoArmReachRandom-v0 --split reachable_train \
  --n 200 --threshold 0.01 --generator-seed 0 --seed-offset 30000
uv run python scripts/generate_reachable_targets.py \
  --env-id myoArmReachRandom-v0 --split reachable_train \
  --n 200 --threshold 0.01 --generator-seed 0 --seed-offset 30000 \
  --max-target-z 1.393

# Update the target_set paths in the configs to the generated locations,
# then collect the three 50-episode transition batches.
for config in reachable_default reachable_lowamp_random reachable_hold; do
  uv run python scripts/collect_episodes.py \
    --config "configs/episodes/${config}.yaml"
done

# 2. Assemble the dataset and train the H=8 forward model.
uv run python scripts/build_dataset.py \
  --episodes runs/episodes/<default-run> \
             runs/episodes/<lowamp-run> \
             runs/episodes/<hold-run> \
  --output runs/datasets/expanded_reachable.npz
uv run python scripts/train_forward_model.py \
  --config configs/models/mlp_reachable_h8.yaml

# 3. Evaluate the fixed-gain closed-loop oracle.
uv run python scripts/evaluate_closed_loop.py \
  --config configs/closed_loop/oracle_k_sweep_r3_v2.yaml

# 4. Train and deploy outcome-adaptive observers.
uv run python scripts/train_reliability_adaptive_v2.py \
  --config configs/train/reliability_adaptive_v2_poc_r3_v2.yaml
uv run python scripts/train_reliability_adaptive_v2.py \
  --config configs/train/reliability_adaptive_v2_fullgrid_r3_v2.yaml

# 5. Regenerate paper figures from the frozen run identifiers.
uv run python scripts/make_r3_paper_figures.py
```

The exact run identifiers used for each claim and figure are frozen in
Appendix C of [`paper/main.tex`](paper/main.tex). Local experimental outputs
under `runs/` are intentionally gitignored; the scripts and configurations
regenerate them.

## Repository layout

```text
src/myoarm_fse/    Environment, models, observers, controllers, and metrics
scripts/           Dataset, training, evaluation, diagnostics, and figures
configs/           YAML configurations for every pipeline stage
tests/             Unit and integration tests
figures/           Current paper figures and selected historical figures
paper/             Biological Cybernetics manuscript source and PDF
docs/              Design, implementation, and release documentation
runs/              Local generated outputs; not committed
```

## Reproducibility notes

- The controller receives only the estimated state. True state is retained
  for evaluation and is never passed into the closed-loop policy.
- The state is the 83-dimensional vector
  `[q, qdot, activation, tip_position, target_position, reach_error]`.
- Muscle excitation in `[0, 1]`, Gym API action in `[-1, 1]`, MuJoCo
  control, and muscle activation are distinct signals throughout the code.
- Random child seeds are derived from a fixed master seed using
  `numpy.random.SeedSequence`.

## Citation and archival

`CITATION.cff` contains machine-readable citation metadata. A versioned
Zenodo DOI will be added after the first R3 GitHub Release is archived.
The bioRxiv DOI identifies the manuscript; the Zenodo DOI identifies the
software snapshot.

## License

Software, configurations, and original project documentation are licensed
under the [Apache License 2.0](LICENSE).

The manuscript in `paper/main.tex` and `paper/main.pdf` is a separate work
and follows the license stated on its bioRxiv record (CC BY-NC-ND 4.0 for
the submitted preprint). Bundled Springer Nature template files and
attributed third-party images retain their upstream terms.

## Contact

Jun Kobayashi, Kyushu Institute of Technology
`kobayashi.jun184@m.kyutech.ac.jp`
