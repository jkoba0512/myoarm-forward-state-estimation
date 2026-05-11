# Phase 1〜4 統合論文 — Outline と研究メッセージ

このドキュメントは Phase 1〜4 の実装結果から論文化に向けた中心メッセージと
構成を **固定** するためのものです。実装中の判断ノートとは別に、論文側の
不可逆な決定 (主張、結論、figure caption) を locked-in 状態で保持します。

## 中心メッセージ

> **Estimator quality propagates to closed-loop reaching performance, but only under specific conditions: a state-coupled controller (joint-space PD with IK precompute) AND an observation-delay regime (≥18 steps). Behaviour-cloned controllers at modest demonstration scale improve reaching but collapse to an effectively open-loop policy, washing out the estimator signal.**

研究問い (Project 1):
> myoArm reaching において、forward prediction と Kalman-like state estimator は、
> sensory delay・observation noise・signal-dependent motor noise 下の制御を改善するか。

答え:
- **改善する条件**: state-coupled controller + observation-delay regime
- **改善が消える条件**: behaviour-cloned policy が per-target generalize しないとき (small demo scale)

## 主結果 3 つ

### R1. Stress eval が blending regime を再現する

Phase 3.3-min (delay [0,2,6] × noise 4 段階 × K {0, 0.5, 0.75, 1.0}) では、improved forward
model 投入後に K-curve がほぼ flat。
Phase 3.2 stress eval (delay [0,6,18,36] × noise 6 段階 × K {0, 0.25, 0.5, 0.75, 1.0}) で、
**(low delay + high noise)** において oracle K が 1.0 → 0.25 まで段階的に低下する領域を再現。
特に (delay=0, xhigh) で oracle K=0.25、K=1 比で tip 推定誤差が 2× 改善。

### R2. condition-level supervised K predictor (Stage A retrained)

stress oracle (72 conditions、5 K) で学習した小さい MLP (8-dim 入力 → 32→32→1 + sigmoid)
が、open-loop estimation で:
- mean delta vs oracle = +0.00109 m
- max delta vs oracle = +0.00586 m

(delay=0, xhigh) で learned 0.0343 m vs K=1.0 0.0635 m vs oracle 0.0298 m。state-aware Stage B
は同等条件で更なる改善を出せず (negative result、controller-engineering implication)。

### R3. closed-loop で estimator quality が伝播する条件

5 つの controller variant で同じ 180-rollout grid を評価:

| controller | reaching success_010 | max\|Δ(L−K1)\|min |
|---|---|---|
| heuristic (random W) | 0% | <0.001 m |
| **joint-space PD + IK** | 0-10% | **0.086 m at (noise=none, delay=18)** |
| BC full (30 demos) | 12% | 0.018 m |
| BC v1 (-target_pos) | 10% | 0.002 m |
| BC v2 (-target_pos, -reach_err) | 10% | 0.002 m |

**joint-space PD + IK が唯一 state-coupled かつ delay regime で機能** → +8.6 cm の learned vs K=1 差を観測。
BC variants は reaching を改善するが open-loop 化して estimator signal が wash out。

## 図表構成

### F1: System overview (block diagram)

forward model + Kalman-like estimator + observation pipeline (noisy / delayed) + controller の
relationship を block diagram で示す。各層で扱う物理量 (true state / noisy obs / x_est / u) を明記。

**Caption**:
"Overview of the myoArm forward state estimation pipeline. The forward model (residual MLP)
maintains a state prediction; the Kalman-like estimator combines it with delayed/noisy observations
via gain K; the controller (joint-space PD or behaviour-cloned) consumes the estimated state.
``true_state`` is used only for evaluation (oracle), never inside the closed loop."

### F2: Stress oracle K heatmap (delay × noise)

Phase 3.2 stress sweep の best_by_condition.csv から、(delay, noise) 平均 oracle K を heatmap で示す。
delay=0 で右下 (xhigh) に向けて 1.0 → 0.25 への gradient、delay≥18 で全 cell が 1.0 になる構造を可視化。

**Caption**:
"Oracle Kalman gain as a function of observation delay and noise level (mean over 3 controllers,
72 cells total). At zero delay the optimal K decreases monotonically from 1.0 (no noise) to 0.25
(xhigh noise). At delay ≥ 18 steps, the optimal K is uniformly 1.0 — forward-model rollout
error compounds faster than observation noise grows, making blending ineffective."

### F3: 7-strategy stress eval comparison

stress grid 72 conditions × 7 strategies の mean tip_err (bar) + delta vs oracle (error bar or
subplot)。learned ≈ best_per_delay > best_per_noise/global_best > K=0 という ordering を示す。

**Caption**:
"Closed-form comparison of 7 gain-selection strategies on the stress evaluation grid. ``learned``
(Stage A retrained) and ``best_per_delay`` (K depending only on delay) achieve mean delta vs oracle
within 0.001 m; ``best_per_noise`` and ``global_best`` are 4-5× worse. K=0 (prediction-only)
diverges, K=1 (observation-only) is the strongest non-adaptive baseline."

### F4: Phase 2 D estimator differentiation (Δ heatmap)

Phase 2 D MVP の 6 cells × (learned − K=1.0) Δ final_tip / Δ min_tip を 2 つの heatmap で示す。
(noise=none, delay=18) が dark blue (−8.6 cm)、他は近 0 を示す。

**Caption**:
"Closed-loop reaching improvement from learned Kalman gain vs K=1.0 baseline, under
joint-space PD + IK controller. Estimator quality differentiation emerges in the (noise=none,
delay=18) cell with a 8.6 cm reduction in final tip-to-target error. Other cells show
sub-centimetre differences."

### F5: Phase 2 D representative trajectories

1 cell (例えば noise=none, delay=18) の代表 episode で、tip_pos の Euclidean error を
time-series で 3 estimator (K=0, K=1, learned) plot。K=0 は発散、K=1 と learned が approach
するが learned の方が target に近づく。

**Caption**:
"Representative tip-to-target error trajectories under the joint-PD controller for one episode
(noise=none, delay=18 steps). The prediction-only estimator (K=0) diverges (>5 m); observation-only
(K=1) approaches the target but stays ~0.74 m away; the learned condition-level estimator
(K_inferred ≈ 0.92) reaches 0.66 m, a 8.6 cm improvement."

### F6: Phase 4 BC trade-off scatter

5 controller variants を 2D scatter: x軸 = reaching success_010 (over 6 cells × estimators 平均)、
y軸 = max|Δ(learned − K=1)|min。D が左上 (reaching 低、Δ 高)、BC が右下 (reaching 高、Δ 低) に
位置し trade-off frontier を示す。

**Caption**:
"Trade-off between reaching success and estimator differentiation across five controller variants.
Joint-space PD + IK (D) sits in the high-differentiation / low-reaching regime; BC variants
improve reaching but lose the estimator signal as the policy becomes effectively open-loop.
No single controller in this study achieves both — the closed-loop benchmark is most informative
in the D regime."

## Results セクション構成

```
3.1 Forward-model + state estimator design
    - residual MLP architecture (Phase 1)
    - fixed-lag Kalman with delay handling (Phase 3.1)
    - improved dataset cycle (3.6k → 51k transitions, h=50 MSE 10.6 → 0.052)

3.2 Where blending matters (Stress eval, F2 + F3)
    - Phase 3.2 stress: oracle K varies 1.0 → 0.25 in (low delay + high noise)
    - delay ≥ 18 forces K=1 due to forward-model rollout limit
    - Stage A retrained recovers most of the oracle gain

3.3 Closed-loop effect of estimator quality (F4 + F5)
    - Phase 2 D: joint-space PD + IK + moment-arm muscle mapping
    - +8.6 cm at (noise=none, delay=18); other cells <2 cm
    - learned beats K=1 most in observation-delay-dominated regime

3.4 Open-loop collapse of behaviour-cloned controllers (F6)
    - Phase 4 BC: ScriptedReach demos + MLP policy
    - Reaching improves (S010 0% → 10-30%) but Δ(L−K1) shrinks by 10×
    - State-sensitivity ablation: removing target_pos / reach_err from BC
      input does not restore differentiation → policy is open-loop in target
```

## Discussion セクション主要点

1. **Trade-off as research insight, not failure mode**: 5 controllers map the
   (reaching success × estimator sensitivity) plane; the existence of this
   trade-off is itself a finding, supporting the conclusion that closed-loop
   estimator benchmarks need state-coupling.
2. **Why BC collapses**: small-scale demos (30 episodes / 30 targets) don't
   constrain the policy to use target features. Future work: target-conditioned
   policy, DAgger, larger demos.
3. **Structural limit of forward-model rollout**: delay ≥ 6 makes K=1 optimal.
   Long-horizon supervision could relax this and widen the K<1 regime in
   future work (Phase B per the implementation plan).
4. **The 8.6 cm signal**: small in absolute terms (vs ~70 cm reach distance)
   but distinguishable from intra-estimator noise and structurally located in
   the regime the theory predicts (delay-dominated, low-noise).

## Future work セクション

```
- Forward model long-horizon supervision (Phase 3.2 structural fix)
- Large-scale demonstration collection + DAgger (Phase 4 BC scale-up)
- Stage B redesign with per-episode K* labels or end-to-end gradient
- Test on additional MyoSuite arm tasks beyond reach
```

## 担当 (locked)

- 図表生成: `scripts/make_paper_figures.py` (matplotlib + seaborn、color-blind safe)
- 図出力: `figures/{F1,F2,...,F6}.{png,pdf}`
- データ summary CSV: `figures/data/{stress_oracle, phase2d, phase4bc, ...}.csv` (再現性)
- caption + 主張は本ドキュメントに locked-in

## 関連 artifacts (図のデータ source)

```
runs/estimators/2026-05-10T11-01-23Z/best_by_condition.csv          # F2
runs/learned_gain_evals/2026-05-10T12-59-43Z/comparison.csv         # F3
runs/closed_loop/2026-05-11T07-08-39Z/metrics.csv                   # F4, F5
runs/closed_loop/2026-05-11T06-15-10Z/metrics.csv                   # F6 (E)
runs/closed_loop/2026-05-11T07-08-39Z/metrics.csv                   # F6 (D)
runs/closed_loop/2026-05-11T08-16-43Z/metrics.csv                   # F6 (BC full)
runs/closed_loop/2026-05-11T08-32-19Z/metrics.csv                   # F6 (BC v1)
runs/closed_loop/2026-05-11T08-41-57Z/metrics.csv                   # F6 (BC v2)
```
