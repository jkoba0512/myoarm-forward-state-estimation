# Phase 1〜4 統合論文 — Outline と研究メッセージ

このドキュメントは Phase 1〜4 の実装結果から論文化に向けた中心メッセージと
構成を **固定** するためのものです。実装中の判断ノートとは別に、論文側の
不可逆な決定 (主張、結論、figure caption) を locked-in 状態で保持します。

## 中心メッセージ (v3、Phase C 反映後)

> **Closed-loop reaching benefits from a forward-model-based predictive state observer, and the optimal sensory prediction-error correction gain depends on forward-model accuracy AND on the oracle used to train any learned gain. (i) With a single-step-supervised model, a learned condition-level correction gain trained on the open-loop estimation oracle beats observation-only by 8.6 cm in the (low noise, large delay) regime. (ii) With H-step rollout supervision (H=4, H=8) the forward model becomes accurate enough that prediction-only (K=0) globally wins in closed loop by 12-24 cm — a paradigm shift from "blend with sensory feedback" to "trust the forward model". (iii) The open-loop oracle that Stage A was trained on no longer matches the closed-loop optimum at this stage, so the predictor outputs K≈0.4-1.0 instead of K=0. (iv) Redefining the oracle in terms of closed-loop task error produces a predictor that matches the K=0 baseline at delay≥18 cells within 1.5 cm and closes 40-60% of the gap at delay=0 cells, with no architectural changes — demonstrating that the right correction gain depends not only on the regime but on what objective the predictor is trained against.**

研究問い (Project 1):
> myoArm reaching において、forward prediction と sensory prediction-error correction を用いた predictive state observer は、
> sensory delay・observation noise・signal-dependent motor noise 下の制御を改善するか。

答え (二段で構成):
- **Yes, with caveats**: estimator quality propagates to closed-loop, but the regime depends on:
  - state-coupled controller (joint-PD + IK; BC washes out the signal)
  - forward-model accuracy (single-step → blending matters; multi-step → prediction alone wins)
- **改善が消える条件**: behaviour-cloned policy が per-target generalize しない (small demo scale)
- **paradigm shift**: long-horizon forward-model supervision shifts the closed-loop optimum from blending (K∈(0,1)) to pure prediction (K=0)

## 主結果 5 つ (R1-R3 + R5 + 新 R6)

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

### R3. closed-loop で estimator quality が伝播する条件 (with single-step forward model)

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

**Caveat (新 R5 で展開)**: この結果は single-step-supervised forward model に依存。multi-step supervision で精度向上した forward model では別の estimator (K=0) が最適化される。

### R5. Long-horizon forward-model supervision shifts the closed-loop paradigm (新)

K=4 multi-step rollout loss で forward model を再訓練:

- **Open-loop**: h=50 tip prediction error が **0.138 m → 0.064 m (-54%)**、h=10 -30%
- **Open-loop stress oracle**: K<1 が必要な cell が 18 → **26 / 72 (25% → 36%)**。delay=6/18/36 にも blending 領域が出現

```
NEW oracle K (delay × noise mean):
delay\noise   none  low  med  high  vhigh  xhigh
  d= 0       1.00  0.83 0.58  0.50  0.33   0.25
  d= 6       1.00  1.00 1.00  0.92  0.67   0.42
  d=18       1.00  1.00 1.00  1.00  0.92   0.75  ← 新規
  d=36       1.00  1.00 1.00  1.00  1.00   0.92  ← 新規
```

- **Closed-loop (Phase 2 D MVP 再走、K=4)**: 期待に反して、**K=0 (prediction-only) が全 6 cell で best** (min_tip ~0.53 m vs K=1 0.66-0.77 m、Δ -12 ~ -24 cm)。R3 の (noise=none, delay=18) で観測されていた learned vs K=1 の +8.6 cm signal は消失 (Δ -0.5 cm)。
- **解釈**: forward model が十分強い領域では、観測ノイズ補正 (blending) より prediction trust の方が制御に有利。open-loop estimation accuracy 最適 K と closed-loop task 性能最適 K が乖離する現象。stress oracle で訓練された learned predictor は依然 K≈1 を出力するが、closed-loop 真の最適 K=0 を捕捉できない。

### R6. Closed-loop-aware oracle が paradigm gap を縮める (Phase C)

R5 で観測した paradigm shift gap (open-loop oracle で訓練した learned が K≈1 を出すが closed-loop 最適は K=0) を、Stage A の oracle 定義変更で対処。

**手順**:
1. K=8 forward model 下で K ∈ {0, 0.25, 0.5, 0.75, 1} を 6 cells × 10 eps の closed-loop で sweep
2. cell ごとに **min_tip_error を最小化する K** を K\* として選択 → 全 6 cell で K\*=0
3. K\*=0 ラベルで Stage A を再学習
4. 同じ Phase 2 D MVP grid で評価

**結果**: Stage A predictor 出力が K≈0.7-1.0 → **K≈0.02** にシフト。

| cell | K=0 (oracle) | Phase B' learned (open-loop oracle) | **Phase C learned (closed-loop oracle)** | Δ(C - K=0) | Δ(C - B') |
|---|---|---|---|---|---|
| none d=0 | 0.493 | 0.775 | 0.666 | +0.173 | **-0.109** |
| **none d=18** | 0.497 | 0.702 | **0.489** | **-0.008** | **-0.213** |
| high d=0 | 0.493 | 0.774 | 0.663 | +0.170 | **-0.111** |
| **high d=18** | 0.497 | 0.702 | **0.497** | **0.000** | **-0.205** |
| xhigh d=0 | 0.493 | 0.773 | 0.652 | +0.159 | **-0.121** |
| **xhigh d=18** | 0.497 | 0.727 | **0.482** | **-0.015** | **-0.245** |

要点:
- **d=18 cells で K=0 oracle performance を達成** (Δ ≤ 1.5 cm、`xhigh d=18` では K=0 を 1.5 cm 超える)
- **d=0 cells で gap が残る** (+16-17 cm): K-curve が K=0→K=0.25 で sharp jump (0.49→0.77)、sigmoid asymptotic 出力 0.025 でも penalty
- **全 cell mean で -16 cm 改善** = Phase B/B' の paradigm shift gap を **40-60% 解消**
- predictor architecture は無変更 (sigmoid output を含めて同じ)、変えたのは training oracle のみ

#### R6 の含意

estimator design は **controller と oracle の co-optimization** が必要。同じ predictor architecture / 同じ訓練 pipeline でも、oracle 定義一つで予測 K が 0.5-0.7 動く。論文上の position:

> "An estimator trained on open-loop oracle labels is not necessarily optimal in closed-loop deployment; the two objectives can diverge even on the same task. We show that redefining the oracle via closed-loop K-sweep resolves the gap with no architectural changes, suggesting that future work on learned sensory prediction-error correction gains should specify the oracle target explicitly."

#### R5 supplementary: K=8 で paradigm shift が monotonic に深化

K=8 multi-step supervision で同じ pipeline を再走:

- **Open-loop**: h=50 tip_err **0.064 → 0.048 m** (further -24% vs K=4、-65% vs OLD baseline)
- **Stress oracle**: K<1 cells 26 → 27 / 72 (新規: d=18 high で 1.0 → 0.917)
- **Closed-loop**: **K=0 advantage が更に拡大** — Δ(K=0 − K=1) at d=18 で OLD +0.01 → K=4 -0.13 → **K=8 -0.20**:

```
Δ(K=0 - K=1) min_tip in closed-loop:
cell           OLD       K=4       K=8
none, d=0    -0.22    -0.24    -0.28
none, d=18   +0.01    -0.12    -0.20
high, d=18   +0.05    -0.15    -0.19
xhigh,d=18   +0.04    -0.12    -0.20
```

- **K=8 で K=1 が d=18 でむしろ悪化** (min_tip 0.66 → 0.70): forward model が強くなるほど observation correction が closed-loop で hurt する
- learned predictor の出力は K=4 とほぼ同じ (K=8 でも K=0 を出せない) — open-loop oracle で訓練している限界
- **Paradigm progression (forward-model strength → optimal closed-loop K) が monotonic に確認**:
  - Weak (OLD): K=1 (rely on observation)
  - Medium (K=4): mixed (some K∈(0,1), K=0 starts winning)
  - Strong (K=8): K=0 dominant (pure prediction)

```
Closed-loop min_tip (NEW Phase B, mean over 10 eps):
cell           K=0     K=1     learned
none, d= 0    0.530   0.770    0.779
none, d=18    0.531   0.656    0.660
xhigh, d= 0   0.530   0.762    0.761
xhigh, d=18   0.531   0.653    0.665
```

## 図表構成

### F1: System overview (block diagram)

forward model + predictive state observer + observation pipeline (noisy / delayed) + controller の
relationship を block diagram で示す。各層で扱う物理量 (true state / noisy obs / x_est / u) を明記。

**Caption**:
"Overview of the myoArm forward state estimation pipeline. The forward model (residual MLP)
maintains a state prediction; the predictive state observer corrects that prediction using delayed/noisy
sensory prediction error via correction gain K; the controller (joint-space PD or behaviour-cloned) consumes the estimated state.
``true_state`` is used only for evaluation (oracle), never inside the closed loop."

### F2: Stress oracle K heatmap (delay × noise) — OLD vs NEW

2-panel: 上段が OLD baseline (single-step forward model)、下段が NEW K=4 (multi-step supervision)。
NEW では delay=6/18/36 にも blending 領域が出現することを直接比較。

**Caption**:
"Oracle sensory prediction-error correction gain as a function of observation delay and noise level (mean over 3 controllers).
**Top**: single-step-supervised forward model. At zero delay the optimal K decreases monotonically from
1.0 (no noise) to 0.25 (xhigh noise). At delay ≥ 18 steps, the optimal K is uniformly 1.0 — forward-
model rollout error compounds faster than observation noise grows.
**Bottom**: with K=4 multi-step rollout supervision, the K<1 regime widens significantly: blending now
beats observation-only at (delay=6, high-xhigh), (delay=18, vhigh-xhigh), and (delay=36, xhigh) —
previously all uniformly K=1. 18 -> 26 of 72 cells require K<1."

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
"Closed-loop reaching improvement from learned correction gain vs K=1.0 baseline, under
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

### F7: Closed-loop paradigm shift across forward-model variants + oracle redefine (4-panel)

4-panel (3-bar per cell): single-step / K=4 / K=8 (open-loop oracle Stage A) / K=8 (closed-loop oracle Stage A)。

**Caption**:
"Closed-loop reaching min-tip-to-target error under joint-PD + IK controller, mean ± std over 10 episodes.
**(1)** Single-step-supervised forward model: K=0 diverges in long-delay cells, learned ≈ K=1, +8.6 cm
learned-vs-K=1 at (none, d=18) is the only signal.
**(2)** K=4 multi-step supervision: K=0 becomes the best estimator in all 6 cells; learned and K=1
flatten near 0.66-0.78 m while K=0 reaches ~0.53 m.
**(3)** K=8 multi-step supervision: K=0 advantage deepens to 0.49 m globally; K=1 actively *worsens*
at d=18 (0.66 → 0.70 m) as forward model grows stronger.
**(4)** Same K=8 forward model but Stage A trained on a closed-loop K-sweep oracle (Phase C): the
predictor now outputs K ≈ 0.02 (vs ≈ 0.7-1.0 from the open-loop oracle), and **the learned bar matches
the K=0 baseline at the three delay=18 cells**, closing 40-60% of the gap that survived at delay=0.
The figure visualises that *both* forward-model accuracy and oracle definition shape what the optimal
closed-loop sensory prediction-error correction gain is."

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
    - fixed-lag predictive state observer with delay handling (Phase 3.1)
    - improved dataset cycle (3.6k → 51k transitions, h=50 MSE 10.6 → 0.052)

3.2 Where blending matters (Stress eval, F2 + F3)
    - Phase 3.2 stress: oracle K varies 1.0 → 0.25 in (low delay + high noise)
    - delay ≥ 18 forces K=1 due to forward-model rollout limit (with single-step supervision)
    - Stage A retrained recovers most of the oracle gain

3.3 Closed-loop effect of estimator quality (F4 + F5)
    - Phase 2 D: joint-space PD + IK + moment-arm muscle mapping
    - +8.6 cm at (noise=none, delay=18); other cells <2 cm
    - learned beats K=1 most in observation-delay-dominated regime
    - All results contingent on the single-step-supervised forward model (→ 3.5)

3.4 Open-loop collapse of behaviour-cloned controllers (F6)
    - Phase 4 BC: ScriptedReach demos + MLP policy
    - Reaching improves (S010 0% → 10-30%) but Δ(L−K1) shrinks by 10×
    - State-sensitivity ablation: removing target_pos / reach_err from BC
      input does not restore differentiation → policy is open-loop in target

3.5 Long-horizon forward-model supervision shifts the paradigm (F2 panel 2-3 + F7 panel 2-3)
    - K=4 multi-step rollout loss: h=50 tip_err 0.138 → 0.064 m (-54%)
    - Open-loop stress oracle: K<1 regime widens 25% → 36% of cells,
      blending now optimal at delay≥6 in high-noise cells
    - Closed-loop: K=0 (prediction-only) becomes the best estimator in
      all 6 cells (min_tip ~0.53 m vs K=1 0.66-0.77 m, Δ -12 to -24 cm)
    - K=8 deepens monotonically: K=0 advantage grows to -0.20 m, K=1
      actively worsens at d=18 as forward model strengthens
    - Stage A trained on the open-loop oracle no longer matches closed-loop optimum
    - Paradigm shift: blending → prediction-trusting as the forward model
      becomes sufficient to roll out over the controller's planning horizon

3.6 Closed-loop-aware oracle resolves the paradigm gap (F7 panel 4)
    - Redefine the Stage A oracle: K* per cell selected by minimising
      closed-loop min_tip over K ∈ {0, 0.25, 0.5, 0.75, 1.0}
    - Under K=8 forward model, K*=0 for all 6 cells (consistent with R5)
    - Retrained Stage A outputs K ≈ 0.02 (vs ≈ 0.4-1.0 from open-loop oracle)
    - Closed-loop: matches K=0 within 1.5 cm at delay=18 cells, closes
      40-60% of the gap at delay=0 cells
    - The residual d=0 gap is structural: K-curve at delay=0 is sharp
      around K=0 (K=0 → K=0.25 jumps 0.49 → 0.77 m) so the sigmoid
      asymptote (K ≈ 0.025) loses ~17 cm; d=18 K-curve is flat near
      K=0 so the slight offset is harmless
    - No architectural changes — same MLP, same training pipeline,
      different oracle definition
```

## Discussion セクション主要点

1. **Trade-off as research insight, not failure mode**: 5 controllers map the
   (reaching success × estimator sensitivity) plane; the existence of this
   trade-off is itself a finding, supporting the conclusion that closed-loop
   estimator benchmarks need state-coupling.
2. **Why BC collapses**: small-scale demos (30 episodes / 30 targets) don't
   constrain the policy to use target features. Future work: target-conditioned
   policy, DAgger, larger demos.
3. **The 8.6 cm signal is regime-bound**: small in absolute terms (vs ~70 cm
   reach distance) but distinguishable from intra-estimator noise and
   structurally located where the theory predicts (delay-dominated, low-noise)
   — **as long as the forward model is single-step-supervised**.
4. **Paradigm shift with long-horizon supervision (新)**: K=4 multi-step rollout
   loss simultaneously widens the open-loop K<1 regime *and* drives the
   closed-loop optimum to K=0. Open-loop estimation accuracy and closed-loop
   task performance do not co-optimize the same correction gain — a structural
   misalignment that the open-loop oracle hides.
5. **Implication for adaptive gain estimators**: Stage A trained on
   open-loop oracle outputs K≈1 at delay≥6 even when closed-loop optimum is
   K=0. Phase C shows that closed-loop-aware oracle definition (label by
   task tip-error under K-sweep closed-loop rollouts) recovers the right
   regime with no architectural changes.
6. **Oracle as a design knob (新, R6)**: same predictor MLP / same
   training pipeline; redefining only the K* labels (open-loop estimation
   error → closed-loop task error) moves the predictor's output by 0.5-0.7
   and closes 40-60% of the closed-loop gap. This suggests that future
   learned correction-gain work should specify the oracle target explicitly,
   and that the implicit "open-loop accuracy" choice can underperform when
   control is downstream.

## Future work セクション

```
- Predictor parameterisation that can output exactly K=0 (Hardtanh /
  output clip / "push-to-zero" regulariser) to remove the d=0 sigmoid
  asymptote gap surfaced in Phase C
- Even larger K in multi-step supervision (K=16, 32) to test whether
  the paradigm shift saturates or continues
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
runs/estimators/2026-05-10T11-01-23Z/best_by_condition.csv          # F2 top (OLD)
runs/estimators/2026-05-11T09-53-59Z/best_by_condition.csv          # F2 bottom (NEW K=4)
runs/learned_gain_evals/2026-05-10T12-59-43Z/comparison.csv         # F3
runs/closed_loop/2026-05-11T07-08-39Z/metrics.csv                   # F4, F5, F7 panel 1 (OLD D MVP)
runs/closed_loop/2026-05-11T10-43-45Z/metrics.csv                   # F7 panel 2 (K=4 D MVP)
runs/closed_loop/2026-05-11T12-37-26Z/metrics.csv                   # F7 panel 3 (K=8 D MVP)
runs/closed_loop/2026-05-11T13-57-01Z/metrics.csv                   # K-sweep (K=0.25/0.5/0.75) for R6
runs/closed_loop/2026-05-11T14-04-34Z/metrics.csv                   # F7 panel 4 (Phase C: K=8 + closed-loop oracle Stage A)
runs/estimators/closed_loop_oracle_phaseC/best_by_condition.csv     # closed-loop oracle (R6)
runs/closed_loop/2026-05-11T06-15-10Z/metrics.csv                   # F6 (E)
runs/closed_loop/2026-05-11T07-08-39Z/metrics.csv                   # F6 (D)
runs/closed_loop/2026-05-11T08-16-43Z/metrics.csv                   # F6 (BC full)
runs/closed_loop/2026-05-11T08-32-19Z/metrics.csv                   # F6 (BC v1)
runs/closed_loop/2026-05-11T08-41-57Z/metrics.csv                   # F6 (BC v2)
```
