---
tags: [research, myoarm, state-estimation, kalman-filter, bayesian-integration, sensorimotor-control, project-idea]
created: 2026-05-09
project: myoarm-probabilistic-state-estimation
status: 新規プロジェクト候補
---

# myoArm 確率的状態推定と Bayesian 統合プロジェクト構想

## 位置づけ

この構想は、[[myoArm_皮質橋核小脳視床ループForwardモデル構想_2026-05-09]] の自然な次段階である。

forward model が「運動指令から次状態を予測する」仕組みだとすれば、state estimation / Kalman-like update と Bayesian integration は、その予測を遅延・ノイズを含む感覚 feedback や prior と統合し、制御に使える現在状態推定を作る仕組みである。

## 大きな問い

myoArm reaching において、真の状態をそのまま controller に渡すのではなく、遅延・ノイズを含む視覚、固有受容、forward prediction、task prior を統合する state estimator を置くことで、ヒトらしい制御制約と robust reaching を同時に説明できるか。

より具体的には:

- 遅延した感覚 feedback だけに頼る制御よりも、forward prediction + Kalman-like update は安定か。
- 視覚・固有受容・予測・prior を不確実性に応じて重みづけする Bayesian integration は、ノイズ条件下の reaching を改善するか。
- 感覚信頼度を変えたとき、モデルの重みづけ・endpoint bias・variability はヒト行動実験と同じ方向に変化するか。

## Project B: Kalman-like state estimation

### 問い

myoArm reaching において、遅延した observation だけではなく、forward prediction と sensory feedback を Kalman-like に統合することで、状態推定と到達制御は改善するか。

### 最小構成

```text
previous state estimate xhat_t
motor command u_t
  ↓
forward model
  ↓
prediction xhat_{t+1|t}
  ↓
delayed / noisy observation y_{t+1}
  ↓
Kalman-like update
  ↓
updated estimate xhat_{t+1|t+1}
  ↓
controller
```

数式の最小形:

```text
x_pred = f(x_est_t, u_t)
y_obs = h(x_true_{t-d}) + noise
x_est = x_pred + K (y_obs - h(x_pred))
u_{t+1} = controller(x_est, target)
```

ここで `K` は Kalman gain 的な重みである。観測が信頼できるときは大きく、観測が不確実なときや遅れが大きいときは小さくする。

### 比較条件

```text
true state controller
delayed observation only
prediction only
prediction + fixed-gain update
prediction + learned Kalman gain
prediction + uncertainty-adaptive gain
```

### 実験操作

- visual feedback delay
- proprioceptive feedback delay
- observation noise
- signal-dependent motor noise
- target jump
- force perturbation
- altered gravity

### 評価指標

- state estimation error
- endpoint error
- minimum / final tip error
- overshoot / oscillation
- perturbation recovery time
- jerk RMS / normalized jerk
- feedback gain requirement
- closed-loop stability

### 研究上の意味

このプロジェクトでは、遅延した feedback だけに頼る高ゲイン制御の不安定性と、forward prediction による補償の有効性を myoArm で検証する。シミュレータでは true state が得られるため、state estimation error を直接測れる点が強い。

## Project C: Bayesian reliability-weighted integration

### 問い

myoArm reaching において、視覚、固有受容、forward prediction、task prior を不確実性に応じて重みづけ統合することで、状態推定・target 推定・運動制御は改善するか。

### 最小構成

```text
prediction prior: x_pred
visual observation: y_vis
proprioceptive observation: y_prop
task prior: p(x) or p(target)
  ↓
Bayesian / reliability-weighted integration
  ↓
state estimate x_est
  ↓
controller
```

実装の最小形:

```text
x_est =
    w_pred * x_pred
  + w_vis  * x_vis
  + w_prop * x_prop
  + w_prior * x_prior
```

重みは信頼度、つまり分散の逆数で決める。

```text
w_i ∝ 1 / σ_i^2
sum_i w_i = 1
```

### 比較条件

```text
vision only
proprioception only
prediction only
fixed average
reliability-weighted integration
learned Bayesian-like integration
wrong prior
adaptive prior
```

### 実験操作

- visual noise を増やす
- proprioceptive noise を増やす
- visual delay を増やす
- proprioceptive delay を増やす
- target distribution prior を偏らせる
- sensory conflict を入れる
- target jump / cursor rotation 的な条件を入れる

### 評価指標

- state estimate error
- target estimate bias
- endpoint bias
- endpoint variability
- adaptation speed
- sensory reliability に応じた重み変化
- prior が強い条件での bias
- uncertainty が大きい条件での robustness

### 研究上の意味

このプロジェクトは Körding & Wolpert 2004 的な Bayesian integration を myoArm reaching に移植する方向である。特に、視覚情報を不確実にすると prior や固有受容への重みが増え、endpoint bias や variability が理論通り変化するかを検証できる。

## 統合プロジェクトとしての全体像

3つの構想は段階的に接続できる。

```text
Project A:
  cortico-ponto-cerebello-thalamo-cortical forward model loop

Project B:
  Kalman-like state estimator for delayed/noisy myoArm control

Project C:
  Bayesian reliability-weighted multisensory integration for myoArm reaching
```

全体アーキテクチャ:

```text
motor command u_t
  ↓
forward model prediction
  ↓
prediction prior x_pred

visual observation y_vis
proprioceptive observation y_prop
vestibular / gravity context g
task prior p(target)
  ↓
Bayesian / Kalman-like state estimator
  ↓
state estimate x_est
  ↓
optimal feedback controller
  ↓
motor command u_{t+1}
```

## 推奨順序

1. Forward model
   - `x_t, u_t -> x_{t+1}` をまず学習する。
2. Kalman-like state estimation
   - prediction と delayed/noisy observation を統合する。
3. Bayesian multisensory / prior integration
   - 視覚、固有受容、prediction、prior の信頼度重みづけを扱う。

理由: Kalman-like update には prediction が必要であり、Bayesian integration には複数の不確実な情報源が必要である。forward model がないと、状態推定プロジェクトが単なる noisy observation averaging になりやすい。

## 最初に作るべき実装

1. `DelayedObservationWrapper`
   - `qpos`, `qvel`, `tip_pos`, `target_pos` に delay を入れる。
2. `NoisyObservationWrapper`
   - 視覚・固有受容に別々の noise level を入れる。
3. `SignalDependentMotorNoise`
   - `neural_command` / `excitation` 側に SDN を入れる。
4. `ForwardModel`
   - one-step prediction baseline。
5. `KalmanLikeEstimator`
   - fixed gain / learned gain / uncertainty-adaptive gain を切り替える。
6. `ReliabilityWeightedIntegrator`
   - `w_i ∝ 1 / σ_i^2` で複数観測を統合する。

## 注意点

- 真の状態を controller に直接渡す条件は oracle baseline として扱う。
- ヒトらしさを主張する条件では、controller は delayed/noisy estimate だけを見る。
- Gym の `api_action` と神経生理学的な `neural_command` / `excitation` を混同しない。
- sensory noise と motor noise を分ける。
- fixed gain で十分な場合、Bayesian / learned gain の必要性を過大主張しない。
- Bayesian integration は「脳が明示的に確率分布を計算している」という主張ではなく、「行動が不確実性重みづけに近い」という計算レベルの主張に留める。

## 既存教材との接続

- [[myoArm_Reaching_Primer]]
- [[myoArm_皮質橋核小脳視床ループForwardモデル構想_2026-05-09]]
- `docs/modern-neuromechanical-control-primer/`
- `docs/myoarm-reaching-primer/`

## 参考文献メモ

- Miall, R. C., & Wolpert, D. M. (1996). Forward models for physiological motor control. *Neural Networks, 9*(8), 1265-1279. https://doi.org/10.1016/S0893-6080(96)00035-4
- Wolpert, D. M., Goodbody, S. J., & Husain, M. (1998). Maintaining internal representations: the role of the human superior parietal lobe. *Nature Neuroscience, 1*, 529-533. https://doi.org/10.1038/2245
- Körding, K. P., & Wolpert, D. M. (2004). Bayesian integration in sensorimotor learning. *Nature, 427*, 244-247. https://doi.org/10.1038/nature02169
- Körding, K. P., & Wolpert, D. M. (2006). Bayesian decision theory in sensorimotor control. *Trends in Cognitive Sciences, 10*(7), 319-326. https://doi.org/10.1016/j.tics.2006.05.003
- Miall, R. C., & King, D. (2008). State estimation in the cerebellum. *The Cerebellum, 7*, 572-576. https://doi.org/10.1007/s12311-008-0072-6
