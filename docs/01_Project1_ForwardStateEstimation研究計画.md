---
tags: [research, myoarm, project-plan, forward-model, state-estimation, kalman-filter]
created: 2026-05-09
status: draft
project: myoarm-forward-state-estimation
---

# Project 1: myoArm Forward Model + Kalman-like State Estimation 研究計画

## 目的

myoArm reaching において、forward model と Kalman-like state estimator が、遅延・観測ノイズ・signal-dependent motor noise 下の制御を改善するかを検証する。

この Project 1 は、後続の Bayesian integration、cortico-cerebellar loop、cortical population dynamics の基盤である。

## 中心仮説

### H1: Forward model は myoArm dynamics を予測できる

`x_t, u_t -> x_{t+1}` を学習する forward model は、one-step prediction だけでなく、短い multi-step rollout でも tip position と関節状態を予測できる。

### H2: Forward prediction は delayed feedback controller を安定化する

遅延した observation だけに頼る controller より、forward prediction を使って現在状態を推定する controller の方が、overshoot、oscillation、endpoint error が小さい。

### H3: Kalman-like update は prediction-only と delayed-feedback-only の両方を上回る

prediction と delayed/noisy observation を fixed gain または learned/adaptive gain で統合すると、state estimation error と reaching performance が改善する。

### H4: CfC / LTC は時系列予測器として有利な可能性がある

MLP より RNN / GRU / CfC / LTC が multi-step rollout で有利かを比較する。旧計画のように CfC を逆動力学近似器として使うのではなく、forward model として使う。

## 研究対象

環境:

- `myoArmReachRandom-v0`
- `myoArmReachFixed-v0` も必要に応じて使用

状態:

```text
x_t = [
  qpos_t,
  qvel_t,
  act_t,
  tip_pos_t,
  target_pos_t,
  reach_err_t
]
```

入力:

```text
u_t = neural_command_t or excitation_t
```

注意:

- Gym の `api_action` は実装上の API 入力。
- 生理学的解釈では `neural_command` / `excitation` を使う。
- `activation` は筋内部状態であり、controller 出力そのものではない。

予測 target:

```text
y_t = x_{t+1}
```

または残差形式:

```text
Δx_t = x_{t+1} - x_t
```

最初は残差形式を推奨する。

## Phase 0: 共通基盤

### 0.1 Episode logger

各 step で保存する。

```text
episode_id
step
time
qpos
qvel
act
tip_pos
target_pos
reach_err
neural_command
excitation
api_action
mj_data.ctrl
reward
terminated
truncated
```

### 0.2 Target set

random target の前に、管理された target set を作る。

分割:

```text
train targets
validation targets
test targets
extrapolation targets
```

target は q0 からの距離、方向、z 高さで層化する。

### 0.3 Noise / delay wrappers

実装:

- `DelayedObservationWrapper`
- `NoisyObservationWrapper`
- `SignalDependentMotorNoise`

SDN:

```python
u = controller(state)
noise = sigma * np.abs(u) * rng.normal(size=u.shape)
u_noisy = np.clip(u + noise, 0.0, 1.0)
api_action = action_adapter(u_noisy)
```

### 0.4 Metrics

制御 metrics:

- minimum tip error
- final tip error
- success rate
- peak speed
- jerk RMS
- normalized jerk
- effort / activation norm
- co-contraction
- overshoot
- oscillation index
- recovery time

推定 metrics:

- one-step prediction MSE
- multi-step rollout MSE
- tip prediction error
- qpos / qvel prediction error
- state estimation error
- uncertainty / error calibration

## Phase 1: Forward model baseline

### 1.1 Dataset 作成

まずは複数 controller から軌跡を集める。

候補:

- random excitation
- low-amplitude random excitation
- simple PD endpoint controller
- static hold controller
- previous controller outputs

重要:

- random だけでは到達運動の分布が偏る。
- policy 1種類だけでは forward model が分布外に弱くなる。
- train / validation / test target を分ける。

### 1.2 モデル比較

比較:

```text
MLP
GRU
LSTM
CfC
LTC
```

入力:

```text
x_t, u_t, target_t, context
```

出力:

```text
Δqpos, Δqvel, Δact, Δtip_pos
```

評価:

- one-step prediction
- 10-step rollout
- 50-step rollout
- target generalization
- altered gravity generalization
- SDN あり/なし

### 1.3 成功判定

Project 1 の次段階へ進む条件:

- one-step tip prediction error が十分小さい。
- short rollout で controller に使える程度の state estimate が得られる。
- MLP / RNN / CfC の性能差が測定できる。

## Phase 2: Delayed feedback control

### 2.1 Baseline controller

まずは単純な controller で比較する。

```text
true state controller
delayed observation controller
prediction-only controller
```

目的:

- 遅延した feedback だけだと何が悪化するかを明確にする。
- forward prediction 単独でどこまで補償できるかを見る。

### 2.2 Delay 条件

候補:

```text
0 ms
20 ms
40 ms
80 ms
120 ms
```

制御周期は `dt=0.02 s` なので、

```text
20 ms = 1 step
40 ms = 2 steps
80 ms = 4 steps
120 ms = 6 steps
```

## Phase 3: Kalman-like estimator

### 3.1 Fixed-gain estimator

最小実装:

```text
x_pred = f(x_est_t, u_t)
y_obs = delayed_noisy_observation
x_est = x_pred + K (y_obs - h(x_pred))
```

まず `K` は scalar または diagonal でよい。

比較:

```text
K = 0.0  prediction only
K = 0.25
K = 0.5
K = 0.75
K = 1.0  observation only
```

### 3.2 Learned / adaptive gain

次に、観測ノイズと delay に応じて gain を変える。

入力:

```text
x_pred
y_obs
innovation = y_obs - h(x_pred)
noise_level
delay_steps
```

出力:

```text
K_t
```

### 3.3 評価

条件:

- observation noise なし/あり
- motor SDN なし/あり
- delay なし/あり
- target jump
- external perturbation

評価:

- state estimation error
- endpoint error
- overshoot
- oscillation
- effort
- recovery time

## Phase 4: 論文化可能な実験セット

最小論文構成:

1. myoArm で遅延 feedback 制御の不安定性を示す。
2. forward model baseline を構築する。
3. prediction-only と delayed-feedback-only の限界を示す。
4. Kalman-like update が state estimation と reaching を改善することを示す。
5. SDN / observation noise / target jump で robustness を評価する。

主張:

> myoArm reaching において、forward prediction と delayed sensory feedback の統合は、高次元筋骨格 plant の安定制御に有効である。これは後続の cortico-cerebellar loop や Bayesian integration を検証するための基盤となる。

## 主要な比較表

| 条件 | forward model | delayed observation | update gain | 目的 |
|---|---|---|---|---|
| Oracle | なし | なし | なし | true state controller の上限 |
| Delayed only | なし | あり | 1.0 | 遅延 feedback の限界 |
| Prediction only | あり | なし | 0.0 | forward prediction 単独の限界 |
| Fixed Kalman-like | あり | あり | 固定 | 最小統合モデル |
| Adaptive Kalman-like | あり | あり | noise/delay 依存 | 不確実性対応 |
| Learned estimator | あり | あり | learned | 上限性能 |

## 失敗時の判断

Forward model がうまく予測できない場合:

- state 表現が足りない可能性。
- `activation` / `mj_data.ctrl` / `qacc` を追加する。
- one-step ではなく local linear model / ensemble にする。
- target 分布や controller 分布を見直す。

Kalman-like update が効かない場合:

- controller が state estimate に鈍感すぎる可能性。
- delay/noise 条件が弱すぎる可能性。
- forward model rollout error が大きすぎる可能性。
- fixed gain では不十分で adaptive gain が必要な可能性。

## 実装順序

```text
1. dataset logger
2. target set generator
3. action adapter / SDN
4. delay/noise wrappers
5. metrics
6. forward model training
7. rollout evaluation
8. delayed-feedback controller
9. fixed-gain estimator
10. adaptive/learned estimator
```

## 新規フォルダ化するときの候補名

```text
myoarm-forward-state-estimation
```

最初のディレクトリ構成案:

```text
myoarm-forward-state-estimation/
  docs/
  src/
    myoarm_fse/
      envs/
      data/
      models/
      estimators/
      controllers/
      metrics/
  scripts/
  configs/
  runs/
  tests/
```
