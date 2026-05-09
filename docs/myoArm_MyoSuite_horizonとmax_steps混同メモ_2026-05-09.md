---
tags: [research, myoarm, myosuite, reproducibility, horizon, experiment-design, bug-risk]
created: 2026-05-09
project: myoarm-lambda-ep
status: 重要再確認事項
---

# myoArm / MyoSuite: `horizon` と `max_steps` 混同メモ (2026-05-09)

## 要点

旧 `myoarm-lambda-ep` プロジェクトでは、多くの実験スクリプトで `run_episode(..., max_steps=600)` としていた。
これは 20 ms 制御周期なら `600 × 0.02 s = 12 s` の episode を意図していた可能性が高い。

しかし、MyoSuite 環境側の `env.unwrapped.horizon` が 150 のままだと、
150 step (`150 × 0.02 s = 3 s`) で `truncated=True` が返る。

旧コードは多くの箇所で:

```python
for _ in range(max_steps):
    ...
    obs, _, term, trunc, info = env.step(a_total)
    ...
    if term or trunc:
        break
```

となっている。

したがって:

```text
max_steps = 600
horizon   = 150
if term or trunc: break
```

なら、実際には 600 step / 12 s ではなく、**150 step / 3 s で終了**する。

## 実測確認

手元の `myosuite 2.12.1` で `myoArmReachRandom-v0` を確認した。

```python
import gymnasium as gym
import myosuite  # noqa
import numpy as np

env = gym.make("myoArmReachRandom-v0")
obs, info = env.reset(seed=0)

print(env.unwrapped.horizon)  # 150
print(env.unwrapped.dt)       # 0.02

for i in range(1, 605):
    obs, rew, term, trunc, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    if term or trunc:
        print(i, term, trunc, env.unwrapped.time)
        break
```

結果:

```text
horizon = 150
dt      = 0.02
ended at 150
term    = False
trunc   = True
time    ≈ 3.0 s
```

## `horizon` と `max_steps` の違い

### `horizon`

MyoSuite / Gym 環境側の episode 上限。

```python
env.unwrapped.horizon
```

`horizon=150` なら、150 control steps で環境が time-limit truncation を返す。

```text
150 steps × 0.02 s = 3 s
```

### `max_steps`

自前の Python loop 側の最大反復数。

```python
for step in range(max_steps):
    ...
```

`max_steps=600` は「環境が止めなければ最大 600 回 step する」という意味であり、
**環境側 `horizon` を変更する効果はない**。

### 実際の episode 長

多くの実装では:

```text
actual steps = min(max_steps, horizon, termination step)
```

になる。

## 旧プロジェクトへの影響

これは旧 preprint / 実験ログの解釈に関わる重要事項。

影響しうる点:

- 「600 control steps = 12 s episode」と書いていた箇所が実際には 3 s だった可能性
- post-target wandering / hold phase / settling phase の解釈
- final error の意味
- movement window の offset 検出
- long-horizon drift / gravity settling の診断
- 旧論文 §2.1 / §2.7 / limitations の episode length 記述
- 教材中の「12 s 評価」記述

ただし、主要 kinematic 指標の多くは movement window 内で計算しているため、
到達運動が 3 s 以内に終わっていれば、`jerk_rms`、peak speed、velocity-peak ratio、
straightness の一部は大きく変わらない可能性もある。

一方、final error や settling / drift の議論には直接影響する。

## 要再確認

旧プロジェクトで実際に `horizon` を 600 に変更していた痕跡は、現時点で見つかっていない。
少なくとも主要スクリプト群では `run_episode(..., max_steps=600)` と `if term or trunc: break` が確認された。

次に確認すべきこと:

- [ ] 主要 JSON / trajectory log に保存された positions の長さを確認
- [ ] `f16_n50.json` などの per-seed log が 150 step か 600 step か確認
- [ ] figure 3 trajectory の raw trajectory 長を確認
- [ ] paper / README / docs 内の `600 steps = 12 s` 記述を洗い出す
- [ ] 新規プロジェクトでは `env.unwrapped.horizon = episode_steps` を明示する

確認コマンド候補:

```bash
rg -n '"positions"|trajectory|tip_pos|max_steps|horizon|600|150' results scripts paper docs
```

## 新規プロジェクトでの推奨

episode 長は一つの変数で管理し、環境側 `horizon` と Python loop 側 `max_steps` を揃える。

```python
episode_steps = 600
env = gym.make("myoArmReachRandom-v0")
env.unwrapped.horizon = episode_steps

for step in range(episode_steps):
    obs, reward, terminated, truncated, info = env.step(api_action)
    if terminated:
        break
    if truncated:
        raise RuntimeError("Unexpected truncation before episode_steps")
```

また、log には必ず以下を保存する。

```python
{
    "dt": env.unwrapped.dt,
    "horizon": env.unwrapped.horizon,
    "episode_steps_requested": episode_steps,
    "episode_steps_recorded": len(positions),
    "terminated": bool(terminated),
    "truncated": bool(truncated),
}
```

## 関連ノート

- [[myoArm λ-EP プロジェクト経緯総まとめ (2026-05-09 時点)]]
- [[myoArm新規プロジェクト構想と教材整備_2026-05-09]]
- [[myoArm-lambda-EP_v3先行研究サーベイ_2026-05-09]]
- [[λ-EP実装の重力補償欠如とGomi-Kawato批判の再現]]

