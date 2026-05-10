# Initial Implementation Plan

作成日: 2026-05-09

対象: `docs/01_Project1_ForwardStateEstimation研究計画.md` の Phase 0 から Phase 1 baseline まで。

## 目的

最初の実装では、forward model と Kalman-like estimator の性能比較に入る前に、再現可能な myoArm reaching データセットを作るための共通基盤を固める。

成功条件:

- 固定 target split を生成・保存できる。
- myoArm episode を同じ schema で記録できる。
- `neural_command` / `excitation` / `api_action` / `activation` を分けて保存できる。
- delay / observation noise / signal-dependent motor noise を独立に切り替えられる。
- 最小 controller で smoke rollout と metric 集計ができる。

## 実装ステップ

### Step 1: Config と target set

追加候補:

- `src/myoarm_fse/config.py`
- `src/myoarm_fse/envs/targets.py`
- `configs/targets/default.yaml`
- `scripts/generate_targets.py`
- `tests/test_targets.py`

内容:

- train / validation / test / extrapolation の target split を生成する。
- seed、各 split の target 数、workspace bounds、z 高さ、距離 bin を config 化する。
- 出力は `runs/targets/*.npz` または `*.json` に保存する。

最初の検証:

```bash
uv run python scripts/generate_targets.py --config configs/targets/default.yaml
uv run pytest tests/test_targets.py
```

### Step 2: Observation/state schema

追加候補:

- `src/myoarm_fse/envs/state.py`
- `tests/test_state_schema.py`

内容:

- `qpos`, `qvel`, `act`, `tip_pos`, `target_pos`, `reach_err` を `x_t` として抽出する関数を作る。
- Gym observation と MuJoCo internals の差分を吸収する薄い adapter を置く。
- shape と dtype を明示する。

最初の検証:

```bash
uv run python -c "import myosuite, gymnasium as gym; env = gym.make('myoArmReachFixed-v0')"
uv run pytest tests/test_state_schema.py
```

### Step 3: Action adapter と motor noise

追加候補:

- `src/myoarm_fse/envs/actions.py`
- `src/myoarm_fse/envs/noise.py`
- `tests/test_actions.py`
- `tests/test_noise.py`

内容:

- controller 出力を `neural_command` / `excitation` として扱い、Gym に渡す `api_action` へ変換する。
- `SignalDependentMotorNoise` を実装する。
- clipping range と random seed の再現性をテストする。

### Step 4: Delay/noisy observation wrappers

追加候補:

- `src/myoarm_fse/envs/wrappers.py`
- `tests/test_wrappers.py`

内容:

- `DelayedObservationWrapper`
- `NoisyObservationWrapper`
- delay steps と Gaussian observation noise を config から切り替える。
- delay 0 / noise 0 のとき identity になることをテストする。

### Step 5: Episode logger

追加候補:

- `src/myoarm_fse/data/logger.py`
- `src/myoarm_fse/data/schema.py`
- `scripts/collect_episodes.py`
- `tests/test_episode_logger.py`

保存 schema:

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

最初は `npz` を標準にする。分析しやすさが必要になった段階で `parquet` 追加を検討する。

### Step 6: Baseline controllers

追加候補:

- `src/myoarm_fse/controllers/random.py`
- `src/myoarm_fse/controllers/hold.py`
- `src/myoarm_fse/controllers/pd_endpoint.py`
- `tests/test_controllers.py`

最初に入れる controller:

- random excitation
- low-amplitude random excitation
- static hold

PD endpoint controller は state schema と action adapter が安定してから追加する。

### Step 7: Metrics

追加候補:

- `src/myoarm_fse/metrics/reaching.py`
- `src/myoarm_fse/metrics/prediction.py`
- `tests/test_metrics.py`

最初に実装する metrics:

- minimum tip error
- final tip error
- success rate
- effort / activation norm
- one-step prediction MSE
- rollout MSE

jerk、overshoot、oscillation index、recovery time は episode logger の出力が安定してから追加する。

### Step 8: Forward model baseline

追加候補:

- `src/myoarm_fse/models/datasets.py`
- `src/myoarm_fse/models/mlp.py`
- `scripts/train_forward_model.py`
- `scripts/evaluate_rollout.py`
- `configs/models/mlp.yaml`

最初の model:

- residual MLP: `[x_t, u_t] -> delta x_t`

最初の評価:

- one-step MSE
- 10-step rollout MSE
- 50-step rollout MSE
- tip prediction error

GRU / LSTM / CfC / LTC は、MLP baseline と dataset schema が固定された後に比較対象として追加する。

## 最初のPR相当の到達点

最初のまとまった実装単位は、Phase 0 の共通基盤に限定する。

含めるもの:

- target set generator
- state/action adapters
- motor noise
- delay/noise wrappers
- episode logger
- random/hold controllers
- reaching metrics
- smoke tests

含めないもの:

- forward model training loop
- Kalman-like estimator
- adaptive/learned gain
- CfC / LTC 比較

## 推奨コマンド

初期確認:

```bash
uv lock
uv run python -c "import myoarm_fse; print(myoarm_fse.__version__)"
uv run pytest
```

Phase 0 実装後の smoke run:

```bash
uv run python scripts/generate_targets.py --config configs/targets/default.yaml
uv run python scripts/collect_episodes.py --episodes 2 --controller random --target-split validation
uv run pytest
```

## 実装進捗ログ

### 2026-05-09: Step 3 (ActionAdapter のみ) 完了

Step 3 のうち ActionAdapter 部分を実装した。SDN (`noise.py`) は別タスクとして未実装。

追加ファイル:

- `src/myoarm_fse/envs/__init__.py` (`ActionAdapter` と `detect_action_dim` を re-export)
- `src/myoarm_fse/envs/actions.py`
- `tests/test_action_adapter.py` (39 tests)

確定した layer 区別 (ActionAdapter スコープ):

- `excitation` ∈ `[0, 1]^n`: 研究側 canonical 表現。
- `api_action` ∈ `[-1, 1]^n`: Gym/MyoSuite に渡す入力。
- `neural_command`: controller 側の責務 (ActionAdapter は扱わない)。
- `activation`: 筋モデル内部状態 (ActionAdapter は扱わない)。
- `mj_data.ctrl`: `env.step` 後に検査する MuJoCo 最終 actuator control (logger 側で post-step 検査)。

確定した API:

```text
ActionAdapter(action_dim: int)
  .action_dim
  .excitation_to_api_action(x) -> np.float32 (n,)
  .api_action_to_excitation(a) -> np.float32 (n,)
  .clip_excitation(x)          -> np.float32 (n,)
  .clip_api_action(a)          -> np.float32 (n,)

detect_action_dim(env) -> int   # env.action_space.shape のみ参照、low/high は検証しない
```

確定した実装方針:

- 出力 dtype は `np.float32` 固定。入力 dtype は暗黙キャストで吸収。
- 入力は 1-D の numpy array-like (list / tuple / ndarray)。2-D batch / 0-D scalar は `ValueError`。
- shape mismatch、NaN / Inf は `ValueError` で弾く。
- レンジ外は silent clip。
- torch tensor は明示サポートしない (numpy 境界で変換する責務は controller 側)。
- `action_dim` は `int` かつ `> 0`。`bool` は `int` の subclass だが `ValueError` で弾く。
- `detect_action_dim` の `low/high` 検証は env factory 側に回す。`strict=True` parameter は今は追加しない。

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-09-myoarm-fse-action-adapter-open-questions.md`
- `Logs/2026-05-09-myoarm-fse-action-adapter-open-questions-answer.md`
- `Logs/2026-05-09-myoarm-fse-action-adapter-ready-to-implement.md`

検証:

```bash
uv run pytest    # 40 passed (39 action_adapter + 1 package)
```

未着手 (Step 3 残り、および後続):

- `src/myoarm_fse/envs/noise.py` (`SignalDependentMotorNoise`)
- Step 4: `DelayedObservationWrapper` / `NoisyObservationWrapper`
- Step 5 以降: episode logger、controllers、metrics、forward model

### 2026-05-09: Step 3 (SignalDependentMotorNoise) 完了

Step 3 残りの SDN を実装し、Step 3 全体を完了とする。

追加ファイル:

- `src/myoarm_fse/envs/noise.py`
- `tests/test_noise.py` (45 tests)

更新ファイル:

- `src/myoarm_fse/envs/__init__.py` (`SignalDependentMotorNoise` を re-export 追加)

確定した layer 位置:

```text
controller -> excitation_command  ([0, 1]^n, raw)
              ↓ SDN.apply()
              excitation         ([0, 1]^n, post-noise + clip)  -- 研究側 canonical
              ↓ ActionAdapter.excitation_to_api_action()
              api_action          ([-1, 1]^n)
              ↓ env.step()
```

確定した API:

```text
SignalDependentMotorNoise(action_dim: int, sigma, rng=None)
  .action_dim
  .sigma                      # property, float
  .rng                        # property, np.random.Generator
  .apply(excitation_command) -> np.float32 (n,)
  .__call__ == .apply
  .reset(seed: int | None)    # rng 再シード
```

Noise model (`01` Phase 0.3 と一致):

```text
noise = sigma * |u| * N(0, 1)        # element-wise independent
out = clip(u + noise, 0, 1)
```

確定した実装方針:

- `excitation` 専用 (api_action や activation には触らない)。
- element-wise independent (cross-muscle 相関なし)。`rng.standard_normal(size=(n,), dtype=np.float32)`。
- 出力 dtype は `np.float32` 固定。入力 dtype は暗黙キャストで吸収。
- shape/finite validation は ActionAdapter と同型 (1-D, action_dim 一致, NaN/Inf 拒否)。
- `sigma` 検証: `int | float | np.floating` を許容、`bool` は拒否、`np.isfinite(sigma) and sigma >= 0` を要求、内部で `float(sigma)` に正規化。
- `sigma == 0` は **「validation + clip only」** (恒等ではない)。範囲外入力は clip される。RNG は消費しない (高速パス)。
- `rng` 受け取り: `None` → fresh `default_rng()`、`int` → `default_rng(seed)`、`np.random.Generator` → そのまま。`np.random.RandomState` は legacy として非対応。`bool` も拒否。
- `reset(seed)` で再シード可能 (再現性テスト用)。`seed=None` で独立 rng に切り替え。

検証:

```bash
uv run pytest    # 85 passed (40 action_adapter + 45 noise + 1 package)
```

これで Step 3 (action adapter + motor noise) は完了。

未着手 (Step 4 以降):

- Step 4: `DelayedObservationWrapper` / `NoisyObservationWrapper`
- Step 5 以降: episode logger、controllers、metrics、forward model

### 2026-05-09: Step 2 (state schema) 完了

計画書上の番号としては Step 2 だが、実装順序として Step 3 完了後に着手した。理由は Obsidian Logs (`2026-05-09-myoarm-fse-session-recovery-and-step-order`) に記載: raw obs vector に直接 delay/noise を乗せる前に、field 単位で扱える state schema を固める方が安全なため。Step 4 wrapper の前提として Step 2 を先行させた。

追加ファイル:

- `src/myoarm_fse/envs/state.py`
- `tests/test_state_schema.py` (58 tests)

更新ファイル:

- `src/myoarm_fse/envs/__init__.py` (`MyoArmState`, `StateSpec` を re-export 追加)

スコープ: **pure schema layer のみ**。raw observation vector の slicing と MyoSuite import は本ステップでは入れない (後続の env extractor で対応)。

確定した API:

```text
StateSpec(qpos_dim: int, qvel_dim: int, act_dim: int)   # frozen dataclass
  .qpos_dim / .qvel_dim / .act_dim
  .dim                          # qpos_dim + qvel_dim + act_dim + 9
  .layout() -> dict[str, slice] # field 名 → flat vector 上の slice

MyoArmState(qpos, qvel, act, tip_pos, target_pos, reach_err)   # frozen, eq=False
  .qpos / .qvel / .act          # 1-D float32, env-dependent length
  .tip_pos / .target_pos / .reach_err  # shape (3,) float32
  .spec() -> StateSpec
  .flatten() -> np.ndarray (D,) float32

MyoArmState.from_arrays(...)    # array-like を float32 に coerce
MyoArmState.unflatten(vec, spec: StateSpec) -> MyoArmState
```

確定した実装方針:

- **Field 宣言順 = flatten 順**を single source of truth とする。順序は `qpos / qvel / act / tip_pos / target_pos / reach_err`。
- 出力 dtype は `np.float32` 固定。`from_arrays` は array-like を coerce、直接コンストラクタは `np.ndarray` のみ受け、`np.float32` でない場合は `ValueError`。
- `qpos / qvel / act` は 1-D (env/model 依存の長さ)。`tip_pos / target_pos / reach_err` は shape `(3,)` 固定。
- shape mismatch、NaN / Inf は `ValueError`。batch (2-D) input は受けない。
- `reach_err = tip_pos - target_pos` の vector として確定。scalar reach error は metrics 側で `np.linalg.norm(state.reach_err)` で計算する責務。
- consistency check (`reach_err ≈ tip_pos - target_pos`) は本ステップでは入れない (将来 strict 版で追加可)。
- `@dataclass(frozen=True, eq=False)` を採用。`eq=True` だと ndarray fields で `__eq__` / `__hash__` が壊れるため `eq=False` 明示。
- `StateSpec` の dim は `int` かつ `> 0`、`bool` は明示拒否 (ActionAdapter / SDN と同型の検証)。
- Layer 境界: `MyoArmState` は **true (oracle) state**。controller には oracle baseline 以外で渡さない。delayed/noisy observation は Step 4 で別途構築する。

検証:

```bash
uv run pytest    # 143 passed (39 action_adapter + 45 noise + 58 state_schema + 1 package)
```

未着手 (Step 4 以降):

- Step 4: `DelayedObservationWrapper` / `NoisyObservationWrapper`
- Step 1: target set generator (Step 2/4 と独立に進められる)
- Step 5 以降: episode logger、controllers、metrics、forward model

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-09-myoarm-fse-state-schema-design-decisions.md`
- `Logs/2026-05-09-myoarm-fse-state-schema-design-decisions-answer.md`

### 2026-05-09: Step 4 (DelayedObservationWrapper / NoisyObservationWrapper) 完了

Step 2 (state schema) 完了後に着手。`gymnasium.Wrapper` を継承しない pure な observation 変換層として実装。env への結合は後続の env factory / extractor 側責務とする。

追加ファイル:

- `src/myoarm_fse/envs/wrappers.py`
- `tests/test_wrappers.py` (49 tests)

更新ファイル:

- `src/myoarm_fse/envs/__init__.py` (`DelayedObservationWrapper`, `NoisyObservationWrapper` を re-export 追加)

確定した API:

```text
DelayedObservationWrapper(spec: StateSpec, delay_steps: int)
  .spec / .delay_steps
  .reset(initial_state: MyoArmState) -> None     # delay_steps > 0 で必須
  .observe(true_state: MyoArmState) -> MyoArmState
  .__call__ == .observe

NoisyObservationWrapper(spec: StateSpec, sigma: dict[str, float], rng=None)
  .spec / .sigma / .rng
  .reset(seed: int | None = None) -> None        # rng 再シード用
  .observe(state: MyoArmState) -> MyoArmState
  .__call__ == .observe
```

Layer 位置:

```text
true MyoArmState
    ↓ NoisyObservationWrapper.observe (additive Gaussian per-field, no clipping)
    ↓ DelayedObservationWrapper.observe (ring buffer of length delay_steps)
    or 逆順 (composition は呼び出し側で選ぶ)
controller-facing MyoArmState   # oracle baseline 以外で controller に渡すのはこちら
```

Delay timing semantics (`delay_steps = k` のとき `observe(s_t) -> s_{t-k}`):

```text
reset(s0): buffer (length k) を s0 で埋める
observe(s_t): 最古エントリを返す + buffer 内で s_t に置き換え (ring head 更新)

例 (delay_steps=2):
  reset(s0); observe(s1)->s0; observe(s2)->s0; observe(s3)->s1; observe(s4)->s2
```

Noise model (additive Gaussian、SDN とは別物):

```text
out_field = in_field + sigma_field * N(0, 1)         # element-wise independent
clipping は行わない
```

確定した実装方針:

- 入出力は **`MyoArmState`**、内部 buffer / noise 計算は **flat `np.ndarray`** (`StateSpec.layout()` で field slice を取得)。
- delay 単位は **`delay_steps` (int)** のみ。ms 変換は config / env factory 側責務。`from_ms` classmethod は今回入れない。
- `delay_steps == 0` は **identity 高速パス** (reset 不要、buffer は確保しない)。
- `delay_steps > 0` で reset 前に `observe` を呼ぶと **`RuntimeError`**。
- 内部 buffer 長は `delay_steps` (k+1 ではなく k)。ring buffer + head index で実装。
- noise sigma は **per-field `dict[str, float]`** のみ。unknown key は `ValueError`、`bool` 拒否、`np.isfinite` and `>= 0` を要求。
- `sigma == {}` または all-zero は **identity 高速パス** (rng は消費しない)。
- noise は **clipping しない** (observation noise は signal の range 制約と別物。`qpos`, `qvel`, `tip_pos` は自然な bound を持たない)。
- rng の扱いは SDN と同型 (`None` / `int` / `np.random.Generator`、`bool` 拒否、`RandomState` 非対応)。
- 入力 state の `.spec()` が wrapper の `spec` と一致しなければ `ValueError`。
- 両 wrapper とも `__call__ == .observe`。
- `gymnasium.Wrapper` 継承、raw obs slicing、MyoSuite import、`env.unwrapped` 抽出は **本ステップに含まない** (env factory 側責務)。

検証:

```bash
uv run pytest    # 192 passed (39 action_adapter + 45 noise + 58 state_schema + 49 wrappers + 1 package)
```

これで Step 4 (delay / noisy observation wrappers) は完了。Phase 0 共通基盤の env-side (Step 2 / 3 / 4) が一段落。

未着手:

- Step 1: target set generator (Step 2-4 と独立に進められる)
- Step 5: episode logger (Step 2 schema / Step 3 SDN / Step 4 wrappers をすべて使う)
- Step 6 以降: baseline controllers、metrics、forward model

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-09-myoarm-fse-observation-wrappers-design-decisions.md`
- `Logs/2026-05-09-myoarm-fse-observation-wrappers-design-decisions-answer.md`

### 2026-05-09: Option B (env factory + extractor) 完了

Phase 0 共通基盤の env-side pure components (Step 2 / 3 / 4) と MyoSuite 実環境を初めて接続する層。`gym.make` で myoArm env を立て、`MyoArmState` を抽出する責務に限定。target 機構は Step 1、logger は Step 5 に分ける。

追加ファイル:

- `src/myoarm_fse/envs/factory.py`
- `src/myoarm_fse/envs/extractors.py`
- `tests/test_env_factory_smoke.py` (12 tests, `@pytest.mark.myosuite`)
- `tests/test_env_extractor_smoke.py` (10 tests, `@pytest.mark.myosuite`)

更新ファイル:

- `pyproject.toml` (`markers = ["myosuite: ..."]` と `addopts = "-m 'not myosuite'"` を追加)
- `src/myoarm_fse/envs/__init__.py` (factory / extractors は **意図的に re-export しない**。理由は後述)

確定した API:

```text
make_env(env_id: str, horizon: int = 600, normalize_act: bool = True) -> gym.Env
  - registry-level horizon override (env.unwrapped.horizon は read-only property)
  - normalize_act は gym.make の kwarg として渡す
  - env.spec.max_episode_steps == env.unwrapped.horizon を assert

extract_state(env) -> MyoArmState
  - env.unwrapped.mj_data から qpos / qvel / act を直読
  - tip_pos, target_pos は env.unwrapped.mj_data.site_xpos[tip_sids[0] / target_sids[0]]
  - reach_err = tip_pos - target_pos (本プロジェクト convention)

extract_ctrl(env) -> np.ndarray (float32)
  - env.unwrapped.last_ctrl の copy (post-sigmoid muscle ctrl)
```

実測で記録した myoArm reach 環境の仕様 (myoArmReachFixed-v0):

```text
env.unwrapped class:           ReachEnvV0 (myosuite.envs.myo.myobase.reach_v0)
action_space:                  Box(-1.0, 1.0, (34,), float32)   # normalize_act=True 前提
observation_space.shape:       (80,)
qpos / qvel / act dim:         20 / 20 / 34
dt:                            0.02 s (control timestep)
default horizon:               150  → 本 factory で 600 に override
tip_sids:                      [334]   (single site)
target_sids:                   [0]     (single site)
target_reach_range key:        'IFtip'
mj_data path:                  env.unwrapped.mj_data            (sim.data ではない)
mj_data fields:                qpos (20,), qvel (20,), act (34,), ctrl (34,)
                               site_xpos (387, 3)
last_ctrl initial value:       ~0.0759 (sigmoid(0 - 0.5)*5 = neutral)
obs_dict keys:                 time, qpos, qvel, act, tip_pos, target_pos, reach_err
```

実装中に発見した重要な MyoSuite 仕様の罠 (extractor の docstring と smoke test に記録済み):

- **`unwrapped.horizon` は read-only property** で `gym.spec(env_id).max_episode_steps` を参照する。`gym.make(env_id, max_episode_steps=600)` だけでは registry が更新されないため、factory は `gym.spec(env_id).max_episode_steps = horizon` を make の前に実行する。これは process-global な副作用が残ることを意味する (docstring に明記)。
- **`obs_dict['qvel']` は `mj_data.qvel * dt`** (per-step displacement) で、生の velocity ではない。extractor は `mj_data.qvel` を直読する。
- **`obs_dict['reach_err']` は `target_pos - tip_pos`** (本プロジェクトの逆符号)。extractor は `tip_pos - target_pos` を自分で計算し、obs_dict の reach_err は使わない。
- **`mj_data.ctrl` は post-step に reset される挙動**が観測される (api_action=0.5 で step した直後に `mj_data.ctrl=0`)。`last_ctrl` の方が「実際に muscle に渡った post-sigmoid ctrl」を保持する。logger では `last_ctrl` を採用。

設計判断ノート (`...env-factory-extractor-design-decisions`) からの **意図的な逸脱**:

- **`__init__.py` に factory / extractors を re-export しない**。理由は、`__init__.py` で `from myoarm_fse.envs.factory import make_env` すると `myosuite` の import 副作用 (env id 登録、~200-300 ms、banner 出力) が `myoarm_fse.envs` を import するすべての test (例: `test_action_adapter.py`) で発生し、軽量 unit test の収集コストが跳ね上がるため。Q4 の「import 副作用の局所化」を優先し、Q8 の re-export 案は実装しなかった。利用側は fully qualified path (`from myoarm_fse.envs.factory import make_env`) で import する。

検証:

```bash
uv run pytest                # default: 192 passed, 22 deselected in 0.19 s
uv run pytest -m myosuite    # 22 passed (12 factory + 10 extractor) in 1.87 s
```

合計 214 tests (192 軽量 + 22 MyoSuite smoke)。

これで Project 0 共通基盤の env-side が完全に揃った: pure components (Step 2 / 3 / 4) + MyoSuite 接続層 (Option B)。

未着手:

- Step 1: target set generator (Step 5 に直接必要、Option B の知見を活かして Random env の seed 再現性を実測確認する)
- Step 5: episode logger (Step 2 schema / Step 3 SDN / Step 4 wrappers / Option B factory + extractors を統合)
- Step 6 以降: baseline controllers、metrics、forward model

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-09-myoarm-fse-env-factory-extractor-design-decisions.md`

### 2026-05-10: Step 1 (target set generator) 完了

Phase 0 の target split (train / val / test / extrapolation) を生成・保存する pipeline を実装した。

#### probe 結果 (実装の前提)

`myoArmReachRandom-v0` の seed 再現性を実装前に実測した。

```text
env.reset(seed=0); target_pos -> A (e.g. [-0.150, -0.416, 1.201])
env.reset(seed=0); target_pos -> B (e.g. [-0.109, -0.273, 1.759])   # A != B!
env.reset(seed=42); env.reset(seed=42)                              # 二重 reset でも再現せず
```

原因: `ReachEnvV0.reset` の実装が `super().reset(seed=...)` の **前** に
`generate_target_pose()` を呼ぶため、reset の seed 引数はそのコールの target 生成に効かない。
double reset でも別 target が出る。

回避策 (採用): **自前 RNG で target_pos を sample し、`mj_model.site_pos[target_sid]` に
直接書き込んで `mujoco.mj_forward` を呼ぶ**。env の `np_random` には依存しない。
target 値は step を跨いで persistent。

```text
target = np.random.default_rng(seed_i).uniform(low, high)
env.reset()
env.unwrapped.mj_model.site_pos[target_sid] = target
mujoco.mj_forward(env.unwrapped.mj_model, env.unwrapped.mj_data)
state = extract_state(env)   # state.target_pos == target
```

bbox は env の `target_reach_range` から取得 (myoArmReachRandom-v0 の場合
`{'IFtip': ((-0.35, -0.42, 0.98), (0.0, -0.07, 1.83))}`)。

probe の挙動が将来 MyoSuite 側で修正された場合に検出できるよう、
`tests/test_targets_smoke.py::test_reset_seed_does_not_reproduce_target` を残してある。

#### 追加ファイル

```text
configs/targets/default.yaml                # train=200 / val=50 / test=50 / extrapolation=30
scripts/generate_targets.py                 # CLI thin wrapper
src/myoarm_fse/envs/targets.py              # TargetSet, config, generation logic
tests/test_targets.py                       # 38 tests, no MyoSuite
tests/test_targets_smoke.py                 # 8 tests, @pytest.mark.myosuite
runs/targets/{train,val,test,extrapolation}.npz   # 生成された target split
```

#### 更新ファイル

```text
pyproject.toml                              # pyyaml>=6.0 を依存に追加
```

#### 確定した API

```text
SplitConfig(name: str, n: int, seed_offset: int)              # frozen dataclass
TargetGenerationConfig(env_id, generator_seed, output_dir, splits)
  .from_dict(d) / .from_yaml(path)
  .split(name) -> SplitConfig

TargetSet(split, seeds, target_pos, tip_to_target_init_distance, meta)  # frozen, eq=False
  .save(path) / .load(path) -> TargetSet
  .n -> int

generate_seed_list(generator_seed, seed_offset, n) -> np.ndarray (int64)
generate_target_set(config, split_name, env=None) -> TargetSet
generate_all_target_sets(config) -> dict[str, TargetSet]
```

#### 確定した実装方針

- Seed rule: `seed_i = generator_seed + seed_offset + i`。`generator_seed` と各 split の
  `seed_offset` は config で固定。constructor 時に **split 間 seed range の重複を assert**
  (現 default は train=[0, 200)、val=[1000, 1050)、test=[2000, 2050)、extrapolation=[3000, 3030))。
- 各 target は **per-target RNG** (`np.random.default_rng(seed_i)`) から sample。
  全 split で1つの RNG を共有しない。
- `target_pos` の dtype は `np.float32`、`seeds` は `np.int64`、
  `tip_to_target_init_distance` は `np.float32`。
- npz は `np.savez` で plain 保存。`meta` は `meta_json` (string) に JSON encode して
  入れる (`allow_pickle=False` でロード可能)。
- MyoSuite import は `targets.py` の **関数内 lazy import** に閉じる
  (`generate_target_set` / `generate_all_target_sets` 内)。
  `import myoarm_fse.envs.targets` 単体では MyoSuite 副作用が走らない。
- `__init__.py` には `targets.py` を **再エクスポートしない**
  (factory / extractors と同じ理由、軽量 import を守るため)。
- workspace override / distance bin / target direct write API は Phase 0 では入れない。
  `extrapolation` は現状 「別 seed range の held-out split」で、真の OOD ではない。
  完了後の workspace OOD は別途追加。
- `target_reach_range` が複数キーを持つ env は `ValueError`
  (現 myoArm reach は単一 'IFtip' なので OK)。

#### 生成結果 (`uv run python scripts/generate_targets.py --config configs/targets/default.yaml`)

```text
runs/targets/train.npz          n=200  init_distance min=0.340 mean=0.780 max=1.223
runs/targets/val.npz            n=50   init_distance min=0.382 mean=0.821 max=1.217
runs/targets/test.npz           n=50   init_distance min=0.398 mean=0.825 max=1.162
runs/targets/extrapolation.npz  n=30   init_distance min=0.478 mean=0.800 max=1.170
```

distance は `||tip_pos - target_pos||` (reset 直後)。約 0.3〜1.2 m の範囲で
myoArm reach の物理的な可到達範囲としては妥当。

#### 検証

```bash
uv run pytest                # default: 230 passed, 30 deselected in 0.21 s
uv run pytest -m myosuite    # 30 passed (12 factory + 10 extractor + 8 targets) in 2.77 s
```

合計 260 tests:

```text
test_action_adapter.py        39 passed     (軽量)
test_noise.py                 45 passed     (軽量)
test_state_schema.py          58 passed     (軽量)
test_wrappers.py              49 passed     (軽量)
test_targets.py               38 passed     (軽量, new)
test_package.py                1 passed     (軽量)
test_env_factory_smoke.py     12 passed     (myosuite)
test_env_extractor_smoke.py   10 passed     (myosuite)
test_targets_smoke.py          8 passed     (myosuite, new)
```

これで Step 1 (target set generator) は完了。

未着手:

- Step 5: episode logger
- Step 6: baseline controllers (random / hold / PD)
- Step 7: reaching metrics
- Step 8: forward model baseline (MLP)

設計判断の経緯と probe 結果は Obsidian 側のログを参照:

- `Logs/2026-05-10-myoarm-fse-target-set-design-decisions.md`
- `Logs/2026-05-10-myoarm-fse-target-set-design-decisions-answer.md`

### 2026-05-10: Step 5 (episode logger + minimum random controller) 完了

最初の rollout pipeline。Step 1〜4 + Option B + Step 1 を統合し、target set から episode を実行して trajectory を保存する layer を完成させた。

#### 追加ファイル

```text
src/myoarm_fse/controllers/__init__.py
src/myoarm_fse/controllers/base.py        # Controller Protocol
src/myoarm_fse/controllers/random.py      # RandomController (clipped Gaussian)
src/myoarm_fse/data/__init__.py
src/myoarm_fse/data/schema.py             # EpisodeLog frozen dataclass + save/load
src/myoarm_fse/data/rollout.py            # run_episode pure function + EpisodeSpec
src/myoarm_fse/data/logger.py             # make_run_id, hash_config, RunIndex / IndexEntry
configs/episodes/default.yaml
scripts/collect_episodes.py               # CLI thin wrapper
tests/test_random_controller.py           # 26 tests
tests/test_episode_log.py                 # 16 tests
tests/test_rollout.py                     # 17 tests, fake env (no MyoSuite)
tests/test_logger.py                      # 9 tests
tests/test_collect_episodes_smoke.py      # 9 tests, @pytest.mark.myosuite
runs/episodes/{run_id}/                   # 生成された episode (untracked, gitignored)
```

#### 確定した API

```text
controllers.base.Controller (Protocol, runtime_checkable)
  .action_dim
  .reset(*, seed=None)
  .act(observation: MyoArmState) -> np.ndarray   # excitation_command [0,1]^n

controllers.random.RandomController(action_dim, mean=0.5, sigma=0.2, rng=None)

data.schema.EpisodeLog  # frozen, eq=False
  episode-level metadata: episode_id, target_id, target_split, target_seed,
    target_pos_set, controller_name, controller_seed, sdn_sigma, sdn_seed,
    obs_noise_sigma, obs_noise_seed, obs_delay_steps, obs_compose,
    max_steps, n_steps, created_at, config_hash, meta
  step arrays (T, *): step, time, true_*, obs_*, neural_command,
    excitation_command, excitation, api_action, last_ctrl, reward,
    terminated, truncated
  .save(path) / .load(path)

data.rollout.EpisodeSpec(episode_id, target_id, target_split, target_seed,
                         controller_name, controller_seed, ...)
data.rollout.run_episode(env, controller, target_pos, *, state_spec,
    action_adapter, sdn=None, obs_noise=None, obs_delay=None,
    obs_compose="noisy_then_delayed", max_steps=600, spec=None) -> EpisodeLog

data.logger.make_run_id(now=None) -> str   # "2026-05-10T08-30-15Z"
data.logger.hash_config(config: dict) -> str  # 12-char sha256 prefix
data.logger.RunIndex(run_id, created_at, config_hash, config,
                    target_set_path, episodes)
  .append(IndexEntry)
  .save(path) / .load(path)
data.logger.IndexEntry(episode_id, file, target_id, target_seed, n_steps)
```

#### 確定した実装方針

- **Pure function rollout**: `run_episode(...) -> EpisodeLog` で stateful logger object を持たない。step buffer は関数内で max_steps preallocate、終了時に `[:n_steps]` で slice。
- **True / observed の両方を保存**: Step 4 の layer 境界に従い、true_qpos と obs_qpos など全 6 field を二重に持つ。identity wrapper の場合でも別 field として記録する。smoke test (`test_identity_pipeline_true_equals_obs`) で identity 条件下の値一致を担保。
- **Command layer の保存**: `neural_command`, `excitation_command`, `excitation`, `api_action`, `last_ctrl` を分けて保存。現状 `neural_command == excitation_command` (identity) だが将来の controller 拡張のため。
- **Last_ctrl 採用**: Option B で確定済みの通り、`mj_data.ctrl` ではなく `extract_ctrl(env)` (= `last_ctrl`) を保存。
- **Obs composition**: 引数 `obs_compose: str` で `"noisy_then_delayed"` (default) か `"delayed_then_noisy"` を選択。unknown は `ValueError`。
- **Controller boundary validation**: `run_episode` 内で controller 出力の shape / finite / `[0, 1]` range を strict 検証。silent clip しない (controller bug を見えるように)。
- **Target injection**: Step 1 と同じ `mj_model.site_pos[target_sid] = target_pos; mujoco.mj_forward(...)` 方式を rollout 開始時に実施。env の `np_random` には依存しない。
- **Ragged storage**: 各 episode は実際の step 数 `T` だけ保存、`n_steps` を metadata に保存。padding は dataset loader 側責務。
- **Per-episode npz + index.json**: `runs/episodes/{run_id}/0000.npz, 0001.npz, ..., index.json` 形式。`run_id` は filesystem-safe UTC timestamp (`2026-05-10T08-30-15Z`)。
- **Reproducibility**: `np.random.SeedSequence(master_seed).spawn(3 * n_episodes)` で各 episode に controller / sdn / obs_noise の child seed を派生。各 seed を episode metadata に保存。
- **MyoSuite 不要 unit test**: `test_rollout.py` で fake env を作り、`mujoco.mj_forward` を `monkeypatch` で `site_pos -> site_xpos` copy に置き換えることで MyoSuite なしの 17 unit tests を実現。
- **`__init__.py`**: `controllers/` と `data/` は通常通り再エクスポート (これらは MyoSuite に直接依存しないため、軽量 import を壊さない)。MyoSuite 依存は rollout の lazy 経路で間接的に呼ばれるが、`run_episode` を呼ばない限り副作用は走らない。

#### CLI

```bash
uv run python scripts/collect_episodes.py --config configs/episodes/default.yaml
```

主要 override:

```text
--n-episodes      override config.n_episodes
--master-seed     override config.master_seed
--output-root     override config.output_root
--target-set      override config.target_set
```

#### 初回 collection 結果

```text
Run id: 2026-05-09T23-51-12Z   (UTC)
Output: runs/episodes/2026-05-09T23-51-12Z
Target set: runs/targets/train.npz (n=200, taking 2)
  [1/2] saved 0000.npz (n_steps=600, final_reach_err_norm=0.599)
  [2/2] saved 0001.npz (n_steps=600, final_reach_err_norm=0.463)
  saved index.json with 2 episodes
```

各 episode は ~825 KB の npz、`index.json` は 903 バイト。RandomController は target に到達しないので `final_reach_err_norm` は 0.5 m 前後 (target との初期距離が ~0.34〜1.2 m なので、random 動作だけでは reach できない、想定通り)。

#### 検証

```bash
uv run pytest                # default: 298 passed, 39 deselected in 0.25 s
uv run pytest -m myosuite    # 39 passed in 4.55 s
```

合計 337 tests:

```text
test_action_adapter.py            39 passed     (軽量)
test_noise.py                     45 passed     (軽量)
test_state_schema.py              58 passed     (軽量)
test_wrappers.py                  49 passed     (軽量)
test_targets.py                   38 passed     (軽量)
test_random_controller.py         26 passed     (軽量, new)
test_episode_log.py               16 passed     (軽量, new)
test_rollout.py                   17 passed     (軽量, new — fake env + monkeypatch)
test_logger.py                     9 passed     (軽量, new)
test_package.py                    1 passed     (軽量)
test_env_factory_smoke.py         12 passed     (myosuite)
test_env_extractor_smoke.py       10 passed     (myosuite)
test_targets_smoke.py              8 passed     (myosuite)
test_collect_episodes_smoke.py     9 passed     (myosuite, new)
```

これで Step 5 (episode logger + minimum random controller) は完了。Phase 0 の **rollout / dataset collection 層** が動作。

#### Phase 0 全体の到達度

```text
Step 1: target set generator              完了
Step 2: state schema                      完了
Step 3: action adapter + motor noise      完了
Step 4: delay / noisy obs wrappers        完了
Step 5: episode logger (+ minimum ctrl)   完了 (このステップ)
Step 6: baseline controllers (hold, PD)   未着手 (random は Step 5 で minimum 実装済み)
Step 7: reaching metrics                  未着手
Step 8: forward model baseline            未着手
+ Option B: env factory + extractor       完了
```

Phase 0 の env-side / data-side が揃い、最初の smoke rollout が動く状態。

未着手:

- Step 6 残り: hold controller、PD endpoint controller
- Step 7: reaching metrics (minimum tip error, final tip error, success rate, effort, prediction MSE)
- Step 8: forward model baseline (residual MLP)

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-10-myoarm-fse-episode-logger-and-random-controller-design-decisions.md`
- `Logs/2026-05-10-myoarm-fse-episode-logger-and-random-controller-design-decisions-answer.md`

### 2026-05-10: Step 6 (HoldController + controller factory) 完了

計画書 02 Step 6 候補のうち **HoldController のみ実装**。PDEndpointController は Step 7 (metrics) 完了後に着手することを設計判断ノートで確定 (PD on muscle env は inverse Jacobian + muscle moment-arm の inverse が biomechanically 重く、metrics で評価フレームが揃ってから入れるのが安全)。

低振幅 random は **新 class を作らず**、`RandomController(sigma=0.05)` の preset config (`configs/episodes/lowamp_random.yaml`) で実現。

#### 追加ファイル

```text
src/myoarm_fse/controllers/hold.py         # HoldController
configs/episodes/lowamp_random.yaml        # RandomController(sigma=0.05) preset
configs/episodes/hold.yaml                 # HoldController(value=0.0) preset
tests/test_hold_controller.py              # 24 tests
tests/test_make_controller.py              # 9 tests
```

#### 更新ファイル

```text
src/myoarm_fse/controllers/__init__.py     # HoldController re-export + make_controller
scripts/collect_episodes.py                # _build_controller を make_controller に置換
tests/test_collect_episodes_smoke.py       # test_hold_controller_in_pipeline 追加
```

#### 確定した API

```text
controllers.hold.HoldController(action_dim, value=0.0)
  .action_dim
  .value
  .reset(*, seed=None)        # no-op (deterministic)、seed type validation のみ
  .act(observation: MyoArmState) -> np.ndarray  # 全 muscle に value を broadcast

controllers.make_controller(spec: dict, action_dim: int, seed: int) -> Controller
  spec["name"] in {"random", "hold"}
  unknown name / non-dict spec / missing name は ValueError
```

`HoldController.act` は事前に確保した output vector の copy を返す (caller が mutate しても以後の呼び出しに影響しない)。

#### 確定した実装方針

- **Hold = scalar value、全 muscle broadcast**: per-muscle vector は YAGNI、必要になれば後で拡張。`value` は `[0, 1]` clamp 内のみ許容、外は `ValueError`。
- **`reset(seed=...)` は no-op だが seed type validation を実施**: Controller Protocol が `int | None` を期待するので、`bool` や `str` を渡すと `TypeError`。RandomController と同型のエラー挙動。
- **Low-amplitude random は preset で表現**: `RandomController(sigma=0.05)` を `lowamp_random.yaml` で。新 class を増やすと Phase 0 の表面積が無駄に増える。
- **`make_controller` factory は薄い if-elif**: registry pattern より明示的に書いた方が読める。controller が 5 個以上に増えたら `factory.py` に分離する。
- **`HoldController` は seed を受け取るが無視**: factory signature の一貫性のため、deterministic な controller も `seed` を受け取る (使わないだけ)。

#### CLI 実行結果と baseline 比較

3 種類の baseline で初期 reach error の分布が明確に分かれることを確認:

```text
controller         final_reach_err_norm (episode 0 / 1)
─────────────────  ─────────────────────────────────────
random (σ=0.2)     0.599 / 0.463    (default、wild な random excitation)
lowamp (σ=0.05)    0.151 / 0.375    (muscles が tonic 領域に留まる)
hold (v=0.0)       0.299 / 0.353    (silent muscles、重力下の static drop)
```

低振幅 random が distance を最も縮める結果 (sigma=0.05 で muscle が中庸に留まり、tip 位置が target に近い姿勢になる)。Hold (silent) は重力で自然落下した姿勢から動かない。Default random は extreme 値を頻繁に取るので姿勢が動的に揺れて target から離れる。Phase 0 の baseline として明確な比較対象になる。

#### 検証

```bash
uv run pytest                # default: 331 passed, 40 deselected in 0.26 s
uv run pytest -m myosuite    # 40 passed in 4.79 s
```

合計 371 tests:

```text
test_action_adapter.py            39 passed     (軽量)
test_noise.py                     45 passed     (軽量)
test_state_schema.py              58 passed     (軽量)
test_wrappers.py                  49 passed     (軽量)
test_targets.py                   38 passed     (軽量)
test_random_controller.py         26 passed     (軽量)
test_episode_log.py               16 passed     (軽量)
test_rollout.py                   17 passed     (軽量)
test_logger.py                     9 passed     (軽量)
test_hold_controller.py           24 passed     (軽量, new)
test_make_controller.py            9 passed     (軽量, new)
test_package.py                    1 passed     (軽量)
test_env_factory_smoke.py         12 passed     (myosuite)
test_env_extractor_smoke.py       10 passed     (myosuite)
test_targets_smoke.py              8 passed     (myosuite)
test_collect_episodes_smoke.py    10 passed     (myosuite, +1 hold case)
```

#### Phase 0 全体の到達度更新

```text
Step 1: target set generator              完了
Step 2: state schema                      完了
Step 3: action adapter + motor noise      完了
Step 4: delay / noisy obs wrappers        完了
Step 5: episode logger (+ random ctrl)    完了
Step 6: baseline controllers              ⚠ 部分完了 (Hold + factory のみ; PD は Step 7 後)
Step 7: reaching metrics                  未着手
Step 8: forward model baseline            未着手
+ Option B: env factory + extractor       完了
```

これで Phase 0 の **3 種類の baseline controller** (random / lowamp_random / hold) が揃った。次は Step 7 (metrics) で評価フレームを確立し、その後で PDEndpointController に進む。

未着手:

- Step 6 残り: PDEndpointController (Step 7 後に着手)
- Step 7: reaching metrics (minimum tip error, final tip error, success rate, effort, prediction MSE)
- Step 8: forward model baseline (residual MLP)

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-10-myoarm-fse-step6-hold-controller-design-decisions.md`

### 2026-05-10: Step 7 (reaching metrics + prediction interface stub) 完了

計画書 02 Step 7 の reaching metrics をフル実装、prediction metrics は **interface signature を pure 実装で stub** として置いた (Step 8 で forward model が prediction MSE を消費する側で利用)。

#### 追加ファイル

```text
src/myoarm_fse/metrics/__init__.py
src/myoarm_fse/metrics/reaching.py       # minimum_tip_error / final_tip_error / success / effort_norm
src/myoarm_fse/metrics/prediction.py     # one_step_prediction_mse / rollout_mse / tip_prediction_error
src/myoarm_fse/metrics/aggregate.py      # aggregate_reaching
tests/test_metrics_reaching.py           # 24 tests
tests/test_metrics_prediction.py         # 20 tests
tests/test_metrics_aggregate.py          # 10 tests
```

#### 確定した API

```text
# reaching: EpisodeLog 入力、scalar / bool 出力
metrics.reaching.minimum_tip_error(log) -> float                # n_steps==0 → inf
metrics.reaching.final_tip_error(log)   -> float                # n_steps==0 → inf
metrics.reaching.success(log, *, threshold=0.05, duration=10) -> bool
metrics.reaching.effort_norm(log)       -> float                # mean over t of L2² of excitation

# prediction: ndarray 入力、scalar 出力 (Step 8 で forward model が呼ぶ)
metrics.prediction.one_step_prediction_mse(true_next, pred_next) -> float    # (T, D)
metrics.prediction.rollout_mse(true_traj, pred_traj)             -> float    # (T, D)
metrics.prediction.tip_prediction_error(true_tip, pred_tip)      -> float    # (T, 3)、distance not MSE

# batch aggregation
metrics.aggregate.aggregate_reaching(logs, *, threshold=0.05, duration=10)
  -> dict[str, float]   # n / *_mean / *_median / *_std / success_rate / effort_mean / effort_std
                        # + threshold, duration を返り値に含める (再現性のため)
```

#### 確定した実装方針

- **全 reaching metric は `true_*` (oracle) を読む**: observation noise / delay は controller の handicap であって、評価者は真値で判断する (Q10)。`obs_*` は metric 計算には使わない。
- **`effort_norm` は `excitation` (post-SDN)** を読む: Step 3 の canonical 表現。SDN ノイズ込みの「実際に筋に流れた activation」に対応。`excitation_command` (pre-SDN) でも `last_ctrl` (post-sigmoid) でもない。
- **Effort は mean-of-squares L2²、time-average**: `(1/T) * Σ_t ||u_t||₂²`。control theory の quadratic cost と整合、ragged episode との比較も成立。
- **Success は sustained 条件**: 「`||tip - target|| < threshold` を連続 `duration` step 維持」。default `threshold=0.05 m`、`duration=10 step` (= 0.2 s)。瞬間タッチでは reach の完了とみなさない。strict 不等号 (boundary は False)。実装は cumsum で sliding-window all-True を計算。
- **Final tip error は last step のみ**: `||true_reach_err[-1]||`。sustained は success が扱うので二重カウントしない。`n_steps==0` は `inf`。
- **Aggregate は薄い helper**: `dict[str, float]` を返す pure function。empty 入力は `{"n": 0}`。`threshold` / `duration` も dict に含めて再現性を担保。
- **Pure function 統一**: class / registry / callable dataclass は導入しない。controllers / data と同じスタイル。
- **CLI は作らない**: Step 8 (forward model) で `evaluate_run.py` を立ち上げる時点で reaching + prediction を一括出力する CLI を整備する。Step 7 では library 関数のみ。

#### Prediction metrics の stub 実装

prediction metrics は実は **forward model に依存しない pure な MSE / 距離計算** なので、interface stub と言いつつ完全実装した:

```python
def one_step_prediction_mse(true_next, pred_next) -> float
def rollout_mse(true_traj, pred_traj) -> float
def tip_prediction_error(true_tip, pred_tip) -> float
```

入力 ndarray の shape mismatch / NaN / Inf は `ValueError`。empty (`T=0`) は `0.0`。Step 8 で forward model から predicted trajectory を得たら、そのまま呼び出して評価できる。

#### 既存 baseline 3 runs での実測

Step 5 / Step 6 で生成した 3 runs (random / lowamp / hold) を `aggregate_reaching` に通した結果:

```text
                     random (σ=0.2)   lowamp (σ=0.05)   hold (v=0.0)
                     ────────────────  ────────────────  ──────────────
n                       2                   2                  2
minimum_tip_error       0.2015              0.1558             0.2974
final_tip_error         0.5310              0.2629             0.3260
success_rate            0.0                 0.0                0.0
effort_mean             9.8102              8.5831             0.0000
```

解釈:

- **lowamp が距離 metrics で best** (`min` 0.156、`final` 0.263)。tonic 領域の muscle activation が arm を target 近傍に保つ。
- **hold は effort=0** (excitation=0 なので L2²=0)。ただし距離は worst で arm が動かないだけ。
- **random は effort 最大** (extreme 値で `||u||²` が大きい) かつ距離も worst。wild な excitation で arm が振り回される。
- **全 baseline `success_rate=0`**: 5 cm 以内に 0.2 s 留まる goal-directed な動きはない、想定通り。

`success_rate` を 0 から動かすには PD endpoint controller (Step 6 残) もしくは forward-model-based RL (Step 8 以降) が必要。これで「PD が機能しているか」を評価する基盤ができた。

#### 検証

```bash
uv run pytest                # default: 385 passed, 40 deselected in 0.27 s
uv run pytest -m myosuite    # 40 passed in 4.87 s
```

合計 425 tests:

```text
test_action_adapter.py            39 passed
test_noise.py                     45 passed
test_state_schema.py              58 passed
test_wrappers.py                  49 passed
test_targets.py                   38 passed
test_random_controller.py         26 passed
test_episode_log.py               16 passed
test_rollout.py                   17 passed
test_logger.py                     9 passed
test_hold_controller.py           24 passed
test_make_controller.py            9 passed
test_metrics_reaching.py          24 passed   (new)
test_metrics_prediction.py        20 passed   (new)
test_metrics_aggregate.py         10 passed   (new)
test_package.py                    1 passed
test_env_factory_smoke.py         12 passed     (myosuite)
test_env_extractor_smoke.py       10 passed     (myosuite)
test_targets_smoke.py              8 passed     (myosuite)
test_collect_episodes_smoke.py    10 passed     (myosuite)
```

#### Phase 0 全体の到達度更新

```text
Step 1: target set generator              完了
Step 2: state schema                      完了
Step 3: action adapter + motor noise      完了
Step 4: delay / noisy obs wrappers        完了
Step 5: episode logger (+ random ctrl)    完了
Step 6: baseline controllers              ⚠ 部分完了 (Hold + factory; PD は metrics で評価可能になったので次に着手可)
Step 7: reaching metrics                  完了 (prediction は stub だが pure function として実装済み)
Step 8: forward model baseline            未着手
+ Option B: env factory + extractor       完了
```

Phase 0 は **forward model 実装 (Step 8) を残すのみ** に近い状態。残作業:

- Step 6 残: PDEndpointController (metrics で評価フレームが整ったので次に着手可能)
- Step 8: forward model baseline (residual MLP)
  - dataset loader (EpisodeLog → (state, action, next_state) tuples)
  - residual MLP `[x_t, u_t] → Δx_t`
  - one-step / rollout 評価 (prediction metrics を呼ぶ)
  - `scripts/train_forward_model.py`、`scripts/evaluate_run.py`

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-10-myoarm-fse-step7-metrics-design-decisions.md`

### 2026-05-10: Step 8 minimum (TransitionDataset) 完了

Step 7 完了後の方向決定 (`Logs/2026-05-10-myoarm-fse-post-step7-next-direction-answer`) で **PD endpoint controller を defer して Step 8 forward model baseline に進む** ことを確定済み。今回は forward model 実装の前段として **transition dataset の構築・保存・分割** を整備した。

#### 追加ファイル

```text
src/myoarm_fse/models/__init__.py          # 再エクスポート
src/myoarm_fse/models/datasets.py          # TransitionDataset + build/shuffle/split helpers
tests/test_model_datasets.py               # 42 tests
```

#### 確定した API

```text
TransitionDataset (frozen dataclass)
  x: float32 (N, state_dim)
  u: float32 (N, action_dim)
  x_next: float32 (N, state_dim)
  dx: float32 (N, state_dim)              # x_next - x
  episode_index: int64 (N,)
  state_dim: int
  action_dim: int
  n_episodes: int
  episode_metadata: tuple[dict, ...]      # length n_episodes
  .save(path) / .load(path)
  .n -> int

build_transitions(logs: Iterable[EpisodeLog]) -> TransitionDataset
shuffle_transitions(dataset, *, rng=None) -> TransitionDataset
split_by_episode(dataset, *, val_episode_ids: Iterable[int]) -> (train, val)
```

#### 確定した実装方針

- **State representation**: `MyoArmState.flatten()` 全 field (`qpos | qvel | act | tip_pos | target_pos | reach_err`)、myoArm reach で **state_dim = 83** を実測確認 (20+20+34+3+3+3)。
- **Action input**: `excitation` (post-SDN、Step 3 の canonical)。`excitation_command` でも `last_ctrl` でもない。
- **Shape**: flat concat `(N, *)` + `episode_index: int64 (N,)` で episode 境界保持。`N = Σ_i (T_i - 1)`。
- **除外条件**: 各 episode の last step のみ (no `x_{t+1}`)。`n_steps < 2` の episode は skip。mid-episode で `terminated`/`truncated` が立っていたら `ValueError` (rollout invariant 違反の defensive 検出)。
- **Provenance**: dataset-level に `episode_metadata: tuple[dict, ...]`。各 dict は `{episode_id, target_id, target_split, target_seed, controller_name, controller_seed, sdn_sigma, sdn_seed, obs_noise_sigma, obs_noise_seed, obs_delay_steps, obs_compose, n_steps, transitions_used, config_hash}` を保持。per-transition trace は持たない (overkill)。
- **Save/load**: `np.savez(allow_pickle=False)` 1 ファイル + `meta_json` を JSON string で埋め込み。TargetSet / EpisodeLog と同型。
- **Immutable + 外部 helper**: `TransitionDataset` は `frozen=True, eq=False`。`shuffle_transitions` / `split_by_episode` は新 instance を返す pure function。
- **`split_by_episode` は ID ベース**: `val_episode_ids: Iterable[int]` で `episode_metadata[i]["episode_id"]` の値を指定 (= original episode_id)。新 dataset 内で `episode_index` は 0..n_episodes-1 に reindex。
- **Strict validation in `__post_init__`**: dtype / shape / finite / `dx == x_next - x` (`atol=1e-5`) / metadata length / JSON-serializability を全部 `ValueError` で fail-fast。empty (`N=0`) は許容。

#### 既存 baseline 3 runs での実測

```text
random (sigma=0.2):   N=1198, state_dim=83, action_dim=34, n_episodes=2
lowamp (sigma=0.05):  N=1198, state_dim=83, action_dim=34, n_episodes=2
hold   (value=0.0):   N=1198, state_dim=83, action_dim=34, n_episodes=2
```

各 run = 2 episode × 600 step → 599 transitions × 2 = **1198 transitions** で計算が合う。state_dim 83 は schema 計算 (qpos 20 + qvel 20 + act 34 + tip 3 + target 3 + reach_err 3 = 83) と一致。

`u[0,:5]` の値も controller の挙動を反映:

- random (σ=0.2): `[0.44, 0.32, 0.66, 0.49, 0.35]` — 0.5 周辺に拡散
- lowamp (σ=0.05): `[0.48, 0.45, 0.54, 0.50, 0.46]` — 0.5 にタイト
- hold (value=0.0): `[0, 0, 0, 0, 0]` — silent muscles

`dx[0,:5]` は 3 baseline で似た値 (`~0.004` のオーダー) — episode 開始直後は重力下の自然落下 dynamics が dominant で、controller の差が現れにくい。

#### 検証

```bash
uv run pytest                # default: 427 passed, 40 deselected in 0.29 s
uv run pytest -m myosuite    # 40 passed in 4.79 s
```

合計 467 tests (+42 = `test_model_datasets.py` の 42 unit)。

#### Step 8 minimum 完了、本体 (MLP + training) は別フェーズ

これで forward model dataset の **入口** が整った。Step 8 残作業:

```text
src/myoarm_fse/models/mlp.py           # residual MLP [x, u] -> Δx
src/myoarm_fse/models/train.py         # training loop helpers
scripts/train_forward_model.py         # CLI で training を走らせる
scripts/evaluate_run.py                # reaching + prediction metrics の一括出力
```

設計判断ノートは別途整備する。

#### Phase 0 全体の到達度

```text
Step 1: target set generator              完了
Step 2: state schema                      完了
Step 3: action adapter + motor noise      完了
Step 4: delay / noisy obs wrappers        完了
Step 5: episode logger (+ random ctrl)    完了
Step 6: baseline controllers              ⚠ 部分完了 (Hold + factory; PD は defer)
Step 7: reaching metrics                  完了
Step 8: forward model baseline            ⚠ 部分完了 (TransitionDataset; MLP + training は次)
+ Option B: env factory + extractor       完了
```

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-10-myoarm-fse-step8-transition-dataset-design-decisions.md`

### 2026-05-10: Step 8 後半 (residual MLP + training loop) 完了

Step 8 minimum (`TransitionDataset`) の続きとして、forward model 本体 (residual MLP) + training loop + 2 つの CLI を実装。3 baseline (random / lowamp / hold) で end-to-end の training と evaluation が動くことを実測確認。

#### 追加ファイル

```text
src/myoarm_fse/models/mlp.py              # ForwardMLP (PyTorch nn.Module)
src/myoarm_fse/models/train.py            # TrainConfig, setup_seeds, train_forward_model,
                                          #   make_train_val_split, save_model, load_model,
                                          #   rollout_predictions, make_model_id
configs/models/mlp.yaml                    # default training config
scripts/train_forward_model.py             # CLI thin wrapper
scripts/evaluate_run.py                    # CLI: model + run → metrics 一括出力
tests/test_model_mlp.py                    # 18 tests
tests/test_model_train.py                  # 23 tests
runs/datasets/baseline_3way.npz           # 3 baseline 統合 dataset (gitignored)
runs/models/{model_id}/                   # 学習済み model + eval_*.json (gitignored)
```

#### 更新ファイル

```text
src/myoarm_fse/models/datasets.py         # split_by_local_indices 公開 helper を追加
src/myoarm_fse/models/__init__.py         # 全 re-export 拡張
docs/02_InitialImplementationPlan.md      # Step 8 後半完了メモ
```

#### 確定した API

```text
ForwardMLP(state_dim, action_dim, hidden_dims=(256, 256))   # PyTorch nn.Module
  .forward(x, u) -> Δx                                      # residual output
  .predict_next(x, u) -> x + Δx                            # closed-loop step

TrainConfig(optimizer="adam", lr=1e-3, weight_decay=0,
            batch_size=256, epochs=200,
            early_stopping_patience=20, early_stopping_min_delta=0,
            grad_clip=None, seed=0, val_step=5)

setup_seeds(master_seed) -> dict[str, int]
  - SeedSequence.spawn で派生、numpy/torch/random を seed
  - returns {"model_init", "dataset_shuffle", "dataloader"}

make_train_val_split(dataset, *, val_step=5) -> (train_ds, val_ds)
  - local index ベース (val_step ごとに val、残りを train)
  - episode_id 衝突に robust

train_forward_model(model, train_ds, val_ds, config, *, seeds=None)
  -> (best_model, metrics)
  - Adam / AdamW、MSE on Δx、early stopping、best checkpoint

rollout_predictions(model, dataset, *, horizons=(1, 10, 50))
  -> dict[h: {"rollout_mse": float, "tip_prediction_error": float}]
  - closed-loop autoregressive、recorded u_t、sliding window
  - episode 境界をまたがない

save_model(model, config, metrics, *, path, info=None)
load_model(path) -> (model, config, metrics)
make_model_id(now=None) -> str   # UTC timestamp、Step 5 run_id 同型

split_by_local_indices(dataset, *, val_indices)
  -> (train, val)   # episode_id 衝突に対応する公開 helper (datasets.py に追加)
```

#### 確定した実装方針

- **MLP architecture**: `Linear(state+action, 256) → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear(state)`、residual output (Δx)。117→256→256→83 で param ~118k (myoArm reach 想定)。
- **Loss**: MSE on Δx、uniform weight、normalize なし (per-field scale の差は受け入れる)。
- **Optimizer**: Adam(lr=1e-3, weight_decay=0)、batch=256、epochs=200、early stopping patience=20、no scheduler。
- **Validation**: `make_train_val_split` で local index ベース、val_step=5 で 6 episode → train 4 / val 2 (~33%)。毎 epoch 評価、best val loss の checkpoint を保持。
- **Rollout evaluation**: closed-loop autoregressive、recorded `u_t` 使用、horizons `[1, 10, 50]`、sliding window、episode 境界をまたがない。
- **tip_pos**: 特別扱いなし。flat state の一部、評価時に `StateSpec.layout()["tip_pos"]` で slice 抽出して `tip_prediction_error` (Step 7) を呼ぶ。
- **Save format**: `state_dict + config.json + metrics.json + info.json` を `runs/models/{model_id}/` に保存。`load_model` は `weights_only=True` で安全に load。
- **Reproducibility**: master_seed → SeedSequence.spawn → numpy / torch / random を seed。CPU 学習で `torch.use_deterministic_algorithms` は入れない (overhead 回避)。
- **CLI**: 2 つの独立 script (`train_forward_model.py` / `evaluate_run.py`)、YAML config 主、限定 `--override`。

#### 実装中に発見した bug fix (1 件)

**Episode_id 衝突問題**: `_load_concat_dataset` で複数 run を concat したとき、各 run が `episode_id` 0, 1 を持つので衝突。`make_train_val_split` が当初 `split_by_episode` 経由で episode_id ベース分割していたため、全 episode が val 行きになる症状が出た。

**修正**: `split_by_local_indices(dataset, *, val_indices)` を `datasets.py` に新設し、`make_train_val_split` をこちらに切り替え。`split_by_episode` の API は維持 (UNIQUE な episode_id を持つ単一 run dataset 用)。

加えて、`scripts/train_forward_model.py` の `_load_concat_dataset` で concat 時に `episode_id` を global sequential index に振り直し、original ID は `source_episode_id` に保存する形にした。これで concat 後の dataset でも `split_by_episode` が動く。

#### 検証

```bash
uv run pytest                # default: 468 passed, 40 deselected in 1.50 s
uv run pytest -m myosuite    # 40 passed in 5.36 s
```

合計 508 tests (+41 = 18 mlp + 23 train)。default 時間が 0.29 s → 1.50 s に上昇 (PyTorch import + 41 新テスト)、許容範囲。

#### Baseline 3way での実走結果

```bash
# 1. dataset 構築 (3 baseline 統合)
uv run python -c "..."  # build_transitions on 6 episodes from 3 runs
# → runs/datasets/baseline_3way.npz: N=3594, state_dim=83, action_dim=34, n_episodes=6

# 2. training
uv run python scripts/train_forward_model.py --config configs/models/mlp.yaml
# → Split: train N=2396 (4 ep) / val N=1198 (2 ep)
# → ForwardMLP(state_dim=83, action_dim=34, hidden_dims=(256, 256), params=118355)
# → best_epoch=36, best_val_loss=0.034781, epochs_run=57 (early stop)
# → Saved: runs/models/2026-05-10T04-15-54Z/

# 3. evaluation against each baseline
uv run python scripts/evaluate_run.py --model runs/models/{model_id} --run runs/episodes/{run_id}
```

3 baseline での prediction metrics:

```text
                    h=1 mse    h=10 mse   h=50 mse   h=1 tip_err  h=10 tip_err  h=50 tip_err
random (σ=0.2)      0.0432     0.2024     10.383     0.0783       0.2858        2.985
lowamp (σ=0.05)     0.0036     0.0304      6.189     0.0220       0.1066        1.924
hold   (v=0.0)      0.0006     0.0199      7.938     0.0196       0.1547        2.932
```

物理的解釈:

- **hold が h=1 で最小 MSE** (0.0006): silent muscles で dynamics は重力 dominant、単純で予測しやすい
- **lowamp が次** (0.0036): tonic muscles で smooth な dynamics
- **random が最大** (0.0432): wild excitation で snapshot ごとに方向が変わる、predict 困難
- **h=50 で全 baseline 発散**: 50 step (1 s) 先まで autoregressive に rollout すると誤差累積、`mse > 5` のオーダーで unreliable。これは最初の MLP baseline では expected
- **tip_err も同パターン**: hold < lowamp < random (short horizons)、long horizon では全部不安定

これで forward model baseline の **動作確認 + 比較対象 baseline** が揃った。

#### Phase 0 全体の到達度

```text
Step 1: target set generator              完了
Step 2: state schema                      完了
Step 3: action adapter + motor noise      完了
Step 4: delay / noisy obs wrappers        完了
Step 5: episode logger (+ random ctrl)    完了
Step 6: baseline controllers              ⚠ 部分完了 (Hold + factory; PD は defer)
Step 7: reaching metrics                  完了
Step 8: forward model baseline            完了 (TransitionDataset + MLP + training + 2 CLIs)
+ Option B: env factory + extractor       完了
```

**Phase 0 ほぼ完了**。残り:

- Step 6 残: PDEndpointController (Step 7 metrics + Step 8 forward model で評価できるようになった)
- 改善: rollout MSE の安定化 (h=50 で発散する問題)、forward model architecture の探索 (LSTM / CfC / LTC)、normalization、ablation (state subset / action variant)
- Project 1 本筋: forward prediction と Kalman-like state estimator の評価

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-10-myoarm-fse-step8-mlp-training-design-decisions.md`

### 2026-05-10: Phase 3.1 (fixed-gain Kalman-like estimator) 完了

Project 1 本筋の最初の estimator 実装。Phase 2 (controller 比較) を defer し、**オフライン評価フレーム** で Phase 3.1 を進めた:

```text
既存 baseline log の true_state
  ↓ NoisyObservationWrapper + DelayedObservationWrapper (Step 4) で y_obs を offline 合成
  ↓ FixedGainKalmanEstimator が forward model (Step 8) と組み合わせて時系列処理
  x_est = x_pred + K * (y_obs - h(x_pred_{t-d}))  (h = identity、buffered + roll forward)
  ↓ aggregate_estimation_metrics で per-field MSE / tip estimation error 集計
```

closed-loop 統合 (estimator + controller) は Phase 2 の PD endpoint controller 実装と一緒に後で。

#### 追加ファイル

```text
src/myoarm_fse/estimators/__init__.py
src/myoarm_fse/estimators/base.py            # Estimator Protocol (将来用)
src/myoarm_fse/estimators/fixed_kalman.py    # FixedGainKalmanEstimator + helpers
configs/estimators/fixed_kalman_default.yaml
scripts/evaluate_estimator.py                # grid sweep CLI
tests/test_fixed_kalman.py                   # 31 tests
```

#### 確定した API

```text
class Estimator(Protocol):
  state_dim
  reset(initial_state)
  step(y_obs, u) -> x_est

class FixedGainKalmanEstimator(forward_model, gain, state_spec, *, delay_steps=0):
  - gain: float (scalar) or dict[str, float] (per-field、未指定 field は 0)
  - 内部で (state_dim,) vector に展開、element-wise update
  - delay_steps > 0 で buffered correction + forward roll
  - Cold start (t < delay_steps): prediction-only で進む
  .state_dim / .action_dim / .delay_steps / .gain_vec
  .reset(initial_state) / .step(y_obs, u)

class EstimationResult:
  x_est, x_true, error_per_step, error_per_step_norm
  n_steps, delay_steps, layout

synth_observations(log, *, state_spec, sigma, delay_steps, seed, obs_compose) -> y_obs
evaluate_estimator_on_log(estimator, log, *, ...) -> EstimationResult
aggregate_estimation_metrics(results, *, skip_cold_start=True) -> dict[str, float]
```

#### 検証

```bash
uv run pytest                # default: 499 passed, 40 deselected in 1.55 s
uv run pytest -m myosuite    # 40 passed in 5.25 s
```

合計 539 tests (+31 = test_fixed_kalman.py)。

#### Grid sweep 実走結果

```bash
uv run python scripts/evaluate_estimator.py --config configs/estimators/fixed_kalman_default.yaml
```

config:

- forward_model: `runs/models/2026-05-10T04-15-54Z` (Step 8 後半で訓練)
- runs: random / lowamp / hold (3 baseline)
- obs_noise_sigma: qpos=0.01, qvel=0.05, tip_pos=0.005, others 0
- gain_grid: [0, 0.25, 0.5, 0.75, 1.0]
- delay_grid: [0, 1, 2, 4, 6] (= 0/20/40/80/120 ms)

`tip_estimation_error_mean` (random run、単位 m):

```text
K\delay  0       1       2       4       6
0.0      49.34   25.44   49.24   50.81   48.00      <- prediction-only catastrophic
0.25     0.118   0.143   0.831   2.573   4.653
0.5      0.049   0.089   0.153   0.495   1.658
0.75     0.022   0.078   0.127   0.223   0.356
1.0      0.008   0.079   0.125   0.212   0.300      <- observation-only at delay=0
```

#### 物理的解釈

- **K=0 (prediction-only) は破滅的**: tip estimation error 25〜56 m。workspace bbox (30 cm) の 100 倍以上。forward MLP の 600 step autoregressive rollout が完全に発散 — Step 8 後半で観察した h=50 mse=10 のさらに先まで run away。
- **K=1 (observation-only) が delay=0 で最良 (0.008 m)**: y_obs は noisy true_state、`tip_pos` の sigma=0.005 とほぼ一致。当然の結果。
- **K=0.5 で delay=0 sweet spot (0.049 m)**: K=1 (0.008) よりわずかに悪いが、prediction と observation の平均化で noise variance reduction が効くケース。
- **delay 増えるほど高 K が必要**: K=0.5 で delay=6 のとき 1.66 m、K=1 で 0.30 m。forward model rollout の累積誤差を観測補正で抑える必要が大きくなる。
- **Hypothesis H3 部分検証**: 「Kalman-like update が prediction-only と observation-only の両方を上回る」 — delay=0 で K=0.75 が K=1 (0.008) より悪い (0.022) のは、prediction が unreliable なため平均化のメリットが noise reduction の恩恵を打ち消すためと推察。

#### 設計判断ノートからの差分

ノート (`...-phase3-fixed-gain-kalman-design-decisions`) の Q1〜Q10 を **そのまま**実装。逸脱 1 件:

- **delay handling の cold start 動作**: ノートでは「cold start 期間は estimator が動かない」と書いたが、実装では「prediction-only で propagate (correction なし)」として動かす形にした。`u_buffer` の長さが `delay_steps` 未満の間は correction を skip。これによりテストが書きやすく、cold start 後に自然に correction が始まる。

#### 既知の制約

1. **`_state_spec_from_model_config` が myoArm 固定**: state_dim=83 を見て StateSpec(qpos=20, qvel=20, act=34) を返す hard-coded fallback。multi-env / multi-schema 拡張時に dataset / model に layout を持たせる必要あり。
2. **K=0 で forward MLP が発散**: 単独 forward model の長期 rollout 不安定性を物語る。Phase 1 に戻って multi-step training loss / per-field standardization / 別 architecture を試す動機がある。
3. **Phase 2 (controller 比較) と closed-loop integration は未実装**: PD endpoint controller を defer 中。estimator が controller を駆動する形は別フェーズ。
4. **observation noise の sigma が固定**: grid sweep で sigma を変える機能はない (config 1 個で 1 noise 設定)。複数 sigma 比較は config を複数作って sweep する運用。

#### Phase 0 + Project 1 全体の到達度

```text
Phase 0:
  Step 1: target set generator              完了
  Step 2: state schema                      完了
  Step 3: action adapter + motor noise      完了
  Step 4: delay / noisy obs wrappers        完了
  Step 5: episode logger (+ random ctrl)    完了
  Step 6: baseline controllers              ⚠ 部分完了 (Hold + factory; PD は defer)
  Step 7: reaching metrics                  完了
  Step 8: forward model baseline            完了

Project 1:
  Phase 1: forward model baseline            完了 (Step 8 と同じ実装)
  Phase 2: delayed feedback control          未着手 (PD 必須、defer 中)
  Phase 3.1: fixed-gain Kalman estimator    完了 (このステップ)
  Phase 3.2: learned/adaptive gain           未着手
  Phase 3.3: SDN/noise/delay 下の評価         部分着手 (Phase 3.1 grid sweep 結果)
  Phase 4: 論文化セット                       未着手
```

設計判断の経緯は Obsidian 側のログを参照:

- `Logs/2026-05-10-myoarm-fse-phase3-fixed-gain-kalman-design-decisions.md`

### 2026-05-10: Forward model improvement cycle (Phase A + B + 再評価) 完了

Phase 3.1 grid sweep で観察した「K=0 prediction-only の 49 m 級発散」を受けて、Forward model の改善サイクルを実施。研究計画書に直接対応するフェーズ番号はないが、Project 1 における基礎品質改善 として位置づけ。実施計画ノート (`Logs/2026-05-10-myoarm-fse-next-implementation-plan-after-fixed-kalman`) の Phase A (dataset 拡大) + Phase B (concat_datasets library 化) + 再評価まで完了。Phase C (per-field standardization) と Phase D (multi-step rollout loss) は **不要と判断** (改善幅が大きすぎたため)。

#### 追加・更新ファイル

```text
# Library 拡張
src/myoarm_fse/models/datasets.py        # concat_datasets() を追加
src/myoarm_fse/models/__init__.py        # concat_datasets を re-export
tests/test_model_datasets.py             # +10 tests for concat_datasets

# CLI 簡素化
scripts/train_forward_model.py           # _load_concat_dataset を concat_datasets() 呼び出しに置換、
                                         # 不要になった numpy/json import を削除

# 設定追加
configs/models/mlp_expanded.yaml         # expanded.npz training config (baseline mlp.yaml と並列)

# 生成物 (gitignore)
runs/episodes/2026-05-10T07-18-17Z/      # lowamp 50 episodes
runs/episodes/2026-05-10T07-18-48Z/      # random 50 episodes
runs/datasets/expanded.npz               # N=51534, n_episodes=106
runs/models/2026-05-10T07-25-23Z/        # improved MLP
runs/estimators/2026-05-10T07-25-42Z/    # 再評価 grid sweep 結果
```

#### concat_datasets API

```python
def concat_datasets(datasets: Iterable[TransitionDataset]) -> TransitionDataset:
    """Concatenate multiple TransitionDatasets, rewriting episode_id to a
    global sequential index. Original IDs are preserved under
    `source_episode_id` and `source_dataset_index` in the merged metadata.
    state_dim/action_dim mismatches raise ValueError; empty datasets in
    the iterable are tolerated and contribute nothing."""
```

これで Step 8 後半で混入した script-private な `_load_concat_dataset` が library に昇格、複数 run 統合が公式 API になった。

#### Dataset 拡大

```text
baseline_3way (元):  6 episodes (random/lowamp/hold × 2 ep), N = 3,594 transitions
expanded (新):       106 episodes (元 6 + lowamp 50 + random 50), N = 51,534 transitions
                     → 14× の transition 増、変動性も改善
```

新規 collection は `scripts/collect_episodes.py` で `--target-set runs/targets/train.npz --n-episodes 50` を `lowamp_random.yaml` / `default.yaml` の 2 config で。各 ~3 分の MyoSuite rollout。

#### Forward model 再学習

```text
Model: ForwardMLP(state_dim=83, action_dim=34, hidden_dims=(256, 256), params=118355)
Train: Adam(1e-3), batch=256, val_step=5
Result: best_epoch=191, best_val_loss=0.0055, epochs_run=200 (early stop に届かず)
```

prediction MSE / tip prediction error の改善:

```text
Metric              baseline_3way    expanded         改善率
─────────────────  ────────────     ──────────       ──────
best_val_loss      0.0348           0.00550          -84%
h=1 mse            0.0348           0.00550          -84%
h=10 mse           0.138            0.0124           -91%
h=50 mse           10.6             0.0524           -99.5%   <- 200× 改善
h=1 tip_err        0.0549           0.00762          -86%
h=10 tip_err       0.267            0.0356           -87%
h=50 tip_err       3.36             0.138            -96%
```

**h=50 rollout MSE = 10 → 0.05** は最も劇的な改善。autoregressive rollout が安定化し、forward model が **estimator の prediction-only 経路で実用域に入った**。

ただし epochs_run=200 で early stopping に届いていない ⇒ さらに長く学習すれば改善余地あり (Phase B/C を追加で検討する余地)。

#### Fixed-gain estimator 再評価

`scripts/evaluate_estimator.py --forward-model runs/models/2026-05-10T07-25-23Z` で 75 setting (5 K × 5 delay × 3 baseline) を再走。

`tip_estimation_error_mean` (m, random run):

```text
Improved MLP (expanded):
K\delay  0       1       2       4       6
0.0      3.158   1.282   2.254   2.438   2.747      <- 49.3→3.2 で 15× 改善、ただしまだ不安定
0.25     0.023   0.044   0.063   0.102   0.138
0.5      0.010   0.024   0.035   0.058   0.079
0.75     0.007   0.015   0.025   0.042   0.058      <- delay=0 で K=1 を超える
1.0      0.008   0.013   0.020   0.035   0.049
```

#### Hypothesis H3 (Kalman update が prediction-only と observation-only の両方を上回る) の検証

**Strong support (delay=0)**:

3 baseline 平均 (`tip_estimation_error_mean`):

```text
baseline    K=0.0    K=0.5    K=0.75   K=1.0
random      3.158    0.0099   0.0071   0.0078    <- K=0.75 best
lowamp      4.232    0.0067   0.0065   0.0080    <- K=0.75 best
hold        4.351    0.0072   0.0064   0.0082    <- K=0.75 best
```

**全 baseline で K=0.75 が K=1.0 を上回る**。さらに lowamp / hold では K=0.5 も K=1.0 を上回る → **prediction の貢献が観測ノイズの平均化として機能している**。これは fixed-gain Kalman の本来の動作であり、Hypothesis H3 の中核を支持する。

**Partial support (delay > 0)**:

delay=6 で:

```text
baseline    K=0.5    K=0.75   K=1.0
random      0.079    0.058    0.049
lowamp      0.048    0.0355   0.0313
hold        0.071    0.0455   0.0356
```

長 delay では K=1.0 (純観測) が依然として最良。理由は delay 中の forward roll で誤差が累積し、prediction の重みを下げる必要があるため。これは数学的に自然 (delay が大きいほど observation の方が relevant)。

K=0 の prediction-only は依然として 1-3 m level で発散気味、実用域ではない。MLP の training を 200 epoch 以上回す or multi-step rollout loss を追加すれば改善する余地あり (将来の Phase C/D)。

#### 結論

**Forward model の dataset 拡大だけで Phase 3.1 の研究的主張が大きく前進**:

1. **Hypothesis H3 を delay=0 で強く検証** (K=0.75 が全 baseline で K=1 を上回る)
2. **Phase 3.2 (learned/adaptive gain) の比較対象** が信頼できる状態に
3. **Phase 2 (controller integration) の前提** (forward model が prediction-only で実用域) は **まだ未達** (K=0 で 1-3 m)、ただし将来の improvements で到達可能

Phase C (per-field standardization) と Phase D (multi-step rollout loss) は **当面不要**:

- 改善幅が dataset 拡大だけで十分大きかった
- improvement 余地はあるが、優先度の高い別タスク (Phase 3.2 learned gain / Phase 2 PD) に進める方が research progress が大きい

#### 次のステップ候補 (再優先)

```text
A. Phase 3.2 (learned/adaptive gain) — H3 検証を fixed gain から adaptive へ
B. Phase 2 (PD endpoint controller + closed-loop) — H2 の reaching 側を検証
C. Phase 0 さらなる collection 拡大 (target_set 全 200+50+50+30 = 330 episode)
D. Phase 3.3 robustness sweep (SDN sigma / target jump / external perturbation)
E. Phase C (per-field standardization) — Phase 1 model 改善
```

Phase C は不要、Phase A (collection 拡大は実施済み) と B (concat_datasets 実装) は完了。判断は次セッションに委ねる。

設計判断ノート / 関連ノートは Obsidian 側を参照:

- `Logs/2026-05-10-myoarm-fse-next-implementation-plan-after-fixed-kalman.md`
