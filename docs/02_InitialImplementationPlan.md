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
