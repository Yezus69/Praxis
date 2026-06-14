
# CSN-PPO Phase 1b — Custom training loop implementation spec (coverage env, 28-D)

You are implementing the CSN-PPO MVP training loop (README §36 cut) wired to the 28-D CoverEnv. Phase 1a pure-functional core is DONE and is REUSED UNCHANGED. Build a CUSTOM single-device PPO loop that reuses brax 0.14.2 primitives and injects: behavioral-memory hinge-KL guard, nullspace gradient projection, holdout train/early-stop, mined high-criticality coverage atoms, and coverage synthetic probes. SKIP (Phase 2+): sentinel worlds, mosaic teacher, adaptive curriculum, two-tier slow/fast priority replacement nuance beyond a simple threshold split. Teacher for every mined/probe atom = the policy's OWN output at mining time under the SAME normalizer.

This spec is self-contained. Implement on CPU (GPU is blocked in this WSL: export `CUDA_VISIBLE_DEVICES=` is fine; the "jax plugin ... Falling back to cpu" stderr is benign).

================================================================================
## 0. GROUND-TRUTH CONSTANTS (from praxis/contract.py — import, never hardcode)
================================================================================
OBS_DIM=28, ACT_DIM=2, N_CELLS=36, K=4, MAX_OBSTACLES=4, GRID_SIZE=6,
ARENA_HALF=3.0, AGENT_MAX_SPEED=2.0, AGENT_RADIUS=0.15, OBSTACLE_RADIUS=0.25,
COLLISION_MARGIN=0.05, EPISODE_LENGTH=600.
Obs slices: AGENT_SLICE=(0,4), OBST_SLICE=(4,20), MASK_SLICE=(20,24),
FRONTIER_SLICE=(24,27), COVERED_SLICE=(27,28).
Collision distance threshold (METERS) = AGENT_RADIUS+OBSTACLE_RADIUS+COLLISION_MARGIN = 0.45.
Per-obstacle rel-pos in obs is divided by ARENA_HALF; rel-vel divided by AGENT_MAX_SPEED.
=> meters_distance_to_obstacle_k = ARENA_HALF * norm(obs[4+4k : 4+4k+2]).
=> normalized collision threshold = 0.45/3.0 = 0.15.
mask[20:24] is ALWAYS all-ones in this env: carries NO information — do NOT use it for criticality (use OBST_SLICE rel-pos instead).
frontier[24:27] = [unit_dir_x, unit_dir_y, dist_to_nearest_unvisited / (2*ARENA_HALF)]; equals [0,0,0] when all cells covered.
covered[27] = visited.sum()/N_CELLS in [0,1].

CONSTANTS the spec introduces (put in config, see §1):
COV_OBS_DIM=28, COV_ACT_DIM=2, COV_COLLISION_THRESH_NORM=0.15 (0.45/3.0),
COV_FRONTIER_NORM=6.0 (2*ARENA_HALF).

================================================================================
## 1. config.py — EDIT (append fields; DO NOT remove existing)
================================================================================
The existing CSNPPOConfig (read it) has README-default nav values. EDIT it to (a) add a coverage namespace and (b) flip defaults to the coverage-baseline-matching values so the killer test is apples-to-apples with praxis/train.py DEFAULTS. Keep all existing fields (other code imports them). Add/override:

```python
# --- coverage env / obs dims (NEW) ---
obs_dim: int = 28
action_dim: int = 2
# --- baseline-matching PPO (OVERRIDE existing defaults to match praxis/train.py DEFAULTS) ---
num_timesteps: int = int(1e7)        # killer test target
num_envs: int = 2048
episode_length: int = 600            # contract.EPISODE_LENGTH
unroll_length: int = 20
batch_size: int = 256
num_minibatches: int = 32
max_updates_per_batch: int = 4       # == baseline num_updates_per_batch
learning_rate: float = 3e-4
entropy_cost: float = 1e-2
discounting: float = 0.97
reward_scaling: float = 1.0
gae_lambda: float = 0.95
clipping_epsilon: float = 0.3
vf_coefficient: float = 0.5
normalize_advantage: bool = True
normalize_observations: bool = True
max_grad_norm: float = 1.0           # baseline injects clip_by_global_norm(1.0)
bootstrap_on_timeout: bool = True    # baseline uses it
policy_hidden_layer_sizes: tuple = (256, 256, 256)
value_hidden_layer_sizes: tuple = (256, 256, 256)
# --- memory sizes: SHRINK for CPU MVP (README values are 1e6/2.6e5 — too big for CPU smoke) ---
memory_size_fast: int = 131_072
memory_size_slow: int = 32_768
memory_batch_size: int = 2048        # split half/half fast/slow
min_memory_size_before_guard: int = 4096   # guard inert until memory has this many atoms
# --- guard coefs (README §6/§18 forms) ---
guard_policy_coef: float = 1.0       # lambda_pi
guard_value_coef: float = 0.25       # lambda_v (also baked into memory_guard_loss as +0.25*value_loss)
guard_lambda_mem: float = 1.0        # lambda_mem in combine_safe_and_guard_grads (§9 add-back)
guard_kl_budget: float = 0.02        # delta0 in §18 delta_m = delta0/(1+c)
critical_kl_budget: float = 0.005
value_budget: float = 0.25           # rho0 in §18 rho_m = rho0/(1+beta*c)
critical_value_budget: float = 0.05
value_budget_beta: float = 1.0
memory_kl_limit_p95: float = 0.05
# --- criticality weights (README §19 FORM; coverage-adapted features, SAME weights) ---
crit_w_advantage: float = 1.0
crit_w_collision: float = 3.0
crit_w_frontier: float = 2.0   # replaces success_proximity weight (2.0)
crit_w_dynamic: float = 1.0
crit_w_novelty: float = 1.0
crit_clip_min: float = 0.1
crit_clip_max: float = 10.0
slow_memory_threshold: float = 3.0   # criticality > this -> slow tier (README §20)
atoms_per_rollout: int = 1024        # top-K mined atoms per update (fixed shape!)
teacher_logstd_floor: float = -6.9   # ~log(0.001), matches brax min_std
# --- probes ---
synthetic_probe_batch_size: int = 512
synthetic_probe_insert_interval: int = 1
# --- holdout (README §21) ---
holdout_fraction: float = 0.2
holdout_eps: float = 1e-4
target_kl: float = 0.03
# --- projection (README §9/§11) ---
enable_gradient_projection: bool = True
projection_eps: float = 1e-8
enable_guard: bool = True
seed: int = 0
log_interval: int = 1
```
Keep `assert (num_envs*unroll_length) % (batch_size*num_minibatches) == 0` as a config-time check (2048*20=40960, 256*32=8192, ratio 5).

================================================================================
## 2. NEW FILE: agent/csn_ppo/criticality_coverage.py
   (coverage-valid replacements for the 27-D nav helpers in synthetic_probes.py)
================================================================================
synthetic_probes.py's `obstacle_distances/collision_proximity/success_proximity/
dynamic_obstacle_score/criticality_score` use 27-D NAV indices (obs[7:23], obs[2:4],
mask obs[23:27]) and goal-reaching semantics — WRONG for 28-D coverage. DO NOT call
them. Implement these vmappable, JIT-safe, per-single-obs functions (all take obs shape (28,)):

```python
import jax, jax.numpy as jnp
from praxis import contract

_OB0, _OB1 = contract.OBST_SLICE          # (4, 20)
_FR0, _FR1 = contract.FRONTIER_SLICE      # (24, 27)
_COV0      = contract.COVERED_SLICE[0]    # 27
_ARENA     = contract.ARENA_HALF          # 3.0
_VMAX      = contract.AGENT_MAX_SPEED     # 2.0
_COLL_THRESH_M = contract.AGENT_RADIUS + contract.OBSTACLE_RADIUS + contract.COLLISION_MARGIN  # 0.45

def _obstacles(obs):                      # -> (K, 4): rel_px_norm, rel_py_norm, vx_norm, vy_norm
    return obs[_OB0:_OB1].reshape(contract.K, 4)

def obstacle_distances_m(obs):            # meters; mask is all-ones so no masking needed
    o = _obstacles(obs)
    return _ARENA * jnp.sqrt(o[:, 0]**2 + o[:, 1]**2)   # (K,)

def collision_proximity(obs):
    # 1.0 at contact (d<=thresh), 0 once min-dist >= 2*thresh; smooth ramp.
    d_min = jnp.min(obstacle_distances_m(obs))
    return jax.nn.relu(_COLL_THRESH_M*2.0 - d_min) / (_COLL_THRESH_M*2.0)

def dynamic_obstacle_score(obs):
    o = _obstacles(obs)
    speeds = jnp.sqrt(o[:, 2]**2 + o[:, 3]**2) * _VMAX  # back to m/s
    return jnp.max(speeds)

def frontier_urgency(obs):
    # high when far from nearest unvisited AND low coverage = hard-to-explore / stuck.
    fdist = obs[_FR1 - 1]            # obs[26], normalized by 2*ARENA in [0,~1]
    covered = obs[_COV0]            # obs[27]
    return fdist * (1.0 - covered)

def coverage_novelty(obs):
    # pure-obs novelty proxy: more unexplored area remaining => more novel.
    return 1.0 - obs[_COV0]

def criticality_score(obs, advantage_abs, cfg):
    c = (cfg.crit_w_advantage * advantage_abs
         + cfg.crit_w_collision * collision_proximity(obs)
         + cfg.crit_w_frontier  * frontier_urgency(obs)
         + cfg.crit_w_dynamic   * dynamic_obstacle_score(obs)
         + cfg.crit_w_novelty   * coverage_novelty(obs))
    return jnp.clip(c, cfg.crit_clip_min, cfg.crit_clip_max)

def memory_weight(c, cfg):           # README §18: w_m = clip(c, w_min, w_max)
    return jnp.clip(c, cfg.crit_clip_min, cfg.crit_clip_max)

def kl_budget_from_c(c, cfg):        # README §18: delta_m = delta0/(1+c)
    return cfg.guard_kl_budget / (1.0 + c)

def value_budget_from_c(c, cfg):     # README §18: rho_m = rho0/(1+beta*c)
    return cfg.value_budget / (1.0 + cfg.value_budget_beta * c)

def cluster_id_for(obs):
    # coverage clusters for bucketing (reuse guarded_loss CLUSTER_* ids):
    #   0 collision_boundary: collision_proximity high
    #   2 dynamic_obstacle:   dynamic score high & not near-collision
    #   3 no_obstacle_straight_line: all obstacles far
    #   1 successful_goal -> REPURPOSED to "high_coverage/frontier-progress" cluster
    coll = collision_proximity(obs) > 0.25
    dyn  = dynamic_obstacle_score(obs) > 0.3
    near = jnp.min(obstacle_distances_m(obs)) < (_COLL_THRESH_M * 4.0)
    cid = jnp.where(coll, 0,
            jnp.where(dyn & near, 2,
              jnp.where(near, 3, 1)))   # 1 = open/progress cluster
    return cid.astype(jnp.int32)
```
NOTE: `guarded_loss.bucket_memory_batches` keys off `cluster_id`/`source_id` exactly as
above (CLUSTER_COLLISION_BOUNDARY=0, SUCCESSFUL_GOAL=1, DYNAMIC=2, NO_OBSTACLE=3;
SOURCE_RECENT_CURRENT=0, SYNTHETIC_PROBE=1, SENTINEL_FAILURE=2). Reuse those ids; the
spec just gives them coverage meanings — no edit to guarded_loss.py needed.

================================================================================
## 3. NEW FILE: agent/csn_ppo/coverage_probes.py
   (replaces 27-D nav probes; teacher = policy-at-mining-time, NOT analytic)
================================================================================
Generate VALID 28-D coverage observations on the semantic manifold. mask MUST be all-ones
(env invariant). Build fixed [4,4] obstacle blocks directly (JIT-safe, no dynamic pad).
Each generator takes an rng key and returns obs shape (28,). Provide a vmapped batch builder.

```python
import jax, jax.numpy as jnp
from praxis import contract

def pack_cover_obs(agent_feat, obstacles, frontier, covered):
    # agent_feat (4,), obstacles (4,4), frontier (3,), covered scalar
    mask = jnp.ones((contract.K,), jnp.float32)          # env invariant
    obs = jnp.concatenate([agent_feat.astype(jnp.float32),
                           obstacles.reshape(-1).astype(jnp.float32),
                           mask,
                           frontier.astype(jnp.float32),
                           jnp.reshape(covered, (1,)).astype(jnp.float32)])
    return jnp.nan_to_num(obs)   # matches env _build_obs

def make_probe_open_explore(rng):
    # no nearby obstacles, clear frontier ahead, low coverage -> "explore freely".
    ka, kf, kc = jax.random.split(rng, 3)
    agent = jax.random.uniform(ka, (4,), minval=-0.8, maxval=0.8)
    far = jnp.full((4,4), 0.0).at[:, 0].set(0.9).at[:, 1].set(0.9)  # rel ~0.9*arena away (norm units)
    fang = jax.random.uniform(kf, (), -jnp.pi, jnp.pi)
    fdir = jnp.array([jnp.cos(fang), jnp.sin(fang)])
    fdist = jax.random.uniform(kf, (), 0.3, 0.9)
    frontier = jnp.concatenate([fdir, fdist[None]])
    covered = jax.random.uniform(kc, (), 0.0, 0.4)
    return pack_cover_obs(agent, far, frontier, covered)

def make_probe_obstacle_ahead(rng):
    # one obstacle near the frontier direction -> "evade while still exploring".
    ka, ko, kf, kc = jax.random.split(rng, 4)
    agent = jax.random.uniform(ka, (4,), -0.6, 0.6)
    fang = jax.random.uniform(kf, (), -jnp.pi, jnp.pi)
    fdir = jnp.array([jnp.cos(fang), jnp.sin(fang)])
    fdist = jax.random.uniform(kf, (), 0.2, 0.7)
    # obstacle near collision threshold along frontier dir (in normalized units)
    near = jax.random.uniform(ko, (), 0.10, 0.20)   # ~ collision_thresh_norm region
    ov = jax.random.uniform(ko, (2,), -0.5, 0.5)
    o0 = jnp.concatenate([fdir*near, ov])
    far = jnp.array([0.9,0.9,0.,0.])
    obstacles = jnp.stack([o0, far, far, far])
    covered = jax.random.uniform(kc, (), 0.1, 0.7)
    frontier = jnp.concatenate([fdir, fdist[None]])
    return pack_cover_obs(agent, obstacles, frontier, covered)

def make_probe_near_complete(rng):
    # high coverage, frontier short or zero -> "fine exploration / finish".
    ka, kf, kc = jax.random.split(rng, 3)
    agent = jax.random.uniform(ka, (4,), -0.9, 0.9)
    far = jnp.array([0.9,0.9,0.,0.])
    obstacles = jnp.stack([far,far,far,far])
    fang = jax.random.uniform(kf, (), -jnp.pi, jnp.pi)
    fdir = jnp.array([jnp.cos(fang), jnp.sin(fang)])
    fdist = jax.random.uniform(kf, (), 0.0, 0.2)
    frontier = jnp.concatenate([fdir, fdist[None]])
    covered = jax.random.uniform(kc, (), 0.7, 0.99)
    return pack_cover_obs(agent, obstacles, frontier, covered)

_PROBE_FNS = (make_probe_open_explore, make_probe_obstacle_ahead, make_probe_near_complete)

def generate_cover_probes(rng, batch_size):
    keys = jax.random.split(rng, batch_size)
    # round-robin generator selection by index (static python loop over 3 fns, then concat)
    per = batch_size // len(_PROBE_FNS)
    chunks = []
    for i, fn in enumerate(_PROBE_FNS):
        sub = jax.lax.map(fn, keys[i*per:(i+1)*per])   # vmappable map
        chunks.append(sub)
    out = jnp.concatenate(chunks, axis=0)
    # pad to exact batch_size with the first generator if not divisible
    if out.shape[0] < batch_size:
        extra = jax.lax.map(_PROBE_FNS[0], keys[out.shape[0]:])
        out = jnp.concatenate([out, extra], axis=0)
    return out   # (batch_size, 28)
```
Probe TEACHER = policy-at-mining-time (README §36, NOT analytic). Probe atoms are labeled in
train.py via `apply_policy_value` (see §6/§7), source_id=SOURCE_SYNTHETIC_PROBE(1),
cluster from `cluster_id_for`, criticality uses advantage_abs=0 (no rollout advantage for a
synthetic obs).

================================================================================
## 4. apply_policy_value — define in agent/csn_ppo/train.py (closure over ppo_network)
================================================================================
EXACT, from VERIFIED MAP. memory_guard_loss calls it as
apply_policy_value(params, normalizer_params, obs); brax .apply signature is
.apply(normalizer_params, network_params, obs) so normalizer is passed FIRST to .apply:

```python
import jax.numpy as jnp
def make_apply_policy_value(ppo_network):
    def apply_policy_value(params, normalizer_params, obs):
        logits = ppo_network.policy_network.apply(normalizer_params, params.policy, obs)
        dist   = ppo_network.parametric_action_distribution.create_dist(logits)
        value  = ppo_network.value_network.apply(normalizer_params, params.value, obs)
        return dist.loc, jnp.log(dist.scale), value   # (pre-tanh mean, logstd, value)
    return apply_policy_value
```
- dist.loc == split(logits,2,-1)[0] (pre-tanh MEAN). dist.scale == softplus(split[1])+0.001 (STD).
- value is ALREADY squeezed to (...,) — do not squeeze again.
- params is brax `ppo_losses.PPONetworkParams(policy, value)`.
- This same function computes teacher labels at mining time (capture loc/log(scale)/value),
  guaranteeing apples-to-apples KL because both go through the same normalizer.

================================================================================
## 5. NEW FILE: agent/csn_ppo/metrics.py
================================================================================
Pure helpers (no jax control flow needed beyond array ops):
```python
def prefix_metrics(d, prefix): return {f"{prefix}/{k}": v for k,v in d.items()}
def merge_metrics(*dicts): out={}; [out.update(d) for d in dicts if d]; return out
def to_float_dict(d):  # jax arrays -> python floats for logging
    import numpy as np
    return {k: (float(np.asarray(v)) if hasattr(v,'shape') else v) for k,v in d.items()}
def generalization_gap(train_surrogate, holdout_surrogate):
    return train_surrogate - holdout_surrogate
def should_stop_epoch(holdout_score, best_holdout_score, memory_kl_p95,
                      memory_kl_limit, approx_kl, target_kl, eps=1e-4):
    # README §21 EXACT
    holdout_bad = holdout_score < best_holdout_score - eps
    memory_bad  = memory_kl_p95 > memory_kl_limit
    kl_bad      = approx_kl > 1.5 * target_kl
    return bool(holdout_bad | memory_bad | kl_bad)
```
Metric keys to log every update (README §30 MVP subset):
`ppo/total_loss, ppo/policy_loss, ppo/v_loss, ppo/entropy_loss, ppo/approx_kl,
ppo/train_surrogate, ppo/holdout_surrogate, ppo/generalization_gap,
memory/kl_mean, memory/kl_p95, memory/policy_loss, memory/value_loss,
memory/policy_violation_frac, memory/fast_size, memory/slow_size, memory/guard_active,
epoch/stopped_at, eval/episode_coverage, eval/episode_collision, env_steps`.

================================================================================
## 6. NEW FILE: agent/csn_ppo/rollout_mining.py
================================================================================
Mine the top-`atoms_per_rollout` highest-criticality states from a fresh rollout into a
BehavioralMemoryBatch, with FIXED output shape (JIT-safe top-k). Teacher = policy-at-mining
(via apply_policy_value). Inputs: flattened rollout obs (M,28), per-state advantage_abs (M,),
params, normalizer_params, apply_policy_value, cfg. Steps:

```python
import jax, jax.numpy as jnp
from agent.csn_ppo import criticality_coverage as cc
from agent.csn_ppo.memory import BehavioralMemoryBatch

def mine_atoms(obs_flat, adv_abs, params, normalizer_params, apply_policy_value, cfg):
    # 1) criticality per state (vmap)
    crit = jax.vmap(lambda o, a: cc.criticality_score(o, a, cfg))(obs_flat, adv_abs)  # (M,)
    # 2) top-K fixed-shape selection
    k = cfg.atoms_per_rollout
    _, idx = jax.lax.top_k(crit, k)                 # (k,)
    obs = obs_flat[idx]                             # (k,28)
    c   = crit[idx]                                 # (k,)
    # 3) teacher labels = policy-at-mining (same normalizer)
    t_mean, t_logstd, t_value = apply_policy_value(params, normalizer_params, obs)
    t_logstd = jnp.maximum(t_logstd, cfg.teacher_logstd_floor)
    # 4) per-atom budgets/weights (README §18 forms)
    w   = jax.vmap(lambda x: cc.memory_weight(x, cfg))(c)
    klb = jax.vmap(lambda x: cc.kl_budget_from_c(x, cfg))(c)
    vb  = jax.vmap(lambda x: cc.value_budget_from_c(x, cfg))(c)
    cid = jax.vmap(cc.cluster_id_for)(obs)
    src = jnp.zeros((k,), jnp.int32)                # SOURCE_RECENT_CURRENT
    batch = BehavioralMemoryBatch(obs=obs, mean=t_mean, logstd=t_logstd, value=t_value,
        weight=w, kl_budget=klb, value_budget=vb, cluster_id=cid, source_id=src)
    # 5) fast/slow split by threshold (README §20): return whole batch + a slow-mask
    slow_mask = c > cfg.slow_memory_threshold       # (k,) bool
    return batch, slow_mask, {"mine/crit_mean": jnp.mean(crit),
                              "mine/crit_p95": jnp.sort(crit)[int(0.95*(crit.shape[0]-1))],
                              "mine/slow_frac": jnp.mean(slow_mask)}

def label_probe_atoms(probe_obs, params, normalizer_params, apply_policy_value, cfg):
    t_mean, t_logstd, t_value = apply_policy_value(params, normalizer_params, probe_obs)
    t_logstd = jnp.maximum(t_logstd, cfg.teacher_logstd_floor)
    n = probe_obs.shape[0]
    c = jax.vmap(lambda o: cc.criticality_score(o, 0.0, cfg))(probe_obs)
    w   = jax.vmap(lambda x: cc.memory_weight(x, cfg))(c)
    klb = jax.vmap(lambda x: cc.kl_budget_from_c(x, cfg))(c)
    vb  = jax.vmap(lambda x: cc.value_budget_from_c(x, cfg))(c)
    cid = jax.vmap(cc.cluster_id_for)(probe_obs)
    src = jnp.ones((n,), jnp.int32)                 # SOURCE_SYNTHETIC_PROBE
    return BehavioralMemoryBatch(obs=probe_obs, mean=t_mean, logstd=t_logstd, value=t_value,
        weight=w, kl_budget=klb, value_budget=vb, cluster_id=cid, source_id=src)
```
NOTE on fast/slow insert: `insert_atoms` needs a FIXED-size batch (ring buffer). For MVP,
insert the WHOLE mined batch into memory_fast every update; insert ONLY-slow atoms into
memory_slow by ZEROING the weight of non-slow atoms (keep shape fixed) before insert into
slow — i.e. build a slow-labeled copy where `weight = where(slow_mask, weight, 0.0)` and
insert that fixed-shape batch into memory_slow. Zero-weight atoms contribute nothing to the
guard loss (weight multiplies the hinge), so this is a correct fixed-shape fast/slow split
without dynamic gather. (This is the simplest JIT-safe realization of README §20.)

================================================================================
## 7. NEW FILE: agent/csn_ppo/train.py — THE CUSTOM CSN-PPO LOOP
================================================================================
Reuse brax primitives EXACTLY as in VERIFIED MAP. Imports:
```python
import functools, jax, jax.numpy as jnp, optax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import losses as ppo_losses
from brax.training import acting, gradients
from brax.training.acme import running_statistics, specs
from brax.training.acting import Evaluator
from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.memory import init_behavioral_memory, insert_atoms, sample_memory
from agent.csn_ppo.guarded_loss import (memory_guard_loss, bucket_memory_batches,
    value_and_grad_guard_loss_by_bucket, MEMORY_BUCKETS, _sorted_p95)
from agent.csn_ppo.gradient_projection import (project_conflicting_gradient,
    combine_safe_and_guard_grads)
from agent.csn_ppo import rollout_mining, coverage_probes, metrics as M
```

### 7.1 Setup (once)
```python
def train(environment, config: CSNPPOConfig, progress_fn=None, eval_env=None):
    rng = jax.random.PRNGKey(config.seed)
    obs_dim, act_dim = config.obs_dim, config.action_dim   # 28, 2
    normalize = (running_statistics.normalize if config.normalize_observations
                 else (lambda x, y: x))
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=config.value_hidden_layer_sizes)
    ppo_network = network_factory(obs_dim, act_dim, preprocess_observations_fn=normalize)
    make_policy = ppo_networks.make_inference_fn(ppo_network, compute_value=True)  # value for bootstrap
    apply_policy_value = make_apply_policy_value(ppo_network)

    rng, kp, kv = jax.random.split(rng, 3)
    params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(kp),
        value=ppo_network.value_network.init(kv))
    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                            optax.adam(config.learning_rate))
    opt_state = optimizer.init(params)
    normalizer_params = running_statistics.init_state(
        specs.Array((obs_dim,), jnp.dtype('float32')), std_eps=0.0, mode='welford')

    memory_fast = init_behavioral_memory(config.memory_size_fast, obs_dim, act_dim)
    memory_slow = init_behavioral_memory(config.memory_size_slow, obs_dim, act_dim)

    # env reset (wrapped env vmaps internally; DO NOT outer-vmap)
    rng, kreset = jax.random.split(rng)
    env_state = jax.jit(environment.reset)(jax.random.split(kreset, config.num_envs))
```
ENV WRAP: caller passes an ALREADY-wrapped env, OR wrap here:
`environment = mujoco_playground.wrapper.wrap_for_brax_training(raw_env, episode_length=config.episode_length, action_repeat=1)`. Build raw_env via `praxis.train.build_env(config.episode_length, reward_overrides)`. Thread episode_length into BOTH (load-bearing for bootstrap).

### 7.2 PPO loss closure (reuse brax compute_ppo_loss)
```python
    ppo_loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss, ppo_network=ppo_network,
        entropy_cost=config.entropy_cost, discounting=config.discounting,
        reward_scaling=config.reward_scaling, gae_lambda=config.gae_lambda,
        clipping_epsilon=config.clipping_epsilon,
        normalize_advantage=config.normalize_advantage,
        vf_coefficient=config.vf_coefficient)
    ppo_value_and_grad = gradients.loss_and_pgrad(ppo_loss_fn, pmap_axis_name=None, has_aux=True)
```

### 7.3 Rollout collection (reuse acting.generate_unroll) — per update
```python
    extra_fields = ('truncation', 'episode_metrics', 'episode_done', 'time_out')
    num_unrolls = (config.batch_size * config.num_minibatches) // config.num_envs  # = 5
    def collect(carry, _):
        es, key = carry
        key, k = jax.random.split(key)
        policy = make_policy((normalizer_params, params.policy, params.value))
        es, data = acting.generate_unroll(environment, es, policy, k,
                       config.unroll_length, extra_fields=extra_fields)
        return (es, key), data
    (env_state, _), data = jax.lax.scan(collect, (env_state, rollout_rng), (), length=num_unrolls)
    # data leading dims: (num_unrolls, T, num_envs, ...). Reshape to [B, T] for compute_ppo_loss:
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)   # (num_unrolls, num_envs, T, ...)
    data = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), data)  # (B, T, ...) B=num_envs*num_unrolls
```
This yields data with leading dims [B, T] = [10240, 20] which compute_ppo_loss expects
(it swapaxes(0,1) internally). VERIFIED layout per MAP.

bootstrap_on_timeout (replicate brax): if config.bootstrap_on_timeout:
```python
    time_out = data.extras['state_extras']['time_out']        # [B,T]
    val      = data.extras['policy_extras']['value']          # [B,T]
    data = data.replace(reward=data.reward + config.discounting * time_out * val)
```

### 7.4 Normalizer update (BEFORE SGD; non-adaptive ordering)
```python
    normalizer_params = running_statistics.update(
        normalizer_params, data.observation, pmap_axis_name=None)
```

### 7.5 Train/holdout split (README §21)
Split along the B axis: `n_hold = int(holdout_fraction * B)`. Shuffle B indices with a key,
slice first (B-n_hold) -> train_data, rest -> holdout_data (both keep [b,T,...] shape via
tree_map gather). holdout is used ONLY to score (surrogate), never to grad-step.

### 7.6 Mine atoms + probes (per update, BEFORE epochs)
- adv for criticality: cheaply recompute per-state advantage_abs by reusing compute_gae OR
  approximate with |reward + discounting*next_value - value| sign-free. SIMPLEST MVP: run
  `ppo_losses.compute_gae` on the [T,B] pre-swap data to get advantages, take abs, flatten to
  match obs_flat. To keep it simple and avoid recompute, instead set advantage_abs from the
  value-target residual: compute values v=value_network.apply over flattened obs, and
  adv_abs = |gae_advantages_flat|. Flatten train obs: `obs_flat = train_data.observation.reshape(-1, 28)`
  and matching adv_abs. (Shapes must align; if recomputing gae is fiddly, pass adv_abs = zeros
  and rely on collision/frontier/dynamic/novelty terms — STILL valid criticality, just no
  advantage weighting. Prefer real |adv| if straightforward.)
- `mined_batch, slow_mask, mine_metrics = rollout_mining.mine_atoms(obs_flat, adv_abs, params, normalizer_params, apply_policy_value, config)`
- gate: only mine/guard once `memory_fast.size >= min_memory_size_before_guard` is reachable;
  ALWAYS insert (to build up memory), but make guard inert until size threshold (see §7.8 gate).
- `memory_fast = insert_atoms(memory_fast, mined_batch)`
- slow-labeled copy: `slow_batch = mined_batch._replace? ` — BehavioralMemoryBatch is a plain
  @dataclass (not NamedTuple): build a new BehavioralMemoryBatch copying fields but
  `weight = jnp.where(slow_mask, mined_batch.weight, 0.0)`; `memory_slow = insert_atoms(memory_slow, slow_batch)`.
- probes (if update % synthetic_probe_insert_interval == 0):
  `probe_obs = coverage_probes.generate_cover_probes(probe_rng, config.synthetic_probe_batch_size)`;
  `probe_batch = rollout_mining.label_probe_atoms(probe_obs, params, normalizer_params, apply_policy_value, config)`;
  `memory_slow = insert_atoms(memory_slow, probe_batch)`.

### 7.7 Guard-active gate
`guard_active = (memory_fast.size >= config.min_memory_size_before_guard) & config.enable_guard`.
When inactive, skip guard grads/projection (g_total = g_ppo); when active, do §7.8. Because
size is a traced scalar but the loop is Python-level (not scanned over updates), you can read
it with `int(memory_fast.size)` ONLY if memory ops are outside jit. RECOMMENDATION: keep the
OUTER update loop in PYTHON (eager), jit only the inner pure functions (collect-scan via
jax.jit, sgd epoch body via jax.jit). Then `int(memory_fast.size)` is a concrete read. This
matches brax-custom-loop practice and keeps holdout/early-stop Python-controllable.

### 7.8 PPO epochs with guard + projection + holdout early-stop (README §21, §27 step 6)
Python loop `for epoch in range(config.max_updates_per_batch)`:
```python
    best_params, best_opt, best_holdout = params, opt_state, -jnp.inf
    stopped_at = config.max_updates_per_batch
    for epoch in range(config.max_updates_per_batch):
        rng, kmem, kloss, kperm = jax.random.split(rng, 4)
        mb_fast = sample_memory(memory_fast, kmem, config.memory_batch_size//2)
        mb_slow = sample_memory(memory_slow, jax.random.split(kmem)[0], config.memory_batch_size//2)
        memory_batch = concat_batches(mb_fast, mb_slow)   # concat each field along axis 0
        # 6a PPO grad on TRAIN data (shuffle minibatches like brax OR single full-batch grad for MVP)
        (ppo_loss, ppo_metrics), g_ppo = ppo_value_and_grad(
            params, normalizer_params, train_data, kloss)
        if guard_active:
            # 6b bucketed guard grads (reuse value_and_grad_guard_loss_by_bucket)
            _, guard_grads, guard_metrics = value_and_grad_guard_loss_by_bucket(
                params, normalizer_params, memory_batch, apply_policy_value, MEMORY_BUCKETS)
            # 6c project conflicting components (README §9/§11) — critical buckets first:
            #   MEMORY_BUCKETS order already puts collision_boundary first; pass guard_grads as-is.
            g_safe = (project_conflicting_gradient(g_ppo, guard_grads, config.projection_eps)
                      if config.enable_gradient_projection else g_ppo)
            # 6d add guard grads back with lambda_mem (README §9: g = g_safe + lambda_mem * sum g_mem)
            coefs = [config.guard_lambda_mem] * len(guard_grads)
            g_total = combine_safe_and_guard_grads(g_safe, guard_grads, coefs)
        else:
            g_total = g_ppo; guard_metrics = {}
        # 6e optimizer step
        updates, opt_state_cand = optimizer.update(g_total, opt_state)
        params_cand = optax.apply_updates(params, updates)
        # 6f holdout surrogate + memory kl p95 on CANDIDATE
        (_, holdout_metrics) = ppo_loss_fn(params_cand, normalizer_params, holdout_data, kloss)
        holdout_surrogate = -holdout_metrics['policy_loss']   # surrogate = -policy_loss (higher=better)
        if guard_active:
            _, gm = memory_guard_loss(params_cand, normalizer_params, memory_batch, apply_policy_value)
            mem_kl_p95 = gm['memory/kl_p95']
        else:
            mem_kl_p95 = jnp.array(0.0)
        approx_kl = ppo_metrics['kl_mean']
        # accept-as-best (README §27 6f)
        accept = (holdout_surrogate > best_holdout) & (mem_kl_p95 < config.memory_kl_limit_p95)
        if bool(accept):
            best_params, best_opt, best_holdout = params_cand, opt_state_cand, holdout_surrogate
        stop = M.should_stop_epoch(float(holdout_surrogate), float(best_holdout),
            float(mem_kl_p95), config.memory_kl_limit_p95, float(approx_kl),
            config.target_kl, config.holdout_eps)
        params, opt_state = params_cand, opt_state_cand
        if stop:
            params, opt_state = best_params, best_opt   # rollback to best epoch
            stopped_at = epoch + 1
            break
```
`surrogate = -policy_loss`: brax compute_ppo_loss returns metrics['policy_loss'] = the
NEGATED clipped surrogate mean (loss = -surrogate), so -policy_loss is the surrogate to
MAXIMIZE. Confirm sign empirically in smoke (train_surrogate should rise over epochs).
`concat_batches`: BehavioralMemoryBatch is a @dataclass — build new one concatenating each
of the 9 arrays along axis 0.

### 7.9 Optional minibatching of the PPO grad (parity with baseline)
For exact baseline parity you may inner-scan num_minibatches over shuffled train_data
(brax sgd_step). For MVP correctness a single full-batch grad over train_data per epoch is
acceptable and simpler; note divisibility still governs the rollout reshape. Document this
deviation. If you minibatch, do the project+combine+update PER MINIBATCH (the §7.8 body moves
inside the minibatch scan) — but then early-stop is evaluated once per epoch after the scan.

### 7.10 Eval (README §30 coverage metric)
Every `log_interval` (or a configurable eval_interval) build an Evaluator OR just run a
deterministic rollout on eval_env and read episode_metrics['coverage']:
```python
    evaluator = Evaluator(eval_env or environment,
        functools.partial(ppo_networks.make_inference_fn(ppo_network), deterministic=True)
            if False else (lambda p: make_policy((normalizer_params, p.policy, p.value))),
        num_eval_envs=128, episode_length=config.episode_length, action_repeat=1, key=eval_key)
    eval_metrics = evaluator.run_evaluation((normalizer_params, params.policy, params.value), {})
```
Read `eval/episode_coverage` (sum of per-step coverage deltas = final coverage fraction) and
`eval/episode_collision`. This is the KILLER-TEST signal (baseline peaks ~0.82 then collapses
to ~0.27; CSN should hold the peak). Simplest robust path: reuse brax `acting.Evaluator`
exactly as praxis baseline does, passing a make_policy that closes over current
normalizer_params/params.

### 7.11 Per-update logging + return
Build metrics dict (§5 keys), include memory/fast_size=int(memory_fast.size),
memory/slow_size=int(memory_slow.size), epoch/stopped_at=stopped_at, env_steps. Call
progress_fn(env_steps, metrics) if provided; else print. Return
`(make_inference_fn(ppo_network), params, normalizer_params, (memory_fast, memory_slow), metrics)`.

================================================================================
## 8. NEW FILE: praxis/train_csn.py (entry; mirrors praxis/train.py env build)
================================================================================
CLI/entry that: builds raw env via `praxis.train.build_env(episode_length, reward_overrides)`,
wraps with `mujoco_playground.wrapper.wrap_for_brax_training(raw, episode_length=EP, action_repeat=1)`
for BOTH train and a separate eval env, constructs CSNPPOConfig (override num_timesteps via
--num-timesteps; default 1e7 killer test), calls `agent.csn_ppo.train.train(...)`, logs to the
same place praxis/train.py does. Mirror praxis/train.py flags for episode_length, num_envs,
seed, reward overrides (--k-cov/--k-coll/--k-time). Add `--enable-guard/--no-enable-guard`,
`--enable-projection/--no`, `--smoke` (num_timesteps=3e5, memory sizes already small).
Print eval/episode_coverage each eval so the collapse curve is visible.
(Equivalent file location alternative per task: agent/train_csn_ppo.py — pick praxis/train_csn.py
to sit beside the baseline; expose `main()`.)

================================================================================
## 9. EDIT agent/csn_ppo/__init__.py
================================================================================
Add exports for the new modules so `from agent.csn_ppo import train` etc. resolve:
`from agent.csn_ppo import train, metrics, rollout_mining, coverage_probes, criticality_coverage`
and add `train` (the module) — keep all existing exports. Do NOT remove the 27-D nav probe
exports (other Phase-1a tests import them); they coexist, just unused by the coverage loop.

================================================================================
## 10. INVARIANTS / REUSE-UNCHANGED (do not reimplement)
================================================================================
- memory.py: BehavioralMemory/Batch, init_behavioral_memory, insert_atoms, sample_memory — REUSE.
- guarded_loss.py: gaussian_kl, memory_guard_loss, bucket_memory_batches,
  value_and_grad_guard_loss_by_bucket, MEMORY_BUCKETS, cluster/source id constants — REUSE.
- gradient_projection.py: tree_dot/tree_add_scaled, project_conflicting_gradient,
  combine_safe_and_guard_grads — REUSE.
- README math is EXACT and unchanged: gaussian_kl (§7), hinge guard (§6/§8), projection (§9),
  budgets delta_m=delta0/(1+c), rho_m=rho0/(1+beta c) (§18), criticality FORM/weights (§19),
  should_stop_epoch (§21). ONLY the obs-derived FEATURES change (coverage, not goal-reaching).
- Pass pmap_axis_name=None everywhere (loss_and_pgrad, running_statistics.update) — single device.
- Keep make_ppo_networks override (256,256,256)/(256,256,256) — brax defaults are wrong.
- memory_guard_loss already returns policy_loss + 0.25*value_loss; do not double-apply value coef.


## SMOKE PLAN (from design)

WSL smoke (CPU). Distro 'praxis', /opt/venv/bin/python, code at /root/praxis (sync /mnt/c repo first).

STEP 0 (sync + import sanity):
  rsync -a --delete /mnt/c/Users/Asav/source/repos/Praxis/ /root/praxis/  (exclude .git, ckpts)
  cd /root/praxis && CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -c "
    import agent.csn_ppo.train, agent.csn_ppo.criticality_coverage as cc, agent.csn_ppo.coverage_probes as cp, agent.csn_ppo.rollout_mining
    import jax, jax.numpy as jnp
    o = cp.generate_cover_probes(jax.random.PRNGKey(0), 12)
    assert o.shape==(12,28), o.shape
    assert jnp.all(o[:,20:24]==1.0), 'mask must be all-ones'   # env invariant
    print('crit', float(cc.criticality_score(o[0], 0.0, __import__('agent.csn_ppo.config',fromlist=['CSNPPOConfig']).CSNPPOConfig())))
    print('IMPORT+PROBE OK')"
  ASSERT: probe shape (N,28), mask slice all ones, criticality finite in [0.1,10].

STEP 1 (short end-to-end smoke, ~5 min CPU):
  CUDA_VISIBLE_DEVICES= /opt/venv/bin/python -m praxis.train_csn --smoke --num-timesteps 300000 --num-envs 256 --seed 0
  (num_envs 256 keeps CPU memory sane; divisibility 256*20=5120 % (batch*minib). Set batch_size=128,
   num_minibatches=40 -> 5120, ratio 1; OR keep batch=256,num_minibatches=20 ->5120. Config must pass assert.)
  ASSERT (parse stdout / returned metrics dict):
    (a) NO NaN/Inf in any logged metric across all updates (grep -i nan; assert none).
    (b) CSN metrics present and finite EVERY update once guard_active flips True:
        memory/kl_p95, memory/kl_mean, memory/policy_loss, memory/guard_active.
    (c) ppo/holdout_surrogate logged and finite; ppo/generalization_gap logged.
    (d) at least one update logs epoch/stopped_at < max_updates_per_batch OR guard inert early
        then active later (size crosses min_memory_size_before_guard).
    (e) memory/fast_size strictly increases over first few updates then saturates at capacity.
    (f) eval/episode_coverage logged, finite, in [0,1], and > 0.05 by end (learning signal).
    (g) gradient projection sanity: with --enable-projection vs --no-enable-projection both run
        without shape errors (compare one update each).

STEP 2 (guard correctness micro-check, no env):
  Construct a memory_batch where teacher == current policy output (zero KL) -> memory_guard_loss
  returns ~0; perturb params -> loss > 0 and kl_p95 > 0. Assert. (Confirms apply_policy_value
  wiring: dist.loc/log(dist.scale)/value order, normalizer passed first to .apply.)

STEP 3 (FULL killer test, 10M, GPU when available; CPU if forced):
  Baseline (already known): praxis/train.py 10M coverage peaks eval/episode_coverage ~0.82 @~1.3M,
  collapses to ~0.27 @10M.
  CSN run: CUDA_VISIBLE_DEVICES=0 python -m praxis.train_csn --num-timesteps 10000000 --num-envs 2048
    --seed 0  (and a paired --no-enable-guard --no-enable-projection ABLATION = should reproduce
    the baseline collapse, proving the harness itself isn't the fix).
  COMPARE / PASS CRITERIA (README §31 acceptance, MVP subset):
    * CSN eval/episode_coverage at 10M >= 0.6 (vs baseline ~0.27) — collapse prevented.
    * CSN peak coverage >= baseline peak (~0.82) within tolerance; held, not collapsed.
    * memory/kl_p95 stays < memory_kl_limit_p95 (0.05) except short windows.
    * holdout early-stop fires on >=some updates (epoch/stopped_at < max) — overfit caught.
    * Ablation (guard+projection OFF) collapses like baseline -> isolates the mechanism.
  Log curves: coverage vs env_steps for {baseline, CSN, CSN-ablation} on one plot.


## RISKS (from design)

1. SURROGATE SIGN (HIGH). holdout_surrogate is derived from brax metrics['policy_loss']. brax's
   policy_loss is the NEGATED clipped surrogate (loss form), so surrogate=-policy_loss. If the sign
   is wrong, should_stop_epoch's "holdout improves" logic inverts and early-stop never/always fires.
   MITIGATION: smoke STEP1 assert train_surrogate rises across epochs on a fresh batch; if it falls,
   flip the sign. Verify empirically before the 10M run.

2. advantage_abs for criticality (MED). The spec offers two paths (real |GAE adv| vs zeros). Recomputing
   GAE on flattened obs to align M states with adv is fiddly (T,B vs B,T axes; flatten order must match
   obs_flat). If misaligned, criticality's advantage term is garbage. MITIGATION: start with adv_abs=0
   (collision/frontier/dynamic/novelty still give a valid coverage criticality) for the first smoke;
   add real |adv| only after confirming flatten-order parity (assert obs_flat[i] corresponds to adv[i]
   by reconstructing one index). Low blast radius since other terms dominate.

3. EAGER OUTER LOOP COST (MED). Keeping the update loop in Python (to read int(memory.size) and do
   Python early-stop) means per-epoch jit dispatch overhead. On CPU at num_envs=2048 this is slow but
   correct. MITIGATION: jit the heavy inner functions (collect scan, ppo_value_and_grad, guard grads);
   accept Python control flow for early-stop. For the 10M run use GPU. Do NOT try to lax.scan the whole
   update loop in the MVP (early-stop + memory.size gate need host control flow).

4. MEMORY RING vs FIXED SHAPE (MED). insert_atoms is fixed-capacity ring; mining must emit FIXED
   atoms_per_rollout via jax.lax.top_k (NOT boolean masking which is dynamic). The slow/fast split uses
   weight-zeroing (fixed shape), NOT gather. If an implementer uses dynamic gather, jit breaks. Spec
   pins top_k + weight-zero; follow exactly.

5. PROBE MANIFOLD VALIDITY (MED). Coverage probes synthesize obs the env never normalizes the same way
   if ranges are off (rel-pos must be in ~[-1,1] arena-normalized units, frontier dist in ~[0,1]). Out-of-
   manifold probes teach the guard to protect nonsense. MITIGATION: keep probe value ranges within the
   ranges the smoke prints from REAL rollout obs (sample a real batch, print min/max per slice, ensure
   probe ranges sit inside). mask MUST be all-ones (asserted in smoke).

6. BOOTSTRAP/TIME_OUT plumbing (MED). bootstrap_on_timeout needs extras['state_extras']['time_out'] AND
   extras['policy_extras']['value']; the latter requires make_inference_fn(compute_value=True) and
   generate_unroll extra_fields including 'time_out'. If either is missing the reward rewrite KeyErrors
   or silently no-ops. Spec sets compute_value=True and extra_fields includes 'time_out' — verify keys
   exist in smoke (print data.extras structure once).

7. GUARD INACTIVE EARLY (LOW). Until memory_fast.size >= min_memory_size_before_guard the guard/projection
   are inert, so the first few updates are plain PPO. That's intended (memory must fill first) but means
   the killer-test protection only engages after ~min_size/atoms_per_rollout updates. Ensure
   min_memory_size_before_guard is small enough (4096 / 1024 = 4 updates) to engage well before the
   ~1.3M-step collapse onset.

8. CLUSTER BUCKETING SEMANTICS (LOW). guarded_loss bucket masks key off cluster_id/source_id integer ids
   that originally meant nav clusters. cluster_id_for reuses ids 0/1/2/3 with coverage meanings; the
   "successful_goal"(1) bucket is repurposed to open/progress. Buckets still partition correctly (masks are
   mutually consistent), so projection-by-bucket works; only the human-readable bucket NAME is slightly
   off. No correctness impact; note it in a comment.

9. GPU STILL BLOCKED (KNOWN). All smoke is CPU-only (CUDA_ERROR_NO_DEVICE in this WSL per repo RESUME.md).
   The 10M killer test (STEP 3) needs GPU; it remains blocked until the driver/WSL issue is resolved.
   Code is CPU-correct; the full comparison is gated on GPU availability. Flag this in the run report.
