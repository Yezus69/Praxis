# CSN-PPO: Contract-Sentinel Nullspace PPO for Praxis

**Purpose:** Build a long-horizon PPO training system for the Praxis navigation agent that resists both catastrophic forgetting and overfitting over 100M+ environment steps.

**Target repo:** `Yezus69/Praxis`
---

## 0. Executive summary

Plain PPO is not enough for 100M+ steps. PPO's trust region protects the policy only on the **latest rollout distribution**. With curriculum, domain randomization, and changing policy-induced state visitation, the latest rollout distribution is a tiny slice of everything the agent has previously learned.

This creates two failure modes:

- **Overfitting:** the policy improves on the current rollout batch but degrades on nearby heldout/current states or synthetic edge cases.
- **Catastrophic forgetting:** the policy improves on the current rollout distribution but destroys behavior on older state distributions.

These are the same failure viewed at different time scales:

```math
\text{overfitting} = \text{short-horizon uncontrolled functional drift}
```

```math
\text{forgetting} = \text{long-horizon uncontrolled functional drift}
```

The solution is to turn PPO from a **single-batch optimizer** into a **longitudinal functional optimizer**.

Every update must improve the current rollout while staying inside a compact, learned envelope of historically useful behavior.

The algorithm is **CSN-PPO: Contract-Sentinel Nullspace PPO**.

Core idea:

> Learn new behavior only in parameter/function directions that do not violate a sparse set of remembered behavioral invariants.

Those invariants are not a huge replay buffer of old PPO data. They are compact **sentinels**:

1. selected real states from old rollouts,
2. synthetic probes generated from the Praxis observation contract,
3. fixed-seed deterministic evaluation worlds,
4. best historical policies per behavior cluster, used as a mosaic teacher.

The optimizer treats these as a long-term trust region.

---

## 1. Repo assumptions

This design assumes the current Praxis MVP architecture:

- Simulator: MuJoCo Playground / MJX.
- Trainer baseline: Brax PPO.
- Agent: small MLP actor-critic.
- Observation: privileged low-dimensional state, not pixels.
- Action: continuous planar velocity `[vx, vy]`.
- Task: reach randomized goal while avoiding static and scripted dynamic obstacles.
- Observation uses fixed `K=4` nearest obstacles, sorted by distance, with active masks.
- Metrics should include at least `success`, `collision`, and reward components.

Expected observation layout:

```text
[ goal: dx, dy, dist, heading_err             (4) ]
[ agent vel: vx, vy, omega                    (3) ]
[ K nearest obstacles: (px, py, vx, vy) each  (4K) ]   # K = 4
[ per-slot active mask                        (K) ]
```

For `K=4`, total dimension is:

```math
4 + 3 + 4K + K = 4 + 3 + 16 + 4 = 27
```

Action dimension:

```math
a \in [-1, 1]^2
```

---

## 2. Why plain PPO forgets

At each PPO update, the optimizer sees only the latest on-policy batch:

```math
D_t = \{o_i, a_i, r_i, A_i, \log \pi_{\theta_t}(a_i|o_i)\}
```

PPO constrains policy change only on the observations in that batch:

```math
D_{KL}\left(\pi_{\theta_t}(\cdot|o_i), \pi_\theta(\cdot|o_i)\right)
```

But after millions of steps, useful behavior lives on a much larger historical distribution:

```math
\bar{\mu}_t = \sum_{k=1}^{t} w_k \mu_k
```

where each \(\mu_k\) is a previous policy's state distribution over different obstacle layouts, starts, goals, curriculum stages, and domain-randomized regimes.

Forgetting happens when:

```math
\text{small KL on } D_t
\quad\text{but}\quad
\text{large KL on } \bar{\mu}_t \setminus D_t
```

Overfitting is the same mechanism on a shorter time scale:

```math
\text{improve current rollout objective}
\quad\text{but}\quad
\text{degrade heldout/current-neighbor states}
```

Therefore, PPO's trust region must be extended from the current rollout distribution to:

1. current rollout states,
2. heldout rollout states,
3. compressed historical behavior states,
4. synthetic contract-space probes,
5. closed-loop fixed-seed sentinel worlds.

---

## 3. Algorithm: Contract-Sentinel Nullspace PPO

CSN-PPO adds eight mechanisms to PPO:

1. **Behavioral Sentinel Memory**
2. **Hinge-KL policy/value guard loss**
3. **Nullspace-protected gradient projection**
4. **Fixed-seed sentinel evaluations**
5. **Mosaic historical teacher**
6. **Contract-space synthetic probes**
7. **PPO holdout early stopping**
8. **Nonzero historical curriculum mixture**

The training objective is still PPO on fresh on-policy rollouts. Old samples are **not** used as off-policy PPO experience. They are used only as functional constraints.

---

## 4. Behavioral Sentinel Memory

Because Praxis uses 27-dimensional low-dimensional observations, memory is cheap.

Each memory atom stores what a historically useful policy did at a specific observation.

```python
BehavioralAtom:
    obs: float32[27]
    teacher_action_mean: float32[2]
    teacher_action_logstd: float32[2]
    teacher_value: float32[]
    teacher_id: int32
    domain_cluster: int32
    criticality: float32
    kl_budget: float32
    value_budget: float32
    age: int32
    source: enum(real_rollout, synthetic_probe, sentinel_failure)
```

Important rule:

> Do not reuse old advantages as PPO data.

That would bias PPO and break its on-policy assumption. Memory is only for:

- policy distillation,
- value consistency,
- behavior protection,
- gradient projection.

---

## 5. Fixed-size JAX memory structure

Inside jitted update paths, avoid Python lists and dynamic shapes. Use fixed-capacity JAX arrays.

```python
from dataclasses import dataclass
import jax
import jax.numpy as jnp


@dataclass
class BehavioralMemory:
    obs: jnp.ndarray              # [N, 27]
    mean: jnp.ndarray             # [N, 2]
    logstd: jnp.ndarray           # [N, 2]
    value: jnp.ndarray            # [N]
    weight: jnp.ndarray           # [N]
    kl_budget: jnp.ndarray        # [N]
    value_budget: jnp.ndarray     # [N]
    cluster_id: jnp.ndarray       # [N]
    source_id: jnp.ndarray        # [N]
    age: jnp.ndarray              # [N]
    write_idx: jnp.ndarray        # scalar int32
    size: jnp.ndarray             # scalar int32


@dataclass
class BehavioralMemoryBatch:
    obs: jnp.ndarray              # [B, 27]
    mean: jnp.ndarray             # [B, 2]
    logstd: jnp.ndarray           # [B, 2]
    value: jnp.ndarray            # [B]
    weight: jnp.ndarray           # [B]
    kl_budget: jnp.ndarray        # [B]
    value_budget: jnp.ndarray     # [B]
    cluster_id: jnp.ndarray       # [B]
    source_id: jnp.ndarray        # [B]
```

Insertion:

```python
def insert_atoms(memory: BehavioralMemory, atoms: BehavioralMemoryBatch) -> BehavioralMemory:
    n = atoms.obs.shape[0]
    idx = (memory.write_idx + jnp.arange(n)) % memory.obs.shape[0]

    return BehavioralMemory(
        obs=memory.obs.at[idx].set(atoms.obs),
        mean=memory.mean.at[idx].set(atoms.mean),
        logstd=memory.logstd.at[idx].set(atoms.logstd),
        value=memory.value.at[idx].set(atoms.value),
        weight=memory.weight.at[idx].set(atoms.weight),
        kl_budget=memory.kl_budget.at[idx].set(atoms.kl_budget),
        value_budget=memory.value_budget.at[idx].set(atoms.value_budget),
        cluster_id=memory.cluster_id.at[idx].set(atoms.cluster_id),
        source_id=memory.source_id.at[idx].set(atoms.source_id),
        age=memory.age.at[idx].set(0),
        write_idx=(memory.write_idx + n) % memory.obs.shape[0],
        size=jnp.minimum(memory.size + n, memory.obs.shape[0]),
    )
```

Sampling:

```python
def sample_memory(memory: BehavioralMemory, rng: jax.Array, batch_size: int) -> BehavioralMemoryBatch:
    max_idx = jnp.maximum(memory.size, 1)
    idx = jax.random.randint(rng, (batch_size,), 0, max_idx)

    return BehavioralMemoryBatch(
        obs=memory.obs[idx],
        mean=memory.mean[idx],
        logstd=memory.logstd[idx],
        value=memory.value[idx],
        weight=memory.weight[idx],
        kl_budget=memory.kl_budget[idx],
        value_budget=memory.value_budget[idx],
        cluster_id=memory.cluster_id[idx],
        source_id=memory.source_id[idx],
    )
```

---

## 6. Behavioral guard loss

For each memory atom \(m\), store the historical teacher policy distribution:

```math
\pi_m^T(a|o_m)
```

For Praxis continuous actions, assume a diagonal Gaussian:

```math
\pi_m^T(a|o_m) = \mathcal{N}(\mu_m^T, \operatorname{diag}((\sigma_m^T)^2))
```

Current policy:

```math
\pi_\theta(a|o_m) = \mathcal{N}(\mu_\theta(o_m), \operatorname{diag}(\sigma_\theta(o_m)^2))
```

Compute the KL:

```math
KL_m = D_{KL}\left(\pi_m^T(\cdot|o_m) \;\|\; \pi_\theta(\cdot|o_m)\right)
```

Do **not** penalize all deviation. That freezes learning. Use a hinge envelope:

```math
L_{\text{guard-policy}}
=
\mathbb{E}_{m \sim M}
\left[
    w_m \cdot \max(0, KL_m - \delta_m)^2
\right]
```

where:

- \(w_m\) is the memory atom's importance weight,
- \(\delta_m\) is the allowed KL drift budget.

Critical states get tight budgets:

```text
near collision boundary       small kl_budget
successful navigation states  small/medium kl_budget
ordinary old states           medium kl_budget
stale weak-policy states      large kl_budget
```

Value function guard:

```math
L_{\text{guard-value}}
=
\mathbb{E}_{m \sim M}
\left[
    w_m \cdot
    \max(0, |V_\theta(o_m) - V_m^T| - \rho_m)^2
\right]
```

where \(\rho_m\) is the value drift budget.

Full training loss:

```math
L =
L_{\text{PPO}}
+ c_v L_{\text{value}}
- c_e H(\pi_\theta)
+ \lambda_\pi L_{\text{guard-policy}}
+ \lambda_v L_{\text{guard-value}}
```

The hinge is crucial:

> If the new policy remains within the behavioral envelope, the memory loss is zero.

This allows progress while preventing destructive drift.

---

## 7. Diagonal Gaussian KL implementation

For diagonal Gaussians:

```math
D_{KL}(\mathcal{N}_0 \| \mathcal{N}_1)
= \frac{1}{2} \sum_i
\left(
\frac{\sigma_{0,i}^2 + (\mu_{0,i} - \mu_{1,i})^2}{\sigma_{1,i}^2}
- 1
+ 2(\log\sigma_{1,i} - \log\sigma_{0,i})
\right)
```

Implementation:

```python
def gaussian_kl(mean0, logstd0, mean1, logstd1):
    """KL[N(mean0, std0) || N(mean1, std1)] for diagonal Gaussians.

    Args:
        mean0: [..., action_dim]
        logstd0: [..., action_dim]
        mean1: [..., action_dim]
        logstd1: [..., action_dim]

    Returns:
        kl: [...]
    """
    var0 = jnp.exp(2.0 * logstd0)
    var1 = jnp.exp(2.0 * logstd1)

    kl_per_dim = 0.5 * (
        (var0 + (mean0 - mean1) ** 2) / (var1 + 1e-8)
        - 1.0
        + 2.0 * (logstd1 - logstd0)
    )

    return jnp.sum(kl_per_dim, axis=-1)
```

---

## 8. Guard loss implementation sketch

`apply_policy_value` should return current policy mean, current policy logstd, and current value for a batch of observations.

```python
def memory_guard_loss(params, normalizer_params, memory_batch, apply_policy_value):
    pred_mean, pred_logstd, pred_value = apply_policy_value(
        params,
        normalizer_params,
        memory_batch.obs,
    )

    kl = gaussian_kl(
        memory_batch.mean,
        memory_batch.logstd,
        pred_mean,
        pred_logstd,
    )

    policy_violation = jax.nn.relu(kl - memory_batch.kl_budget)
    policy_loss = jnp.mean(
        memory_batch.weight * policy_violation ** 2
    )

    value_error = jnp.abs(pred_value - memory_batch.value)
    value_violation = jax.nn.relu(value_error - memory_batch.value_budget)
    value_loss = jnp.mean(
        memory_batch.weight * value_violation ** 2
    )

    metrics = {
        "memory/kl_mean": jnp.mean(kl),
        "memory/kl_p95": jnp.percentile(kl, 95),
        "memory/policy_violation_frac": jnp.mean(policy_violation > 0),
        "memory/value_violation_frac": jnp.mean(value_violation > 0),
        "memory/policy_loss": policy_loss,
        "memory/value_loss": value_loss,
    }

    return policy_loss + 0.25 * value_loss, metrics
```

Implementation note:

- If `jnp.percentile` creates compatibility issues under JIT, replace it with an approximate percentile using `jnp.sort` and integer indexing.

---

## 9. Nullspace-protected PPO updates

Regularization alone is too passive.

If the current PPO gradient directly conflicts with old behavior, the optimizer should not merely add a penalty and hope Adam balances it. It should modify the update direction.

Let:

```math
g_{\text{ppo}} = \nabla_\theta L_{\text{PPO-current}}
```

and:

```math
g_{\text{mem}} = \nabla_\theta L_{\text{guard}}
```

Assume we are minimizing loss. If:

```math
g_{\text{ppo}}^\top g_{\text{mem}} < 0
```

then the PPO update would increase the memory guard loss. Project out the destructive component:

```math
g_{\text{safe}}
=
g_{\text{ppo}}
-
\frac{
\min(0, g_{\text{ppo}}^\top g_{\text{mem}})
}{
\|g_{\text{mem}}\|^2 + \epsilon
}
g_{\text{mem}}
```

Then optimize with:

```math
g = g_{\text{safe}} + \lambda_{\text{mem}} g_{\text{mem}}
```

Interpretation:

> PPO may update freely in the nullspace of remembered behavior.  
> It may update weakly inside protected directions.  
> It may not update in directions that actively damage protected behavior.

This is the algorithmic core of CSN-PPO.

---

## 10. Gradient tree utilities

```python
def tree_dot(a, b):
    return sum(
        jnp.vdot(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def tree_add_scaled(a, b, scale):
    return jax.tree_util.tree_map(lambda x, y: x + scale * y, a, b)


def tree_scalar_mul(a, scale):
    return jax.tree_util.tree_map(lambda x: scale * x, a)


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)
```

---

## 11. Project conflicting gradients

Project by bucket, not by one giant memory gradient.

Suggested buckets:

```text
bucket 1: collision-boundary memories
bucket 2: successful-goal memories
bucket 3: dynamic-obstacle memories
bucket 4: no-obstacle straight-line memories
bucket 5: recent-current memories
bucket 6: synthetic contract probes
bucket 7: sentinel regression states
```

Critical buckets should be projected first.

```python
def project_conflicting_gradient(g_ppo, memory_grads, eps=1e-8):
    """Removes gradient components that would increase memory losses.

    Args:
        g_ppo: pytree gradient of current PPO loss.
        memory_grads: list[pytree], each gradient of a guard bucket.
        eps: numerical stabilizer.

    Returns:
        safe PPO gradient pytree.
    """
    g = g_ppo

    for g_mem in memory_grads:
        dot = tree_dot(g, g_mem)
        norm = tree_dot(g_mem, g_mem) + eps

        # If dot < 0, the PPO gradient conflicts with the memory gradient.
        coeff = jnp.minimum(dot, 0.0) / norm

        # Remove only the conflicting component.
        g = tree_add_scaled(g, g_mem, -coeff)

    return g
```

Combine with guard gradients:

```python
def combine_safe_and_guard_grads(g_safe, memory_grads, memory_coefs):
    g = g_safe
    for g_mem, coef in zip(memory_grads, memory_coefs):
        g = tree_add_scaled(g, g_mem, coef)
    return g
```

---

## 12. Why this is different from EWC, replay, and KL-to-old-policy

### Not replay

Replay tries to train PPO on stale data. That violates the on-policy assumption if old advantages/actions are reused as RL targets.

CSN-PPO does **not** use old samples as PPO experience. It uses them as functional constraints.

### Not EWC

EWC protects parameters. In overparameterized networks, parameter distance is a poor proxy for behavior.

CSN-PPO protects policy/value outputs on important observations and protects gradient directions that preserve those outputs.

### Not PPO KL

PPO already has local KL control. The missing piece is longitudinal KL control across historical state distributions.

CSN-PPO adds that missing longitudinal trust region.

### Not checkpointing

Checkpoints remember old policies, but they do not make the current policy retain anything.

CSN-PPO distills the best historical behaviors into the live policy.

### Not domain randomization alone

Domain randomization helps generalization, but if the distribution shifts over training, old regimes can still vanish.

CSN-PPO keeps a nonzero historical mixture and sentinel bank.

---

## 13. Sentinel worlds: closed-loop forgetting detection

Pointwise memory is not enough.

A policy can match old actions on old observations and still fail closed-loop because small action differences shift future states.

CSN-PPO therefore keeps a bank of deterministic evaluation worlds.

```python
SentinelSeed:
    reset_rng: PRNGKey
    domain_randomization_params
    cluster_id: int32
    difficulty: float32
    best_return: float32
    best_success_rate: float32
    best_collision_rate: float32
    champion_policy_id: int32
```

Every `sentinel_eval_interval` PPO updates:

1. Run deterministic evaluation on the sentinel bank.
2. Track success rate, collision rate, return, and episode length.
3. Compare against the best historical policy for each cluster.
4. If a cluster regresses, mine failed trajectory states into memory with high criticality.
5. Temporarily increase memory guard weight for that cluster.
6. Stop curriculum advancement until the regression recovers.

Regression rule:

```python
def sentinel_regressed(current, best, success_tol=0.05, collision_tol=0.03):
    success_bad = current.success_rate < best.success_rate - success_tol
    collision_bad = current.collision_rate > best.collision_rate + collision_tol
    return success_bad | collision_bad
```

Cluster-level rule:

```text
if sentinel_success(cluster) < best_success(cluster) - tolerance:
    increase lambda_mem for cluster
    add failed states to memory
    reduce learning rate temporarily
    stop curriculum advancement
```

---

## 14. Mosaic teacher

A naive anti-forgetting system clones the previous policy. That is wrong.

Early policies are bad. You do not want to preserve bad behavior forever.

Instead, use a **mosaic teacher**.

For each sentinel/domain cluster, maintain the best historical policy:

```text
cluster 0: no obstacles                 champion = checkpoint 120
cluster 1: static clutter               champion = checkpoint 680
cluster 2: crossing dynamic obstacle    champion = checkpoint 1440
cluster 3: narrow passage               champion = checkpoint 2100
cluster 4: near-goal precision          champion = checkpoint 1880
```

When memory atoms are inserted, label them with the best known teacher for their cluster, not necessarily the current policy.

The current student is trained to become the union of best historical behaviors:

```math
\pi_\theta
\approx
\operatorname{distill}\left(
\pi^*_{\text{cluster 1}},
\pi^*_{\text{cluster 2}},
\ldots
\right)
```

This avoids the stability-plasticity trap:

- Do not freeze all old behavior.
- Preserve historically excellent behavior.
- If the current policy beats the champion on a cluster for several evaluations, it becomes the new champion.
- Then memory targets for that cluster can be refreshed.

Champion update rule:

```python
def maybe_update_champion(cluster_metrics, current_policy_id, champions, min_margin=0.02, patience=3):
    """Update champion when current policy reliably beats previous cluster champion."""
    for cluster_id, metrics in cluster_metrics.items():
        champ = champions[cluster_id]

        better_success = metrics.success_rate > champ.best_success_rate + min_margin
        no_collision_regression = metrics.collision_rate <= champ.best_collision_rate + 0.01
        better_return = metrics.mean_return > champ.best_return

        if better_success and no_collision_regression and better_return:
            champ.consecutive_wins += 1
        else:
            champ.consecutive_wins = 0

        if champ.consecutive_wins >= patience:
            champ.policy_id = current_policy_id
            champ.best_success_rate = metrics.success_rate
            champ.best_collision_rate = metrics.collision_rate
            champ.best_return = metrics.mean_return
            champ.consecutive_wins = 0

    return champions
```

---

## 15. Contract-space synthetic probes

Praxis has a semantic low-dimensional observation contract. That means we can generate valid edge-case observations without rolling out the simulator.

Synthetic probes fight overfitting because the agent cannot merely memorize the latest sampled layouts. It must remain consistent on the semantic manifold implied by the task.

Generate probes such as:

```text
1. no obstacle, goal straight ahead
2. obstacle directly between agent and goal
3. moving obstacle crossing left-to-right
4. moving obstacle crossing right-to-left
5. obstacle behind agent, should mostly ignore
6. near-goal, low-speed precision case
7. high-speed collision-boundary case
8. empty mask / partial mask cases
9. K=4 all active, correctly sorted by distance
10. padded inactive obstacle slots
11. goal behind agent, turn/redirect case
12. narrow passage with two active obstacles
13. obstacle moving away, should not overreact
14. obstacle moving toward agent, should evade
15. high angular error with low linear velocity
```

Synthetic probes must preserve:

- fixed shape,
- `K=4`,
- nearest-obstacle sorting,
- active masks,
- zero padding for inactive obstacle slots,
- plausible ranges for positions and velocities.

---

## 16. Synthetic probe generator sketch

```python
def pack_contract_obs(goal, agent_vel, obstacles, mask):
    """Pack Praxis 27D observation.

    Args:
        goal: tuple/list (dx, dy, dist, heading_err)
        agent_vel: tuple/list (vx, vy, omega)
        obstacles: array-like [4, 4], each row (px, py, vx, vy)
        mask: array-like [4]
    """
    return jnp.concatenate([
        jnp.asarray(goal, dtype=jnp.float32),
        jnp.asarray(agent_vel, dtype=jnp.float32),
        jnp.asarray(obstacles, dtype=jnp.float32).reshape(-1),
        jnp.asarray(mask, dtype=jnp.float32),
    ], axis=0)


def sort_and_pad_obstacles(obstacles, max_k=4):
    """Sort active obstacles by distance and pad to K=4."""
    # obstacles: [N, 4] where columns are px, py, vx, vy
    d = jnp.sqrt(obstacles[:, 0] ** 2 + obstacles[:, 1] ** 2)
    order = jnp.argsort(d)
    obstacles = obstacles[order]

    n = jnp.minimum(obstacles.shape[0], max_k)
    clipped = obstacles[:max_k]

    pad_n = max_k - clipped.shape[0]
    padded = jnp.pad(clipped, ((0, pad_n), (0, 0)))

    # For pure JIT use, prefer fixed-size generation to avoid dynamic pad_n.
    mask = jnp.concatenate([
        jnp.ones((clipped.shape[0],), dtype=jnp.float32),
        jnp.zeros((pad_n,), dtype=jnp.float32),
    ])

    return padded, mask
```

JIT-safe version should avoid dynamic obstacle count. Prefer generating fixed `[4, 4]` arrays directly with masks.

Example blocked-path probe:

```python
def make_probe_blocked_path(rng):
    rng_goal, rng_obs, rng_vel = jax.random.split(rng, 3)

    # Sample goal vector.
    angle = jax.random.uniform(rng_goal, (), minval=-jnp.pi, maxval=jnp.pi)
    dist = jax.random.uniform(rng_goal, (), minval=2.0, maxval=8.0)
    dx = dist * jnp.cos(angle)
    dy = dist * jnp.sin(angle)
    heading_err = angle

    # Unit vector toward goal and perpendicular vector.
    gx = dx / (dist + 1e-8)
    gy = dy / (dist + 1e-8)
    nx = -gy
    ny = gx

    # Obstacle lies near the line from agent to goal.
    alpha = jax.random.uniform(rng_obs, (), minval=0.25, maxval=0.75)
    lateral = jax.random.normal(rng_obs, ()) * 0.15
    ox = alpha * dx + lateral * nx
    oy = alpha * dy + lateral * ny

    # Obstacle velocity crosses the path.
    speed = jax.random.uniform(rng_vel, (), minval=0.2, maxval=1.0)
    direction = jnp.where(jax.random.bernoulli(rng_vel), 1.0, -1.0)
    ovx = direction * speed * nx
    ovy = direction * speed * ny

    obstacles = jnp.array([
        [ox, oy, ovx, ovy],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=jnp.float32)

    mask = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    agent_vel = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

    return pack_contract_obs(
        goal=(dx, dy, dist, heading_err),
        agent_vel=agent_vel,
        obstacles=obstacles,
        mask=mask,
    )
```

No-obstacle probe:

```python
def make_probe_no_obstacle(rng):
    angle = jax.random.uniform(rng, (), minval=-jnp.pi, maxval=jnp.pi)
    dist = jax.random.uniform(rng, (), minval=1.0, maxval=8.0)

    dx = dist * jnp.cos(angle)
    dy = dist * jnp.sin(angle)
    heading_err = angle

    return pack_contract_obs(
        goal=(dx, dy, dist, heading_err),
        agent_vel=(0.0, 0.0, 0.0),
        obstacles=jnp.zeros((4, 4), dtype=jnp.float32),
        mask=jnp.zeros((4,), dtype=jnp.float32),
    )
```

---

## 17. Synthetic probe teachers

For probes, the teacher can be one of three sources:

1. mosaic historical teacher,
2. analytic geometric controller,
3. safer of the two.

For no-obstacle probes, the analytic teacher is simple:

```math
a^* = \operatorname{clip}\left(k \cdot \frac{g}{\|g\|}, [-1,1]\right)
```

Implementation:

```python
def analytic_no_obstacle_teacher(obs, speed=1.0):
    dx, dy, dist, _heading_err = obs[0], obs[1], obs[2], obs[3]
    direction = jnp.array([dx, dy]) / (dist + 1e-8)
    return jnp.clip(speed * direction, -1.0, 1.0)
```

For obstacle-blocked path, penalize action projected toward an obstacle that lies between agent and goal.

Sketch:

```python
def analytic_obstacle_teacher(obs):
    dx, dy, dist, _heading_err = obs[0], obs[1], obs[2], obs[3]
    goal_dir = jnp.array([dx, dy]) / (dist + 1e-8)

    obstacle_block = obs[7:7 + 16].reshape(4, 4)
    mask = obs[23:27]

    desired = goal_dir

    for k in range(4):
        px, py, ovx, ovy = obstacle_block[k]
        active = mask[k]
        rel = jnp.array([px, py])
        obs_dist = jnp.linalg.norm(rel) + 1e-8
        obs_dir = rel / obs_dist

        # Obstacle is considered blocking if it is in front of the agent and close to goal ray.
        forward = jnp.dot(obs_dir, goal_dir)
        lateral_dist = jnp.linalg.norm(rel - jnp.dot(rel, goal_dir) * goal_dir)
        blocking = (active > 0.5) & (forward > 0.3) & (lateral_dist < 0.75)

        # Evade perpendicular to obstacle direction.
        perp = jnp.array([-obs_dir[1], obs_dir[0]])
        evade_strength = active * blocking.astype(jnp.float32) * jnp.clip(1.5 / obs_dist, 0.0, 2.0)
        desired = desired + evade_strength * perp

    desired = desired / (jnp.linalg.norm(desired) + 1e-8)
    return jnp.clip(desired, -1.0, 1.0)
```

For synthetic memory atoms, store a Gaussian teacher with small but nonzero logstd:

```python
def make_teacher_distribution(action_mean, logstd=-2.0):
    return action_mean, jnp.full_like(action_mean, logstd)
```

---

## 18. Memory selection: what gets preserved?

Do not store random transitions uniformly. That wastes memory and protects irrelevant behavior.

Insert states with high long-term value:

```text
1. success states
2. near-collision states
3. actual collision precursor states
4. dynamic obstacle crossing states
5. high advantage magnitude states
6. high policy disagreement states
7. rare observation clusters
8. sentinel regression states
9. synthetic contract probes
```

Criticality score:

```math
c(o)
=
\alpha_1 |\hat{A}(o,a)|
+
\alpha_2 \cdot \text{collision\_proximity}(o)
+
\alpha_3 \cdot \text{success\_proximity}(o)
+
\alpha_4 \cdot \text{novelty}(o)
+
\alpha_5 \cdot \text{sentinel\_failure}(o)
```

Memory weight:

```math
w_m = \operatorname{clip}(c(o), w_{\min}, w_{\max})
```

KL budget:

```math
\delta_m = \frac{\delta_0}{1 + c(o)}
```

Value budget:

```math
\rho_m = \frac{\rho_0}{1 + \beta c(o)}
```

More critical memories get stronger protection.

---

## 19. Practical criticality features for Praxis

Given the 27D observation:

```python
def obstacle_distances(obs):
    obstacles = obs[7:23].reshape(4, 4)
    mask = obs[23:27]
    d = jnp.sqrt(obstacles[:, 0] ** 2 + obstacles[:, 1] ** 2)
    # Inactive obstacles get large distance.
    return jnp.where(mask > 0.5, d, 1e6)


def collision_proximity(obs, radius=0.75):
    d_min = jnp.min(obstacle_distances(obs))
    return jax.nn.relu(radius - d_min) / radius


def success_proximity(obs, goal_radius=0.5):
    dist_to_goal = obs[2]
    return jax.nn.relu(goal_radius - dist_to_goal) / goal_radius


def dynamic_obstacle_score(obs):
    obstacles = obs[7:23].reshape(4, 4)
    mask = obs[23:27]
    speeds = jnp.sqrt(obstacles[:, 2] ** 2 + obstacles[:, 3] ** 2)
    return jnp.max(mask * speeds)
```

Example criticality:

```python
def criticality_score(obs, advantage_abs, novelty, sentinel_failure):
    c = (
        1.0 * advantage_abs
        + 3.0 * collision_proximity(obs)
        + 2.0 * success_proximity(obs)
        + 1.0 * dynamic_obstacle_score(obs)
        + 1.0 * novelty
        + 5.0 * sentinel_failure
    )
    return jnp.clip(c, 0.1, 10.0)
```

---

## 20. Novelty and reservoir policy

Use two memory tiers:

```text
fast memory: recent/high-turnover, e.g. 1,048,576 atoms
slow memory: rare/critical/champion, e.g. 262,144 atoms
```

Fast memory captures recent behavior and adapts quickly.

Slow memory protects rare high-value states.

Slow-memory replacement should favor keeping high criticality and rare clusters:

```math
P(\text{replace } m) \propto \frac{1}{\epsilon + w_m} \cdot \frac{1}{1 + \text{rarity\_bonus}_m}
```

Simpler implementation:

```python
def should_insert_slow_memory(criticality, threshold=3.0):
    return criticality > threshold
```

Then use ring buffers for both memories initially. Add priority replacement later if needed.

---

## 21. Overfitting control inside PPO epochs

PPO often overfits because it performs multiple update epochs over the same rollout batch.

Split each rollout batch:

```text
current_train: 80%
current_holdout: 20%
```

After each PPO epoch, compute:

```text
train_surrogate
holdout_surrogate
train_approx_kl
holdout_approx_kl
memory_kl_p95
sentinel_guard_loss
```

Stop PPO epochs early when:

```text
train objective improves
but holdout objective degrades
```

or:

```text
holdout KL rises too fast
```

or:

```text
memory KL p95 exceeds budget
```

Rollback rule:

```python
if holdout_surrogate < best_holdout_surrogate - eps:
    rollback_to_best_epoch_params()
    stop_epochs_for_this_batch()
```

More complete sketch:

```python
def should_stop_epoch(
    holdout_score,
    best_holdout_score,
    memory_kl_p95,
    memory_kl_limit,
    approx_kl,
    target_kl,
    eps=1e-4,
):
    holdout_bad = holdout_score < best_holdout_score - eps
    memory_bad = memory_kl_p95 > memory_kl_limit
    kl_bad = approx_kl > 1.5 * target_kl
    return holdout_bad | memory_bad | kl_bad
```

This attacks overfitting while sentinel/nullspace machinery attacks forgetting. Same principle: prevent unmeasured functional drift.

---

## 22. Curriculum without distribution replacement

Do not train on an advancing curriculum that replaces old worlds. That manufactures forgetting.

Use a standing mixture:

```text
70% current frontier difficulty
20% uniformly sampled historical difficulties
10% adversarial/sentinel failures
```

Training world distribution:

```math
p_t(\text{world})
=
0.7 p_{\text{frontier}}
+
0.2 p_{\text{history}}
+
0.1 p_{\text{sentinel-failures}}
```

This prevents the data stream itself from erasing older behaviors.

MJX requires fixed model topology, so randomize continuous fields while keeping maximum obstacle count declared in the model.

Recommended randomized fields:

```text
start position
start velocity
goal position
static obstacle position
static obstacle size
moving obstacle phase
moving obstacle speed
moving obstacle direction
friction/contact parameters if exposed safely
active obstacle masks
```

Do not add/remove geoms dynamically under JIT. Declare max-N obstacles and mask or move extras.

---

## 23. Expected module layout

Add this under `agent/`:

```text
agent/
  train_csn_ppo.py
  csn_ppo/
    __init__.py
    README.md
    config.py
    memory.py
    sentinel.py
    synthetic_probes.py
    guarded_loss.py
    gradient_projection.py
    train.py
    metrics.py
    rollout_mining.py
    mosaic_teacher.py
    curriculum.py
```

File responsibilities:

```text
config.py
    CSNPPOConfig dataclass with all hyperparameters.

memory.py
    BehavioralMemory, BehavioralMemoryBatch, insertion, sampling, aging.

sentinel.py
    SentinelSeed, sentinel bank creation, deterministic evaluation, regression detection.

synthetic_probes.py
    Generate 27D contract-space probes and analytic teachers.

guarded_loss.py
    Gaussian KL, hinge-KL policy guard, value guard, bucketed guard losses.

gradient_projection.py
    tree_dot, tree_add_scaled, project_conflicting_gradient.

train.py
    Forked/wrapped PPO loop with CSN additions.

metrics.py
    Aggregation and logging of PPO, memory, sentinel, and curriculum metrics.

rollout_mining.py
    Select high-criticality states from fresh rollouts and failed sentinels.

mosaic_teacher.py
    Track champion checkpoints per cluster and label memory atoms.

curriculum.py
    Mixture of frontier/history/sentinel-failure worlds.
```

---

## 24. Config sketch

```python
from dataclasses import dataclass


@dataclass
class CSNPPOConfig:
    # PPO baseline
    num_timesteps: int = int(1e8)
    num_envs: int = 2048
    episode_length: int = 1000
    unroll_length: int = 20
    batch_size: int = 256
    num_minibatches: int = 32
    max_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.97
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    seed: int = 0

    # Behavioral memory
    memory_size_fast: int = 1_048_576
    memory_size_slow: int = 262_144
    memory_batch_size: int = 4096
    min_memory_size_before_guard: int = 16_384

    # Guard budgets
    guard_policy_coef: float = 1.0
    guard_value_coef: float = 0.25
    guard_kl_budget: float = 0.02
    critical_kl_budget: float = 0.005
    value_budget: float = 0.25
    critical_value_budget: float = 0.05
    memory_kl_limit_p95: float = 0.05

    # Gradient projection
    enable_gradient_projection: bool = True
    projection_eps: float = 1e-8

    # Sentinel evaluation
    sentinel_eval_interval: int = 25
    sentinel_bank_size: int = 4096
    sentinel_success_tolerance: float = 0.05
    sentinel_collision_tolerance: float = 0.03
    sentinel_patience: int = 3

    # Synthetic probes
    synthetic_probe_batch_size: int = 4096
    synthetic_probe_insert_interval: int = 1

    # Holdout overfit control
    holdout_fraction: float = 0.2
    holdout_eps: float = 1e-4
    target_kl: float = 0.03

    # Curriculum mixture
    frontier_fraction: float = 0.70
    history_fraction: float = 0.20
    sentinel_failure_fraction: float = 0.10

    # Mosaic teacher
    enable_mosaic_teacher: bool = True
    champion_min_margin: float = 0.02
    champion_patience: int = 3
```

---

## 25. Integration into Praxis training

Current baseline likely resembles:

```python
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper
import functools

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(256, 256, 256),
    value_hidden_layer_sizes=(256, 256, 256),
)

make_inference_fn, params, metrics = ppo.train(
    environment=wrapper.wrap_for_brax_training(env),
    network_factory=network_factory,
    num_timesteps=int(2e7),
    num_envs=2048,
    episode_length=1000,
    unroll_length=20,
    batch_size=256,
    num_minibatches=32,
    num_updates_per_batch=4,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.97,
    reward_scaling=1.0,
    normalize_observations=True,
    num_evals=10,
    log_training_metrics=True,
    save_checkpoint_path="ckpts/",
    seed=0,
)
```

Replace with:

```python
from agent.csn_ppo import train as csn_ppo
from agent.csn_ppo.config import CSNPPOConfig
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import wrapper
import functools

network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(256, 256, 256),
    value_hidden_layer_sizes=(256, 256, 256),
)

config = CSNPPOConfig(
    num_timesteps=int(1e8),
    num_envs=2048,
    episode_length=1000,
    unroll_length=20,
    batch_size=256,
    num_minibatches=32,
    max_updates_per_batch=4,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.97,
    reward_scaling=1.0,
    normalize_observations=True,
    seed=0,
)

make_inference_fn, params, memory, metrics = csn_ppo.train(
    environment=wrapper.wrap_for_brax_training(env),
    network_factory=network_factory,
    config=config,
    save_checkpoint_path="ckpts_csn/",
)
```

---

## 26. 100M-step arithmetic

With:

```text
num_envs = 2048
unroll_length = 20
```

one rollout batch contains:

```math
2048 \times 20 = 40{,}960
```

environment steps.

For 100M steps:

```math
100{,}000{,}000 / 40{,}960 \approx 2441
```

PPO update batches.

If sentinel evaluation runs every 25 updates:

```math
2441 / 25 \approx 98
```

sentinel evaluations.

This is practical.

---

## 27. Full CSN-PPO training loop pseudo-code

```python
def train(environment, network_factory, config, save_checkpoint_path):
    rng = jax.random.PRNGKey(config.seed)

    networks = network_factory(
        observation_size=environment.observation_size,
        action_size=environment.action_size,
        preprocess_observations_fn=...,  # use Brax normalizer if enabled
    )

    params = initialize_params(networks, rng)
    optimizer_state = initialize_optimizer(params, config.learning_rate)
    normalizer_params = initialize_normalizer(environment.observation_size)

    memory_fast = init_behavioral_memory(config.memory_size_fast, obs_dim=27, action_dim=2)
    memory_slow = init_behavioral_memory(config.memory_size_slow, obs_dim=27, action_dim=2)

    sentinel_bank = create_sentinel_bank(
        rng=rng,
        size=config.sentinel_bank_size,
        environment=environment,
    )

    champions = init_mosaic_champions()
    adaptive_guard_coefs = init_guard_coefficients(config)
    curriculum_state = init_curriculum_state(config)

    num_steps_per_update = config.num_envs * config.unroll_length
    num_updates = config.num_timesteps // num_steps_per_update

    env_state = reset_vectorized_env(environment, rng, config.num_envs, curriculum_state)

    for update in range(num_updates):
        rng, rollout_rng, mine_rng, probe_rng, opt_rng = jax.random.split(rng, 5)

        # 1. Collect fresh on-policy rollout.
        rollout, env_state, normalizer_params = collect_on_policy_rollout(
            environment=environment,
            env_state=env_state,
            params=params,
            normalizer_params=normalizer_params,
            networks=networks,
            rng=rollout_rng,
            unroll_length=config.unroll_length,
            curriculum_state=curriculum_state,
        )

        # 2. Compute advantages/returns using current rollout only.
        rollout = compute_gae_and_returns(
            rollout=rollout,
            params=params,
            normalizer_params=normalizer_params,
            networks=networks,
            discounting=config.discounting,
        )

        # 3. Split rollout into train/holdout to detect PPO epoch overfitting.
        train_batch, holdout_batch = split_rollout_train_holdout(
            rollout,
            holdout_fraction=config.holdout_fraction,
            rng=opt_rng,
        )

        # 4. Mine high-value behavior atoms from fresh rollout.
        candidate_atoms = mine_behavioral_atoms(
            rollout=rollout,
            params=params,
            normalizer_params=normalizer_params,
            networks=networks,
            rng=mine_rng,
            champions=champions,
        )
        memory_fast = insert_atoms(memory_fast, candidate_atoms.fast_atoms)
        memory_slow = insert_atoms(memory_slow, candidate_atoms.slow_atoms)

        # 5. Add synthetic contract-space probes.
        if update % config.synthetic_probe_insert_interval == 0:
            probe_obs = generate_contract_probes(
                rng=probe_rng,
                batch_size=config.synthetic_probe_batch_size,
            )
            probe_atoms = label_probe_atoms(
                obs=probe_obs,
                params=params,
                normalizer_params=normalizer_params,
                networks=networks,
                champions=champions,
                config=config,
            )
            memory_slow = insert_atoms(memory_slow, probe_atoms)

        best_epoch_params = params
        best_epoch_optimizer_state = optimizer_state
        best_holdout_score = -jnp.inf

        # 6. PPO epochs with guard losses and projection.
        for epoch in range(config.max_updates_per_batch):
            rng, mem_rng, minibatch_rng = jax.random.split(rng, 3)

            memory_batch_fast = sample_memory(memory_fast, mem_rng, config.memory_batch_size // 2)
            memory_batch_slow = sample_memory(memory_slow, mem_rng, config.memory_batch_size // 2)
            memory_batches_by_bucket = bucket_memory_batches(
                memory_batch_fast,
                memory_batch_slow,
            )

            # 6a. PPO loss and gradient on current fresh train data.
            ppo_loss_value, g_ppo, ppo_metrics = value_and_grad_ppo_loss(
                params=params,
                normalizer_params=normalizer_params,
                networks=networks,
                batch=train_batch,
                config=config,
            )

            # 6b. Guard losses and gradients by memory bucket.
            guard_loss_values = []
            guard_grads = []
            guard_metrics = {}

            for bucket_name, memory_batch in memory_batches_by_bucket.items():
                loss_value, grad_value, metrics_value = value_and_grad_guard_loss(
                    params=params,
                    normalizer_params=normalizer_params,
                    networks=networks,
                    memory_batch=memory_batch,
                    config=config,
                )
                guard_loss_values.append(loss_value)
                guard_grads.append(grad_value)
                guard_metrics.update(prefix_metrics(metrics_value, f"memory/{bucket_name}"))

            # 6c. Remove PPO gradient components that damage protected behavior.
            if config.enable_gradient_projection:
                g_safe = project_conflicting_gradient(
                    g_ppo,
                    guard_grads,
                    eps=config.projection_eps,
                )
            else:
                g_safe = g_ppo

            # 6d. Add guard gradients back with adaptive coefficients.
            g_total = combine_safe_and_guard_grads(
                g_safe,
                guard_grads,
                memory_coefs=adaptive_guard_coefs,
            )

            # 6e. Optimizer update.
            params_candidate, optimizer_state_candidate = optimizer_update(
                params=params,
                optimizer_state=optimizer_state,
                grads=g_total,
            )

            # 6f. Evaluate holdout/generalization and memory drift.
            holdout_score, holdout_metrics = evaluate_holdout_surrogate(
                params=params_candidate,
                normalizer_params=normalizer_params,
                networks=networks,
                batch=holdout_batch,
                config=config,
            )

            memory_kl_p95 = evaluate_memory_kl_p95(
                params=params_candidate,
                normalizer_params=normalizer_params,
                networks=networks,
                memory_batches=memory_batches_by_bucket,
            )

            approx_kl = ppo_metrics["ppo/approx_kl"]

            accept_as_best = (
                (holdout_score > best_holdout_score)
                & (memory_kl_p95 < config.memory_kl_limit_p95)
            )

            if accept_as_best:
                best_epoch_params = params_candidate
                best_epoch_optimizer_state = optimizer_state_candidate
                best_holdout_score = holdout_score

            stop_now = should_stop_epoch(
                holdout_score=holdout_score,
                best_holdout_score=best_holdout_score,
                memory_kl_p95=memory_kl_p95,
                memory_kl_limit=config.memory_kl_limit_p95,
                approx_kl=approx_kl,
                target_kl=config.target_kl,
                eps=config.holdout_eps,
            )

            params = params_candidate
            optimizer_state = optimizer_state_candidate

            if stop_now:
                params = best_epoch_params
                optimizer_state = best_epoch_optimizer_state
                break

        # 7. Fixed-seed sentinel evaluation.
        if update % config.sentinel_eval_interval == 0:
            sentinel_metrics, failed_trajectories = evaluate_sentinel_bank(
                environment=environment,
                sentinel_bank=sentinel_bank,
                params=params,
                normalizer_params=normalizer_params,
                networks=networks,
                deterministic=True,
            )

            regressions = detect_sentinel_regressions(
                sentinel_metrics=sentinel_metrics,
                sentinel_bank=sentinel_bank,
                config=config,
            )

            if any_regressions(regressions):
                failed_atoms = mine_failed_sentinel_states(
                    failed_trajectories=failed_trajectories,
                    regressions=regressions,
                    params=params,
                    normalizer_params=normalizer_params,
                    networks=networks,
                    criticality_bonus=5.0,
                )
                memory_slow = insert_atoms(memory_slow, failed_atoms)

                adaptive_guard_coefs = raise_cluster_guard_weights(
                    adaptive_guard_coefs,
                    regressions,
                )

                curriculum_state = freeze_or_slow_curriculum(curriculum_state)

            else:
                champions = maybe_update_mosaic_champions(
                    sentinel_metrics=sentinel_metrics,
                    params=params,
                    champions=champions,
                    config=config,
                    checkpoint_path=save_checkpoint_path,
                )

                adaptive_guard_coefs = slowly_relax_guard_weights(adaptive_guard_coefs)
                curriculum_state = maybe_advance_curriculum(curriculum_state, sentinel_metrics)

        # 8. Logging and checkpointing.
        metrics = merge_metrics(
            ppo_metrics,
            guard_metrics,
            holdout_metrics,
            sentinel_metrics if update % config.sentinel_eval_interval == 0 else {},
        )
        log_metrics(metrics, update)

        if should_checkpoint(update, metrics):
            save_checkpoint(save_checkpoint_path, params, normalizer_params, optimizer_state, update)

    return make_inference_fn(networks, normalizer_params), params, (memory_fast, memory_slow), metrics
```

---

## 28. PPO loss reminder

Standard PPO clipped objective:

```math
r_t(\theta) = \frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{old}}(a_t|o_t)}
```

```math
L_{\text{clip}}(\theta)
=
-\mathbb{E}_t
\left[
\min
\left(
    r_t(\theta) \hat{A}_t,
    \operatorname{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t
\right)
\right]
```

Value loss:

```math
L_V = \mathbb{E}_t[(V_\theta(o_t) - R_t)^2]
```

Entropy bonus:

```math
H(\pi_\theta) = \mathbb{E}_t[-\log \pi_\theta(a_t|o_t)]
```

Baseline PPO loss:

```math
L_{\text{PPO}}
=
L_{\text{clip}} + c_v L_V - c_e H(\pi_\theta)
```

CSN-PPO adds the guard terms and gradient projection around this baseline.

---

## 29. Constrained optimization view

The clean formulation:

```math
\min_\theta
\quad
L_{\text{PPO}}(\theta; D_t)
```

subject to:

```math
\mathbb{E}_{o \sim M_t}
\left[
D_{KL}(\pi^T(\cdot|o), \pi_\theta(\cdot|o))
\right]
\leq
\epsilon_M
```

```math
\text{SentinelSuccess}_c(\theta)
\geq
\text{BestSentinelSuccess}_c - \epsilon_S
\quad
\forall c
```

```math
\text{HoldoutPPO}(\theta)
\geq
\text{HoldoutPPO}(\theta_{\text{best epoch}}) - \epsilon_H
```

Practical optimizer:

```text
PPO gradient
-> remove components that increase memory loss
-> add hinge memory gradient
-> accept only if holdout and sentinels do not regress
```

---

## 30. Metrics to log

Do not judge success by training reward alone. Training reward is where overfitting hides.

Log these:

```text
current/eval_episode_reward
current/eval_success_rate
current/eval_collision_rate

ppo/train_surrogate
ppo/holdout_surrogate
ppo/generalization_gap
ppo/approx_kl
ppo/clip_fraction
ppo/entropy
ppo/value_loss

memory/kl_mean
memory/kl_p95
memory/policy_violation_frac
memory/value_violation_frac
memory/guard_loss
memory/fast_size
memory/slow_size

sentinel/success_rate_mean
sentinel/success_rate_min_cluster
sentinel/collision_rate_mean
sentinel/collision_rate_max_cluster
sentinel/regression_count
sentinel/worst_cluster_id

curriculum/frontier_success
curriculum/history_success
curriculum/adversarial_success
curriculum/current_difficulty

mosaic/num_champions
mosaic/champion_updates
mosaic/cluster_win_streak
```

Hard gates:

```text
1. sentinel success may not drop more than 5% from cluster best
2. sentinel collision may not rise more than 3% from cluster best
3. memory KL p95 must stay below budget
4. holdout PPO surrogate must not degrade while train surrogate improves
5. historical curriculum slice must remain above threshold
```

---

## 31. Acceptance criteria

A build should be considered successful when all are true:

```text
A. The agent trains for at least 100M env steps without NaNs or metric collapse.

B. Current eval success improves over the baseline PPO run.

C. Sentinel success mean does not collapse after curriculum advances.

D. Sentinel min-cluster success remains within 5% absolute of the best historical cluster score after the cluster is learned.

E. Collision rate does not regress more than 3% absolute on learned sentinel clusters.

F. Memory KL p95 remains below the configured budget except during short recovery windows.

G. Holdout surrogate catches at least some overfit epochs and triggers rollback/early-stop.

H. Failed sentinel trajectories are mined into memory and later recover.

I. Training logs make forgetting visible instead of silently hiding it.
```

---

## 32. Implementation phases for Codex/Claude Code

### Phase 1: Minimal viable CSN-PPO

Implement:

```text
memory.py
guarded_loss.py
gradient_projection.py
synthetic_probes.py
train.py fork/wrapper with memory guard loss
```

Skip temporarily:

```text
mosaic teacher
sentinel champions
advanced priority replacement
adaptive curriculum
```

Goal:

```text
PPO + memory hinge-KL + synthetic probes + gradient projection + holdout early stop
```

### Phase 2: Sentinel bank

Implement:

```text
sentinel.py
fixed-seed deterministic eval
sentinel regression detection
failed trajectory mining
cluster-specific guard coefficient increases
```

### Phase 3: Mosaic teacher

Implement:

```text
champion checkpoints per cluster
memory labeling from cluster champion
champion update rules
teacher refresh
```

### Phase 4: Curriculum mixture

Implement:

```text
70/20/10 frontier/history/sentinel-failure sampling
curriculum freeze on sentinel regression
curriculum advancement only when current + historical slices pass
```

### Phase 5: Optimization and cleanup

Implement:

```text
JIT compilation boundaries
metric cleanup
checkpoint metadata
unit tests
smoke tests
100M-step launch config
```

---

## 33. Unit tests

Add tests before long runs.

```text
tests/test_csn_memory.py
tests/test_csn_guarded_loss.py
tests/test_csn_gradient_projection.py
tests/test_csn_synthetic_probes.py
tests/test_csn_sentinel.py
```

Test cases:

```python
def test_gaussian_kl_zero_for_identical_distributions():
    mean = jnp.zeros((8, 2))
    logstd = jnp.zeros((8, 2))
    kl = gaussian_kl(mean, logstd, mean, logstd)
    assert jnp.allclose(kl, 0.0, atol=1e-6)


def test_guard_loss_zero_inside_budget():
    # Current distribution equals teacher distribution.
    # KL below budget, value error below budget.
    # Guard loss should be zero or near-zero.
    pass


def test_guard_loss_positive_outside_budget():
    # Current policy differs from teacher beyond KL budget.
    # Guard loss should be positive.
    pass


def test_projection_removes_conflict():
    g_ppo = {"x": jnp.array([-1.0, 0.0])}
    g_mem = {"x": jnp.array([1.0, 0.0])}
    g_safe = project_conflicting_gradient(g_ppo, [g_mem])
    assert tree_dot(g_safe, g_mem) >= -1e-6


def test_projection_leaves_non_conflict_alone():
    g_ppo = {"x": jnp.array([1.0, 0.0])}
    g_mem = {"x": jnp.array([1.0, 0.0])}
    g_safe = project_conflicting_gradient(g_ppo, [g_mem])
    assert jnp.allclose(g_safe["x"], g_ppo["x"])


def test_probe_obs_shape():
    obs = make_probe_no_obstacle(jax.random.PRNGKey(0))
    assert obs.shape == (27,)


def test_probe_masks_valid():
    obs = make_probe_no_obstacle(jax.random.PRNGKey(0))
    mask = obs[23:27]
    assert jnp.all((mask == 0.0) | (mask == 1.0))
```

---

## 34. Practical hyperparameter starting point

```text
memory_size_fast              1,048,576
memory_size_slow              262,144
memory_batch_size             4096
synthetic_probe_batch_size    4096
sentinel_bank_size            4096
sentinel_eval_interval        25 updates

guard_policy_coef             1.0
guard_value_coef              0.25
guard_kl_budget               0.02
critical_kl_budget            0.005
memory_kl_limit_p95           0.05

holdout_fraction              0.2
target_kl                     0.03
max_updates_per_batch         4

frontier/history/sentinel     70/20/10
```

If learning becomes too conservative:

```text
increase guard_kl_budget
reduce guard_policy_coef
reduce memory_batch_size
project only critical buckets first
relax slow-memory insertion
```

If forgetting persists:

```text
lower critical_kl_budget
increase sentinel_failure_fraction
increase guard_policy_coef for regressing clusters
increase sentinel bank diversity
increase synthetic probe diversity
```

If overfitting persists:

```text
increase holdout_fraction
lower target_kl
reduce max_updates_per_batch
increase synthetic_probe_batch_size
increase entropy_cost modestly
```

---

## 35. Sharp implementation warnings

1. **Do not train PPO on old advantages.**
   Old memory is for functional constraints only.

2. **Do not preserve all old behavior equally.**
   Preserve high-criticality and historically successful behavior.

3. **Do not clone the immediately previous policy forever.**
   Use the mosaic teacher: best checkpoint per cluster.

4. **Do not let curriculum replace old regimes.**
   Keep a permanent historical slice.

5. **Do not rely only on pointwise KL.**
   Use closed-loop sentinel worlds.

6. **Do not use dynamic shapes in JAX/MJX update paths.**
   Use fixed-size buffers and fixed-shape probes.

7. **Do not use training reward as the main success metric.**
   Use success, collision, sentinel regression, memory KL, and holdout surrogate.

8. **Do not freeze learning with hard behavior cloning.**
   Use hinge KL budgets so useful drift is allowed.

9. **Do not protect stale bad behavior.**
   Replace weak teachers when better champions emerge.

10. **Do not advance curriculum during sentinel regression.**
    Recover first.

---

## 36. Minimal MVP cut if time is limited

If implementing this in one shot is too much, build this subset first:

```text
1. fork Brax PPO training loop
2. add BehavioralMemory
3. mine high-advantage, success, near-collision states
4. add Gaussian hinge-KL guard loss
5. add gradient projection against guard gradient
6. split rollout into train/holdout and early-stop PPO epochs
7. add synthetic no-obstacle and blocked-path probes
8. log memory KL p95 and holdout surrogate
```

This gets the central mechanism working.

Then add sentinel worlds and mosaic teacher.

---

## 37. Final design invariant

The entire system exists to enforce one invariant:

> A PPO update is allowed only if it improves the current rollout without causing unbounded functional drift on remembered, synthetic, or closed-loop sentinel behavior.

Equivalently:

```math
\Delta L_{\text{PPO-current}} < 0
\quad\text{and}\quad
\Delta D_{\text{memory}} \leq \epsilon_M
\quad\text{and}\quad
\Delta R_{\text{sentinel}} \geq -\epsilon_S
```

That is the core of CSN-PPO.

---

## 38. Expected result

For the current Praxis setup - low-dimensional privileged state, fixed-K obstacles, continuous `[vx, vy]`, Brax/MJX PPO, and thousands of parallel environments - CSN-PPO should substantially reduce the usual 100M-step failure pattern:

```text
learn simple navigation
advance curriculum
specialize to recent obstacle distribution
forget old obstacle regimes
collision rate silently rises on old seeds
training reward hides the damage
```

CSN-PPO makes that failure measurable and correctable:

```text
sentinel regression detects it
failed states enter memory
cluster guard weight rises
conflicting PPO gradient components are projected out
policy recovers old behavior while continuing to learn new behavior
```

This is not a formal guarantee against arbitrary adversarial distribution shift. It is a concrete, implementable system that directly attacks the mechanism causing both catastrophic forgetting and overfitting: uncontrolled functional drift outside the latest rollout batch.
