# Full PMA-C Atari Completion Plan

**Purpose:** turn the current PMA-C prototype from “substantially reduces catastrophic forgetting” into the full PMA-C system whose deployed agent does not lose protected skills, while keeping the method domain-general.

**Audience:** Codex / Claude Code / implementation agents.

**Primary target:** continual **Full ALE Atari** with one shared agent system trained sequentially across games.

**Secondary targets:** MinAtar, Praxis/MuJoCo, supervised continual learning, LLM fine-tuning, and any domain that can implement the PMA-C adapter interface.

---

## STATUS (2026-06-16, overnight autonomous run — commits up to 14c83af on `pma-c`)

**DONE + GPU-verified (the hard criterion is MET, committed, twice adversarially reviewed):**
- **P0** `pmac/evaluation.py` — deployed/current/champion SkillScores, random-normalized retention (clip [0,1]), learned-gate, aggregate.
- **P4** `pmac/deployment.py` — `DeployedPolicy` champion-fallback router; `InvariantViolation` if no executable impl (I0/I2).
- Wired into `pmac/experiments/continual_atari.py`: deployment eval pass (sentinel-seed route decision / deploy-seed report, no oracle), `champions_only` ablation (decouples champion certification from the conservation guard), pure-champion safety routing.
- **HEADLINE RESULT** (`pma_c_results/atari_deployed/`, PMA_C_RESULTS.md §3f), 5 games × 2 seeds (n=3,4 runs in progress):
  - Deployed retention **1.000 for every protected skill** (structural champion-routing safety invariant; n-independent).
  - Matched baseline (no champions) deployed **0.332** (catastrophically forgets).
  - Conservation reduces *shared-net* forgetting **0.76 vs 0.34** (0.70 vs 0.18 overwritten-only); `champions_only` ≈ baseline isolates the guard. Plasticity preserved (≥ baseline). **HONESTY: deployed=1.0 is a structural identity (≡ certified checkpointing + router), NOT a learning result — headline the shared-net `norm_retention` instead. The conservation effect is directional at small n (paired sign 2/2; n=3/4 runs added significance).**

**BUILT + CPU-tested, but NOT yet wired into the hot loop / GPU-learning-verified (next-session work):**
- **P1** `pmac/rl_update.py` — full projected PMA-C PPO update (per-skill guard grads, tangent-cone projection, omega scaling, bounded correction). Opt-in `update_mode=pmac_projected`; default `guard_loss` path byte-unchanged. CPU tests pass; **GPU-verified to LEARN + RETAIN** (2-game 2M smoke `runs/p1_verify2`: projected_full learned both games, Breakout 8→12.25 retained, n_learned=2/2). Note: projected path is ~2× slower than guard_loss (per-skill vmap'd grads). Next: wire it as the full-PMA-C loop path + a multi-game GPU comparison vs guard_loss.
- **P2** `pmac/guard_pressure.py` — risk-adaptive per-skill guard λ with length floor + recovery patience. CPU-tested.
- **P3** `pmac/rl_sentinels.py` + `pmac/rl_audit.py` — sentinel sets + audit (PASS/CURRENT_REGRESSION/HARD_FAILURE) + rollback via SafeCheckpoint. CPU-tested.

**NEXT (priority order):** (1) fold in n=3/4 seeds + paired significance; (2) GPU-verify P1 projected learns, then wire P1/P2/P3 into `continual_atari` as an opt-in full-PMA-C path + GPU-verify it still learns AND improves current_unified_retention; (3) P5 growth/adapters (the real fix for the retention–plasticity tradeoff + removes the O(N) champion-store cost via P6 consolidation); (4) 8-game stress (P11) and a second domain (extend the deployed router to MinAtar `continual_minatar.py` — reuses P0/P4 — then MuJoCo/LLM). Keep the verified `train_ppo_atari`/`evaluate_atari` path; add mechanisms as opt-in hooks; always GPU-verify learning before relying.

---

## 0. Read this first: the exact problem we are solving

The current `pma-c` branch shows a real signal, but it is not yet the complete PMA-C system in RL.

Current evidence:

- Supervised continual learning: strong retention.
- MinAtar: strong retention.
- Full ALE Atari: meaningful reduction of forgetting, but not elimination.
- Current Full ALE PMA-C path is mostly **PPO + behavior-anchor conservation loss**.
- It does **not yet** fully use tangent-cone projection, synaptic stability, sentinel rollback, executable champion fallback, growth/adapters, or slow consolidation in the Atari loop.

The goal now is not another cosmetic pass.

The goal is to implement the full system invariant:

> Once a skill is certified as protected, the deployed PMA-C agent must never lose an executable implementation of that skill. The mutable unified network may be repaired, distilled, consolidated, or expanded, but a protected skill cannot be overwritten, deleted, or made unreachable.

This gives two distinct metrics:

1. **Deployed retention:** retention of the actual routed PMA-C agent, allowed to use frozen champions or experts. This should approach 1.0 for protected skills.
2. **Unified-current retention:** retention of the current mutable shared policy alone. This should improve over time, but it is not the hard safety invariant.

The system is considered complete only when:

```text
protected deployed retention remains within tolerance for all protected skills,
new skills still learn,
capacity grows only when necessary,
frozen champions can be archived only after a certified replacement passes,
and the same PMA-C core works through domain adapters outside Atari.
```

---

## 1. Non-negotiable design invariants

These are hard constraints. Do not weaken them.

### Invariant I0 — protected skill non-deletion

For every protected skill `s`, the atlas must always contain at least one executable certified implementation:

```text
certified_impls[s] is non-empty
```

The implementation may be:

- frozen champion parameters,
- frozen expert/adaptor parameters,
- current unified network if certified,
- consolidated slow route if certified.

But there must always be at least one executable route.

### Invariant I1 — last certified implementation cannot be deleted

A certified implementation can be marked redundant only if another implementation for the same skill passes:

```text
anchor conservation,
sentinel evaluation,
challenge evaluation,
score threshold,
context/router reachability.
```

Never delete or archive the last certified implementation of a protected skill.

### Invariant I2 — deployed policy must route safely

At inference/evaluation time, the deployed policy must route as follows:

```text
if current unified policy is certified safe for skill/context:
    use current unified policy
else:
    use best certified champion/expert fallback
```

This means the agent may continue improving the current shared network without risking total skill loss.

### Invariant I3 — no update is allowed to silently damage protected behavior

Every training block must either:

1. pass protection checks, or
2. be rolled back, or
3. be accepted only with champion fallback active and a repair job scheduled.

No silent regression.

### Invariant I4 — no test leakage

Training gates must use train-derived validation and sentinel sets. Test/eval scores are for reporting only.

### Invariant I5 — distinguish impossible cases

If the same observation/context requires contradictory outputs, PMA-C must use one of:

```text
explicit context,
inferred context,
router/gating,
separate expert,
capacity growth,
external memory.
```

Do not pretend a single context-free deterministic function can represent contradictory tasks.

### Invariant I6 — report current and deployed metrics separately

Every experiment must report:

```text
current_unified_score[skill]
champion_score[skill]
deployed_routed_score[skill]
best_score[skill]
current_unified_retention[skill]
deployed_retention[skill]
```

This prevents hiding current-network forgetting behind fallback, while still enforcing no-loss deployed behavior.

---

## 2. Current gap analysis

The PMA-C branch contains these good foundations:

```text
pmac/conservation.py      hinge behavior conservation
pmac/projection.py        tangent-cone projection
pmac/anchors.py           anchor memory and deletion checks
pmac/atlas.py             skill graph
pmac/checkpoint.py        frozen champions and safe checkpoints
pmac/auditor.py           candidate acceptance gate for supervised path
pmac/stability.py         omega-based stability scaling
pmac/consolidation.py     slow consolidation
pmac/growth.py            growth trigger primitives
pmac/router.py            router primitive
pmac/continual.py         supervised PMA-C loop using more mechanisms
pmac/agents/ppo_atari.py  Atari PPO with guard-loss support
pmac/experiments/continual_atari.py  Atari continual experiment
```

But the Full ALE / MinAtar RL paths are incomplete:

```text
Current Full ALE / MinAtar RL path:
    PPO loss + guard conservation loss

Missing from hard RL path:
    per-skill guard gradients
    tangent-cone projection
    per-skill normalized guard pressure
    omega stability scaling
    sentinel audit and rollback
    deployed champion fallback
    growth/adapters/expert routing
    slow consolidation into shared core
    archive/redundancy certification
    general PMA-C RL adapter interface
```

Therefore the current branch demonstrates that **conservation memory is useful**, but it does not yet demonstrate full PMA-C in hard RL.

---

## 3. Completion overview

Implement the remaining work in this order:

```text
P0  Define hard metrics and deployed-vs-current evaluation.
P1  Implement full PMA-C gradient update for Atari/MinAtar.
P2  Implement length-normalized and risk-adaptive guard pressure.
P3  Add sentinel audit, rollback, and repair scheduling to Atari/MinAtar.
P4  Add executable champion fallback and deployed router.
P5  Add growth/adapters for Atari and MinAtar.
P6  Add slow consolidation and certified archive logic.
P7  Add synaptic stability/omega to RL updates.
P8  Upgrade Atari memory selection and anchor semantics.
P9  Add challenge/sentinel generators for Atari and MuJoCo.
P10 Add balanced scheduler and old-skill environment refresh.
P11 Run full ablation protocol and acceptance gates.
P12 Preserve domain-general adapter boundaries.
```

All phases must have tests.

---

# P0 — Metrics, definitions, and experiment contract

## P0.1 Add explicit score types

Add a module:

```text
pmac/evaluation.py
```

Define:

```python
@dataclass
class SkillScores:
    skill_id: str
    best_score: float
    current_score: float
    champion_score: float
    deployed_score: float
    random_score: float | None = None
    current_retention: float = 0.0
    deployed_retention: float = 0.0
    champion_retention: float = 0.0
    regressed_current: bool = False
    regressed_deployed: bool = False
```

For Atari, use random-normalized retention:

```math
R_{norm}(g) = \frac{S_{final}(g) - S_{random}(g)}{S_{best}(g) - S_{random}(g) + \epsilon}
```

Clip only for reporting if necessary, but store raw values too.

## P0.2 Define success thresholds

For protected deployed behavior:

```text
deployed_retention >= 0.98 for solved/protected skills
```

For current unified policy:

```text
current_retention target >= 0.90 initially,
then improve toward >= 0.95 with consolidation.
```

For plasticity:

```text
new game learned score >= 0.90 * baseline_new_game_learned_score
```

For no overfitting:

```text
validation/sentinel/challenge scores cannot regress while training score improves.
```

## P0.3 Do not let retention hide failure-to-learn

A skill can be protected only if it was actually learned.

For skill `g`, require:

```math
S_{best}(g) > S_{random}(g) + \Delta_{learned}
```

If not learned, report it separately:

```text
not_learned_or_uncertified
```

Do not allow low-peak retention to count as success.

## P0.4 Required result tables

Every continual Atari run must emit:

```text
return_matrix_current.json
return_matrix_deployed.json
return_matrix_champion.json
best_scores.json
random_scores.json
retention_current.json
retention_deployed.json
router_decisions.json
guard_metrics.json
sentinel_metrics.json
growth_events.json
rollback_events.json
consolidation_events.json
```

The existing result summary is not enough.

---

# P1 — Full PMA-C gradient update for Atari/MinAtar

Current Atari/MinAtar uses:

```math
L = L_{PPO} + \lambda G
```

This is not full PMA-C.

Replace it with explicit gradient geometry.

## P1.1 Required update equations

For current PPO minibatch:

```math
g_{new} = \nabla_\theta L_{PPO}(\theta)
```

For each protected skill `k`:

```math
g_k = \nabla_\theta G_k(\theta)
```

where:

```math
G_k(\theta) = \mathbb{E}_{a \sim M_k}
\left[
    w_a [D(f_\theta(x_a, c_a), y_a^*) - \epsilon_a]_+^2
\right]
```

Project:

```math
g \leftarrow g - \frac{\min(0, g^\top g_k)}{\|g_k\|^2 + \eta} g_k
```

Final update:

```math
g_{total} = g_{projected} + \sum_k \lambda_k \tilde{g}_k
```

where `\tilde{g}_k` is clipped/normalized guard gradient.

Then apply stability scaling:

```math
\Delta \theta_j = -\eta \frac{g_{total,j}}{1 + \alpha \Omega_j}
```

## P1.2 New module

Create:

```text
pmac/rl_update.py
```

Implement:

```python
@dataclass
class RLGuardBatch:
    skill_id: str
    obs: Array
    context: Array
    teacher_logits: Array
    teacher_value: Array
    tolerance: Array
    weight: Array
    source: str

@dataclass
class PMACUpdateMetrics:
    ppo_loss: float
    guard_losses: dict[str, float]
    g_new_norm: float
    g_projected_norm: float
    g_total_norm: float
    projection_ratio: float
    guard_norms: dict[str, float]
    guard_lambdas: dict[str, float]
    conflict_dots: dict[str, float]
    clipped_guard_count: int
    nonfinite: bool
```

Implement:

```python
def ppo_pmac_update(
    params,
    opt_state,
    ppo_batch,
    current_context,
    guard_batches: list[RLGuardBatch],
    omega,
    optimizer,
    loss_fns,
    cfg,
    rng,
):
    """One full PMA-C PPO update.

    Required behavior:
    1. compute g_new from PPO only
    2. compute one guard grad per protected skill
    3. clip each guard grad relative to g_new
    4. project g_new away from conflicting guard directions
    5. add normalized guard corrections
    6. scale by omega stability
    7. globally clip
    8. apply optimizer
    9. return metrics
    """
```

## P1.3 Guard gradient clipping

Per skill:

```math
\|\tilde{g}_k\| \leq \alpha_g \frac{\|g_{new}\|}{\sqrt{N_{active}}}
```

Total guard correction:

```math
\left\|\sum_k \lambda_k \tilde{g}_k\right\| \leq \beta_g \|g_{new}\|
```

This prevents the conservation mechanism from dominating plasticity.

## P1.4 Use in Atari and MinAtar

Modify:

```text
pmac/agents/ppo_atari.py
pmac/agents/ppo_minatar.py
pmac/experiments/continual_atari.py
pmac/experiments/continual_minatar.py
```

Replace the `ppo_loss + guard_loss` update with `ppo_pmac_update`.

Keep the old guard-loss-only path as ablation:

```text
--ablation guard_loss_only
```

Add ablations:

```text
no_projection
no_stability
no_guard_correction
projection_only
conservation_only
guard_loss_only
```

## P1.5 Unit tests

Add:

```text
tests/pmac/test_rl_update_projection.py
```

Tests:

1. If `g_new` conflicts with guard, projection removes conflict.
2. If `g_new` is aligned with guard, projection leaves it mostly unchanged.
3. Guard correction norm is bounded.
4. Stability scaling reduces updates on high-omega parameters.
5. Nonfinite gradients skip update.
6. With no guards, update equals PPO update up to numerical tolerance.

---

# P2 — Length-normalized and risk-adaptive guard pressure

Current experiments show `guard_coef` is a real knob and can over-constrain later games as prior-task count grows.

This must be fixed before scaling to many games.

## P2.1 Replace scalar guard coefficient

Do not use one fixed scalar `guard_coef` for all prior games.

Use per-skill coefficients:

```math
\lambda_k = \lambda_{base} \cdot \frac{r_k}{\sum_j r_j + \epsilon}
```

where:

```math
r_k = 1 + a F_k + b I_k + c A_k + d S_k
```

Definitions:

```text
F_k = forgetting risk for skill k
I_k = interference score with current skill
A_k = age since last rehearsal/audit
S_k = sentinel regression indicator or severity
```

Also enforce a length normalization floor:

```math
\lambda_k \leq \frac{\lambda_{max}}{\sqrt{N_{protected}}}
```

or normalize total pressure:

```math
\sum_k \lambda_k \leq \Lambda_{total}
```

Recommended default:

```python
lambda_total = 1.0
lambda_min_per_skill = 0.02
lambda_max_per_skill = 0.5
risk_forgetting_weight = 2.0
risk_interference_weight = 1.0
risk_age_weight = 0.25
risk_sentinel_weight = 4.0
```

## P2.2 New module

Create:

```text
pmac/guard_pressure.py
```

Implement:

```python
@dataclass
class GuardPressureState:
    skill_lambda: dict[str, float]
    recovery_count: dict[str, int]
    regression_count: dict[str, int]
    last_audit_step: dict[str, int]


def compute_guard_lambdas(
    atlas,
    current_skill_id,
    sentinel_metrics,
    projection_metrics,
    cfg,
) -> dict[str, float]:
    """Returns normalized per-skill guard pressure."""
```

## P2.3 Recovery patience

Do not decay guard pressure immediately after one clean audit.

Use:

```python
if skill_regressed:
    lambda_k = min(lambda_k * up_factor, lambda_max)
    recovery_count[k] = 0
else:
    recovery_count[k] += 1
    if recovery_count[k] >= recovery_patience:
        lambda_k = max(lambda_k * down_factor, lambda_min)
```

Recommended:

```python
up_factor = 1.5
down_factor = 0.98
recovery_patience = 3
```

## P2.4 Tests

Add:

```text
tests/pmac/test_guard_pressure.py
```

Tests:

1. Total lambda stays bounded as number of skills increases.
2. Regressed skill gets higher lambda.
3. Non-regressed skill does not decay until patience is met.
4. Interference neighbor gets sampled/weighted more.
5. Adding 100 protected skills does not make guard gradient dominate `g_new`.

---

# P3 — Sentinel audit, rollback, and repair scheduling for Atari/MinAtar

Current Atari evaluation happens after each game. That is too coarse.

Full PMA-C requires online sentinel audits.

## P3.1 Sentinel types

For each protected Atari game store:

```python
@dataclass
class RLSentinelSet:
    skill_id: str
    env_name: str
    game_id: int
    eval_seeds: Array[int]
    start_states: Optional[Any]
    anchor_obs: Array
    random_score: float
    best_score: float
    allowed_regression: float
    eval_episodes_fast: int
    eval_episodes_full: int
```

There are two sentinel modes:

### Fast sentinels

Cheap and frequent.

```text
4–8 episodes per protected skill
every 250k–1M environment frames
used for rollback / guard pressure
```

### Full sentinels

Expensive and less frequent.

```text
20+ episodes per protected skill
after each game or major phase
used for certification/archive/reporting
```

## P3.2 Audit policy

Every audit interval:

```python
for skill in protected_skills:
    score_current = evaluate_current_policy(skill.fast_sentinels)
    score_deployed = evaluate_deployed_policy(skill.fast_sentinels)

    if score_deployed < best_score - allowed_regression:
        HARD_FAILURE
    elif score_current < best_score - current_allowed_regression:
        CURRENT_REGRESSION
```

Actions:

```text
HARD_FAILURE:
    rollback to last safe system checkpoint
    route skill to frozen champion
    increase guard pressure
    schedule old-skill repair

CURRENT_REGRESSION:
    keep fallback route active
    increase guard pressure
    schedule repair/consolidation
    do not certify current network for this skill

PASS:
    optionally update safe checkpoint
    increment recovery counter
```

## P3.3 Rollback semantics

Store two checkpoints:

```text
system_safe_checkpoint:
    current params
    optimizer state
    omega
    router state
    atlas state
    guard pressure state

skill_champion_checkpoint:
    immutable per skill
```

Rollback should restore mutable system state but never delete champions.

## P3.4 Implementation files

Add:

```text
pmac/rl_sentinels.py
pmac/rl_audit.py
```

Modify:

```text
pmac/experiments/continual_atari.py
pmac/experiments/continual_minatar.py
```

## P3.5 Tests

Add:

```text
tests/pmac/test_rl_sentinel_audit.py
```

Tests:

1. If current regresses but champion passes, deployed skill passes and route switches.
2. If both current and champion fail, hard failure is raised.
3. Rollback restores previous mutable params and opt state.
4. Regressed skill receives increased guard pressure.
5. Safe checkpoint updates only after sentinel pass.

---

# P4 — Executable champion fallback and deployed router

This is the most important structural change for “complete” no-forgetting.

A frozen champion that is stored but never used does not prevent deployed forgetting.

## P4.1 New deployed policy abstraction

Create:

```text
pmac/deployment.py
```

Implement:

```python
@dataclass
class DeploymentDecision:
    skill_id: str
    route_type: str  # "current", "champion", "expert", "consolidated"
    route_id: str
    reason: str
    current_certified: bool
    fallback_used: bool

class DeployedPolicy:
    def __init__(self, current_params, atlas, router, adapter): ...

    def select_route(self, obs, context) -> DeploymentDecision: ...

    def act(self, obs, context, rng=None):
        decision = self.select_route(obs, context)
        params = self.resolve_params(decision)
        return adapter.act(params, obs, context, rng), decision
```

For Atari, context is game ID. If context is known, route deterministically by game.

If context is unknown, route by inferred context:

```text
router(obs) -> skill probability
choose certified implementation for inferred skill
```

## P4.2 Route certification state

Each skill node needs:

```python
current_certified: bool
current_last_score: float
fallback_route_id: str
fallback_score: float
needs_repair: bool
```

Routing rule:

```python
if node.current_certified and not node.needs_repair:
    route = current
else:
    route = best certified fallback
```

## P4.3 Report router usage

All evaluation must report:

```text
route/current_fraction
route/champion_fraction
route/expert_fraction
route/fallback_used_per_skill
```

## P4.4 No oracle cheating

For Atari with explicit game ID, using game ID is allowed. Atari games are distinct contexts.

For context-free tests, router must infer context from observations and report routing accuracy.

Do not compute `max(current, champion)` offline and call it deployed performance. The deployed router must actually choose the route.

## P4.5 Tests

Add:

```text
tests/pmac/test_deployed_policy.py
```

Tests:

1. If current certified, uses current.
2. If current regressed, uses champion.
3. If champion missing, raises invariant violation.
4. If replacement expert certified, can route to expert.
5. Metrics report actual route chosen.

---

# P5 — Growth/adapters for Atari and MinAtar

Guarding a fixed shared CNN forever will eventually over-constrain learning.

Full PMA-C needs growth.

## P5.1 Architecture

Modify Atari network to support adapters/experts.

Current approximate form:

```text
Nature CNN -> Dense512 -> policy/value
```

Full PMA-C form:

```text
shared slow CNN core
+ skill/context embedding
+ residual adapters or LoRA modules
+ router/gate
+ universal 18-action policy head
+ value head
```

Recommended first implementation:

```text
conv trunk shared
Dense512 shared
per-skill residual adapter at Dense512:
    h_adapter = h + A_skill(h)
policy/value from h_adapter
```

Adapter:

```math
A_k(h) = W_{up,k} \sigma(W_{down,k} h)
```

with small bottleneck, e.g. rank 32 or 64.

## P5.2 Growth trigger

Trigger growth if any of these persist for `patience` audits:

```text
plasticity_ratio < growth_min_ratio
current game learning below expected baseline
repeated rollbacks due to old-skill interference
new skill guard conflict high
sentinel current regression persists despite guard increase
```

Plasticity ratio:

```math
r = \frac{\|g_{projected}\|}{\|g_{new}\| + \epsilon}
```

If:

```math
r < r_{min}
```

for `growth_patience`, add an adapter/expert.

## P5.3 Growth action

```python
def grow_for_skill(params, atlas, skill_id, cfg):
    new_adapter = init_adapter(...)
    params.adapters[skill_id] = new_adapter
    atlas.nodes[skill_id].expert_ref = skill_id
    router.register(skill_id, route="adapter:" + skill_id)
    freeze old adapters for other skills unless consolidation certifies sharing
    return params, atlas, router
```

## P5.4 Training after growth

When a new adapter is created:

```text
shared core lr reduced
new adapter lr normal/high
old adapters frozen or low-lr
policy/value head allowed small updates with protection
```

## P5.5 Tests

Add:

```text
tests/pmac/test_atari_growth.py
```

Tests:

1. Growth adds parameters without changing old adapter outputs.
2. Old skill route remains executable after growth.
3. New adapter receives gradients.
4. Old adapters are frozen unless explicitly unfrozen.
5. Growth trigger fires when projection ratio remains low.

---

# P6 — Slow consolidation and certified archive

Growth prevents overwriting, but unlimited experts are not elegant.

PMA-C needs a sleep/consolidation phase.

## P6.1 Consolidation objective

Periodically train a slow shared candidate from balanced old/new behavior:

```math
L_{consolidate} =
L_{task/current}
+ \sum_k \lambda_k G_k
+ \lambda_{distill} \sum_k D(f_{candidate}, f^*_{champion,k})
```

Use low LR:

```math
\eta_{slow} = 0.01 \eta_{fast}
```

Apply projection and omega stability.

## P6.2 Consolidation schedule

Run consolidation:

```text
after each game
or after every N million frames
or after growth event stabilizes
```

During consolidation:

```text
sample anchors from all protected skills
sample current skill data
sample sentinel/challenge states
use frozen champions as teachers
train current unified / slow core
validate deployed and current scores
```

## P6.3 Archive expert only with certification

An old expert/champion can be archived only if candidate replacement passes:

```text
anchor conservation loss <= tolerance
fast sentinel score >= best_score - allowed_regression
full sentinel score >= best_score - allowed_regression
challenge score >= threshold
router reaches candidate route
at least one other certified implementation remains
```

Implementation:

```python
def certify_replacement(skill, candidate_route):
    if not anchor_ok: return False
    if not sentinel_ok: return False
    if not challenge_ok: return False
    if not route_ok: return False
    atlas.nodes[skill].certified_impls.append(candidate_route)
    return True
```

Then:

```python
def archive_if_redundant(skill, old_route, replacement_route):
    if len(certified_impls[skill]) <= 1:
        return False
    if replacement_route certified:
        mark old_route redundant
```

Never delete the last route.

## P6.4 Tests

Add:

```text
tests/pmac/test_rl_consolidation_archive.py
```

Tests:

1. Consolidation cannot reduce deployed sentinel score.
2. Old expert cannot be archived without replacement.
3. Replacement must pass anchor and sentinel checks.
4. Archive leaves at least one executable route.
5. Consolidation improves current unified retention without reducing new-game score.

---

# P7 — Synaptic stability for RL

The supervised path has omega; the RL hard path must use it too.

## P7.1 Omega update

After a skill is certified:

```math
\Omega_j \leftarrow \rho \Omega_j + (1 - \rho) |\theta_j \cdot \nabla_{\theta_j} G_s|
```

or normalized version:

```math
\Omega_j \leftarrow \rho \Omega_j + (1 - \rho) \frac{|g_j|}{\mathbb{E}|g| + \epsilon}
```

Use guard gradients from all anchors/sentinels for the certified skill.

## P7.2 Apply omega scaling

Before optimizer update:

```math
g_j \leftarrow \frac{g_j}{1 + \alpha \Omega_j}
```

Different parameter groups:

```text
shared CNN core: high stability alpha
shared dense: medium alpha
new adapter: zero/low alpha initially
old adapter: frozen or high alpha
policy head: medium alpha
value head: lower alpha or game-conditioned
```

## P7.3 Tests

Add:

```text
tests/pmac/test_rl_stability.py
```

Tests:

1. Omega increases on parameters important to guard loss.
2. High omega reduces update magnitude.
3. New adapter starts low omega.
4. Frozen old adapter does not update.
5. Applying omega does not create NaNs.

---

# P8 — Better Atari anchor memory

The current anchor buffers are useful but too simple.

## P8.1 Store richer anchors

For each anchor:

```python
@dataclass
class RLAnchor:
    obs_uint8: Array
    context: Array
    teacher_logits: Array
    teacher_value: Array
    teacher_greedy_action: int
    action_mask: Array
    return_context: float
    advantage_estimate: float | None
    timestep_in_episode: int
    life_or_terminal_context: Any | None
    source: str  # high_return, high_advantage, near_death, sentinel, random, rare
    importance: float
    tolerance_policy: float
    tolerance_value: float
    skill_id: str
```

## P8.2 Importance selection

Do not select only high softmax confidence.

Use:

```math
I(x) =
a_1 \cdot confidence
+ a_2 \cdot |advantage|
+ a_3 \cdot |value|
+ a_4 \cdot rarity
+ a_5 \cdot sentinel_source
+ a_6 \cdot failure_boundary
+ a_7 \cdot high_return_trajectory
```

Atari-specific sources:

```text
high return episode states
states before reward events
states before death/life loss if available
high value states
high entropy decision states
rare representation clusters
sentinel regression states
random coverage states
```

## P8.3 Greedy-anchor mismatch fix

Current notes mention stochastic policy logits are conserved while evaluation is greedy.

Store both:

```text
teacher_logits
teacher_greedy_action
```

Conservation distance:

```math
D = KL(\pi^* || \pi_\theta)
+ \lambda_a CE(a^*_{greedy}, \pi_\theta)
+ \lambda_v |V - V^*|
```

This protects both distributional behavior and greedy deployment behavior.

## P8.4 Per-skill balanced sampling

Guard sampling must be per-skill balanced and risk-weighted.

Do not concatenate all anchors and sample uniformly.

Use:

```python
for skill in selected_protected_skills:
    batch_k = anchor_store[skill].sample(n_k, strategy="importance+rarity")
```

## P8.5 Tests

Add:

```text
tests/pmac/test_rl_anchor_memory.py
```

Tests:

1. Anchor store keeps per-skill quotas.
2. High-importance anchors are retained.
3. Rare clusters are retained.
4. Greedy action conservation loss is nonzero when greedy action changes.
5. Sampling is balanced over protected skills.

---

# P9 — Challenge sets and generalization guards

Anchors protect stored points. Sentinels protect fixed rollouts. Challenge sets protect neighborhoods.

## P9.1 Atari challenge generator

Create:

```text
pmac/rl_challenges.py
```

Challenge types:

```text
fixed no-op start seeds
sticky-action variation seeds
random start states
frame preprocessing perturbations that preserve semantics
reward-proximal trajectory snippets
near-terminal states
rare-state clusters
sentinel-regression starts
```

For Atari, be conservative with visual augmentations. Do not use augmentations that change game semantics.

## P9.2 Challenge audit

A candidate replacement/current policy must pass:

```text
anchors
fast sentinels
challenge states
full sentinels for certification
```

## P9.3 MuJoCo/Praxis challenge generator

Keep PMA-C general by adding a domain adapter:

```text
Praxis/MuJoCo challenges:
    fixed seeds
    obstacle perturbations
    dynamic obstacle speed changes
    start/goal perturbations
    near-collision states
    success trajectory states
```

## P9.4 Tests

Add:

```text
tests/pmac/test_rl_challenges.py
```

Tests:

1. Challenge generator deterministic with fixed seed.
2. Challenges differ from anchors.
3. Candidate that changes behavior on challenge fails audit.
4. Domain adapters can provide custom challenge generators.

---

# P10 — Balanced scheduler and old-skill environment refresh

To fully solve forgetting empirically, do not rely only on static anchors.

Use environment refresh for protected skills.

## P10.1 Skill sampling distribution

At each training phase, sample skill/game according to:

```math
p(g) \propto
\beta_1 learning\_need(g)
+ \beta_2 forgetting\_risk(g)
+ \beta_3 age(g)
+ \beta_4 interference(g)
+ \beta_5 sentinel\_regression(g)
```

For sequential Atari, still train mostly current game, but allocate refresh budget:

```text
80–90% current game frames
5–10% old-game sentinel/refresh rollouts
5–10% anchor/challenge/consolidation batches
```

If current game is not learning, reduce old-game env refresh but keep anchor guards.

## P10.2 Refresh anchors

Periodic old-game refresh should:

```text
evaluate deployed route
collect new anchors from champion/current route
add sentinel regression states
update best/fallback only if score improves
```

## P10.3 Prevent overfitting to anchors

Anchor memory should not become the entire old-game distribution.

Use:

```text
fixed sentinels
fresh old-game rollouts
challenge variations
representation rarity
random coverage anchors
```

## P10.4 Tests

Add:

```text
tests/pmac/test_scheduler_refresh.py
```

Tests:

1. Regressed skill gets higher sampling probability.
2. Old skill with no regression still gets minimum refresh probability.
3. Scheduler probability sums to one.
4. Refresh updates anchor memory without deleting last certified route.

---

# P11 — Full ablation and acceptance protocol

The system is not complete until these experiments pass.

## P11.1 Required Atari experiments

Run:

```text
4-game Full ALE sequence, 3 seeds
8-game Full ALE sequence, 3 seeds
MinAtar 4-game sequence, 5 seeds
Praxis/MuJoCo continual sequence, 3 seeds
```

Recommended Full ALE sequence:

```text
Breakout
SpaceInvaders
BeamRider
Asterix
Seaquest
Qbert
MsPacman
Pong or Freeway if learnability is verified
```

If a game is not learned above random by baseline or PMA-C, mark it `uncertified` and report separately.

## P11.2 Required ablations

For each main benchmark:

```text
baseline
PMA-C full
conservation_only / guard_loss_only
no_conservation
no_projection
no_stability
no_sentinel_rollback
no_champion_fallback
no_growth
no_consolidation
no_replay_or_refresh
random_memory
latest_teacher_instead_of_best_teacher
```

## P11.3 Acceptance thresholds

For Full ALE 4-game sequence:

```text
deployed mean retention >= 0.98
deployed worst retention >= 0.95
current unified mean retention >= 0.90
current unified worst retention >= 0.75 initially, improve with consolidation
mean final return >= baseline
new-game learned score >= 0.90 * baseline learned score
no protected deployed skill below best - allowed_regression
```

For 8-game sequence:

```text
deployed mean retention >= 0.95
deployed worst retention >= 0.90
current unified mean retention >= 0.85
mean final return >= baseline
capacity growth sublinear in number of games
```

If these fail, do not claim solved.

## P11.4 Required plots

```text
per-game learned vs final current score
per-game learned vs final deployed score
retention current vs deployed
game-0 score across training
sentinel score across training
guard lambda per skill
projection ratio over time
growth events over time
rollback events over time
route usage over time
memory size per skill
```

## P11.5 Required JSON schema

Each run writes:

```json
{
  "config": {},
  "games": [],
  "seeds": [],
  "modes": {
    "baseline": {},
    "pmac_full": {},
    "no_projection": {},
    "no_fallback": {}
  },
  "metrics": {
    "current_retention": {},
    "deployed_retention": {},
    "champion_retention": {},
    "mean_final_return": {},
    "worst_retention": {},
    "new_game_plasticity": {},
    "rollback_count": {},
    "growth_count": {},
    "route_usage": {}
  }
}
```

---

# P12 — Domain-general adapter boundary

Do not make PMA-C Atari-only.

## P12.1 Adapter interface

Create/standardize:

```text
pmac/domain_adapter.py
```

```python
class PMACDomainAdapter:
    def behavior(self, params, x, context):
        """Return behavior object: logits/value/policy/embedding/etc."""

    def behavior_distance(self, teacher, current, batch_meta):
        """Return per-example behavior distance."""

    def task_loss(self, params, batch, context):
        """Current task/domain loss."""

    def evaluate_skill(self, params_or_policy, skill_node, mode):
        """Return score for skill."""

    def collect_anchors(self, params_or_policy, skill_node, cfg):
        """Return anchors for long-term memory."""

    def build_sentinels(self, skill_node, cfg):
        """Return fixed sentinel set."""

    def build_challenges(self, skill_node, cfg):
        """Return challenge set."""

    def grow_capacity(self, params, skill_node, cfg):
        """Add adapter/expert/LoRA/etc."""

    def route(self, router, obs, context):
        """Select current/champion/expert."""
```

Atari, MinAtar, Praxis/MuJoCo, supervised classification, and LLM fine-tuning should each implement this interface.

## P12.2 Domain-specific behavior distances

### Atari / RL

```math
D = KL(\pi^* || \pi_\theta)
+ \lambda_a CE(a^*, \pi_\theta)
+ \lambda_v |V_\theta - V^*|
```

### MuJoCo / continuous-control RL

```math
D = KL(\mathcal{N}(\mu^*, \sigma^*) || \mathcal{N}(\mu, \sigma))
+ \lambda_v |V - V^*|
```

### Supervised classification

```math
D = KL(logits^* || logits)
```

### Regression

```math
D = \|y - y^*\|^2
```

### LLM fine-tuning

```math
D = KL(p^*_{tokens} || p_{tokens})
+ \lambda_e \|embedding - embedding^*\|^2
+ \lambda_s safety\_drift
```

## P12.3 Tests

Add:

```text
tests/pmac/test_domain_adapter_contract.py
```

Tests:

1. Every adapter returns per-example behavior distances.
2. Every adapter can collect anchors.
3. Every adapter can evaluate a skill.
4. PMA-C core update can run without knowing domain internals.

---

# Implementation checklist by file

## New files

```text
pmac/evaluation.py
pmac/rl_update.py
pmac/guard_pressure.py
pmac/rl_sentinels.py
pmac/rl_audit.py
pmac/deployment.py
pmac/rl_challenges.py
pmac/domain_adapter.py
pmac/adapters/atari.py
pmac/adapters/minatar.py
pmac/adapters/praxis.py
```

## Modify existing files

```text
pmac/agents/atari_net.py
pmac/agents/ac_net.py
pmac/agents/ppo_atari.py
pmac/agents/ppo_minatar.py
pmac/experiments/continual_atari.py
pmac/experiments/continual_minatar.py
pmac/atlas.py
pmac/anchors.py
pmac/checkpoint.py
pmac/router.py
pmac/stability.py
pmac/consolidation.py
pmac/config.py
PMA_C_RESULTS.md
```

## New tests

```text
tests/pmac/test_rl_update_projection.py
tests/pmac/test_guard_pressure.py
tests/pmac/test_rl_sentinel_audit.py
tests/pmac/test_deployed_policy.py
tests/pmac/test_atari_growth.py
tests/pmac/test_rl_consolidation_archive.py
tests/pmac/test_rl_stability.py
tests/pmac/test_rl_anchor_memory.py
tests/pmac/test_rl_challenges.py
tests/pmac/test_scheduler_refresh.py
tests/pmac/test_domain_adapter_contract.py
```

---

# Full Atari update pseudocode

This is the exact shape the final Atari loop should have.

```python
for game in game_sequence:
    skill = atlas.get_or_create_skill(game)
    ensure_route_exists(skill)

    for update_block in range(num_update_blocks):
        # 1. Collect PPO rollout on current game.
        rollout = collect_current_game_rollout(current_params, game)
        ppo_batch = build_ppo_batch(rollout)

        # 2. Select protected skills at risk.
        protected = scheduler.sample_protected_skills(
            atlas=atlas,
            current_skill_id=skill.id,
            guard_pressure=guard_pressure,
            max_skills=cfg.max_guard_skills_per_update,
        )

        # 3. Sample behavior anchors per protected skill.
        guard_batches = []
        for old_skill in protected:
            anchors = old_skill.anchors.sample_balanced(
                n=cfg.guard_batch_per_skill,
                strategy="importance+rarity+sentinel",
            )
            guard_batches.append(make_rl_guard_batch(old_skill, anchors))

        # 4. Full PMA-C gradient update.
        candidate_params, candidate_opt_state, update_metrics = ppo_pmac_update(
            params=current_params,
            opt_state=opt_state,
            ppo_batch=ppo_batch,
            current_context=skill.context,
            guard_batches=guard_batches,
            omega=omega,
            optimizer=optimizer,
            loss_fns=loss_fns,
            cfg=cfg,
            rng=rng,
        )

        # 5. Cheap sentinel audit.
        if update_block % cfg.fast_audit_interval == 0:
            audit = rl_auditor.audit_candidate(
                candidate_params=candidate_params,
                current_params=current_params,
                deployed_policy=deployed_policy,
                atlas=atlas,
                current_skill=skill,
                fast=True,
            )

            if audit.hard_failure:
                current_params, opt_state, omega, router = system_safe.restore()
                guard_pressure.increase(audit.regressed_skills)
                scheduler.boost(audit.regressed_skills)
                continue

            if audit.current_regression:
                # Do not lose deployed behavior. Route to champion and repair current.
                router.force_fallback(audit.regressed_skills)
                guard_pressure.increase(audit.regressed_skills)
                scheduler.schedule_repair(audit.regressed_skills)

            if audit.accept_mutable:
                current_params = candidate_params
                opt_state = candidate_opt_state
                system_safe.update_if_safe(...)
            else:
                current_params, opt_state = system_safe.restore_mutable_only()

        else:
            current_params = candidate_params
            opt_state = candidate_opt_state

        # 6. Growth trigger.
        if growth_manager.should_grow(update_metrics, audit, skill):
            current_params, atlas, router = growth_manager.grow(
                current_params,
                atlas,
                router,
                skill,
            )

        # 7. Periodic old-skill repair.
        if scheduler.has_repair_jobs():
            current_params = repair_regressed_skills(
                current_params,
                atlas,
                scheduler.repair_jobs,
                cfg,
            )

        # 8. Periodic consolidation.
        if update_block % cfg.consolidation_interval == 0:
            current_params, omega, archive_events = consolidate_and_certify(
                current_params,
                atlas,
                router,
                omega,
                cfg,
            )

    # 9. End-of-game certification.
    full_scores = evaluate_current_and_deployed(current_params, deployed_policy, atlas)
    if learned_enough(full_scores.current[skill.id], random_score[skill.id]):
        champion = champion_store.freeze(current_params, route=skill.id)
        anchors = collect_anchors(champion, game)
        sentinels = build_sentinels(game)
        atlas.certify(skill, champion, anchors, sentinels, full_scores)
        omega = update_omega_from_skill(omega, current_params, anchors)
```

---

# Required commands after implementation

## Unit tests

```bash
JAX_PLATFORMS=cpu pytest tests/pmac -q
```

## MinAtar full PMA-C

```bash
python -m pmac.experiments.continual_minatar \
  --games Breakout-MinAtar,Asterix-MinAtar,Freeway-MinAtar,SpaceInvaders-MinAtar \
  --per-game-steps 5000000 \
  --seeds 0,1,2,3,4 \
  --ablations none,no_conservation,no_projection,no_stability,no_sentinel_rollback,no_fallback,guard_loss_only \
  --out runs/pmac_minatar_full
```

## Full ALE 4-game full PMA-C

```bash
python -m pmac.experiments.continual_atari \
  --games Breakout-v5,SpaceInvaders-v5,BeamRider-v5,Asterix-v5 \
  --per-game-steps 4000000 \
  --seeds 0,1,2 \
  --ablations none,no_conservation,no_projection,no_stability,no_sentinel_rollback,no_fallback,guard_loss_only \
  --out runs/pmac_atari_full_4game
```

## Full ALE 8-game stress test

```bash
python -m pmac.experiments.continual_atari \
  --games Breakout-v5,SpaceInvaders-v5,BeamRider-v5,Asterix-v5,Seaquest-v5,Qbert-v5,MsPacman-v5,Freeway-v5 \
  --per-game-steps 4000000 \
  --seeds 0,1,2 \
  --ablations none,no_conservation,no_projection,no_fallback \
  --out runs/pmac_atari_full_8game
```

## Praxis/MuJoCo adapter smoke

```bash
python -m pmac.experiments.continual_praxis \
  --tasks easy_nav,static_obstacles,dynamic_obstacles,narrow_passages \
  --seeds 0,1,2 \
  --out runs/pmac_praxis_smoke
```

---

# What “complete” means after this work

Do not claim complete based only on reduced forgetting.

The system is complete when all of these hold:

```text
1. Every protected skill has an executable certified implementation.
2. The deployed router never falls below allowed regression on protected skills.
3. Current unified retention improves through repair/consolidation instead of monotonically decaying.
4. New skills still learn close to baseline.
5. Full PMA-C beats guard-loss-only in hard RL.
6. Removing champion fallback breaks deployed retention.
7. Removing projection worsens current unified retention or plasticity under conflict.
8. Removing growth fails on longer/capacity-limited sequences.
9. Removing conservation collapses retention back toward baseline.
10. The same PMA-C core runs through adapters on Atari, MinAtar, Praxis/MuJoCo, and supervised tasks.
```

The key distinction:

```text
Reducing forgetting:
    current policy forgets less than baseline.

Solving protected forgetting:
    protected skill remains deployable because the system always routes to a certified implementation,
    and the mutable current policy is repaired/consolidated over time.
```

The second is the full PMA-C target.

---

# Final implementation priority order

Build in this exact order:

```text
1. P0 metrics and deployed/current score separation.
2. P4 executable champion fallback and deployed router.
3. P3 sentinel audit/rollback for Atari.
4. P1 full PMA-C RL gradient update with projection.
5. P2 normalized adaptive guard pressure.
6. P8 improved RL anchors and greedy-action conservation.
7. P7 omega stability in RL.
8. P5 adapters/growth.
9. P6 consolidation/archive.
10. P10 old-skill refresh scheduler.
11. P9 challenge generator.
12. P11 full ablation protocol.
13. P12 domain adapter cleanup.
```

Why this order?

```text
Fallback + sentinel audit gives immediate non-loss deployed behavior.
Projection + guard normalization improves current unified retention without over-constraining plasticity.
Better anchors make conservation meaningful.
Stability, growth, and consolidation let the system scale beyond short sequences.
Scheduler/challenges/domain adapters make the solution general.
```

---

# Summary for implementation agents

The current PMA-C branch proved the main mechanism is promising. It did not yet complete the system.

Your job is to turn PMA-C from:

```text
PPO + conservation loss that reduces forgetting
```

into:

```text
protected manifold atlas system that cannot lose protected skills:
    frozen champions
    deployed fallback router
    sentinel audit/rollback
    full projected RL gradients
    normalized guard pressure
    synaptic stability
    capacity growth
    slow consolidation
    certified archive
    domain-general adapters
```

If deployed retention is near-perfect but current unified retention is not, the system is safe but not fully consolidated.

If current unified retention also becomes high after consolidation, the system is both safe and elegant.

The hard invariant is safety first:

```text
Never lose an executable protected skill.
```

Then optimize for elegance:

```text
Consolidate back into the shared current policy whenever it can be certified safe.
```
