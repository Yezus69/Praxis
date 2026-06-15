# PMA-C: Protected Manifold Atlas with Consolidation

**Purpose:** A general continual-learning system for reinforcement learning, supervised learning, imitation learning, world models, multimodal agents, and language models that prevents catastrophic forgetting and controls overfitting by treating learned capabilities as protected behavior manifolds.

**Core claim:** With long-term memory, immutable/certified skill implementations, modular capacity growth, slow consolidated weights, balanced rehearsal, and regression-gated updates, catastrophic forgetting becomes a systems invariant rather than a hyperparameter accident.

**Important honesty boundary:** No finite mutable model can guarantee perfect retention for arbitrary contradictory tasks without memory, context, or capacity growth. PMA-C removes that restriction. It preserves skills at the **system level** by never deleting or overwriting the last certified implementation of a capability. Generalization to unseen states is empirical and depends on sentinel/anchor coverage, but loss of previously certified behavior is prevented by construction.

---

## 0. Executive summary

Catastrophic forgetting and overfitting are both forms of uncontrolled functional drift.

- **Overfitting:** short-horizon drift away from the true local data manifold while improving the current training batch.
- **Catastrophic forgetting:** long-horizon drift out of previously learned behavior manifolds while optimizing a new task or distribution.

A trained skill is not a single point in weight space. It is a **tube** or **manifold region** of parameter values that implement acceptable behavior for a capability. Forgetting occurs when training leaves one of these protected tubes.

PMA-C solves this by maintaining a **Protected Manifold Atlas**:

1. Each capability is a node in a graph.
2. Each node owns behavioral anchors, sentinel environments, challenge probes, a best-known teacher, and optionally a frozen expert/module.
3. New learning happens in a plastic workspace.
4. Updates are projected away from directions that damage protected behavior.
5. If the protected constraints block learning, the system grows new capacity.
6. Useful behavior is periodically consolidated into a slow shared core.
7. A certified frozen implementation is never deleted until another implementation passes the same sentinel/anchor tests.

The biological analogy is:

```text
fast plastic workspace     = rapid learning
slow consolidated core     = long-term semantic memory
frozen/certified experts   = hard-to-update skill memory
sentinel replay            = rehearsal / sleep replay
stability scores           = synaptic consolidation
router/gate                = context-dependent behavior selection
skill graph                = cortical-style capability atlas
```

---

## 1. The geometric view

Let a model be parameterized by:

```math
\Theta
```

The model induces a behavior function:

```math
B_\Theta : (x, c, h) \mapsto y
```

where:

- `x` is an input, observation, state, prompt, or trajectory prefix.
- `c` is a context, task ID, game ID, domain ID, or inferred latent context.
- `h` is optional recurrent/memory state.
- `y` is a behavior object: logits, action distribution, value estimate, embedding, decoded output, Q-values, etc.

A capability or skill `s` is defined by:

```math
s = (\mu_s, d_s, J_s, \epsilon_s)
```

where:

- `mu_s` is the distribution of states/inputs for that skill.
- `d_s` is the behavior distance metric.
- `J_s` is the task performance metric.
- `epsilon_s` is the allowed behavior drift tolerance.

A learned skill is not a point. It is a protected behavior tube:

```math
\mathcal{T}_s
=
\left\{
\Theta :
\mathbb{E}_{x \sim \mu_s}
\left[
    d_s(B_\Theta(x,c_s), B^*_s(x,c_s))
\right]
\leq
\epsilon_s
\right\}
```

where `B_s*` is the best-known teacher behavior for skill `s`.

The current model is safe only if:

```math
\Theta \in \bigcap_{s \in \mathcal{S}_{protected}} \mathcal{T}_s
```

Forgetting is leaving this intersection.

Overfitting is leaving the local generalization tube around the current task manifold.

Therefore, continual learning should be formulated as constrained movement through weight space:

```math
\min_\Theta L_{new}(\Theta)
\quad
\text{subject to}
\quad
\Theta \in \bigcap_s \mathcal{T}_s
```

PMA-C implements this constrained movement with memory, projection, growth, and consolidation.

---

## 2. Impossibility boundary

The following setting is impossible to solve generally:

```text
fixed finite capacity
one uniformly mutable parameter vector
no memory of old behavior
no access to old environments/data
no task/context signal
arbitrary conflicting tasks
```

Example:

```text
Task A: input x -> action left
Task B: same input x -> action right
```

If the input and context are exactly identical, one deterministic function cannot implement both. Any honest general solution must include at least one of:

```text
context or inferred latent context
long-term memory of old behavior
capacity growth / modular experts
frozen/certified old implementations
balanced replay / old environment resampling
slow consolidated weights
```

PMA-C explicitly includes all of these. The guarantee is therefore system-level:

> A protected capability is not forgotten because the system never mutates or deletes the last certified implementation of that capability.

---

## 3. High-level architecture

```text
                         ┌────────────────────────────┐
                         │ Protected Manifold Atlas    │
                         │ skill graph + certificates  │
                         └──────────────┬─────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
┌─────────▼──────────┐       ┌──────────▼─────────┐       ┌───────────▼───────────┐
│ Slow Core           │       │ Plastic Workspace   │       │ Frozen Skill Experts   │
│ consolidated shared │       │ fast adapters/new   │       │ immutable champions     │
│ features/weights    │       │ modules/new skills  │       │ hard long-term memory   │
└─────────┬──────────┘       └──────────┬─────────┘       └───────────┬───────────┘
          │                             │                             │
          └───────────────┬─────────────┴───────────────┬─────────────┘
                          │                             │
                 ┌────────▼────────┐           ┌────────▼────────┐
                 │ Router/Gate      │           │ Sentinel Auditor │
                 │ context selection│           │ regression gates │
                 └────────┬────────┘           └────────┬────────┘
                          │                             │
                          └──────────────┬──────────────┘
                                         │
                                  ┌──────▼──────┐
                                  │ Live Agent   │
                                  │ deployable   │
                                  └─────────────┘
```

PMA-C has four memory strata:

| Layer | Name | Mutable? | Purpose |
|---|---:|---:|---|
| L0 | Episodic memory | append/compress | raw trajectories, examples, prompts, env seeds |
| L1 | Anchor memory | append/replace only with certificate | behavior probes and teacher outputs |
| L2 | Frozen experts/champions | immutable after certification | guaranteed recoverable skill implementations |
| L3 | Slow consolidated core | very slow/projected updates | shared reusable knowledge |

---

## 4. The central invariant

For each protected skill `s`, PMA-C stores at least one certified implementation:

```text
impl_s ∈ {slow core route, frozen expert, adapter, champion checkpoint, composed expert set}
```

The implementation is certified if:

```math
J_s(impl_s) \geq J_s^{best} - \delta_s
```

and:

```math
G_s(impl_s) \leq \epsilon_s
```

and all sentinel tests pass.

**Invariant:**

```text
Never mutate or delete the last certified implementation of any protected skill.
```

This is the part that makes the system robust. A live model may temporarily regress during experimentation, but the system has not forgotten because the certified implementation still exists and can be routed or restored.

---

## 5. Protected Manifold Atlas

The atlas is a graph:

```math
\mathcal{A} = (V, E)
```

Each node `v_s ∈ V` represents a skill/capability/domain region.

### 5.1 Skill node schema

```python
@dataclass
class SkillNode:
    skill_id: str
    context_key: Any                  # task ID, game ID, domain ID, or latent context prototype
    status: Literal["learning", "protected", "archived", "merged"]

    # Behavior memory
    anchors: AnchorStore              # x, context, teacher behavior, tolerance, importance
    sentinels: SentinelStore          # env seeds, validation episodes, challenge probes
    episodic: EpisodicStore           # optional raw trajectories/examples

    # Implementations
    champion_ref: ModelRef            # best frozen/certified teacher
    expert_ref: Optional[ModuleRef]   # frozen adapter/expert if needed
    slow_route_ref: Optional[RouteRef] # route through slow core if certified

    # Metrics
    best_score: float
    current_score: float
    retention: float
    allowed_regression: float
    last_certified_step: int

    # Geometry
    prototype_embedding: Array
    local_radius: float
    stability: float

    # Relations
    positive_transfer_neighbors: set[str]
    interference_neighbors: set[str]
```

### 5.2 Edge schema

Edges encode transfer/interference between skills.

```math
I_{ij}
=
\cos(g_i, g_j)
=
\frac{g_i^\top g_j}{\|g_i\|\|g_j\| + \eta}
```

Interpretation:

```text
I_ij > 0  -> gradients tend to transfer
I_ij < 0  -> gradients interfere
I_ij ≈ 0  -> mostly independent
```

The graph is used to:

```text
sample memory from likely-interfering skills
reuse experts from related skills
decide when to split or merge skill nodes
schedule old-task rehearsal
route contexts to modules
```

---

## 6. Behavior distances

PMA-C is domain-general because it only requires a behavior distance.

### 6.1 Classification / supervised learning

Teacher logits `z*`, current logits `z`:

```math
D(y^*, y)
=
D_{KL}(\text{softmax}(z^*/T) \| \text{softmax}(z/T))
```

### 6.2 Regression

```math
D(y^*, y) = \|y - y^*\|_2^2
```

### 6.3 RL policy

```math
D_\pi
=
D_{KL}
\left(
\pi^*(\cdot | o,h,c)
\|\
\pi_\Theta(\cdot | o,h,c)
\right)
```

### 6.4 RL value

```math
D_V = |V_\Theta(o,h,c) - V^*(o,h,c)|
```

### 6.5 Q-learning

```math
D_Q = \|Q_\Theta(o,c) - Q^*(o,c)\|_2^2
```

### 6.6 Language models

For a prompt/context `x`:

```math
D_{LM}
=
\frac{1}{T}\sum_{t=1}^{T}
D_{KL}
\left(
P^*(\cdot | x_{<t})
\|\
P_\Theta(\cdot | x_{<t})
\right)
```

### 6.7 Embedding / representation learning

```math
D_{emb} = 1 - \cos(e_\Theta(x), e^*(x))
```

---

## 7. Conservation loss

For anchor `i`:

```text
x_i       input/observation/state/prompt
c_i       context/task/domain
h_i       optional recurrent state or trajectory prefix
y_i*      teacher behavior
ε_i       allowed drift
w_i       importance
b_i       behavior bucket / skill ID
```

The anchor loss is:

```math
\ell_i(\Theta)
=
 w_i
\left[
D(B_\Theta(x_i,c_i,h_i), y_i^*) - \epsilon_i
\right]_+^2
```

where:

```math
[z]_+ = \max(0,z)
```

For a skill `s`:

```math
G_s(\Theta)
=
\frac{1}{|M_s|}
\sum_{i \in M_s}
\ell_i(\Theta)
```

Total conservation loss:

```math
G(\Theta)
=
\sum_s \lambda_s G_s(\Theta)
```

The hinge is essential. It prevents freezing useful plasticity. Behavior can drift within tolerance, but violations are corrected.

---

## 8. Tangent-cone gradient projection

Current objective gradient:

```math
g_{new} = \nabla_\Theta L_{new}(\Theta)
```

Conservation gradient for protected skill/bucket `s`:

```math
g_s = \nabla_\Theta G_s(\Theta)
```

The candidate update is:

```math
\Theta' = \Theta - \eta g
```

First-order conservation condition:

```math
G_s(\Theta')
\approx
G_s(\Theta) - \eta \nabla G_s^\top g
```

To avoid increasing conservation loss:

```math
\nabla G_s^\top g \geq 0
```

Therefore the safe gradient is the closest gradient to `g_new` that satisfies the active constraints:

```math
\min_g \frac{1}{2}\|g - g_{new}\|^2
\quad
\text{subject to}
\quad
g_s^\top g \geq 0
\quad \forall s \in \mathcal{S}_{active}
```

A cheap sequential projection is:

```math
g \leftarrow g -
\frac{\min(0, g^\top g_s)}{\|g_s\|^2 + \eta_0} g_s
```

Final gradient:

```math
g_{total}
=
g_{projected}
+
\lambda_G \nabla_\Theta G(\Theta)
```

The projection removes destructive components. The guard term actively repairs drift.

### 8.1 Pseudocode

```python
def project_conflicts(g_new, guard_grads, eps=1e-8):
    g = g_new
    for g_guard in guard_grads:
        dot = tree_dot(g, g_guard)
        norm = tree_dot(g_guard, g_guard) + eps
        coeff = min(dot, 0.0) / norm
        g = tree_add_scaled(g, g_guard, -coeff)
    return g
```

---

## 9. Synaptic stability: hard-to-update weights

Each parameter receives a stability score:

```math
\Omega_j \geq 0
```

Stable parameters receive smaller effective learning rates:

```math
\eta_j
=
\frac{\eta}{1 + \alpha \Omega_j}
```

Update:

```math
\Delta \Theta_j = -\eta_j g_j
```

A simple stability update after certification:

```math
\Omega_j
\leftarrow
\rho \Omega_j
+
(1-\rho)
\left|
\theta_j
\frac{\partial G_s}{\partial \theta_j}
\right|
```

Interpretation:

```text
If changing a weight would damage protected behavior, that weight becomes harder to update.
```

This creates a slow consolidated substrate without freezing the whole network.

---

## 10. Growth: when projection blocks learning

Projection can protect old skills so strongly that the new task cannot learn. That means the current parameterization lacks enough free degrees of freedom.

Measure plasticity ratio:

```math
r
=
\frac{\|g_{projected}\|}{\|g_{new}\| + \eta_0}
```

If:

```math
r < r_{min}
```

for `growth_patience` steps, grow capacity.

Growth options:

```text
new adapter
new expert
new LoRA rank
new skill head
new context embedding
new recurrent memory slot
new router branch
new latent option/policy
```

The new module is initialized as no-op so old behavior is unchanged:

```math
B_{\Theta + \phi_{new}}(x,c) \approx B_\Theta(x,c)
```

Then the new task can learn in `φ_new` while old protected behavior remains intact.

---

## 11. Consolidation

PMA-C has fast learning and slow consolidation.

### 11.1 Fast phase

The plastic workspace learns the current task quickly:

```math
\phi_{fast} \leftarrow \phi_{fast} - \eta_{fast} \nabla_\phi L_{new}
```

### 11.2 Certification

When the new skill reaches threshold:

```math
J_{new}(\Theta, \phi_{fast}) \geq J_{threshold}
```

freeze a champion:

```text
champion_new = snapshot(Θ, φ_fast, route, normalizer, context)
```

Add a new skill node to the atlas.

### 11.3 Slow consolidation phase

Periodically train the slow core to absorb reusable behavior from the atlas:

```math
\min_{\theta_{slow}}
\sum_{s \in \mathcal{A}}
\mathbb{E}_{x \sim M_s}
D(B_{\theta_{slow}}(x,c_s), B^*_s(x,c_s))
+
\lambda_{task} L_{task}
+
\lambda_\Omega \sum_j \Omega_j (\Delta \theta_j)^2
```

subject to sentinel gates:

```math
J_s(\theta_{slow}) \geq J_s^{best} - \delta_s
```

for every protected skill.

### 11.4 Deletion rule

Never delete an expert/champion unless another implementation passes the same certificate.

```python
if candidate_impl.passes_all(skill.sentinels, skill.anchors):
    mark_old_impl_redundant()
else:
    keep_old_impl()
```

This is the non-forgetting invariant.

---

## 12. Router / context gate

The live model may use:

```text
known context        e.g. Atari game ID, task ID, environment ID
inferred context     e.g. from observation/history/prompt
hybrid context       known ID + inferred subskill
```

Router:

```math
\alpha = R_\omega(x,h,c)
```

Top-k composed behavior:

```math
B(x,c)
=
\alpha_0 B_{slow}(x,c)
+
\sum_{k \in TopK(\alpha)} \alpha_k B_{expert_k}(x,c)
```

For policies, combine logits or distributions carefully:

```math
z = \alpha_0 z_{slow} + \sum_k \alpha_k z_k
```

then:

```math
\pi = \text{softmax}(z)
```

### 12.1 Ambiguous context

If two skills require different behavior for identical input and no context/history distinguishes them, no function can solve both. PMA-C requires either:

```text
explicit context
inferred context from history
active probing
separate route selected by external task/environment ID
```

For Atari, use game ID if available. If game ID is hidden, infer it from frames/history.

---

## 13. Memory selection

Do not store random examples only. Store manifold-defining anchors.

Anchor importance:

```math
q(x)
=
\alpha_1 I(x)
+
\alpha_2 N(x)
+
\alpha_3 B(x)
+
\alpha_4 U(x)
+
\alpha_5 F(x)
+
\alpha_6 R(x)
```

where:

| Term | Meaning |
|---|---|
| `I(x)` | loss/advantage/return importance |
| `N(x)` | novelty in representation space |
| `B(x)` | boundary proximity / decision margin |
| `U(x)` | uncertainty |
| `F(x)` | failure or near-failure relevance |
| `R(x)` | rarity / underrepresented region |

### 13.1 RL anchors

Store:

```text
high-return trajectory states
high-advantage states
near-death / near-failure states
near-success states
exploration breakthrough states
rare states
sentinel states
states where old and new policies disagree
recurrent hidden states or trajectory prefixes
```

### 13.2 Supervised anchors

Store:

```text
class-boundary examples
rare classes
hard examples
high-gradient examples
high-confidence exemplars
calibration probes
out-of-distribution probes
```

### 13.3 LLM anchors

Store:

```text
skill-defining prompts
safety prompts
format-following prompts
tool-use prompts
reasoning prompts
previously failed prompts now solved
rare instruction types
```

---

## 14. Memory deletion and compression

Deletion must be certificate-based, not FIFO.

An anchor `a` can be deleted only if there exists a replacement set `R` such that:

```math
\forall x \in neighborhood(a):
D(B_{teacher}(x), B_{student}(x)) \leq \epsilon_a
```

Practical rule:

```python
def can_delete_anchor(anchor, candidate_cover):
    same_skill = candidate_cover.skill_id == anchor.skill_id
    close_in_representation = dist(candidate_cover.embedding, anchor.embedding) < anchor.radius
    matching_teacher = behavior_distance(candidate_cover.teacher, anchor.teacher) < anchor.epsilon
    sentinel_still_passes = run_local_sentinel(anchor.skill_id)
    return same_skill and close_in_representation and matching_teacher and sentinel_still_passes
```

For experts:

```python
def can_archive_expert(skill, candidate_impl):
    return (
        passes_anchor_tests(candidate_impl, skill.anchors)
        and passes_sentinel_tests(candidate_impl, skill.sentinels)
        and score(candidate_impl, skill) >= skill.best_score - skill.allowed_regression
    )
```

Never delete the last certified implementation.

---

## 15. Scheduler

Training distribution should prevent replacement of old manifolds.

For skills/tasks `s`:

```math
p(s)
\propto
\beta_1 \cdot learning\_need(s)
+
\beta_2 \cdot forgetting\_risk(s)
+
\beta_3 \cdot uncertainty(s)
+
\beta_4 \cdot rarity(s)
```

Where:

```text
learning_need     = current task is not solved
forgetting_risk   = current score below best or rising guard loss
uncertainty       = high variance / low confidence
rarity            = under-sampled skill node
```

For RL, this means balanced environment resampling:

```text
current/new tasks
old tasks near regression
old tasks with sparse coverage
sentinel seeds
challenge seeds
```

For Atari sequential learning, a starting schedule:

```text
50% current game
20% old-game environment rollouts
20% old-game sentinel/replay states
10% challenge/synthetic states
```

For many simultaneous tasks, use the scheduler above.

---

## 16. Acceptance gate

Every candidate update must pass gates.

```python
def accept(candidate):
    return (
        current_validation_ok(candidate)
        and conservation_ok(candidate)
        and sentinel_ok(candidate)
        and challenge_ok(candidate)
    )
```

Mathematically:

```math
J_{current}(\Theta') \geq J_{current}(\Theta) - \delta_{current}
```

```math
G_s(\Theta') \leq G_s(\Theta) + \delta_s
\quad \forall s \in protected\_sample
```

```math
J_s(\Theta') \geq J_s^{best} - \Delta_s
\quad \forall s \in sentinel\_sample
```

If a candidate fails:

```text
rollback
increase guard pressure for regressed skills
sample regressed skills more often
mine failure states into memory
grow capacity if projection is repeatedly blocking learning
```

---

## 17. Full PMA-C training loop

```python
def train_pmac(system, stream):
    for step in range(max_steps):
        # 1. Select current learning source.
        source = system.scheduler.sample(system.atlas)
        batch_new = source.sample_batch()

        # 2. Compute current learning gradient.
        loss_new = system.adapter.current_loss(system.model, batch_new)
        g_new = grad(loss_new, system.model.trainable_params())

        # 3. Sample protected behavior memory from high-risk/interfering atlas nodes.
        protected_nodes = system.atlas.sample_protected_nodes(
            current_source=source,
            strategy="interference_and_risk",
        )
        guard_batches = [node.anchors.sample() for node in protected_nodes]

        guard_grads = []
        guard_losses = []
        for node, guard_batch in zip(protected_nodes, guard_batches):
            loss_guard = hinge_conservation_loss(
                model=system.model,
                batch=guard_batch,
                distance=system.adapter.distance,
            )
            guard_losses.append(loss_guard)
            guard_grads.append(grad(loss_guard, system.model.trainable_params()))

        # 4. Project away destructive components.
        g_projected = project_conflicts(g_new, guard_grads)

        # 5. Add corrective guard pressure.
        g_total = g_projected
        for node, g_guard in zip(protected_nodes, guard_grads):
            g_total = tree_add_scaled(g_total, g_guard, node.guard_lambda)

        # 6. Apply synaptic stability scaling.
        g_total = scale_by_stability(g_total, system.omega)

        # 7. Candidate update.
        candidate = system.optimizer.apply_candidate(system.model, g_total)

        # 8. Acceptance gate.
        audit = system.auditor.evaluate_candidate(candidate, source, protected_nodes)
        if audit.accept:
            system.model = candidate
            system.safe_checkpoint.update_if_safe(candidate, audit)
        else:
            system.model = system.safe_checkpoint.restore()
            system.atlas.handle_regression(audit)
            system.scheduler.boost(audit.regressed_nodes)

        # 9. Capacity growth if projection kills plasticity.
        plasticity_ratio = norm(g_projected) / (norm(g_new) + 1e-8)
        if system.growth_controller.should_grow(plasticity_ratio, audit):
            system.model = system.growth_controller.grow(system.model, source)

        # 10. Memory update.
        important = system.memory_selector.select(batch_new, system.model, audit)
        system.atlas.insert_anchors(source.skill_id, important)

        # 11. Certification of new skill.
        if source.is_new_skill and system.auditor.skill_solved(system.model, source):
            node = system.atlas.create_or_update_node(source)
            node.champion_ref = system.snapshot.freeze(system.model, route=source.route)
            node.status = "protected"

        # 12. Slow consolidation.
        if step % system.config.consolidation_interval == 0:
            system = consolidate(system)
```

---

## 18. Consolidation loop

```python
def consolidate(system):
    frozen_teachers = system.atlas.certified_teachers()
    student = system.model.slow_core_candidate()

    for epoch in range(system.config.consolidation_epochs):
        batches = []
        for node in system.atlas.protected_nodes():
            batches.append(node.anchors.sample())

        loss = 0.0
        guard_grads = []

        for node, batch in zip(system.atlas.protected_nodes(), batches):
            teacher = node.champion_ref.load()
            teacher_behavior = teacher.behavior(batch.x, batch.context)
            student_behavior = student.behavior(batch.x, batch.context)

            node_loss = behavior_distance(student_behavior, teacher_behavior)
            loss += node.consolidation_weight * node_loss
            guard_grads.append(grad(node_loss, student.params()))

        g = grad(loss, student.params())
        g = project_conflicts(g, guard_grads)
        g = scale_by_stability(g, system.omega)
        candidate_student = optimizer_step(student, g)

        if system.auditor.consolidation_candidate_ok(candidate_student):
            student = candidate_student
        else:
            break

    # Merge only if certified.
    for node in system.atlas.protected_nodes():
        if system.auditor.impl_certified(student, node):
            node.slow_route_ref = student.route_for(node)
            # Do not delete the expert yet; mark as redundant candidate.
            node.expert_ref.mark_redundant_if_any()

    system.model.install_slow_core(student)
    system.omega = update_stability(system.omega, system.model, system.atlas)
    return system
```

---

## 19. Domain adapters

PMA-C is implemented through an adapter interface.

```python
class DomainAdapter:
    def behavior(self, model, batch):
        """Return behavior object: logits, policy/value, embedding, etc."""
        raise NotImplementedError

    def distance(self, behavior_current, behavior_teacher, batch):
        """Behavior drift metric."""
        raise NotImplementedError

    def current_loss(self, model, batch):
        """Task objective for new data/env."""
        raise NotImplementedError

    def evaluate_skill(self, model, skill_node):
        """Return task score for certification/sentinel."""
        raise NotImplementedError

    def make_challenge_batch(self, skill_node):
        """Generate stress probes / augmentations / adversarial cases."""
        raise NotImplementedError

    def grow_capacity(self, model, source):
        """Add adapter/expert/head/etc. appropriate for domain."""
        raise NotImplementedError
```

### 19.1 RL adapter

Behavior:

```python
Behavior = {
    "policy_logits": logits,
    "policy_dist": dist,
    "value": value,
    "recurrent_state": h,
}
```

Distance:

```math
D = D_{KL}(\pi^* \| \pi) + \lambda_V |V - V^*|
```

Memory anchors include recurrent state or enough trajectory prefix to reconstruct it.

Certification uses:

```text
return/score
success rate
collision/death/failure rate
sentinel seeds
policy KL on anchors
value drift on anchors
```

### 19.2 Atari adapter

Architecture:

```text
shared visual encoder, slow
recurrent core, slow+partly plastic
game/context embedding
router/gate
adapters/experts
universal 18-action policy head
game-conditioned value normalization/head
```

Memory per game:

```text
best checkpoint / champion
high-score trajectories
near-life-loss states
high-advantage states
rare room/screen states
exploration breakthrough states
sentinel seeds
behavior anchors with policy logits and value
```

Routing:

```text
if game ID is available: use it
if game ID is hidden: infer from frame/history
if uncertain: evaluate top-k routes or use ensemble until context confidence is high
```

### 19.3 MuJoCo / robotics adapter

Behavior:

```text
continuous action distribution
value function
latent state estimate
contact/safety predictions
```

Memory:

```text
successful trajectories
near-fall / near-collision states
terrain/task seeds
high-torque/stability boundary cases
rare dynamics states
```

Distance:

```math
D = D_{KL}(\mathcal{N}(\mu^*,\Sigma^*) \| \mathcal{N}(\mu,\Sigma)) + \lambda_V |V - V^*|
```

### 19.4 Supervised adapter

Behavior:

```text
class logits
calibrated probabilities
embedding
```

Memory:

```text
class prototypes
boundary examples
hard examples
rare examples
calibration set
```

### 19.5 LLM adapter

Behavior:

```text
token logits
completion distribution
embedding
tool-call schema behavior
reward-model score
safety classifier behavior
```

Memory:

```text
prompts
expected behavior distribution
format/tool-call probes
safety probes
reasoning/coding/math probes
previously solved failures
```

---

## 20. Atari-specific build recommendation

A practical Atari PMA-C agent:

```text
Encoder: shared CNN or ViT-style visual encoder
Core: recurrent state model / GRU / transformer memory
Context: game embedding or inferred game latent
Router: top-k sparse router
Experts: adapters or LoRA blocks per game/skill cluster
Policy: universal 18-action action head
Value: game-conditioned value head or normalized value head
Memory: per-game anchor/sentinel/champion stores
```

Training protocol:

```text
1. Train game 1 to threshold.
2. Freeze game-1 champion.
3. Add anchors/sentinels for game 1.
4. Train game 2 in plastic workspace.
5. Project gradients against game-1 conservation loss.
6. Grow adapter if projection blocks learning.
7. Certify game 2.
8. Continue through all games.
9. Periodically consolidate into slow core.
10. Never delete a champion until slow/core route passes the same tests.
```

Success metrics:

```text
average retention = current_score / best_score averaged over games
worst-game retention
number of regressed sentinel nodes
new-game learning speed
number of grown modules
policy KL on protected anchors
value drift on protected anchors
```

Target empirical criterion:

```text
average retention >= 80% to 90%
worst-game retention does not collapse
new games still learn
full PMA-C beats ablations across multiple seeds
```

---

## 21. Why this system is more general than CSN-PPO

CSN-PPO was one PPO-specific implementation of the principles.

PMA-C is the general system:

```text
Protected Manifold Atlas    -> general skill graph
Behavior anchors            -> any domain
Projection                  -> any differentiable learner
Growth                      -> adapters/experts/heads/modules
Consolidation               -> slow core / hard long-term memory
Sentinel auditor            -> env seeds, validation sets, prompts, probes
Router                      -> task IDs or inferred context
```

It does not assume PPO, Atari, MuJoCo, pixels, language, or labels.

---

## 22. Why this should work empirically

The usual learner fails because it uses only:

```math
\min_\Theta L_{current}(\Theta)
```

PMA-C solves the missing constraints:

```math
\min_\Theta L_{current}(\Theta)
```

subject to:

```math
\Theta \in \bigcap_s \mathcal{T}_s
```

and if that intersection cannot support new learning:

```text
grow a new chart/module instead of overwriting old charts
```

and if consolidation fails:

```text
keep the old certified expert
```

Therefore:

```text
old behavior is protected by memory and projection
new behavior is learned in plastic capacity
shared knowledge is compressed slowly
old implementations are never destroyed without certification
```

This is the harmony of the system.

---

## 23. What must never happen

These are hard engineering invariants:

```text
1. Never update frozen champion weights.
2. Never delete the last certified implementation of a skill.
3. Never consolidate into slow core unless old sentinel gates pass.
4. Never trust current-task train loss as evidence of generalization.
5. Never label old-failure memory with the currently failing policy.
6. Never sample only the current task for long periods.
7. Never allow a router change that breaks old context routing without certification.
8. Never reduce memory through FIFO deletion alone.
9. Never let all capacity be uniformly plastic forever.
10. Never claim a skill is protected without anchors and sentinels.
```

---

## 24. Minimal implementation modules

```text
pmac/
  adapter.py               # DomainAdapter interface
  atlas.py                 # Skill graph, nodes, edges, certificates
  anchors.py               # Anchor stores, coreset selection, deletion rules
  sentinels.py             # Env seeds / validation tasks / prompt probes
  memory_selector.py       # importance scoring and anchor insertion
  behavior_distance.py     # KL/MSE/cosine/etc.
  conservation.py          # hinge conservation loss
  projection.py            # tangent-cone gradient projection / QP fallback
  stability.py             # Ω importance and learning-rate scaling
  growth.py                # adapter/expert/rank/head growth
  router.py                # context routing and route certification
  scheduler.py             # balanced task/skill sampling
  auditor.py               # acceptance gates, rollback, regression detection
  consolidation.py         # slow-core distillation and certification
  checkpoint.py            # immutable champions and safe checkpoints
  train.py                 # PMA-C training loop
```

---

## 25. Pseudocode for core modules

### 25.1 Conservation loss

```python
def conservation_loss(model, batch, adapter):
    current = adapter.behavior(model, batch)
    d = adapter.distance(current, batch.teacher_behavior, batch)
    violation = relu(d - batch.tolerance)
    return mean(batch.weight * violation ** 2)
```

### 25.2 Projection

```python
def project_conflicts(g_new, guard_grads, eps=1e-8):
    g = g_new
    for g_guard in guard_grads:
        dot = tree_dot(g, g_guard)
        norm = tree_dot(g_guard, g_guard) + eps
        coeff = minimum(dot, 0.0) / norm
        g = tree_add_scaled(g, g_guard, -coeff)
    return g
```

### 25.3 Stability scaling

```python
def scale_by_stability(g, omega, alpha):
    return tree_map(lambda gj, oj: gj / (1.0 + alpha * oj), g, omega)
```

### 25.4 Growth trigger

```python
def should_grow(g_new, g_projected, history, cfg):
    ratio = tree_norm(g_projected) / (tree_norm(g_new) + 1e-8)
    history.append(ratio)
    return len(history) >= cfg.growth_patience and mean(history[-cfg.growth_patience:]) < cfg.growth_min_ratio
```

### 25.5 Certification

```python
def certify_impl(impl, skill_node, adapter):
    anchor_ok = evaluate_anchor_drift(impl, skill_node.anchors, adapter) <= skill_node.anchor_tolerance
    score_ok = adapter.evaluate_skill(impl, skill_node) >= skill_node.best_score - skill_node.allowed_regression
    sentinel_ok = run_sentinels(impl, skill_node.sentinels).passed
    router_ok = route_still_selects_impl(impl, skill_node.context_key)
    return anchor_ok and score_ok and sentinel_ok and router_ok
```

---

## 26. Recommended defaults

```python
PMAConfig(
    anchor_memory_per_skill=10_000,
    episodic_memory_per_skill=100_000,
    sentinel_count_per_skill=128,
    challenge_count_per_skill=512,

    drift_budget_policy_kl=0.01,
    drift_budget_value=0.05,
    guard_lambda=1.0,
    guard_lambda_max=64.0,

    projection_enabled=True,
    stability_enabled=True,
    consolidation_enabled=True,
    growth_enabled=True,

    growth_min_ratio=0.10,
    growth_patience=100,

    slow_lr_multiplier=0.01,
    stability_alpha=10.0,
    stability_decay=0.99,

    consolidation_interval=10_000,
    consolidation_epochs=1_to_10,

    old_skill_sample_fraction=0.30,
    sentinel_eval_interval=1_000_to_10_000_steps,
    full_audit_interval=large_but_regular,
)
```

For Atari or RL, use environment-step-based audit intervals, not update-count-based intervals.

---

## 27. Required ablations

To prove PMA-C works, compare full system against:

```text
no conservation memory
no projection
no growth
no slow consolidation
no frozen champions
no router/context
no sentinel rollback
random memory instead of critical anchors
latest teacher instead of best teacher
uniform task sampling instead of risk scheduler
```

The full system should dominate these ablations in:

```text
average retention
worst-skill retention
new-task learning speed
old-task regression count
final score
area under performance curve
```

---

## 28. Empirical acceptance protocol

### 28.1 Continual Atari

```text
for game in AtariGames:
    train current game to threshold or budget
    certify champion
    add anchors and sentinels
    evaluate all previous games
    record retention
```

Accept if:

```text
average retention >= 0.80
worst-game retention >= 0.60 initially, target >= 0.75
new games still learn above baseline
ablation forgets substantially more
results hold across >= 3 seeds
```

### 28.2 MuJoCo / robotics

```text
train tasks sequentially or with shifting distributions
freeze champions per task/domain
evaluate old-task returns and safety events
require old-task return retention and safety retention
```

### 28.3 Supervised continual learning

```text
sequential class/task/domain training
track average accuracy, backward transfer, calibration, forgetting
```

### 28.4 LLM fine-tuning

```text
new task improves
old skills/safety/tool-use/format/reasoning probes remain within tolerance
router/context does not regress
```

---

## 29. Guarantees under system assumptions

Assumptions:

```text
1. Each protected skill has at least one certified implementation.
2. Certified implementations are immutable unless copied.
3. The router can select the correct implementation from context or inferred context.
4. The last certified implementation is never deleted.
5. Sentinels/anchors are representative enough for the desired empirical domain.
```

Then:

### 29.1 Non-overwrite guarantee

A protected skill cannot be overwritten because its certified implementation is immutable.

### 29.2 Non-deletion guarantee

A protected skill cannot be deleted because deletion is forbidden unless another implementation is certified.

### 29.3 Regression recovery guarantee

If the live route regresses but the champion remains, the system can route back or roll back.

### 29.4 Plasticity guarantee

If protected constraints block learning, growth creates new trainable degrees of freedom initialized as no-op, so old behavior is unchanged while new learning gets capacity.

### 29.5 Generalization caveat

No finite anchor set can guarantee behavior on every unseen state. PMA-C controls this empirically through sentinel coverage, environment resampling, challenge generation, and conservative deletion.

---

## 30. Build order for Codex / Claude Code

### Phase 1: Framework core

Build:

```text
adapter.py
behavior_distance.py
projection.py
conservation.py
stability.py
```

Tests:

```text
KL/MSE distances correct
hinge loss zero inside tolerance
projection removes conflicting gradient component
stability scaling reduces updates to high-Ω params
```

### Phase 2: Atlas and memory

Build:

```text
atlas.py
anchors.py
sentinels.py
memory_selector.py
checkpoint.py
```

Tests:

```text
cannot delete last certified implementation
anchor insertion uses importance scores
sentinel regression creates high-risk event
skill graph tracks interference edges
```

### Phase 3: Training loop

Build:

```text
scheduler.py
auditor.py
growth.py
train.py
```

Tests:

```text
candidate update rejected on protected regression
guard pressure increases on regression
growth triggers when projected gradient ratio stays low
old-skill sampling increases after regression
```

### Phase 4: Consolidation

Build:

```text
consolidation.py
router.py
```

Tests:

```text
slow core distills from champions
old expert is not archived unless slow route certifies
router certification blocks bad route changes
```

### Phase 5: Domain adapters

Build one adapter first:

```text
AtariAdapter or MuJoCoAdapter
```

Then build others.

---

## 31. Final position

The simple version is:

```text
Protect old behavior as manifolds.
Move through weight space only in safe tangent directions.
Grow a new chart when the current manifold intersection cannot support learning.
Consolidate slowly into hard-to-update shared memory.
Never delete the last certified implementation of a skill.
```

This is the general algorithm.

The critical insight is that long-term knowledge should not live only in ordinary mutable weights. It should live in a protected atlas made of:

```text
frozen champions
behavior anchors
sentinel environments/probes
slow consolidated weights
stability-weighted parameters
modular experts/adapters
a certified router
```

That is what makes the system robust across RL, supervised learning, LLMs, Atari, MuJoCo, and future domains.

