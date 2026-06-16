# Living Memory PMA-C for Full ALE Atari

## Purpose

This document specifies the **complete Living Memory PMA-C architecture** for solving catastrophic forgetting in sequential Full ALE Atari training without relying on one full checkpoint per game at runtime.

The target is a single live Atari agent trained across many Atari games, including the full 57-game setting, while retaining previously learned games. The system may use memory, replay, environment resampling, adapters, routing, and slow consolidated weights, but it must not become a library of independent per-game checkpoints.

The central design rule is:

> Do not store full brains as memory. Store compressed, indexed, behavior-relevant memories that the live brain can retrieve, train against, consolidate, merge, and safely delete.

The system must satisfy this deployment invariant:

```text
At evaluation time, the agent may use:
    the current live model,
    bounded compressed memory,
    small bounded adapters,
    the router,
    and the memory index.

At evaluation time, the agent must not load a separate full model checkpoint for the requested game.
```

Short-lived safe checkpoints are allowed only for rollback during training. Temporary teacher snapshots are allowed only for labeling and certification. They are not the final memory system and are not used as runtime policies.

---

## 1. Problem Definition

Let there be a set of Atari games:

```text
G = {g_1, g_2, ..., g_N}
```

For Full ALE:

```text
N can be 57
observation o_t ∈ uint8[4, 84, 84]
action space A has 18 discrete actions
game context c_g is known during training and evaluation
```

The agent trains sequentially or with a curriculum over games. After learning game `g_i`, the agent must not lose the ability to play any previously protected game `g_j`.

For each game `g`, define:

```text
S_g^best      best certified score achieved so far
S_g^current   current score using the live model + memory, not an old checkpoint
R_g           normalized retention
```

Use random-normalized retention for games with different score scales:

```math
R_g =
\frac{S_g^{current} - S_g^{random}}
     {S_g^{best} - S_g^{random} + \epsilon}
```

Clamp only for reporting, not for acceptance logic:

```math
R_g^{report} = \mathrm{clip}(R_g, 0, 1.5)
```

A game is protected when:

```math
R_g \ge R_{min}
```

Recommended hard target:

```text
mean retention >= 0.95
worst-game retention >= 0.90
new-game score >= 90% of the unprotected baseline score
no per-game full checkpoint loaded at evaluation time
```

The system is not complete until the old-game score is preserved by the live model plus memory, not by swapping in saved full weights.

---

## 2. Core Principle

Forgetting is uncontrolled functional drift.

Overfitting is short-horizon uncontrolled functional drift.

Catastrophic forgetting is long-horizon uncontrolled functional drift.

The solution is to constrain learning geometrically:

```text
New learning may move the model only inside the intersection of protected behavior tubes.
If no safe direction exists, grow small capacity or use memory retrieval instead of overwriting old behavior.
```

A protected behavior tube for game `g` is:

```math
\mathcal{T}_g =
\left\{
\theta :
D(f_\theta(m_i, c_g), y_i^*) \le \epsilon_i
\;\;\forall i \in \mathcal{M}_g
\right\}
```

where:

```text
m_i      compressed memory atom
c_g      game context
y_i*     protected teacher behavior
D        behavior distance
epsilon  allowed local drift
```

The live model must stay in:

```math
\theta \in \bigcap_{g \in G_{protected}} \mathcal{T}_g
```

while still improving on the current game.

---

## 3. High-Level Architecture

The complete system has these parts:

```text
1. Live Atari agent
2. Stable memory-key encoder
3. Bounded compressed episodic memory
4. Memory index and retrieval reader
5. Memory-conditioned policy/value network
6. Protected behavior conservation loss
7. Tangent-cone gradient projection
8. Synaptic stability scaling
9. Risk-normalized guard scheduler
10. Closed-loop old-game review scheduler
11. Sentinel evaluation and rollback gate
12. Adapter growth for real conflicts
13. Slow consolidation phase
14. Memory merge, pruning, and certification
```

All parts are required for the Full ALE solution. Do not implement only the guard loss and call the system complete.

---

## 4. Live Atari Agent

The live agent is one model. It contains a shared visual trunk, a memory reader, a policy/value head, and a small adapter bank.

For an Atari observation:

```math
o_t \in \mathbb{R}^{4 \times 84 \times 84}
```

and game context:

```math
c_g \in \mathbb{R}^{d_c}
```

compute:

```math
z_t = E_{key}(o_t)
```

```math
h_t = E_{policy}(o_t, c_g)
```

Retrieve memory:

```math
m_t = \mathrm{ReadMemory}(z_t, c_g)
```

Choose a small adapter mixture:

```math
a_t^{adapter} = \mathrm{Router}(h_t, m_t, c_g)
```

Then compute policy logits and value:

```math
\ell_t, V_t =
F_\theta(h_t, m_t, c_g, a_t^{adapter})
```

The deployed policy is:

```math
\pi_\theta(a|o_t,g) = \mathrm{softmax}(\ell_t)
```

The value estimate is:

```math
V_\theta(o_t,g) = V_t
```

The key point:

> The policy is memory-conditioned. Memory is not only a training regularizer. It participates in inference.

---

## 5. Stable Memory-Key Encoder

The memory index requires a stable latent space. If the encoder drifts freely, old memory keys become meaningless.

Use a separate key encoder:

```math
E_{key}
```

It may share early layers with the policy encoder, but its output space must be stabilized.

Recommended rule:

```text
E_key changes slowly.
Memory keys are periodically re-embedded or aligned.
Key-space drift is explicitly penalized on visual sentinels.
```

The key is normalized:

```math
k_t =
\frac{E_{key}(o_t)}
     {\|E_{key}(o_t)\|_2 + \epsilon}
\in \mathbb{R}^{d_k}
```

Recommended dimension:

```text
d_k = 64, 128, or 256
```

The key encoder has a slow-update rule:

```math
\theta_{key} \leftarrow
(1-\tau_{key})\theta_{key}
+
\tau_{key}\theta_{policy-key}
```

with:

```text
tau_key small, e.g. 0.001 to 0.01
```

Additionally, for visual sentinel memories:

```math
L_{key} =
\mathbb{E}_{(o_i,k_i^*)}
\left[
1 -
\cos(E_{key}(o_i), k_i^*)
\right]
```

This prevents the retrieval space from silently drifting.

---

## 6. Memory Atom

The main memory unit is a compressed behavior prototype.

A memory atom stores:

```text
key             normalized latent key k_i
context         game/context embedding c_i
teacher_policy  protected action distribution p_i*
teacher_value   protected normalized value v_i*
successor_key   optional next-state latent key
action          optional action taken
reward          optional reward
done            optional termination flag
return_trace    optional short return estimate
importance      scalar priority
age             scalar or integer
count           number of merged memories represented
game_id         protected game identifier
cluster_id      local prototype cluster
radius          local coverage radius
eps_policy      allowed policy drift
eps_value       allowed value drift
source_flags    high-return, near-life-loss, sentinel, novelty, failure-recovery, etc.
```

For Atari, do not store every raw frame forever. Store mostly latent prototypes.

Recommended representation:

```text
key:             int8 or fp16 vector
teacher_policy:  fp16[18] as probabilities or logits
teacher_value:   fp16 scalar, normalized per game
context:         small int or fp16 embedding
metadata:        compact scalars
```

Use a small separate visual-sentinel memory for encoder grounding and audits. Visual sentinels may store compressed observation codes or a bounded number of raw frame stacks. This is not the main memory.

---

## 7. Memory Tiers

Use a bounded memory hierarchy.

### 7.1 Hot memory

Used every training update and optionally every inference step.

```text
Lives in VRAM.
Fixed capacity.
Stores the highest-priority memory prototypes.
```

Recommended scale:

```text
50k to 200k atoms
```

### 7.2 Warm memory

Larger memory bank used for refresh, consolidation, and sampling into hot memory.

```text
Lives in CPU RAM or memory-mapped storage.
Indexed by approximate nearest neighbor search.
Fixed global capacity.
```

Recommended scale:

```text
1M to 5M compressed atoms
```

### 7.3 Visual sentinels

Small set of actual or compressed observations used to keep the key encoder and policy grounded.

```text
Bounded per game.
Used for audits and key-space consistency.
Not the main memory.
```

### 7.4 Temporary teacher snapshots

A short-lived teacher snapshot may be kept immediately after a game is learned to label memories and certify consolidation.

It is not deployed as game memory.

It must become discardable once:

```text
live model + memory passes behavior anchors,
live model + memory passes old-game sentinels,
and the memory index contains sufficient compressed prototypes.
```

---

## 8. Memory Write Rule

Do not write every frame. Write only important experience.

During rollouts, compute:

```math
\delta_t =
r_t + \gamma V(o_{t+1}, g) - V(o_t, g)
```

and advantage estimate:

```math
A_t
```

Define novelty:

```math
N_t =
1 -
\max_{i \in \mathcal{M}_{g}}
\cos(k_t, k_i)
```

Define entropy:

```math
H_t =
-\sum_a \pi(a|o_t,g)\log\pi(a|o_t,g)
```

Define life/failure proximity:

```text
L_t = 1 if the transition is near life loss, death, severe negative reward, or sentinel failure.
```

Define return contribution:

```text
Q_t = normalized contribution to high-return trajectory.
```

Define forgetting risk for game `g`:

```math
F_g =
\max\left(
0,
\frac{S_g^{best} - S_g^{current}}
     {|S_g^{best} - S_g^{random}| + \epsilon}
\right)
```

Importance score:

```math
I_t =
w_A \widehat{|A_t|}
+
w_\delta \widehat{|\delta_t|}
+
w_N N_t
+
w_H H_t
+
w_L L_t
+
w_Q Q_t
+
w_F F_g
```

The hats denote normalization by running robust statistics per game.

Recommended default weights:

```text
w_A      1.0
w_delta  1.0
w_N      1.5
w_H      0.25
w_L      3.0
w_Q      2.0
w_F      3.0
```

Write if:

```text
I_t is in the top percentile of the current rollout,
or the state belongs to a rare cluster,
or it comes from a sentinel regression,
or it is needed to satisfy a minimum per-game memory quota.
```

At write time, store teacher behavior from the current best live policy before it is overwritten:

```math
p_t^* = \mathrm{softmax}(\ell_t / T)
```

```math
v_t^* = \frac{V_t - \mu_g}{\sigma_g + \epsilon}
```

where `mu_g` and `sigma_g` are running value/return normalization statistics for game `g`.

Use temperature `T >= 1` for policy targets. Recommended:

```text
T = 1 for exact action preference preservation
T = 2 for smoother behavior preservation
```

---

## 9. Memory Retrieval

Given current key `k_t` and context `c_g`, retrieve top `K` memories.

Use similarity:

```math
s_i =
\frac{k_t^\top k_i}{\tau_r}
+
\beta_c \cdot \mathrm{sim}(c_g, c_i)
+
\beta_I \cdot \log(I_i + \epsilon)
-
\beta_a \cdot \mathrm{agePenalty}_i
```

Default:

```text
same-game memories receive strong positive context bias
cross-game memories are allowed only through positive-transfer links or high latent similarity
```

Select:

```math
\mathcal{N}_K(k_t,c_g) = \mathrm{TopK}_i(s_i)
```

Attention weights:

```math
\alpha_i =
\frac{\exp(s_i)}
     {\sum_{j \in \mathcal{N}_K}\exp(s_j)}
```

Memory summary:

```math
m_t =
\sum_{i \in \mathcal{N}_K}
\alpha_i W_v [k_i, c_i, p_i^*, v_i^*, source_i]
```

Memory policy distribution:

```math
p_{mem}(a|o_t,g)
=
\sum_{i \in \mathcal{N}_K}
\alpha_i p_i^*(a)
```

Memory value:

```math
v_{mem}(o_t,g)
=
\sum_{i \in \mathcal{N}_K}
\alpha_i v_i^*
```

Retrieval confidence:

```math
\rho_t =
\max_{i \in \mathcal{N}_K} \cos(k_t,k_i)
```

Use memory only when retrieval confidence is high enough:

```math
b_t =
\sigma
\left(
w_\rho \rho_t
+
w_c \mathrm{contextMatch}
-
b_0
\right)
```

where `b_t` is a memory-blend coefficient in `[0,1]`.

Final deployed distribution can be either:

### Internal conditioning

```math
\ell_t, V_t = F_\theta(h_t, m_t, c_g, a_t^{adapter})
```

or, for stronger direct recall:

### Explicit probability blend

```math
p_{final}
=
(1-b_t)\,p_{net}
+
b_t\,p_{mem}
```

```math
\ell_{final} = \log(p_{final} + \epsilon)
```

```math
V_{final}
=
(1-b_t)V_{net}
+
b_t(\sigma_g v_{mem} + \mu_g)
```

The explicit blend is recommended for old-game retention because it lets memory directly affect action selection without requiring the base weights to perfectly internalize every memory.

Use the final policy for both training log-probabilities and evaluation.

---

## 10. Current-Game PPO Objective

For the current game, use standard PPO on rollouts from the live model plus memory.

For each transition:

```math
r_t(\theta) =
\frac{\pi_\theta(a_t|o_t,g)}
     {\pi_{\theta_{old}}(a_t|o_t,g)}
```

Policy loss:

```math
L_{\pi}
=
-\mathbb{E}_t
\left[
\min
\left(
r_t(\theta)A_t,
\mathrm{clip}(r_t(\theta),1-\epsilon_{ppo},1+\epsilon_{ppo})A_t
\right)
\right]
```

Value loss:

```math
L_V =
\mathbb{E}_t
\left[
(V_\theta(o_t,g)-R_t)^2
\right]
```

Entropy bonus:

```math
L_H =
-\mathbb{E}_t[H(\pi_\theta(\cdot|o_t,g))]
```

PPO objective to minimize:

```math
L_{PPO}
=
L_\pi
+
c_V L_V
+
c_H L_H
```

where `c_H` is negative if implemented as an entropy reward, or positive if `L_H` is already negative.

---

## 11. Behavior Conservation Loss

For memory atom `i`, define current model behavior from the stored latent key and context.

The model must support a latent behavior head:

```math
\ell_i^\theta, v_i^\theta =
B_\theta(k_i, c_i, \mathrm{ReadMemory}(k_i,c_i))
```

This is necessary so conservation can be applied to compressed latent memories without requiring full raw frames for every memory.

Teacher policy:

```math
p_i^*
```

Current policy:

```math
p_i^\theta =
\mathrm{softmax}(\ell_i^\theta)
```

Policy distance:

```math
D_{\pi,i}
=
D_{KL}
\left(
p_i^* \;\|\; p_i^\theta
\right)
=
\sum_a p_i^*(a)
\left[
\log(p_i^*(a)+\epsilon)
-
\log(p_i^\theta(a)+\epsilon)
\right]
```

Value distance using normalized value:

```math
D_{V,i}
=
\mathrm{Huber}
\left(
v_i^\theta - v_i^*
\right)
```

Combined behavior distance:

```math
D_i =
D_{\pi,i}
+
\lambda_V D_{V,i}
```

Hinge conservation loss:

```math
L_{cons}
=
\mathbb{E}_{i \sim \mathcal{M}}
\left[
w_i
\left[
D_i - \epsilon_i
\right]_+^2
\right]
```

where:

```math
[x]_+ = \max(x,0)
```

This loss is zero inside the protected behavior tube and grows quadratically only when the live model drifts outside tolerance.

---

## 12. Visual Sentinel Loss

Latent-only memory is not enough because the visual encoder can drift. Maintain a bounded set of visual sentinels per protected game.

For visual sentinel `(o_j, k_j^*, p_j^*, v_j^*)`:

```math
k_j^\theta =
\frac{E_{key}(o_j)}
     {\|E_{key}(o_j)\|+\epsilon}
```

Key consistency:

```math
L_{key}
=
\mathbb{E}_j
\left[
1 - k_j^\theta \cdot k_j^*
\right]
```

Behavior consistency on visual sentinels:

```math
L_{visual-beh}
=
\mathbb{E}_j
\left[
D_{KL}(p_j^* \| p_\theta(o_j,c_j))
+
\lambda_V \mathrm{Huber}(v_j^\theta - v_j^*)
\right]
```

Use visual sentinels sparingly. They exist to prevent encoder drift and to audit real observation behavior.

---

## 13. Retrieval Alignment Loss

The memory reader must retrieve the correct memories.

For a visual sentinel or current memory atom with positive key `k_i`, query:

```math
q_i = Q_\theta(o_i,c_i)
```

Use contrastive retrieval loss:

```math
L_{retr}
=
-\mathbb{E}_i
\log
\frac{
\exp(q_i^\top k_i / \tau)
}{
\exp(q_i^\top k_i / \tau)
+
\sum_{j \in \mathcal{N}^-_i}
\exp(q_i^\top k_j / \tau)
}
```

Negatives should include:

```text
same-game nearby but behavior-different memories
different-game memories
high-confusion memories from previous retrieval failures
```

This prevents the memory reader from retrieving irrelevant old behaviors.

---

## 14. Full Training Loss Components

The system uses these losses:

```math
L_{task}       = current/review PPO loss
L_{cons}       = memory conservation loss
L_{visual}     = visual sentinel behavior + key consistency
L_{retr}       = retrieval alignment
L_{adapter}    = adapter sparsity and routing regularization
```

However, the update must not simply minimize the weighted sum. The correct update uses gradient projection.

---

## 15. Tangent-Cone Gradient Projection

Let:

```math
g_{task} = \nabla_\theta L_{task}
```

For each protected game or memory bucket `b`:

```math
g_b = \nabla_\theta L_{cons,b}
```

If:

```math
g_{task}^\top g_b < 0
```

then taking a gradient descent step on `g_task` would increase protected loss `L_cons,b`, because:

```math
\Delta L_{cons,b}
\approx
-\eta g_b^\top g_{task}
```

So remove the destructive component:

```math
g
\leftarrow
g
-
\frac{
\min(0, g^\top g_b)
}{
\|g_b\|^2+\epsilon
}
g_b
```

Sequential projection:

```text
g_safe = g_task
for each protected bucket b:
    if dot(g_safe, g_b) < 0:
        remove conflicting component
```

Then add bounded guard correction:

```math
g_{total}
=
g_{safe}
+
\sum_b \lambda_b \cdot \mathrm{clipNorm}(g_b, \kappa \|g_{task}\|)
+
\lambda_{visual}\nabla L_{visual}
+
\lambda_{retr}\nabla L_{retr}
+
\lambda_{adapter}\nabla L_{adapter}
```

The guard gradient must be clipped relative to the task gradient:

```math
\tilde g_b =
g_b
\cdot
\min
\left(
1,
\frac{\kappa \|g_{task}\|}
     {\|g_b\|+\epsilon}
\right)
```

This prevents conservation from exploding and destroying plasticity.

Recommended default:

```text
kappa = 0.5 to 1.0
```

---

## 16. Risk-Normalized Guard Pressure

A fixed guard coefficient fails as the number of protected games grows.

Use a fixed total guard budget allocated by forgetting risk.

For each protected game `g`:

```math
r_g =
\max
\left(
0,
\frac{
S_g^{best} - S_g^{current}
}{
|S_g^{best} - S_g^{random}|+\epsilon
}
\right)
```

Let:

```math
u_g =
r_g
+
\alpha_v \cdot \mathrm{violationRate}_g
+
\rho
```

where:

```text
violationRate_g = fraction of memory samples outside tolerance
rho = small floor so every protected game gets some guard pressure
```

Allocate:

```math
\lambda_g =
\Lambda_{total}
\frac{u_g}
     {\sum_{h \in G_{protected}}u_h+\epsilon}
```

This keeps total guard pressure approximately constant even as the number of protected games grows.

Do not use:

```text
same guard_coef for every prior game without normalization
```

because accumulated guard pressure grows with the number of old games and will eventually block new learning.

---

## 17. Synaptic Stability Scaling

For each parameter `theta_j`, maintain stability score:

```math
\Omega_j
```

Update after protected behavior gradients:

```math
\Omega_j
\leftarrow
\rho_\Omega \Omega_j
+
(1-\rho_\Omega)
\left|
\theta_j \cdot g_{cons,j}
\right|
```

or use squared gradients:

```math
\Omega_j
\leftarrow
\rho_\Omega \Omega_j
+
(1-\rho_\Omega)
g_{cons,j}^2
```

Apply stability-scaled update:

```math
\Delta \theta_j
=
-\eta
\frac{
g_{total,j}
}{
1+\alpha_\Omega \Omega_j
}
```

This makes parameters important for protected behavior harder to change while leaving unused capacity plastic.

---

## 18. Closed-Loop Old-Game Review

Memory anchors alone do not cover all old-game states. Use actual Full ALE environments for old-game review.

During training on a current game, allocate a fraction of environment interaction to protected-game review.

Review sampling probability:

```math
P_{review}(g)
=
\frac{
u_g
}{
\sum_h u_h+\epsilon
}
```

where `u_g` is the same risk score used for guard allocation.

Training mixture:

```text
mostly current-game PPO rollouts
plus small review rollouts from high-risk protected games
plus memory conservation minibatches
```

Recommended initial ratio:

```text
current game: 80% to 90%
old-game review: 10% to 20%
```

If a game regresses, temporarily increase review for that game.

Review rollouts use the live model plus memory. They are not loaded from old checkpoints.

For review games, use actual PPO/A2C-style loss from environment rewards, but with a smaller coefficient:

```math
L_{task}
=
L_{PPO,current}
+
\lambda_{review}
\sum_{g \in G_{review}}
P_{review}(g) L_{PPO,g}
```

This keeps closed-loop behavior alive, not just pointwise memory behavior.

---

## 19. Sentinel Evaluation and Rollback Gate

Every fixed number of update blocks, evaluate all protected games or a high-risk subset.

The evaluation policy is:

```text
current live model + memory + adapters
```

not an old checkpoint.

For each protected game:

```math
S_g^{current}
```

Compute retention:

```math
R_g =
\frac{S_g^{current}-S_g^{random}}
     {S_g^{best}-S_g^{random}+\epsilon}
```

Regression condition:

```math
R_g < R_{min}
```

or:

```math
S_g^{current} < S_g^{best} - \Delta_g
```

If any protected game regresses:

```text
reject the candidate update block
restore last safe live model + memory + adapter state
increase that game's risk score
increase review sampling for that game
write failure memories
increase retrieval confidence requirement for confusing states
```

A candidate update is accepted only if:

```text
current game does not regress beyond allowed tolerance
protected-game sentinel scores pass
memory conservation violation rate remains bounded
retrieval alignment does not collapse
new-game learning remains above minimum progress threshold
```

This gate is required. Without it, the model can match anchors while failing closed-loop trajectories.

---

## 20. Adapter Growth

If protected memory constraints and new-game learning conflict, do not overwrite old skills. Grow small capacity.

Compute plasticity ratio:

```math
r_{plastic}
=
\frac{
\|g_{safe}\|
}{
\|g_{task}\|+\epsilon
}
```

If:

```math
r_{plastic} < r_{min}
```

for `P` consecutive update blocks and the current game is not improving, grow a small adapter.

Adapter types:

```text
residual MLP adapter after visual trunk
LoRA-style low-rank adapter in dense layers
small game-conditioned policy/value adapter
small memory-reader adapter
```

The adapter must be tiny compared with the base model.

Routing:

```math
a^{adapter}
=
\sum_{k \in TopS}
\gamma_k A_k(h,m,c)
```

Sparse router:

```math
\gamma =
\mathrm{TopS}\left(\mathrm{softmax}(R_\phi(h,m,c))\right)
```

Adapter regularization:

```math
L_{adapter}
=
\lambda_{sparse}\|\gamma\|_1
+
\lambda_{load} L_{load-balance}
+
\lambda_{norm}\sum_k\|A_k\|^2
```

Growth is allowed only when:

```text
projection repeatedly blocks learning,
guard violations are high,
new-game score is below target,
and existing adapters cannot solve the conflict.
```

---

## 21. Slow Consolidation

Consolidation is the process that turns episodic memory into slow weights and reduces dependence on memory.

After each game and periodically during long training, run a consolidation phase.

Consolidation data mixture:

```text
current game high-return memories
protected game memory prototypes
visual sentinels
old-game review rollouts
retrieval-confusion examples
```

Consolidation objective:

```math
L_{consolidate}
=
L_{cons}
+
\lambda_{visual}L_{visual}
+
\lambda_{retr}L_{retr}
+
\lambda_{review}L_{review}
+
\lambda_{adapter-distill}L_{adapter-distill}
```

Use slow learning rate:

```math
\eta_{slow} \ll \eta_{task}
```

Update slow core only if the post-consolidation model passes all sentinels.

Adapter distillation:

```math
D_{KL}
\left(
\pi_{adapter}(a|o,g)
\;\|\;
\pi_{base+memory}(a|o,g)
\right)
```

If the base model plus memory can reproduce adapter behavior, reduce adapter routing probability or merge adapter into the slow core.

---

## 22. Memory Merge

Memory must be bounded.

Two memories `i` and `j` can be merged if:

```math
\cos(k_i,k_j) > 1 - r_{merge}
```

and:

```math
D_{KL}(p_i^* \| p_j^*) < \epsilon_{\pi,merge}
```

and:

```math
|v_i^* - v_j^*| < \epsilon_{V,merge}
```

and:

```text
same game or certified positive-transfer context
```

Merge count:

```math
n = n_i + n_j
```

Merged key:

```math
k_{new}
=
\frac{n_i k_i + n_j k_j}
     {\|n_i k_i + n_j k_j\|+\epsilon}
```

Merge policy probabilities, not raw logits:

```math
p_{new}
=
\frac{n_i p_i^* + n_j p_j^*}{n_i+n_j}
```

Store logits as:

```math
\ell_{new}
=
\log(p_{new}+\epsilon)
```

Merged value:

```math
v_{new}
=
\frac{n_i v_i^* + n_j v_j^*}{n_i+n_j}
```

Merged importance:

```math
I_{new}
=
\max(I_i,I_j)
+
\lambda_{count}\log(1+n_i+n_j)
```

---

## 23. Memory Eviction

Use fixed total memory budget.

Never evict purely by age.

For memory atom `i`, define utility:

```math
U_i =
I_i
+
\lambda_s \mathbf{1}_{sentinel}
+
\lambda_r \mathrm{rarity}_i
+
\lambda_f F_{g_i}
+
\lambda_c \mathrm{teacherConfidence}_i
-
\lambda_m \mathrm{modelCoverage}_i
-
\lambda_a \mathrm{agePenalty}_i
```

Model coverage:

```math
\mathrm{modelCoverage}_i = 1
```

only if:

```math
D_i < \epsilon_i
```

for `H` consecutive audits and neighboring sentinels pass.

Evict lowest utility memories subject to hard constraints:

```text
do not evict the last memory cluster for a protected game
do not evict sentinel-failure memories unless certified covered
do not evict rare high-risk clusters
do not evict if removal causes retrieval or sentinel regression
```

Memory budget allocation across games:

```math
B_g =
B_{min}
+
(B_{total}-N B_{min})
\frac{
u_g
}{
\sum_h u_h+\epsilon
}
```

If `N B_min > B_total`, reduce `B_min` and increase merge pressure. The global budget remains fixed.

---

## 24. Memory Deletion Certification

Before deleting or compressing a memory cluster `C`, run a local deletion audit.

Delete cluster `C` only if all hold:

```math
L_{cons,C}^{without C} \le \epsilon_C
```

```text
retrieval quality for neighboring visual sentinels does not degrade
```

```text
protected game sentinel score remains above threshold
```

```text
old-game review return does not drop
```

```text
no protected game loses its last cluster in that behavior region
```

This is the memory equivalent of safe synaptic pruning.

---

## 25. Checkpoints Are Not Long-Term Memory

The system may use short-lived snapshots for:

```text
rollback of recent unsafe updates
immediate teacher labeling after a game is learned
offline certification
```

But Full Atari evaluation must not load a full checkpoint for the requested game.

Long-term knowledge must live in:

```text
slow shared weights
compressed memory
retrieval index
small adapters
router state
```

A temporary teacher snapshot must be retired when:

```text
memory anchors are labeled
visual sentinels are collected
live model + memory passes certification
adapter/slow-core path can reproduce the protected behavior
```

---

## 26. Overfitting Control

Overfitting is handled by the same protected-memory machinery plus current-game validation.

For the current game, maintain validation seeds distinct from training seeds.

A candidate update block is rejected if:

```text
training return improves
but current-game validation return drops beyond tolerance
```

or:

```text
protected-game sentinel return drops
```

or:

```text
memory conservation violation rate rises
```

or:

```text
retrieval starts choosing incorrect-game memories at high confidence
```

Define current-game validation regression:

```math
S_{current}^{val,new}
<
S_{current}^{val,best}
-
\Delta_{current}
```

This prevents over-specialization to a narrow training seed distribution.

---

## 27. Training Loop

The complete training loop is:

```text
Initialize live model.
Initialize bounded memory.
Initialize router/adapters.
Initialize random score baselines for games.

For each training phase:
    choose current game or game curriculum target
    collect current-game rollouts with live model + memory
    collect review rollouts for high-risk protected games
    write important current/review memories
    sample balanced memory batches by risk and rarity
    compute PPO task gradient
    compute per-game memory conservation gradients
    project task gradient against guard gradients
    add normalized clipped guard corrections
    add visual sentinel, retrieval, adapter regularization gradients
    apply synaptic stability scaling
    perform candidate update
    periodically evaluate sentinel games
    accept only if sentinels and validation pass
    rollback and increase risk/review if regression occurs
    periodically consolidate
    merge/prune memory only after certification
    grow small adapters only when projection blocks learning persistently
```

No step in this loop uses a per-game full checkpoint as the runtime policy.

---

## 28. Pseudocode

```python
for phase in training_phases:

    current_game = scheduler.choose_current_game()

    current_rollouts = collect_rollouts(
        model=live_model,
        memory=memory,
        game=current_game,
    )

    review_games = scheduler.sample_review_games(risk_scores)

    review_rollouts = [
        collect_rollouts(
            model=live_model,
            memory=memory,
            game=g,
        )
        for g in review_games
    ]

    write_important_memories(current_rollouts)
    write_important_memories(review_rollouts)

    task_loss = ppo_loss(current_rollouts)

    for review in review_rollouts:
        task_loss += review_weight(review.game) * ppo_loss(review)

    g_task = grad(task_loss)

    guard_grads = []
    guard_losses = []

    for game in protected_games:
        memory_batch = memory.sample_balanced(game)
        loss_g = conservation_loss(live_model, memory_batch)
        g_g = grad(loss_g)
        g_g = clip_relative_to_task(g_g, g_task)
        guard_grads.append((game, g_g))
        guard_losses.append((game, loss_g))

    g_safe = g_task

    for game, g_g in guard_grads:
        if dot(g_safe, g_g) < 0:
            g_safe = project_out_conflict(g_safe, g_g)

    g_total = g_safe

    for game, g_g in guard_grads:
        g_total += guard_lambda(game) * g_g

    g_total += lambda_visual * grad(visual_sentinel_loss())
    g_total += lambda_retrieval * grad(retrieval_alignment_loss())
    g_total += lambda_adapter * grad(adapter_regularization())

    g_total = scale_by_synaptic_stability(g_total, omega)

    candidate = optimizer_step(live_model, g_total)

    if audit_due:
        audit = run_sentinel_audit(candidate, memory, adapters)

        if audit.passes:
            live_model = candidate
            save_short_term_safe_state()
            update_best_scores(audit)
        else:
            restore_short_term_safe_state()
            scheduler.increase_risk(audit.regressed_games)
            memory.write_failure_memories(audit.failure_states)
            adapter_controller.maybe_grow(audit)

    else:
        live_model = candidate

    if consolidation_due:
        candidate_consolidated = consolidate(live_model, memory, adapters)

        if run_sentinel_audit(candidate_consolidated).passes:
            live_model = candidate_consolidated
            memory.mark_covered_memories()
            memory.merge_and_prune_certified()

    omega = update_synaptic_stability(omega, guard_grads)
```

---

## 29. Completion Criteria

The system is complete only if all are true:

```text
1. Full ALE evaluation uses no per-game full checkpoint.
2. Old games are played by the live model + memory + adapters.
3. Memory has a fixed global budget.
4. Adapter capacity has a fixed or explicitly reported budget.
5. Mean normalized retention is high.
6. Worst-game normalized retention is high.
7. New-game learning remains competitive with baseline.
8. Removing memory/conservation causes forgetting.
9. Removing retrieval harms old-game performance.
10. Removing projection or sentinel rollback worsens stability or retention.
11. Memory compression/pruning does not reduce protected scores.
12. The system scales to longer game sequences without guard pressure exploding.
```

Recommended Full Atari target:

```text
mean normalized retention >= 0.95
worst-game normalized retention >= 0.90
new-game learned score >= 90% of unprotected PPO baseline
fixed memory budget
checkpoint-free evaluation
```

---

## 30. Required Ablations

To know whether the system works for the right reason, compare:

```text
full Living Memory PMA-C
no memory read at inference
no conservation loss
no gradient projection
no visual sentinels
no retrieval alignment
no old-game review
no adapter growth
no consolidation
no memory compression
guard loss only
full checkpoint fallback oracle
plain PPO baseline
```

The important comparison is:

```text
full Living Memory PMA-C
vs.
guard loss only
```

because the current partial system is essentially guard loss only in hard RL.

---

## 31. Failure Modes and Required Fixes

### Failure: old-game anchors pass but old-game score collapses

Cause:

```text
pointwise memory coverage is insufficient
```

Fix:

```text
increase closed-loop old-game review
write failure memories
increase visual sentinels
increase retrieval confidence
grow adapter if needed
```

### Failure: new game cannot learn

Cause:

```text
guard pressure too high or no available safe tangent direction
```

Fix:

```text
normalize guard budget by number of games
use projection instead of raw sum only
grow small adapter
increase current-game allocation temporarily
```

### Failure: memory retrieval returns wrong game

Cause:

```text
context gating too weak or key space collapsed
```

Fix:

```text
increase context bias
train retrieval contrastive loss
add hard negatives from confused games
stabilize key encoder
```

### Failure: memory grows too large

Cause:

```text
write policy too aggressive or merge thresholds too strict
```

Fix:

```text
increase merge pressure
evict certified covered memories
lower per-game minimum budget
consolidate more frequently
```

### Failure: model matches memory but cannot act well

Cause:

```text
memory not integrated into inference or closed-loop distribution shifted
```

Fix:

```text
use explicit memory-policy blending
increase old-game environment review
evaluate more sentinel seeds
add successor-key memories
```

---

## 32. Non-Negotiable Design Choices

These are required.

```text
No per-game full checkpoint as runtime policy.
Memory must be bounded.
Memory must be compressed.
Memory must be used during inference.
Memory must be used during training.
Old games must be evaluated closed-loop.
Unsafe updates must be rejected or rolled back.
Guard pressure must be normalized as games increase.
Encoder/key-space drift must be controlled.
Memory deletion must require certification.
```

Do not remove any of these and still call the result complete.

---

## 33. Final Summary

The complete system is not:

```text
train game
save checkpoint
load checkpoint when evaluating that game
```

The complete system is:

```text
one live model
bounded compressed latent memory
memory-conditioned inference
memory conservation during learning
gradient projection against protected behavior
closed-loop old-game review
sentinel rollback
slow consolidation
safe memory merge/prune
small adapter growth only when necessary
```

This architecture is designed to make learned Atari behavior live in the same agent as compressed, retrievable, consolidatable memory rather than as a pile of frozen full models.

The final test is simple:

> After training many Full ALE games sequentially, the agent must play old games using only the live model, bounded memory, router, and adapters. If it needs to load the old full checkpoint, the system has not solved forgetting.
