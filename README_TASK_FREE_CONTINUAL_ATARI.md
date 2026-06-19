# Task-Free Synaptic Null-Space Continual PPO for Atari

## Research-grade build specification

This document is the implementation contract for replacing the current game-conditioned, memory-action-blending continual Atari design with a single task-free recurrent agent whose old behavior is retained in its live neural weights.

The first validation target is a sequential curriculum of five Atari games. The same system must then extend unchanged to eight games. The architecture, optimizer, memory format, update rules, and evaluation logic must be identical for every game. The implementation must not depend on external game identities at policy inference, memory lookup, adapter routing, value normalization, or gradient protection.

The design is intentionally conservative. It combines several complementary protection mechanisms because no one mechanism is sufficient in deep reinforcement learning:

1. Recurrent task/context inference from experience rather than a supplied game identifier.
2. Bounded episodic sequence memory rather than isolated latent states.
3. Behavior conservation on replayed sequences.
4. Layerwise protected activation subspaces that constrain the actual parameter update.
5. Multiple non-cancelling behavior constraints on high-risk memory clusters.
6. Exact post-update sentinel checks and rollback.
7. Delayed-credit estimation for memory selection and safe reward shaping.
8. Fixed, preallocated residual capacity that is activated only when the shared network has exhausted safe plasticity.

The resulting system is not a library of game policies. It is one recurrent policy, one value function, one fixed 18-action output space, one bounded memory, and one set of protected synaptic subspaces.

---

## 1. North-star requirement

After the agent learns a sequence of Atari games, every previously learned game must be played by the current live network alone.

At action-selection time, the policy may use only:

- The current observation.
- Its recurrent hidden state.
- The previous action.
- The previous reward after a fixed task-independent transform.
- The true reset mask.
- Content-conditioned residual modules that are part of the single live network.

At action-selection time, the policy must not use:

- A game ID, task ID, environment name, curriculum index, or one-hot task vector.
- A per-game policy head, value head, optimizer, normalization state, or parameter tree.
- A frozen full-network checkpoint.
- A router that selects a full policy or checkpoint.
- Stored teacher logits as the final action distribution.
- A nearest-neighbor policy cache.
- A blend between the live policy and a memorized policy.
- An externally supplied indication of which learned skill should be recalled.

Episodic memory, protected subspaces, visual sentinels, and temporary rollback snapshots are allowed during training and continued learning. They are not allowed to become alternate runtime policies.

The final deployment artifact is one live neural network. The continued-learning state additionally contains one bounded episodic memory, one bounded set of protected bases, one optimizer state, one slow target encoder, one return-prediction model, and one fixed residual-capacity bank.

---

## 2. What can and cannot be guaranteed

A finite memory and finite network cannot prove unchanged behavior on every reachable state in every Atari game. A rigorous implementation can guarantee a narrower and useful invariant:

> Every committed optimizer step must remain in the protected layerwise update space and pass a risk-prioritized recurrent sentinel check. Every accepted training block and consolidation phase must additionally pass the complete protected sentinel set and closed-loop environment gates.

This provides a hard update invariant on represented activation subspaces, a tolerance-bounded invariant on protected memories, and an empirical retention test on complete games.

The system must never claim universal zero forgetting merely because stored anchors are unchanged. It must report both:

- Exact or tolerance-bounded conservation on protected sequence memories.
- Closed-loop score retention in the actual environments.

If the protected subspaces leave no useful update direction and all preallocated residual capacity is used, the correct behavior is to report capacity exhaustion or stop the unsafe update. The system must not silently overwrite an old skill.

---

## 3. Core first-principles model

A behavior is not stored in individual scalar weights. It is stored in coordinated mappings between activation patterns and downstream responses.

For a linear operation

$$
y = W x,
$$

an old behavior on input activation $x$ is unchanged after a parameter update $\Delta W$ when

$$
\Delta W x = 0.
$$

If important old activations lie in the column span of an orthonormal basis $U$, then preserving all represented activations requires

$$
\Delta W U = 0.
$$

A candidate update can be projected into the safe null space as

$$
\Delta W_{safe}
=
\Delta W - (\Delta W U)U^\top
=
\Delta W(I-UU^\top).
$$

This permits coordinated changes among weights while preserving old computations. It is strictly more expressive than freezing individually important scalar weights.

The overall memory model is:

- **Episodic memory:** compact causal sequences used to reconstruct old computations and losses.
- **Protected synaptic memory:** low-rank activation bases that encode which coordinated weight directions old behavior depends on.
- **Recurrent context:** an internally inferred belief about the current environment and state of play.
- **Consolidation:** replay old sequences, update protected bases, and remove redundant episodic details only after certification.

This is the operational analogy to hippocampal replay and cortical consolidation. It is not intended as a literal neuroscience model.

---

## 4. Mandatory changes to the current design

The new path must make the following semantic changes. Legacy mechanisms may remain only as explicit baselines and must never be mixed into the headline result.

### 4.1 Remove external task conditioning

Remove all game embeddings and every policy/value dependency on an external game index. Remove same-game retrieval masks, per-game value statistics, per-game adapter masks, and any memory priority or quota computed from a supplied game label.

The external experiment driver may know which environment it instantiated so that it can record scores and run closed-loop evaluations. That identity must never enter the agent, its memory keys, its router, its losses, or its optimizer.

### 4.2 Remove memorized-action deployment

Stored teacher policies are training targets only. They must not be mixed into live logits or sampled directly during evaluation. The action path must remain unchanged when episodic memory contents are shuffled, removed, or corrupted at inference.

### 4.3 Replace feed-forward policy execution with recurrent execution

Four stacked Atari frames are useful but insufficient as the sole context mechanism. Add recurrent state so the agent can infer the active environment and temporal mode from observation, action, reward, and transition history.

### 4.4 Replace isolated latent atoms with sequence memories

The fast path must store actions, rewards, true terminals, life-loss boundaries, predecessor observations, teacher behavior, and causal-credit estimates. Optional sequence fields must not remain unused placeholders.

### 4.5 Replace one averaged guard gradient

A single average conservation gradient can cancel conflicting old constraints. Protection must use layerwise activation null spaces plus multiple behavior constraints from separate high-risk memory clusters.

### 4.6 Protect the applied optimizer delta

Projecting only a raw gradient is not sufficient with Adam or another coordinate-wise preconditioner. The actual optimizer-proposed parameter delta must be projected immediately before application. The optimizer's first-moment state should also be projected. No unprojected decoupled weight decay may modify protected matrices.

### 4.7 Make long-term protection state persistent

Any synaptic-importance or protected-subspace state must survive every minibatch, PPO update, training block, replay phase, environment switch, and consolidation phase. It must be serialized as part of the one continued-learning state.

### 4.8 Replace global top-importance eviction

The memory manager must preserve diversity and coverage. A burst of high-priority samples from the current stream must not evict the sole representatives of older contexts.

### 4.9 Use correctly oriented safety gates

Every gate must compare a loss to a maximum allowed loss or a score to a minimum allowed score. Do not negate a loss and then compare it to a positive score threshold. Gate names and telemetry must make the direction unambiguous.

### 4.10 Do not protect unlearned behavior

A policy is not consolidated merely because a training interval ended. It must first exceed a meaningful learned-score threshold and remain stable over repeated evaluations. Weak or random behavior remains transient and evictable.

---

## 5. Hard architectural invariants

The implementation must satisfy all of the following:

1. One shared visual encoder.
2. One recurrent state transition.
3. One 18-action policy head.
4. One scalar value head.
5. No task-specific output parameters.
6. No game identity in the forward signature.
7. No game identity in memory records.
8. No game identity in adapter routing.
9. No per-game reward or value normalization.
10. Fixed network tensor shapes for all five and eight-game experiments.
11. Fixed global episodic-memory byte budget.
12. Fixed maximum protected-basis budget.
13. Fixed maximum residual-capacity budget allocated at initialization.
14. Teacher behavior is never used as an action source at inference.
15. Temporary candidate snapshots are discarded after acceptance or rejection.
16. Every state mutation involved in an update is committed or rolled back atomically.

A static and runtime leakage test must verify that changing or permuting external environment labels cannot change logits, recurrent states, memory admission, replay sampling, or adapter routing for an identical trajectory.

---

## 6. Agent architecture

### 6.1 Inputs

At time $t$, the agent receives:

- Atari observation $o_t \in \text{uint8}^{4\times84\times84}$.
- Previous action $a_{t-1}$ in the common 18-action space.
- Previous reward transformed as $\bar r_{t-1}=\operatorname{clip}(r_{t-1},-1,1)$.
- A true reset flag indicating a real environment reset or game switch.
- Recurrent hidden state $h_{t-1}$.

The reward fed into the recurrent network must use the same clipping transform during training and evaluation even though score accounting uses unclipped evaluation rewards.

Pseudo-terminal life loss may be used for PPO return construction, but it must not automatically erase recurrent context. Recurrent state resets only on a true full-episode reset, an actual environment reset, or an explicit stream switch. The implementation must distinguish PPO terminal masks from recurrent reset masks.

### 6.2 Visual encoder

Use the existing Nature-style Atari convolutional geometry unless single-task baselines prove a necessary change:

- Convolution with 32 outputs, kernel 8, stride 4.
- Convolution with 64 outputs, kernel 4, stride 2.
- Convolution with 64 outputs, kernel 3, stride 1.
- Dense visual feature of width 512.

Use deterministic activations. Do not use batch normalization, dropout, or any layer whose train/evaluation statistics can drift independently of the protected mapping.

Standard ReLU is the compatibility default. CReLU must be implemented as a controlled plasticity ablation because it has evidence in continual Atari settings. The activation choice must be fixed across all games in a run and must first match the single-task baseline within the accepted tolerance.

Let the visual feature be

$$
e_t=E_\theta(o_t)\in\mathbb{R}^{512}.
$$

### 6.3 Recurrent core

Embed the previous action into 32 dimensions and concatenate it with the visual feature, clipped previous reward, and true reset indicator:

$$
x_t=[e_t;\operatorname{Emb}(a_{t-1});\bar r_{t-1};d^{reset}_{t-1}].
$$

Use a GRU hidden size of 512:

$$
h_t=\operatorname{GRU}_\theta(x_t,h_{t-1}).
$$

If the framework's built-in recurrent cell hides gate matrices in a form that cannot be projected and audited, implement an explicit GRU cell with exposed update, reset, and candidate matrices.

### 6.4 Policy and value

Use one shared policy and one shared value function:

$$
\ell_t=W_\pi h_t+b_\pi,
\qquad
\pi_\theta(a\mid H_t)=\operatorname{softmax}(\ell_t),
$$

$$
V_t=W_V h_t+b_V.
$$

There is no alternate final-logit path. The logits above are the only action logits.

### 6.5 Context key

Produce a normalized, content-derived context key:

$$
q_t=\frac{W_qh_t+b_q}{\|W_qh_t+b_q\|_2+\epsilon}
\in\mathbb{R}^{128}.
$$

The context key is used for sequence-memory diversity, internal context clustering, drift detection, and replay sampling. It is not a game label and is not used to retrieve an action distribution.

Maintain a slow target encoder with exponential moving average updates only after accepted parameter updates. Stored key anchors are generated by the slow encoder.

### 6.6 Predictive auxiliary heads

To force the recurrent state to represent dynamics and task context, add small auxiliary predictions:

- Next slow-encoder visual feature from $(h_t,a_t)$.
- Clipped reward category $-1,0,+1$.
- True terminal probability.

Recommended losses are cosine or normalized regression for the next feature, cross-entropy for reward category, and binary cross-entropy for true terminal. Their default total gradient contribution should remain small relative to PPO and should be logged separately.

Targets from the slow encoder must be stop-gradient. The predictor must not receive an environment identity.

---

## 7. Recurrent PPO requirements

The current flattened-transition PPO update is incompatible with the new recurrent model. PPO training must preserve temporal order.

### 7.1 Rollout state

For each vectorized environment, store:

- Observations.
- Previous actions and rewards.
- Policy actions.
- Old log probabilities.
- Value predictions.
- Rewards.
- PPO terminal masks.
- True recurrent reset masks.
- Initial recurrent state for each contiguous rollout chunk.

### 7.2 Sequence minibatches

Do not randomly flatten individual transitions. Split rollouts into contiguous sequence chunks and batch those chunks. Recompute recurrent states in order with reset masks.

For current on-policy PPO data, the stored initial hidden state from the rollout is valid because the data and old log probabilities came from the same policy version.

For replayed old data, never trust a stored hidden state as authoritative. Reconstruct it by burn-in from raw observations, previous actions, rewards, and reset masks.

Recommended initial values:

- Rollout length: 128 transitions.
- Replay sequence length: 64 transitions.
- Replay burn-in: 16 transitions.
- Protected-loss region: the remaining 48 transitions.

These values are configuration defaults, not separate per-game settings.

### 7.3 PPO loss

Use standard clipped PPO policy loss, clipped or robust value loss, and entropy regularization. Compute GAE using the PPO terminal mask while recurrent resets use the true reset mask.

Replay sequences must never be inserted into the PPO importance-ratio objective. Their behavior policy is stale. They are used only for behavior conservation, representation consistency, predictive learning, and protected-subspace construction.

---

## 8. Task-free context inference and stream-change detection

The recurrent network is the primary context inference system. No discrete task selection is required for action production.

For memory management and automatic consolidation, compute a sequence context signature from a window of slow-encoder context keys:

$$
s_j=\operatorname{normalize}\left(\frac{1}{|j|}\sum_{t\in j}q_t^{EMA}\right).
$$

The memory manager may maintain internal content clusters using only these signatures and predictive-error statistics. Such cluster IDs:

- Are created online.
- Are not initialized from game labels.
- Are not supplied to the policy.
- Are not used as hard action routes.
- May split one game into multiple modes or merge similar modes across games.

Use persistent change detection rather than a single novelty spike. A regime change should require both:

- Sustained increase in predictive error relative to robust running statistics.
- Sustained distance from recent context-signature support.

A robust Page-Hinkley, CUSUM, or equivalent detector is acceptable. Use median/MAD or another outlier-resistant scale estimate. Include a cooldown so one novel room or rare Atari event does not repeatedly trigger consolidation.

The initial five-game blocked experiment may expose stream boundaries to the outer training controller solely to schedule evaluation and consolidation. The agent must still receive no identity. The completed system must also support fully automatic boundary detection and an unannounced-switch evaluation.

---

## 9. Bounded episodic sequence memory

### 9.1 Purpose

The episodic memory serves four roles:

1. Reconstruct old recurrent computations.
2. Supply soft behavior/value targets.
3. Estimate behavior drift and context risk.
4. Build and audit protected activation subspaces.

It is not an inference-time policy.

### 9.2 Memory record

Each stored sequence must contain enough information to exactly reconstruct the model inputs over the sequence:

- Frame-efficient observations. Store the initial frame stack and one new grayscale frame per transition, or another lossless representation that reconstructs every 4-frame observation exactly.
- Actions.
- Clipped rewards used by the policy.
- Optional raw rewards for analysis only.
- PPO terminal/life-loss masks.
- True reset masks.
- Teacher policy logits or probabilities for each protected transition.
- Teacher value targets in the shared clipped-return scale.
- Slow-encoder context-key anchors.
- Estimated causal contribution and backward credit trace.
- Advantage magnitude and TD-error magnitude at write time.
- Dynamics-surprise score.
- Teacher confidence or entropy.
- Sequence importance.
- Internal content-cluster identifier, if clustering is enabled.
- Episode identifier and ordering metadata needed to link adjacent chunks.
- Protection status: transient, candidate, protected, failure-recovery, or deletion-pending.

Do not store a game ID, environment name, curriculum index, or external skill label.

Do not store a recurrent hidden state as the only means of replay. A hidden state may be cached for diagnostics, but replay must be correct after reconstructing it with burn-in under the current network.

### 9.3 Global budget

Use one fixed byte budget across all experiences. A practical initial research budget is 1 GiB of CPU-resident compressed sequence memory, with the exact number of records determined by encoded size. The budget must remain fixed as the number of learned games grows.

GPU memory contains only the currently sampled replay batch, protected basis matrices, and small index/cache structures. It must not contain one permanent policy-sized object per game.

### 9.4 Admission priority

For transition $t$, compute robustly normalized terms:

- $|A_t|$: advantage magnitude.
- $|\delta_t|$: TD-error magnitude.
- $C_t$: delayed-credit contribution.
- $N_t$: novelty in context/representation space.
- $S_t$: predictive dynamics surprise.
- $F_t$: failure, life-loss, or terminal proximity.
- $H_t$: policy entropy.
- $D_t$: current conservation drift or cluster risk.

A default priority form is

$$
I_t=
1.0\widehat{|A_t|}
+1.0\widehat{|\delta_t|}
+2.0\widehat{C_t}
+1.5N_t
+1.0\widehat{S_t}
+2.0F_t
+0.25H_t
+3.0D_t.
$$

Aggregate transition priorities into a sequence score using a mixture of mean, upper quantile, and maximum so that one crucial event is not diluted by an otherwise ordinary sequence.

Weights are global defaults. They are never tuned separately by game.

### 9.5 Diversity and coverage

Admission and eviction must combine importance with marginal coverage. Content coverage is measured in sequence-signature space and, after consolidation, by the residual activation energy not already represented by the protected bases.

A new record is valuable when it:

- Covers a previously unsupported context region.
- Contains high delayed-credit transitions.
- Exposes a conservation failure.
- Adds activation directions not represented by existing bases.
- Represents a rare failure/recovery path.

A record is redundant when a nearby record has similar context signature, teacher policy, value, causal contribution, and activation coverage.

### 9.6 Eviction

Never evict solely by global scalar priority. Use a utility that rewards importance, risk, coverage, and causal value and penalizes redundancy:

$$
U_i=I_i
+\lambda_{risk}R_i
+\lambda_{cover}C^{marginal}_i
+\lambda_{causal}C^{causal}_i
-\lambda_{red}R^{redundancy}_i
-\lambda_{age}A^{redundant}_i.
$$

Age may reduce utility only when the record is already redundant. Old age alone is not evidence that a memory is unimportant.

Maintain a minimum representation for each internally discovered context cluster, but do not allow unbounded cluster growth. Merge close clusters by content and behavior similarity. Internal cluster budgeting must be independent of external game labels.

### 9.7 Protected sentinels

A small subset of raw sequences and frames is marked as protected sentinels. A protected sentinel cannot be evicted until deletion certification demonstrates all of the following:

- Its behavior is covered by other sentinels.
- Its activation directions are covered by the retained protected bases.
- Removing it does not worsen held-out conservation metrics.
- Closed-loop retention remains within threshold when the corresponding content region is tested.

---

## 10. Teacher targets and behavior tubes

Teacher targets are generated only from an accepted live model after a behavior is certified as learned. A temporary full-model snapshot may be used long enough to label candidate memories and evaluate the candidate, then it must be discarded.

For each protected transition store:

$$
p_t^*=\operatorname{softmax}(\ell_t^*/T),
\qquad
V_t^*=V_{\theta^*}(H_t),
\qquad
q_t^*=q_{\theta_{EMA}^*}(H_t).
$$

Use $T\geq1$, normally $T=1$. Store soft policy targets, not only argmax actions.

Define behavior distance

$$
D_t(\theta)=
D_{KL}(p_t^*\|\pi_\theta(\cdot\mid H_t))
+\lambda_V\operatorname{Huber}(V_\theta(H_t)-V_t^*)
+\lambda_q(1-\cos(q_t,q_t^*)).
$$

Use behavior tubes rather than an always-active quadratic penalty:

$$
L_{tube}=
\mathbb{E}_t
\left[
 w_t\,\operatorname{ReLU}(D_t-\epsilon_t)^2
\right].
$$

Recommended initial tolerances:

- Policy KL tolerance: 0.01.
- Value absolute/Huber tolerance: 0.1 in clipped-return value units.
- Context-key cosine-distance tolerance: 0.02.

The replay objective must include both mean violation and a worst-tail term such as the mean of the top 10% violations. An average alone can hide a small catastrophically forgotten region.

Teacher targets may be replaced only when a newer live model is certified to be at least as good in closed-loop evaluation and does not increase protected behavior violations.

---

## 11. Delayed reward and causal credit assignment

Plain GAE remains the reliable fallback. Delayed-credit machinery must improve memory selection first and modify training rewards only after it is validated.

### 11.1 Return predictor

Train a separate causal sequence model with two task-free heads:

1. A prefix-return head that predicts the complete discounted episode return from the history observed so far:

$$
F_\psi(H_t)\approx
\mathbb{E}[G_0\mid H_t],
\qquad
G_0=\sum_{k=0}^{T-1}\gamma^k r_k.
$$

2. A remaining-return potential head:

$$
\Phi_\psi(H_t)\approx
\mathbb{E}\left[
\sum_{k=t}^{T-1}\gamma^{k-t}r_k
\mid H_t
\right].
$$

The heads may share one small causal recurrent trunk. The predictor must receive observation features, actions, rewards, and reset masks but no task identity. It may consume stop-gradient features from the policy encoder or maintain its own encoder. Gradients from this predictor must not bypass the protected-update machinery into the policy network.

Train it on completed episodes and linked memory chunks. Use held-out episodes or held-out chunks for validation. Freeze its parameters for the duration of each rollout block so reward shaping and memory priorities are internally consistent.

### 11.2 Causal contribution for memory

Use the innovation in the prefix-return prediction after transition $t$ as the default causal-importance signal:

$$
c_t=F_\psi(H_{t+1})-F_\psi(H_t),
\qquad
C_t=|c_t|.
$$

The chronology must define $H_t$ before action $a_t$ and $H_{t+1}$ after observing its consequence, so the innovation is attributed to $a_t$. For accounting, define an initial baseline contribution $c_{init}=F_\psi(H_0)$, the action-attributed innovations above, and a terminal residual $c_{term}=G_0-F_\psi(H_T)$. Their sum must equal $G_0$ within numerical tolerance. Only the action-attributed innovations are used as causal priorities. They are not directly substituted for the environment reward when $\gamma<1$.

Propagate importance backward with an eligibility trace:

$$
I_{t-k}\mathrel{+}=(\gamma\lambda_c)^k C_t.
$$

This gives earlier decisions credit when a reward arrives much later. The implementation must verify action/contribution indexing with deterministic synthetic delayed-reward tests.

### 11.3 Safe reward shaping

Do not naively replace rewards with prefix-prediction differences when $\gamma<1$; that is not generally discounted-return equivalent.

The safe default is potential-based shaping using a frozen predictor for the duration of a rollout block:

$$
\tilde r_t
=
r_t+
\eta\left[
\gamma\Phi_\psi(H_{t+1})-
\Phi_\psi(H_t)
\right],
$$

with terminal potential set to zero. This preserves the optimal policy under the standard potential-shaping conditions while moving information about future return closer to causal actions.

Enable shaping only when the return predictor beats a constant-return baseline on held-out data for multiple consecutive validation windows. A safe automatic coefficient is

$$
\eta=
\operatorname{clip}
\left(
1-\frac{\operatorname{MSE}_{val}}{\operatorname{Var}(G)+\epsilon},
0,
0.5
\right).
$$

If validation degrades, set $\eta=0$ immediately. Memory prioritization may continue using conservative contribution estimates.

The system must log the potential-shaping telescoping residual and verify terminal handling. Reward shaping is an ablation; failure of the predictor must not break PPO.

---

## 12. Protected activation subspaces

### 12.1 Protected modules

Maintain an orthonormal input-activation basis for every behavior-relevant affine operation:

- Each convolution.
- The dense visual layer.
- Every GRU gate matrix.
- Policy head.
- Value head.
- Context-key head.
- Adapter routers and adapter matrices once they contribute to protected behavior.

Pure auxiliary heads that cannot affect policy, value, context, or routing need not be protected.

### 12.2 Bias handling

Protect weights and biases jointly by augmenting each presynaptic activation with a constant one:

$$
\bar x=[x;1].
$$

Treat the bias as an extra row or column of the affine matrix. This permits coordinated weight-bias updates that preserve old preactivations instead of freezing every bias independently.

### 12.3 Dense layers

For conceptual orientation $y=W x$, with $W\in\mathbb{R}^{d_{out}\times d_{in}}$, project the actual update as

$$
\Delta W_{safe}
=
\Delta W-(\Delta WU)U^\top.
$$

If the framework stores dense kernels as $[d_{in},d_{out}]$, apply the equivalent left projection:

$$
\Delta K_{safe}
=
\Delta K-U(U^\top\Delta K).
$$

Do not materialize a full identity matrix.

### 12.4 Convolutions

Flatten each convolutional kernel into a matrix whose input dimension is

$$
d_{in}=k_hk_wc_{in}+1
$$

including the bias dimension. Build the basis from sampled input patches using the same padding and stride semantics as the convolution. Project the flattened update and reshape it back.

Patch collection must include high-credit, high-gradient, high-surprise, and coverage-diverse spatial locations rather than uniformly storing every patch.

### 12.5 GRU

For each GRU time step form the augmented presynaptic vector

$$
\xi_t=[x_t;h_{t-1};1].
$$

Each update, reset, and candidate gate matrix is projected against a basis spanning protected $\xi_t$ vectors. The same input basis may be shared across gates because their presynaptic vectors are identical, but every gate update must be projected.

If the initial hidden state and every protected gate preactivation are unchanged, the hidden trajectory is unchanged by induction over time. Recurrent invariance tests must verify this over complete stored sequences, including reset masks.

### 12.6 Basis construction

After a candidate behavior is certified, replay its selected sequences through the accepted model with teacher-forced stored actions. Collect weighted presynaptic activations for every protected module.

Let $A_l\in\mathbb{R}^{d_l\times n}$ contain normalized activation columns weighted by the square root of causal/behavioral importance. Given old basis $U_l$, compute the residual

$$
R_l=A_l-U_l(U_l^\top A_l).
$$

Compute an SVD or randomized SVD of $R_l$. Append the smallest number of residual singular vectors needed to capture a configurable fraction of residual energy. A recommended initial threshold is 99.5%.

Reorthogonalize the concatenated basis with QR or an equivalent stable procedure. Basis computation is float32 or higher precision. Basis storage may use float16 only after verifying that reloaded projection error remains within tolerance.

Do not silently truncate a basis and declare old behavior protected. If a rank limit would discard necessary directions, the system must either:

- Prove the discarded directions are redundant through sentinel tests.
- Activate unused residual capacity.
- Reject further learning and report capacity exhaustion.

### 12.7 Approximate protection bound

When an activation $x$ is only approximately represented by $U$, decompose

$$
x=UU^\top x+r.
$$

Then

$$
\|\Delta W_{safe}x\|
\leq
\|\Delta W_{safe}\|\,\|r\|.
$$

Track residual norms for protected activations and include them in the basis-quality gate. This makes the approximation error explicit.

### 12.8 Basis update timing

Do not add current-task activations to protected bases while the policy is still learning or unstable. Otherwise poor intermediate behavior becomes permanently protected. Update bases only after certification.

---

## 13. Optimizer-safe parameter updates

This section is a hard implementation invariant.

### 13.1 Why raw-gradient projection is insufficient

Adam transforms a raw gradient using elementwise first and second moments. Elementwise preconditioning can rotate a matrix update out of a protected activation null space. Therefore, projecting the gradient and then applying Adam does not guarantee that the applied parameter delta is safe.

### 13.2 Required update sequence

For every optimizer step:

1. Compute the on-policy recurrent PPO loss.
2. Compute predictive auxiliary losses.
3. Sample replay sequences and compute behavior-tube losses.
4. Compute and log each gradient component separately.
5. Form the combined raw gradient using global, risk-adaptive coefficients.
6. Project raw gradients for protected affine parameters before they enter Adam. This keeps first moments cleaner but is not the final safety operation.
7. Ask Adam for a candidate optimizer state and candidate parameter delta without committing either.
8. Project the actual candidate parameter delta for every protected affine parameter.
9. Project the updated Adam first-moment matrices against the same bases.
10. Apply the multiple-constraint correction described below in the already projected update space.
11. Enforce a final global update-norm bound.
12. Run post-update sentinel backtracking checks.
13. Commit parameters, optimizer state, slow encoder, and mutable counters atomically only if the candidate passes.
14. On rejection, restore all pre-step state. Do not advance the slow encoder or memory teacher targets.

Second-moment statistics may remain coordinate-wise, but the final applied parameter delta must always be projected. Whenever a protected basis expands, immediately project stored first moments. Repeated rejection may trigger first-moment reset for the affected modules.

Use Adam without unprojected decoupled weight decay. If weight decay is desired, include its delta in the candidate update before the final projection.

---

## 14. Multiple non-cancelling behavior constraints

Layerwise null-space protection is the hard structural mechanism. A second functional guard catches approximation error, unprotected nonlinear effects, and coverage gaps.

Do not average all old memories into one gradient. Maintain separate constraints for the highest-risk content clusters.

For cluster $i$, define an unhinged behavior distance $D_i(\theta)$, its gradient

$$
g_i=\nabla_\theta D_i(\theta),
$$

and remaining tolerance

$$
m_i=D_i^{max}-D_i(\theta).
$$

Given candidate update $\Delta_0$, require the first-order condition

$$
g_i^\top\Delta\leq m_i.
$$

Project each $g_i$ into the same layerwise allowed update space so a correction cannot reintroduce a forbidden parameter direction. Select the worst-risk clusters, recommended initial maximum eight per constrained update.

Solve

$$
\min_\Delta\frac12\|\Delta-\Delta_0\|_2^2
\quad\text{subject to}\quad
G\Delta\leq m,
$$

where rows of $G$ are the separate projected gradients.

The small dual problem is

$$
\min_{\lambda\geq0}
\frac12\lambda^\top(GG^\top+\mu I)\lambda
-\lambda^\top(G\Delta_0-m),
$$

and

$$
\Delta=\Delta_0-G^\top\lambda.
$$

Use a small ridge $\mu$ for numerical stability. Include only constraints that the candidate is predicted to violate. If the solver fails or produces non-finite values, reject the update.

The constrained correction may run once per PPO update or at another documented cadence chosen for throughput. Layerwise delta projection still runs on every optimizer step.

---

## 15. Exact post-update checks and backtracking

First-order constraints are not sufficient in a nonlinear recurrent policy. Every candidate update must be checked on a sampled sentinel batch before commitment.

Evaluate the candidate parameters on complete recurrent sentinel sequences after burn-in and require:

- Policy KL remains below each sequence or cluster tolerance.
- Value error remains below tolerance.
- Context-key drift remains below tolerance.
- Router output drift remains below tolerance for protected contexts.
- No non-finite activations, logits, values, losses, or recurrent states.

Try step scales

$$
\alpha\in\{1,1/2,1/4,1/8,1/16,1/32\}.
$$

Apply the largest scale that passes. If none passes, reject the step and restore the old optimizer state as well as parameters.

Do not accept based only on a mean. Check maximum or high quantile per protected cluster. A tiny cluster with severe regression must not be hidden by a large stable cluster.

---

## 16. Risk-adaptive replay and protection pressure

Let each internal content cluster have risk

$$
u_c=
\rho_0
+\lambda_D\,\text{behavior-violation}_c
+\lambda_Q\,\text{high-quantile-drift}_c
+\lambda_R\,\text{basis-residual}_c
+\lambda_A\,\text{time-since-replay}_c.
$$

Normalize replay sampling and any soft conservation coefficient across clusters so the total protection budget remains bounded as experience grows:

$$
P(c)=\frac{u_c}{\sum_j u_j+\epsilon}.
$$

The default replay transition count should begin near 25% of the current on-policy transition count and rise toward parity when protected risk increases. It must not grow linearly with the number of games.

Cluster risk also contributes to memory admission and prevents a currently regressing context from being evicted.

---

## 17. Plasticity and fixed residual capacity

Protection can eventually remove most useful update directions. The system must measure this directly.

For module $l$, track:

$$
r_l^{free}=1-\frac{\operatorname{rank}(U_l)}{d_l},
$$

and applied-update plasticity

$$
\rho_l=
\frac{\|\Delta_l^{safe}\|}
{\|\Delta_l^{candidate}\|+\epsilon}.
$$

Also track activation rank, dead-unit fraction, gradient norm, and current-environment score progress.

### 17.1 Preallocated residual modules

Allocate a fixed maximum bank of small low-rank residual adapters at initialization. Recommended initial capacity is eight adapters of rank 32, inserted at two locations:

- After the dense visual feature.
- After the recurrent hidden state before the policy/value heads.

An adapter is

$$
A_k(x)=U_k\sigma(V_kx),
$$

with the up projection initialized to zero so activating a dormant adapter initially changes no output.

The visual adapter router is computed from the unadapted visual feature, previous recurrent state, previous action embedding, and previous clipped reward; this avoids a circular dependency. The post-recurrent adapter router is computed from the new recurrent state. Both routers produce sparse top-two mixtures, receive no task label, and are protected/distilled on old sentinels like policy behavior.

Adapters are not allocated per game. One adapter may serve several content regions, and one game may use several adapters. The global adapter count and ranks are fixed before training.

### 17.2 Activation rule

Activate a dormant adapter only when all are true for multiple consecutive training blocks:

- Median protected-update ratio is below 0.1.
- Current-environment score is not improving.
- Replay and conservation losses are finite and active.
- The blockage is attributable to protected constraints rather than a broken optimizer or environment.

A default patience of three blocks is reasonable.

Once an adapter contributes to protected behavior, its relevant parameters and router inputs become part of protected-subspace construction. Do not freeze or route adapters by external game identity.

If no dormant capacity remains, reject unsafe learning and report capacity exhaustion.

---

## 18. Consolidation lifecycle

Consolidation is the transition from transient experience to protected long-term skill.

### 18.1 Candidate phase

During current-environment learning, new sequence records are transient. They may be replayed for predictive learning and current-policy stability but do not constrain future behavior permanently.

### 18.2 Learning certification

A behavior becomes eligible for protection only when:

- Closed-loop score exceeds random by a meaningful margin.
- Its random-normalized progress reaches at least 90% of the matched single-task progress, or another predeclared learned threshold. For score $S$, use $(S-S^{random})/(S^{single}-S^{random}+\epsilon)$ so negative-score games are handled correctly.
- Performance is stable across at least two independent evaluation windows.
- The policy and value functions have no non-finite behavior.

The experimental controller may use the environment name to associate a score with a benchmark column. This identity is never supplied to the model.

### 18.3 Label memories

Using the accepted live model, generate teacher policy, value, context-key, and activation data for the selected candidate sequences. This temporary teacher is the same live model at the certification point, not a permanently retained policy.

### 18.4 Expand protected bases

Replay selected sequences, collect weighted presynaptic activations, update residual SVD bases, reorthogonalize, and verify residual coverage.

### 18.5 Slow replay consolidation

Run a bounded low-learning-rate replay-only phase using behavior tubes, predictive losses, context alignment, and optional adapter-to-shared-path distillation. All updates still pass through projection, multiple constraints, and post-update checks.

### 18.6 Closed-loop acceptance gate

Evaluate every previously learned benchmark environment using:

- Current live weights.
- Zero-initialized recurrent state at true episode reset.
- No game identifier.
- No episodic-memory action read.
- No checkpoint route.

Accept consolidation only if protected retention and current behavior pass. Otherwise roll back parameters, optimizer, slow encoder, protected bases, adapter state, memory metadata, context clusters, and return-predictor state atomically.

### 18.7 Prune only after certification

Merge or delete redundant episodic records only after the new protected bases and remaining sentinels reproduce their behavior within tolerance. Raw sentinel coverage must remain sufficient to detect representation drift.

---

## 19. Closed-loop rollback policy

Temporary snapshots are allowed only as transactional safety state.

Before each candidate training block or consolidation phase, snapshot every mutable component:

- Live parameters.
- Optimizer state.
- Slow encoder.
- Protected bases.
- Episodic-memory metadata and protection states.
- Internal context centroids.
- Active adapter state.
- Return predictor and its optimizer.
- Running robust statistics.

On rejection:

1. Restore the complete snapshot.
2. Increase replay probability for violating content clusters.
3. Promote the failed sequences as failure-recovery candidates, but do not make their poor teacher behavior authoritative.
4. Reduce candidate learning rate or update norm.
5. Activate dormant residual capacity only if the plasticity criteria are met.
6. Retry within a fixed retry budget.
7. If safe progress remains impossible, stop and report the conflict.

A parameter rollback without optimizer, basis, or memory rollback is invalid.

---

## 20. Inference and continued-learning separation

The forward policy must not take episodic memory as an argument. Protected bases are update constraints and do not participate in forward inference.

Two modes must be clearly separated:

### Inference mode

Uses live parameters, recurrent state, and active content-routed residual modules only.

### Continued-learning mode

Additionally uses episodic replay, protected bases, optimizer state, slow encoder, context index, return predictor, and transactional snapshots.

A final invariance test must show that evaluation scores and action traces are unchanged when the episodic memory is unavailable during inference.

---

## 21. Five-game and eight-game validation protocol

### 21.1 Primary five-game suite

Use:

1. Space Invaders.
2. Breakout.
3. Beam Rider.
4. Asterix.
5. Q*bert.

Use the common full 18-action space for every game.

### 21.2 Eight-game extension

Add:

6. Pong.
7. Seaquest.
8. Demon Attack.

The extension must change only the curriculum list and training duration. It must not change the architecture, memory schema, loss definitions, protection rules, or hyperparameters.

### 21.3 Single-task references

Train the exact same recurrent architecture independently on each game using the same PPO hyperparameters and maximum environment-step budget, but without continual protection. These runs define matched single-task learning references.

A sequential result is not meaningful if the game was never learned. Protect a game only after it meets the learned threshold.

### 21.4 Sequential curriculum

Train games in blocked sequence for the first proof. Do not interleave raw old-environment transitions into the core training stream. Episodic sequence replay is allowed and required.

Run at least three random seeds. Use more than one game order, including a substantially different or reversed order, before claiming order robustness.

### 21.5 Evaluation

After each training block and after each consolidation:

- Evaluate every learned game.
- Use true un-clipped episode returns.
- Use at least 30 completed episodes per game for the research result.
- Report stochastic policy evaluation as primary and deterministic greedy evaluation as a diagnostic.
- Use fixed evaluation seed sets plus a disjoint held-out seed set.
- Report mean, standard error or bootstrap confidence interval, and per-game score.

### 21.6 Retention

For game $g$, define

$$
R_g=
\frac{S_g^{current}-S_g^{random}}
{S_g^{best}-S_g^{random}+\epsilon}.
$$

Do not clamp this value for acceptance logic. Clamping is allowed only in a clearly labeled visualization.

Report:

- Mean retention.
- Worst-game retention.
- Per-game retention.
- Normalized forgetting.
- Current-game plasticity relative to matched single-task training.
- Area under the retention curve over the full sequence.
- Protected-subspace rank and free-update ratio.
- Episodic-memory usage in bytes.
- Active adapter count and routing entropy.

### 21.7 Initial success targets

The research target for the five-game suite is:

- Mean retention at least 0.95.
- Worst-game retention at least 0.90.
- Each newly learned game reaches at least 90% of its matched single-task random-normalized progress.
- No game/task identity enters the policy.
- No per-game full checkpoint exists at runtime.
- Episodic memory is not used for action selection.
- Global memory and residual-capacity budgets remain fixed.

Failure to reach these thresholds is an experimental result, not permission to weaken the invariants or use game-conditioned routing.

### 21.8 Unannounced-switch evaluation

After sequential learning, construct a stream of episodes drawn from learned games in random order. The agent receives no identity and starts with zero recurrent state at each true reset.

Measure:

- First-episode score after a switch.
- Frames required for context-key stabilization.
- Score over the next several episodes.
- Router/adaptor stability.
- False context-change detections.

Also include a harder diagnostic in which environment switches occur without a task cue at valid reset boundaries.

---

## 22. Required ablations

At minimum compare:

1. Recurrent PPO only.
2. Recurrent PPO plus episodic behavior replay.
3. Recurrent PPO plus protected activation subspaces only.
4. Replay plus subspaces without multiple constraints.
5. Full system without delayed-credit shaping.
6. Full system without residual adapters.
7. Full system without raw visual sentinels.
8. Full system.

Include a diagnostic that projects only raw gradients but not Adam deltas. It should not be used as a valid protection implementation; it exists to demonstrate why applied-delta projection matters.

A legacy game-ID or checkpoint-routing system may be reported as a structural upper-bound baseline, but its retention must not be compared as though it were the same problem.

---

## 23. Unit and property tests

The build is incomplete until all of the following are tested.

### 23.1 Identity leakage

- Policy forward function has no task argument.
- Permuting external benchmark labels leaves all logits and hidden states unchanged for identical inputs.
- Memory admission and replay sampling are unchanged by label permutation.
- Serialized parameters contain no per-game policy/value trees.

### 23.2 Linear invariance

For random protected activations and random candidate updates, verify after projection:

$$
\|\Delta WU\|\leq\text{numerical tolerance}.
$$

Verify affine outputs on represented activations remain unchanged after the actual applied update.

### 23.3 Bias invariance

Verify augmented weight-bias projection preserves affine outputs while permitting coordinated nonzero weight and bias changes.

### 23.4 Convolution invariance

Collect exact input patches, project a candidate kernel update, and verify convolution outputs on those patches remain unchanged.

### 23.5 GRU invariance

For protected input/action/reward sequences and reset masks, verify gate preactivations and complete hidden trajectories remain unchanged after projected updates.

### 23.6 Adam safety

Construct a raw gradient whose Adam-preconditioned delta leaves the raw-gradient null space. Verify:

- Raw-gradient projection alone can fail.
- Final delta projection restores invariance.
- Projected first-moment state remains in the allowed space.
- Rejected updates restore the old optimizer state.

### 23.7 Basis construction

Verify:

- Orthonormality.
- Monotonic represented energy.
- Stable residual SVD updates.
- No duplicate directions after repeated consolidation.
- Serialization/reload preserves projection tolerance.

### 23.8 Multiple-constraint solver

For synthetic conflicting constraints, verify the returned update satisfies every active inequality. Verify non-finite or failed solves reject the update.

### 23.9 Backtracking

Create a nonlinear case in which the first-order constraint passes but the full candidate violates behavior tolerance. Verify line search reduces the step or rejects it.

### 23.10 Replay burn-in

Verify replay losses after burn-in match a full unroll from the episode start within tolerance. Stored hidden-state corruption must not affect reconstructed replay.

### 23.11 Terminal semantics

Verify life-loss PPO boundaries do not incorrectly reset recurrent context, while true game resets and environment switches do.

### 23.12 Delayed credit

Use deterministic delayed-reward environments to verify:

- Contribution indexing assigns credit to the causal action.
- Eligibility traces propagate backward correctly.
- Potential shaping uses a frozen predictor within a rollout.
- Terminal potential is zero.
- The shaping telescope matches the expected boundary shift.
- Shaping disables itself when validation quality falls.

### 23.13 Memory budget and diversity

Verify the byte budget is never exceeded, current-stream bursts cannot evict sole old-context representatives, and protected sentinels require deletion certification.

### 23.14 No inference memory path

Evaluation actions must be identical when episodic-memory teacher policies are shuffled or unavailable.

### 23.15 Adapter safety

Activating a zero-initialized dormant adapter must initially change no output. After training, old-sentinel router and policy tolerances must still pass.

### 23.16 Atomic rollback

Inject a deliberate old-skill regression and verify parameters, optimizer, slow encoder, bases, memory metadata, clusters, adapter state, and return predictor all restore exactly.

### 23.17 Gate orientation

Every maximum-loss gate must fail when its loss increases beyond the threshold. Every minimum-score gate must fail when its score falls below the threshold. Include explicit regression tests against sign inversions.

---

## 24. Integration tests

Before expensive Atari runs, require:

1. A small partially observable two-context environment in which identical instantaneous observations require different actions depending on recent history. The recurrent agent must solve it without a task ID.
2. A two-task sequential environment with deliberately conflicting gradients. Plain training must forget; projected training must preserve stored behavior.
3. A delayed-reward environment in which memory importance identifies the early causal action.
4. A two-game Atari smoke test verifying end-to-end recurrent rollout, sequence replay, basis collection, applied-delta projection, certification, rollback, and memory-disabled evaluation.
5. The full five-game protocol.
6. The unchanged eight-game extension only after the five-game system passes functional and safety tests.

Stub-only tests are insufficient for the final claim. At least one real environment test must exercise every protection mechanism.

---

## 25. Telemetry and audit output

Every training block must expose enough data to diagnose why learning succeeded or failed:

- Current PPO policy, value, entropy, and approximate-KL losses.
- Replay behavior-tube mean and worst-tail violations.
- Per-cluster policy KL, value drift, and key drift.
- Raw gradient norm, Adam candidate-delta norm, projected-delta norm, and final applied norm.
- Per-layer protected rank, free-rank fraction, activation residual, and update plasticity ratio.
- Number of active functional constraints and QP residuals.
- Backtracking scale and rejection count.
- Return-predictor validation error and shaping coefficient.
- Context-change detector statistics.
- Memory bytes, sequence count, cluster count, coverage, merges, evictions, and deletion certifications.
- Adapter activation events, usage distribution, and routing drift.
- Closed-loop score matrix after each learned game.
- Temporary snapshot creation and deletion counts.

The report must distinguish:

- Behavior retained by the live network.
- Behavior conserved on replay memories.
- Closed-loop environment retention.

There is no deployed-memory-retention metric because the memory does not produce actions.

---

## 26. Failure handling

### Current behavior never becomes learned

Do not consolidate it. Continue training within the declared budget or report failure to learn. Never define a low score as a protected success merely to obtain high retention.

### Old behavior regresses on sentinels

Reject the update, restore all state, raise replay risk for the violating clusters, and reduce the candidate step.

### Sentinels pass but closed-loop score regresses

Rollback the training block. Mine failure/recovery sequences from the regressed rollout, expand coverage, and do not weaken the closed-loop gate.

### Protected basis consumes nearly all input rank

Attempt redundancy-certified basis compression. If that fails, activate dormant residual capacity. If no capacity remains, stop unsafe learning.

### Context keys drift

Use raw stored sequences and visual sentinels to realign the current key encoder with fixed old key anchors. Recompute only mutable search-index keys; do not overwrite the original anchors without certification.

### Return predictor is inaccurate

Set shaping coefficient to zero. Continue with PPO, GAE, replay, and subspace protection.

### Constraint solver is unstable

Reject the candidate update. Never fall back silently to an unconstrained update.

### Evaluation is too noisy

Increase completed evaluation episodes and use confidence intervals. Do not clamp retention or choose the maximum of incompatible evaluation runs.

---

## 27. End-to-end algorithm

```text
initialize one recurrent policy/value network
initialize one slow target encoder from the live network
initialize one Adam optimizer state
initialize empty episodic sequence memory
initialize empty protected activation bases for all protected modules
initialize all residual adapters dormant
initialize return predictor and task-free context-change detector

for each incoming environment stream block:
    collect recurrent on-policy rollouts
    preserve true-reset masks separately from PPO life-loss masks
    compute standard GAE
    train/evaluate the delayed-return predictor on completed sequences
    if predictor is certified:
        compute potential-shaped rollout rewards with a frozen predictor
        recompute GAE from shaped rewards
    compute causal contribution and memory-admission signals

    for each recurrent PPO update:
        sample on-policy sequence minibatches
        sample risk-balanced replay sequence minibatches
        reconstruct replay recurrent states by burn-in
        compute PPO, predictive, and behavior-tube losses
        compute separate gradient components
        project raw gradients for protected affine parameters
        obtain candidate Adam state and candidate parameter delta
        project the actual candidate parameter delta
        project Adam first moments
        apply multi-cluster constrained correction in projected space
        bound final update norm
        backtrack against full recurrent sentinel losses
        if accepted:
            atomically commit parameters and optimizer state
            update slow encoder
        else:
            restore all pre-step state

    admit high-value, coverage-diverse transient sequence memories
    update task-free context statistics and change detector

    when a behavior is eligible for consolidation:
        snapshot the complete mutable state
        verify the current behavior is genuinely learned
        label selected sequences with accepted teacher behavior
        collect protected presynaptic activations
        expand and reorthogonalize protected bases
        run bounded replay-only slow consolidation
        evaluate all learned benchmark environments with the live network only
        if every gate passes:
            mark candidate memories protected
            certify safe memory merges/deletions
            discard the temporary snapshot
        else:
            restore the complete snapshot
            increase risk for failed content regions
            optionally activate dormant residual capacity

return one live network plus bounded continued-learning state
```

---

## 28. Definition of complete

The new system is complete only when all of the following are true:

- It sequentially learns the five-game suite with one architecture.
- The policy never receives a game identity.
- The final live network plays every learned game without an action-memory read.
- No full per-game checkpoint is available to the runtime policy.
- Recurrent context is reconstructed from experience.
- Episodic memory stores causal sequences and remains within a fixed global byte budget.
- Protected activation bases persist across the entire curriculum.
- The actual Adam parameter delta is projected before application.
- Multiple old behavior constraints cannot cancel into one mean gradient.
- Every accepted update passes recurrent sentinel checks.
- Every accepted training block passes closed-loop retention gates.
- Delayed-credit shaping is mathematically discount-correct and automatically disabled when unreliable.
- Residual capacity is content-routed, preallocated, bounded, and not assigned by game ID.
- Memory-disabled inference produces the same action path used in the reported scores.
- The five-game result is replicated across multiple seeds and more than one order.
- The unchanged implementation can be extended to the eight-game suite.
- Failures and capacity exhaustion are reported honestly rather than hidden by routing, score clamping, or checkpoint fallback.

---

## 29. Research basis

The implementation should be informed by, but not blindly copy, the following ideas:

- Elastic Weight Consolidation: scalar synaptic importance is a useful baseline but does not represent coordinated safe weight changes. ArXiv 1612.00796.
- Gradient Episodic Memory: separate old constraints and project candidate updates rather than averaging them. ArXiv 1706.08840.
- Experience Replay for Continual Learning / CLEAR: replay plus behavioral cloning can reduce forgetting in task-agnostic continual reinforcement learning. ArXiv 1811.11682.
- Orthogonal Gradient Descent: preserve old outputs by restricting parameter-space updates. ArXiv 1910.07104.
- Gradient Projection Memory: construct protected subspaces from layer activations. ArXiv 2103.09762.
- Recurrent Experience Replay in Distributed Reinforcement Learning: reconstruct recurrent context with sequence replay and burn-in while accounting for representational drift. OpenReview r1lyTjAqYX.
- RUDDER: use return prediction and contribution analysis to address delayed rewards. ArXiv 1806.07857.
- Policy Invariance under Reward Transformations: use discount-correct potential shaping $\gamma\Phi(H_{t+1})-\Phi(H_t)$ instead of arbitrary reward redistribution when $\gamma<1$. Ng, Harada, and Russell, 1999.
- Loss of Plasticity in Continual Deep Reinforcement Learning: continually trained Atari agents can lose the ability to learn, so plasticity must be measured independently of forgetting. ArXiv 2303.07507.
- Maintaining Plasticity in Deep Continual Learning: dead or low-utility representations require explicit monitoring and possibly controlled renewal in unprotected capacity. ArXiv 2306.13812.

The build must preserve the invariants in this specification even where a referenced method uses task IDs, classification labels, raw-gradient-only projection, unbounded replay, or a different optimizer.
