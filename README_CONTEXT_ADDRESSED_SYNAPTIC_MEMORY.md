# Context-Addressed Synaptic Tensor Memory
## Research and Build Specification for Task-Free Continual Atari Learning

**Working name:** CASTM — Context-Addressed Synaptic Tensor Memory  
**Immediate target:** five Atari games sampled once at random from the canonical Atari-57 suite  
**Long-term target:** one agent and one architecture that can learn Atari-57 sequentially, infer context from experience, retain prior behavior, and remain plastic without game IDs, per-game checkpoints, frozen policies, or inference-time episodic action lookup  
**Hardware available:** two RTX 4090 GPUs  
**Implementation standard:** research-grade, test-driven, numerically verified, and optimized for rapid experimental iteration

---

# 1. Executive directive

Do not continue tuning the current null-space/sentinel system as the primary solution. That system reached the expected stability-plasticity local minimum:

- Strong global protection preserves old behavior but removes most useful update directions.
- Weak protection restores plasticity but allows old behavior to drift.
- A hard behavioral sentinel can reject nearly every update and silently turn continual learning into a frozen policy.
- A single mutable weight vector cannot simultaneously receive an arbitrary new-task update and remain unchanged for all old tasks unless the update happens to lie in the remaining old-task null space.

The next system must change the representation of long-term weights.

The central design is:

> The network does not have one fixed effective weight matrix per layer. It has a compact synaptic memory tensor. A recurrently inferred, content-derived address retrieves the effective weights needed for the current behavioral context. New learning is written into an address direction that is algebraically orthogonal to previously stored addresses, so the full new update is retained while previously decoded weights remain unchanged.

The invariant is not that old scalar parameters never move. The invariant is:

$$
W^{\text{effective,new}}_\ell(k_i)
=
W^{\text{effective,old}}_\ell(k_i)
\quad\text{for every protected address } k_i.
$$

All underlying storage may remain live, resident in VRAM or pageable memory, writable on recall, transportable under representation changes, and recompressible. Unrelated experience must not alter the decoded function of an old memory.

---

# 2. Required outcome

The final system must satisfy all of the following.

1. **One policy architecture.** No game-specific actor, critic, encoder, checkpoint, or manually selected head.
2. **No external task identity.** Game names, curriculum indices, environment IDs, and task labels may exist only in the experiment harness for environment construction and reporting. They must never enter policy inference, context inference, routing, memory addressing, optimization, or replay selection.
3. **No inference-time episodic action memory.** Episodic memory may train, validate, audit, and index the network. It may not return an action distribution that is blended into the deployed policy.
4. **No frozen full policies.** A complete network copy per game is prohibited.
5. **No permanent scalar freezing as the main retention mechanism.** Contextual memories must remain intentionally revisable when recalled. Global reparameterization and compression may change every stored number while preserving decoded behavior.
6. **Full new-context plasticity.** New learning must not be restricted to the shrinking null space of all prior network activations.
7. **Exact synaptic noninterference at protected addresses.** A committed write for context $i$ must not change the decoded layer weights for context $j\ne i$, up to a stated floating-point tolerance.
8. **Compactness.** Long-term storage must be low-rank and bounded. The five-game prototype should consume far less than the current gigabyte-scale episodic replay bank.
9. **Task-free recall.** The active address must be inferred from observation, action, reward, termination, and recurrent history.
10. **Fast experimentation.** The design must support fixed JAX shapes, sparse memory execution, early stopping, cached references, and parallel experimentation on two independent GPUs.

The initial five-game experiment is successful only when the same live agent both learns each new game and retains every previously learned game.

---

# 3. What to preserve from the current branch

Keep and reuse the validated repair work:

- Recurrent PPO with separate PPO terminal and true recurrent reset semantics.
- Correct temporal replay alignment using $(o_t,a_{t-1},r_{t-1},\text{reset}_t)$.
- Admission only when recurrent context can be reconstructed from raw history.
- Forced-FIRE masking out of the PPO importance-ratio objective.
- True unclipped reward storage.
- Sequence replay records with raw frames and teacher outputs.
- Transactional snapshots and rollback.
- Exact completed-episode evaluation infrastructure after fixing the remaining partial-return fallback.
- Unit tests for task-identity leakage.
- Existing Atari environment wrappers and the fixed 18-action policy interface.
- Plain recurrent PPO as a matched baseline.

Keep the existing null-space/sentinel implementation only as an ablation and diagnostic baseline. Do not layer the new system on top of hard per-update sentinel rejection.

---

# 4. What must be replaced

Replace these as primary retention mechanisms:

- Global layerwise null-space projection against all old activations.
- Hard behavioral sentinel acceptance on every PPO minibatch.
- Global residual adapter bases that are recomputed over all protected behaviors.
- Adapter updates that touch every active adapter.
- Any per-game adapter mask used by evaluation.
- One-gigabyte replay as the primary behavioral store.

The new primary store is a content-addressed field of low-rank weight deltas.

---

# 5. Core mathematical model

## 5.1 Full synaptic tensor

For an ordinary affine layer,

$$
y = W x + b.
$$

Replace the fixed effective matrix with an address-conditioned matrix:

$$
y = W(k)x+b(k),
$$

where $k\in\mathbb R^{d_k}$ is an internally inferred canonical memory address.

The conceptual full tensor is

$$
\mathcal W\in
\mathbb R^{d_{\text{out}}\times d_{\text{in}}\times d_k}.
$$

The effective matrix is the contraction

$$
W(k)
=
W_0+
\mathcal W\times_3 k
=
W_0+
\sum_{a=1}^{d_k}k_a\mathcal W_{:,:,a}.
$$

The bias has an analogous addressed memory:

$$
b(k)=b_0+\mathcal Bk,
\qquad
\mathcal B\in\mathbb R^{d_{\text{out}}\times d_k}.
$$

$W_0,b_0$ are shared weights. They are live and may be changed transactionally with exact compensation described later.

## 5.2 Protected address set

Let the canonical protected addresses be columns of

$$
K=[k_1,\ldots,k_n]
\in\mathbb R^{d_k\times n}.
$$

The preferred implementation uses an orthonormal canonical codebook:

$$
K^\top K=I.
$$

The codebook is internal. A content router maps experience to these addresses. It is not an externally supplied task ID.

The general implementation must also support nonorthogonal addresses. Define the dual address matrix

$$
D=K(K^\top K+\lambda I)^{-1}.
$$

With full rank and $\lambda=0$,

$$
D^\top K=I.
$$

Let $d_i$ be column $i$ of $D$. Then

$$
d_i^\top k_j=\delta_{ij}.
$$

For an orthonormal codebook, $D=K$ and $d_i=k_i$.

## 5.3 Exact context-specific write

Suppose PPO produces an **applied optimizer delta** $\Delta W_i$ that should modify context $i$.

Write it to synaptic memory as

$$
\mathcal W'
=
\mathcal W+
\Delta W_i\otimes d_i.
$$

The decoded update at context $j$ is

$$
W'(k_j)-W(k_j)
=
\Delta W_i(d_i^\top k_j)
=
\Delta W_i\delta_{ij}.
$$

Therefore:

$$
W'(k_i)=W(k_i)+\Delta W_i,
$$

while

$$
W'(k_j)=W(k_j)
\quad\forall j\ne i.
$$

This is the primary no-forgetting guarantee.

The same rule applies to bias:

$$
\mathcal B'
=
\mathcal B+
\Delta b_i\otimes d_i.
$$

## 5.4 Allocating a genuinely novel address

For a general candidate address $u$, remove the span of existing addresses:

$$
P_\perp=I-KK^+,
$$

$$
r=P_\perp u.
$$

If

$$
\|r\|>\epsilon_{\text{novel}},
$$

allocate

$$
k_{n+1}=\frac{r}{\|r\|}.
$$

When $K$ is orthonormal, the new dual is simply

$$
d_{n+1}=k_{n+1}.
$$

The implementation should normally allocate from a precomputed orthonormal canonical codebook. The residual formula remains required for tests, continuous-address experiments, and future dynamic address growth.

## 5.5 Reconsolidating an existing context

When a familiar context is recalled and improved, commit its new applied delta with the existing dual $d_i$:

$$
\mathcal W'
=
\mathcal W+
\Delta W_i\otimes d_i.
$$

The memory is therefore live and revisable. It is not a frozen checkpoint.

## 5.6 Conditional guarantee

The algebra guarantees synaptic noninterference **conditional on correct address retrieval**. It does not guarantee that an ambiguous history can always identify the correct latent context. Synaptic retention and context inference must be tested separately.

---

# 6. Compact factorized synaptic memory

A full tensor is unnecessary. Store a contextual low-rank factorization:

$$
W(k)
=
W_0+
\sum_{m=1}^{M}
(c_m^\top k)B_mA_m,
$$

where

$$
A_m\in\mathbb R^{r_m\times d_{\text{in}}},
\qquad
B_m\in\mathbb R^{d_{\text{out}}\times r_m},
\qquad
c_m\in\mathbb R^{d_k}.
$$

Each block $m$ is a low-rank synaptic memory component. It is not a game head. Components are addressed by content and may later be merged, rotated, shared across contexts, or rewritten.

The effective output is

$$
y
=
W_0x+b_0
+
\sum_m
(c_m^\top k)
B_m(A_mx)
+
\sum_m(c_m^\top k)\beta_m.
$$

For a write to context $i$, use

$$
c_m=d_i.
$$

If a scratch delta is already represented as

$$
\Delta W_i=B_sA_s,
$$

then committing it only requires appending

$$
(A_m,B_m,c_m)=(A_s,B_s,d_i).
$$

No dense materialization or SVD is required. SVD is used only for recompression.

## 6.1 Dense layers

Use the formula directly. Bias memory is stored separately or by augmenting the input with a constant one. A separate addressed bias is simpler and should be used initially.

## 6.2 Convolutional layers

Represent a rank-$r$ convolutional delta as:

1. A spatial convolution $A_m$ with the original kernel size producing $r$ channels.
2. A $1\times1$ convolution $B_m$ mapping $r$ channels to output channels.

For kernel $K$,

$$
\Delta K_m
\approx
B_m\circ A_m.
$$

This is equivalent to a low-rank factorization of the kernel flattened as

$$
(k_hk_wc_{\text{in}})\times c_{\text{out}}.
$$

## 6.3 GRU gates

For each gate $g\in\{z,r,n\}$, concatenate input and previous hidden state:

$$
\xi_t=[x_t;h_{t-1}].
$$

Use

$$
p_g(k)
=
W^0_g\xi_t+b^0_g
+
\sum_m(c_{g,m}^\top k)
B_{g,m}(A_{g,m}\xi_t)
+
\beta_g(k).
$$

All three gates must have independent factor banks.

## 6.4 Policy and value heads

Both heads use addressed low-rank factors. The policy remains one 18-action head. The value remains one scalar head.

## 6.5 Contextualized layers

The implementation must support contextual deltas at:

- Convolution 1.
- Convolution 2.
- Convolution 3.
- Encoder dense layer.
- All three GRU gates.
- Policy head.
- Value head.

For rapid pilot runs, a configuration may begin with encoder dense, GRU, policy, and value enabled. If an oracle-address experiment cannot reach the new-game progress gate, contextual capacity must be expanded into convolution 2 and 3 before changing the memory mathematics.

---

# 7. Fast plastic scratchpad

Do not append a persistent memory component for every PPO minibatch.

Each contextualized layer has one temporary low-rank scratch delta:

$$
S_\ell=B^s_\ell A^s_\ell.
$$

During a stable context segment, the effective layer is

$$
W_\ell^{\text{active}}
=
W_{0,\ell}
+
W_{\text{memory},\ell}(k)
+
S_\ell.
$$

PPO updates the scratch factors freely. Old contexts do not constrain the scratch optimization because the scratch is not yet part of long-term memory and is used only while the current address is active.

At a commit event:

1. Freeze the optimizer transaction momentarily.
2. Compute the **actual applied scratch delta**, including optimizer scaling.
3. Append the scratch factors to synaptic memory with address factor $d_i$.
4. Verify decoded-weight invariance for every protected address.
5. Verify functional sentinels.
6. Commit atomically.
7. Reset scratch factors and scratch optimizer state.

Use LoRA-style initialization:

- $A^s$: small orthogonal or Gaussian initialization.
- $B^s$: zero initialization.

This gives a zero initial delta while preserving a usable gradient path.

Scratch optimizer state is local to the currently locked context segment. Do not use one global Adam momentum state across unrelated contexts. The simplest correct first implementation is Adam for scratch factors, reset at context switch. A later implementation may store addressed optimizer moments with the same dual-write rule.

---

# 8. One live shared network

## 8.1 Initial learning

Before any context is protected, train the shared network normally with recurrent PPO. This gives the agent a useful first set of visual and control features.

After the first context is certified:

- Preserve its effective weights through the addressed tensor.
- Learn later contexts primarily through scratch deltas.
- Continue improving shared weights only through exact compensated consolidation.

This is not permanent freezing. The shared substrate remains mutable through a safe coordinate-preserving operation.

## 8.2 Exact compensated shared update

Suppose a low-rank component $S$ should be moved into shared weights:

$$
W_0'=W_0+S.
$$

To preserve all protected addresses, find $g$ satisfying

$$
K^\top g=\mathbf 1.
$$

The minimum-norm solution is

$$
g=K(K^\top K)^{-1}\mathbf 1.
$$

Update memory as

$$
\mathcal W'
=
\mathcal W-S\otimes g.
$$

For every protected address $k_i$:

$$
W_0'+\mathcal W'\times_3 k_i
=
W_0+S+\mathcal W\times_3 k_i-S(g^\top k_i)
=
W_0+\mathcal W\times_3 k_i.
$$

Thus the shared weights and memory tensor both change while every old decoded weight remains identical.

## 8.3 Common-structure consolidation

Periodically decode residual weights for several contexts and identify a common low-rank component $S$, for example from a joint SVD or robust low-rank mean.

Move $S$ into $W_0$ with exact compensation. Then recompress contextual residuals.

This is the mechanism by which reusable structure migrates from context-specific memory into the shared substrate without forgetting.

Do not implement shared consolidation in the first two-game proof until addressed writes and routing pass independently. It becomes required before the full five-game run.

---

# 9. Context inference without task IDs

Separate the **content query** from the **canonical memory address**.

## 9.1 Query encoder

Use a dedicated recurrent context encoder:

$$
h^c_t
=
F_\psi(
 h^c_{t-1},
 E_c(o_t),
 a_{t-1},
 r_{t-1},
 d_{t-1}
),
$$

$$
q_t
=
\frac{G_\psi(h^c_t)}
{\|G_\psi(h^c_t)\|+\epsilon}.
$$

The context encoder must not consume game IDs or curriculum position.

Train it using self-supervised signals:

- Next latent prediction.
- Reward-sign or reward-bin prediction.
- True terminal prediction.
- Temporal consistency.
- Contrastive separation between distant incompatible histories.
- Prediction-error change detection.

During the initial proof, policy gradients should not directly update the context query encoder. Its representation should change through its self-supervised objective, followed by prototype refresh. This prevents routing collapse while retaining a live, trainable address system.

## 9.2 Prototype index

For each discovered context $i$, keep $M_p$ normalized content prototypes:

$$
P_i=\{p_{i,1},\ldots,p_{i,M_p}\}.
$$

These prototypes are computed from raw anchor histories, not game labels.

The score is

$$
s_i(t)
=
\max_j\frac{q_t^\top p_{i,j}}{\tau}
+
\lambda_h\log(\epsilon+\pi_{t-1,i}).
$$

The posterior is

$$
\pi_t=\operatorname{softmax}(s(t)).
$$

Use an exponential moving average of scores and a minimum dwell time to prevent route chattering.

## 9.3 Canonical address

Each discovered context is assigned one unused orthonormal canonical code $k_i$. The effective address is

$$
z_t=\sum_i\pi_{t,i}k_i.
$$

Use hard top-1 routing for the first implementation after a brief evidence-accumulation period. Soft top-2 routing is a later compositional extension.

The canonical address is an internal memory coordinate. It is not a game ID because:

- It is allocated from experience.
- It is retrieved by content.
- Multiple contexts may arise inside one game.
- Different games may share a context if their dynamics and required computation are sufficiently similar.
- External label permutations have no effect.

## 9.4 Unknown-context detection

Declare a context novel only when all of the following hold for a persistence window:

1. Maximum prototype posterior is below a calibrated confidence threshold.
2. Nearest-prototype cosine distance is above a novelty threshold.
3. Global dynamics-prediction error is elevated relative to its robust running baseline.
4. The condition persists for at least $H_{\text{novel}}$ frames.

Do not allocate a new address from a single surprising frame.

## 9.5 Context switch transaction

When a stable switch is detected:

1. Commit the current scratchpad to the previous locked address.
2. Reset scratch optimizer state.
3. If the destination context is known, retrieve its address.
4. If it is novel, allocate an unused canonical address and initialize prototypes from the recent history window.
5. Lock the destination address after hysteresis.
6. Resume PPO learning.

## 9.6 Ambiguous histories

If two contexts are observationally identical but require different actions, no agent can infer the correct behavior before receiving disambiguating history, reward, goal, or dynamics evidence.

During uncertainty:

- Use the shared/base policy or posterior mixture.
- Do not commit scratch updates.
- Accumulate evidence until confidence is sufficient.

---

# 10. Query drift and memory-coordinate transport

The canonical codebook decouples memory coordinates from query-encoder drift. When the query encoder changes:

1. Re-encode stored raw anchor histories.
2. Recompute content prototypes.
3. Re-evaluate routing accuracy and margins.
4. Accept the query-encoder update only if old contexts remain correctly retrieved.

The synaptic tensor does not need transport when canonical codes remain unchanged.

If canonical addresses themselves are ever remapped, implement exact coordinate transport.

Let old decoded residual weights at protected contexts be columns of

$$
V=[\operatorname{vec}(W(k_1)-W_0),\ldots,
\operatorname{vec}(W(k_n)-W_0)].
$$

For a new address matrix $K'$, find a transported memory operator $M'$ satisfying

$$
M'K'=V.
$$

A minimum-change solution is

$$
M'
=
V{K'}^+
+
M(I-K'{K'}^+).
$$

Then

$$
M'K'=V.
$$

Transport must be transactional and must pass exact decoded-weight tests before commit.

---

# 11. Sparse and efficient execution

A naive implementation that evaluates every memory component at every step is unacceptable.

## 11.1 Preallocated component pools

Use fixed-shape pools for JAX compilation. Each layer owns:

- Input factors.
- Output factors.
- Address factors.
- Addressed bias factors.
- Active mask.
- Component-to-context index metadata.
- A fixed-size gather table per canonical address.

Use FP16 or BF16 for factor storage and FP32 for:

- Address codes.
- Dual computations.
- Orthogonalization.
- Accumulation where numerical drift matters.

## 11.2 Sparse gather

For top-1 routing, gather only components associated with the selected canonical address plus any explicitly shared components.

For a batch with several addresses:

- Group rows by selected address, or
- Use fixed-size indexed gather and vectorized batched multiplication.

Forward cost should scale with the number of active components for the selected address, not with the total number of stored contexts.

## 11.3 Initial capacities

Use these initial values unless memory profiling justifies a change:

- Canonical address dimension: 128.
- Maximum discovered contexts in the Atari prototype: 64.
- Content query dimension: 128.
- Prototypes per context: 8.
- Scratch rank: 8 for convolutional layers, 16 for dense and GRU layers, 8 for policy and value heads.
- Maximum active rank per context per large layer before mandatory recompression: 64.
- Maximum active rank per context per small head: 32.

The storage implementation must expose exact bytes used by:

- Shared parameters.
- Synaptic factors.
- Address book.
- Content prototypes.
- Episodic anchors.
- Scratch state.
- Optimizer state.

## 11.4 Expected scale

A rank-16 contextual residual across all major 512-dimensional layers is orders of magnitude smaller than a full network checkpoint. A preallocated factor pool supporting Atari-57 should fit comfortably within a small fraction of a 24 GB GPU if inactive factors are compactly stored and sparsely gathered.

Do not claim compactness without reporting bytes per learned context and total bytes after every game.

---

# 12. Episodic memory after the redesign

Episodic memory is no longer the primary behavior store.

Keep only enough raw history to support:

- Content prototypes.
- Router calibration.
- Context-encoder prototype refresh.
- Functional sentinels.
- Compression validation.
- Delayed-credit research after the retention mechanism works.

Per discovered context retain approximately:

- 8 content prototypes.
- 16 to 32 short raw anchor sequences.
- Teacher policy and value outputs on those sequences.
- A small set of ambiguous or high-error routing histories.

Enable frame compression. The five-game prototype should target a total episodic budget of 64 to 128 MB, not one gigabyte.

Memory records must contain no game or task field.

---

# 13. Delayed reward and credit assignment

Do not make delayed-credit shaping a dependency of the first retention proof. PPO already demonstrated that it can learn the selected Atari games under adequate plasticity. The immediate blocker is parameter interference.

The system must retain the corrected infrastructure for later use:

- True raw returns.
- Correct recurrent sequences.
- Return predictor.
- Causal contribution estimates.
- Eligibility traces.
- Discount-correct potential shaping.

After the two-game and five-game retention gates pass, integrate delayed credit as follows.

Let a causal return predictor estimate final discounted return from prefix history:

$$
\hat G_t=F(H_t).
$$

Define contribution

$$
c_t=\hat G_{t+1}-\hat G_t.
$$

Use the backward trace

$$
I_t
=
|c_t|+\gamma\lambda_c I_{t+1},
$$

cut at true episode boundaries.

Use $I_t$ for anchor selection and commit importance. Potential shaping is

$$
r'_t
=
r_t+
\eta(\gamma\Phi_{t+1}-\Phi_t),
$$

with terminal potential zero.

Enable shaping only after held-out complete-episode prediction error beats the constant-return baseline for multiple windows.

---

# 14. Persistent state

The continual state must include at least:

## 14.1 Shared policy state

- Shared CNN, dense, GRU, policy, and value parameters.
- Shared optimizer state used only before first protection and during compensated shared consolidation.

## 14.2 Synaptic memory state per contextualized layer

- Input factors.
- Output factors.
- Address factors.
- Bias factors.
- Active masks.
- Component ranks.
- Context-to-component gather indices.
- Free component list.
- Compression statistics.

## 14.3 Scratch state

- Scratch input and output factors.
- Scratch bias.
- Scratch optimizer state.
- Locked address.
- Segment start time.
- Pending applied delta metadata.

## 14.4 Address book

- Canonical address codebook.
- Used-address mask.
- Dual matrix.
- Content prototypes.
- Prototype counts and robust covariance estimates.
- Raw anchor references.
- Current posterior.
- Locked address.
- Hysteresis state.
- Unknown-context detector state.

## 14.5 Audit state

- Decoded-weight fingerprints per address.
- Functional sentinel baselines.
- Routing accuracy history.
- Memory byte history.
- Compression transactions.
- Experiment seed and suite definition.

All state must serialize and restore exactly.

---

# 15. Required algorithms

## 15.1 Route

```text
input: observation, previous action, previous reward, reset, context hidden
q, next_context_hidden = context_encoder(...)
score each stored prototype set
apply posterior smoothing and hysteresis
if known-context confidence is sufficient:
    lock or retain canonical address
else if novelty persists and prediction error is elevated:
    emit NOVEL
else:
    emit UNCERTAIN
return canonical address posterior, lock state, next_context_hidden
```

## 15.2 Allocate address

```text
input: recent stable query window, canonical codebook, used mask
select one unused orthonormal canonical code
build initial content prototypes from the recent history window
append raw routing anchors
update used mask and dual matrix
verify address rank and condition number
return new canonical address
```

## 15.3 Commit scratch to known address

```text
input: scratch factors, address k_i, dual d_i, persistent factor bank
for each contextualized layer:
    append scratch factors with address factor d_i
    append addressed bias delta with d_i
materialize decoded kernels for every protected address before and after
require unchanged kernels for all j != i within tolerance
require current address changed by the intended scratch delta
run functional sentinels
if all checks pass:
    atomically commit and reset scratch
else:
    rollback and report exact failing invariant
```

## 15.4 Commit scratch to novel address

The procedure is identical after allocating $k_{n+1}$. For an orthonormal canonical codebook, $d_{n+1}=k_{n+1}$.

## 15.5 Recompress one address

```text
decode the complete residual operator for one canonical address
for every layer:
    materialize or implicitly compose its addressed low-rank blocks
    compute truncated SVD or structured low-rank refactorization
    select smallest rank satisfying reconstruction tolerance
replace old blocks transactionally
verify decoded weights and functional sentinels
commit only on success
```

## 15.6 Shared consolidation

```text
decode residual operators for all protected addresses
find a common low-rank component S
propose W0 <- W0 + S
compute g satisfying K^T g = 1
propose memory <- memory - S tensor g
verify every protected decoded operator is unchanged
recompress residuals
run all functional and routing audits
commit atomically or rollback
```

---

# 16. Numerical invariants

The implementation must check these continuously.

1. Address normalization:

$$
|\|k_i\|_2-1|<10^{-5}.
$$

2. Address orthogonality for the canonical codebook:

$$
\|K^\top K-I\|_{\max}<10^{-5}.
$$

3. Duality:

$$
\|D^\top K-I\|_{\max}<10^{-5}.
$$

4. Noninterfering write for every protected address $j\ne i$:

$$
\frac{
\|W'_\ell(k_j)-W_\ell(k_j)\|_F
}{
\|W_\ell(k_j)\|_F+\epsilon
}
<\epsilon_{\text{write}}.
$$

Use $\epsilon_{\text{write}}=10^{-6}$ in FP32 unit tests and no worse than $10^{-4}$ in mixed-precision integration.

5. Intended current-context write:

$$
\frac{
\|[W'_\ell(k_i)-W_\ell(k_i)]-\Delta W_\ell\|_F
}{
\|\Delta W_\ell\|_F+\epsilon
}
<\epsilon_{\text{write}}.
$$

6. Address rank must equal the number of used canonical addresses.

7. No NaN or Inf in factors, addresses, posteriors, duals, scratch deltas, or decoded weights.

8. Compression may never be accepted solely on average error. Maximum per-layer, per-context error must pass.

---

# 17. Tests required before Atari training

## 17.1 Synthetic conflicting linear memories

Construct several contexts with identical input distributions but mutually incompatible target matrices.

Sequentially learn matrices $W_1,\ldots,W_N$. After every write, require:

$$
\hat W(k_i)=W_i
$$

for every prior context within numerical tolerance.

Alternate updates among contexts for thousands of steps. Verify reconsolidation changes only the recalled context.

## 17.2 Nonorthogonal dual-address test

Use deliberately nonorthogonal addresses. Verify

$$
D^\top K=I
$$

and exact context-specific writes.

## 17.3 New-address residual test

Allocate a new address from

$$
r=(I-KK^+)u.
$$

Verify orthogonality and noninterference.

## 17.4 Dense, convolutional, GRU, policy, and value tests

For every contextualized layer type:

- Compare factorized forward output against explicitly materialized effective weights.
- Verify old-address invariance after a write.
- Verify intended current-address update.
- Verify gradients flow through scratch factors.

## 17.5 Compression test

Append redundant factors, recompress, and verify all protected decoded weights and functional outputs remain within tolerance.

## 17.6 Shared-consolidation test

Move a common component into shared weights using exact compensation. Verify every protected effective operator is unchanged while the underlying shared and memory parameters both change.

## 17.7 Query-drift test

Update the context query encoder, rebuild prototypes from raw anchors, and verify routing remains correct. If canonical addresses are remapped, test coordinate transport.

## 17.8 Task-identity leakage test

Static and runtime tests must prove that policy, router, memory write, replay, and optimizer APIs have no game/task/label argument.

## 17.9 Sparse execution test

Changing an unselected memory component must not change the output. Instrument execution to prove that only selected components are evaluated.

## 17.10 Serialization test

Save and restore the complete continual state. Require byte-identical addresses, factors, optimizer state, routing state, and deterministic action traces.

---

# 18. Atari suite definition

The research target is a fixed five-game suite sampled once from the canonical Atari-57 list.

Requirements:

1. Use a fixed suite seed of **57057**.
2. Sample five games without replacement from the canonical Atari-57 set.
3. Persist the sampled ordered list before training.
4. Use the same list for all methods, seeds, references, and ablations.
5. Do not replace a difficult game after observing results.
6. The first two sampled games are the two-game diagnostic pair.
7. Final experiments must include multiple orders of the same five games.

The experiment harness may know game names for environment creation and evaluation. The agent and memory system may not.

Use the full 18-action output convention consistently.

---

# 19. Evaluation correctness

The existing partial-return fallback must be removed from all scientific gates.

An evaluation is valid only when the requested number of complete episodes has finished.

For every evaluation report:

- Mean true unclipped return.
- Standard deviation.
- Standard error.
- Number of completed episodes.
- Total environment transitions.
- Whether a cap was reached.

If fewer than the requested episodes complete, return an invalid result. Do not substitute partial in-progress returns.

Pilot evaluation: 10 completed episodes.  
Final evaluation: 30 completed episodes.

Use stochastic policy evaluation as the primary score and greedy evaluation as a secondary diagnostic.

---

# 20. Metrics

For game $i$, let:

- $S_i(t)$: current score.
- $S_i^{\text{random}}$: random-policy score.
- $S_i^{\text{single}}$: matched single-task reference.
- $S_i^{\text{best}}(t)$: best score observed after learning game $i$.

Normalized progress:

$$
P_i(t)
=
\frac{
S_i(t)-S_i^{\text{random}}
}{
S_i^{\text{single}}-S_i^{\text{random}}+\epsilon
}.
$$

Retention:

$$
R_i(t)
=
\frac{
S_i(t)-S_i^{\text{random}}
}{
S_i^{\text{best}}(t)-S_i^{\text{random}}+\epsilon
}.
$$

Forgetting:

$$
F_i(t)
=
\max_{\tau\le t}P_i(\tau)-P_i(t).
$$

Report:

- Per-game normalized progress.
- Per-game retention.
- Worst-game retention.
- Mean retention.
- Harmonic mean of current-game progress and worst old-game retention.
- Retention area under the curriculum curve.
- Forward transfer relative to matched single-task learning curves.
- Router top-1 accuracy and confidence margin.
- Context lock latency after a switch.
- Number of route changes per episode.
- Address count, rank, and condition number.
- Active factor rank per context and layer.
- Synaptic-memory bytes.
- Episodic-memory bytes.
- Throughput.
- Commit, compression, and rollback counts.
- Exact decoded-weight drift.

---

# 21. Acceptance gates

## 21.1 Mathematical memory gate

Before Atari, the synthetic tests must demonstrate exact noninterference for at least 32 conflicting contexts.

## 21.2 Two-game oracle-address diagnostic

This is a diagnostic only, not a valid final task-free result.

- Use the first two sampled games.
- The harness may force the correct canonical address but may not provide the address to the policy through any other path.
- The purpose is to isolate synaptic-memory capacity from router failure.

Pass only if after learning game 2:

$$
P_2\ge0.90
$$

and

$$
R_1\ge0.90.
$$

If this fails, do not tune the router. Fix contextual capacity, scratch optimization, write correctness, or compression.

## 21.3 Two-game inferred-address gate

Remove the oracle. Use only content inference.

Pass only if:

$$
P_2\ge0.90,
\qquad
R_1\ge0.90,
$$

with router top-1 accuracy at least 99% on held-out histories and no external identity.

## 21.4 Five-game blocked curriculum gate

After every game, evaluate all games seen so far.

Final pass requires:

$$
\min_i P_i\ge0.90
$$

or, when a game is intrinsically undertrained at the matched budget,

$$
\min_i R_i\ge0.90
$$

and the current game must still satisfy

$$
P_{\text{current}}\ge0.90.
$$

Report both criteria; do not hide a failure behind a mean.

## 21.5 Unannounced-switch gate

After the blocked curriculum passes, train and evaluate with unannounced switches among the five games. The agent receives only experience, not a boundary signal beyond ordinary environment termination/reset.

Measure route-lock latency and transient loss after each switch.

## 21.6 Replication

Final claims require at least three seeds and at least two curriculum orders. The two GPUs should run independent paired experiments.

---

# 22. Experiment ladder and GPU strategy

Do not begin with a full five-game, five-million-step run.

## Stage A: CPU and short GPU correctness

- Complete every mathematical and layer-level test.
- Run synthetic switching environments.
- Verify sparse execution and state serialization.

No long Atari run is allowed until this stage passes.

## Stage B: matched single-task references

Train one fresh recurrent PPO agent per sampled game with the same backbone and environment budget. Cache all references.

Use both GPUs independently to generate references in parallel.

## Stage C: two-game oracle-address pilot

GPU 0: CASTM with oracle address.  
GPU 1: matched plain recurrent PPO or a second CASTM seed.

Use a pilot budget of 500,000 environment steps per game, evaluate every 100,000 steps, and escalate to 2 million only if the learning curve is positive.

Escalate to the final 5-million-step budget only when the mechanism is functioning.

## Stage D: two-game inferred address

GPU 0 and GPU 1 run independent seeds. Stop immediately if routing accuracy or decoded-weight invariants fail.

## Stage E: five-game pilot

Use 1 million environment steps per game, evaluating all prior games every 250,000 steps.

The purpose is to expose rank, routing, and compression failures before a final run.

## Stage F: five-game final

Use the matched final budget, 30 completed episodes per evaluation, three seeds, and multiple orders.

## Parallelism rule

Prefer one independent process per GPU. Do not spend engineering time on multi-GPU data parallelism until the algorithm passes. Independent paired experiments provide faster scientific feedback.

---

# 23. Early termination rules

Terminate a run immediately when any of these occurs:

1. A protected decoded weight changes beyond tolerance after commit.
2. Address rank is lost or the dual becomes ill-conditioned.
3. Any NaN or Inf appears.
4. Router accuracy on old held-out histories falls below 99% for two consecutive audits.
5. A supposedly unselected component affects outputs.
6. The requested evaluation episodes do not complete and the run attempts to use a partial score.
7. Current-game normalized progress stays below 0.20 for three pilot evaluations while the matched single-task curve is improving.
8. Scratch gradient norm is nonzero but applied scratch delta remains effectively zero.
9. Memory rank or component capacity is exhausted without a successful compression transaction.
10. Throughput falls below an established floor because all memory components are being evaluated instead of sparsely gathered.

Do not monitor a structurally failed run for hours.

---

# 24. Required ablations

Run only after the core system works.

1. Plain recurrent PPO.
2. Existing null-space protection.
3. CASTM with oracle address.
4. CASTM with inferred address.
5. CASTM without shared consolidation.
6. CASTM without episodic behavior replay.
7. CASTM with late layers only.
8. CASTM with all affine layers contextualized.
9. Hard routing versus soft top-2 routing.
10. Different scratch ranks.
11. Different compression tolerances.

The most important comparison is oracle versus inferred address. It cleanly separates memory failure from context-inference failure.

---

# 25. Failure diagnosis matrix

## Old decoded weights drift, oracle address correct

Cause: write algebra, optimizer-delta extraction, factor append, bias handling, or compression is incorrect.

Action: stop all RL experiments and repair the invariant.

## Old decoded weights are exact, old score falls, oracle address correct

Cause: recurrent hidden-state reset, stochastic evaluation, environment semantics, uncontextualized mutable pathway, or functional dependence outside contextualized layers.

Action: contextualize the missing pathway or verify state handling.

## Oracle passes but inferred routing fails

Cause: content query, prototype index, hysteresis, unknown detection, or query drift.

Action: work only on context inference. Do not alter synaptic memory.

## Old retention passes but new game does not learn under oracle

Cause: scratch capacity, contextualized layer coverage, optimizer, or backbone representation.

Action: increase scratch rank or contextualize deeper layers. Do not loosen retention because old contexts are already algebraically isolated.

## Memory grows too quickly

Cause: commit frequency, redundant scratch writes, or failed recompression.

Action: commit less frequently, merge repeated writes to the same address, and improve transactional low-rank compression.

## Contexts proliferate inside one game

Cause: novelty detector too sensitive, query instability, or insufficient hysteresis.

Action: tighten persistent novelty conditions and update existing prototypes before allocating a new address.

---

# 26. Prohibited shortcuts

The following invalidate the experiment:

- Supplying a game or task ID.
- Selecting a checkpoint by game name.
- One full policy per game.
- A separate game-specific action head.
- Reading an episodic action distribution at inference.
- Evaluating with a per-game adapter mask from metadata.
- Resampling the five-game suite after seeing performance.
- Using incomplete-episode returns in gates.
- Reporting mean retention while a worst game is forgotten.
- Calling an oracle-address run task-free.
- Claiming success when the old game is retained but the new game fails to learn.
- Claiming success when the new game learns but any old game falls below the retention gate.

---

# 27. Implementation order

Implement in this order. Do not skip ahead.

1. Fix evaluation so only completed episodes are valid.
2. Add a standalone addressed linear-memory module with exact dual writes.
3. Add factorized dense memory and scratch commit.
4. Add addressed bias.
5. Add convolutional factor memory.
6. Add GRU-gate factor memory.
7. Integrate policy and value heads.
8. Add exact decoded-weight audit APIs.
9. Add canonical address book and orthonormal codebook.
10. Add content query encoder and prototype router.
11. Add context switch transaction.
12. Add sparse gather execution.
13. Add per-address recompression.
14. Run synthetic conflicting-context tests.
15. Run two-game oracle-address Atari.
16. Run two-game inferred-address Atari.
17. Add exact compensated shared consolidation.
18. Run five-game pilot.
19. Run unannounced switching.
20. Only then reintroduce delayed-credit shaping.

---

# 28. Deliverables

The completed implementation must produce:

1. A self-contained mathematical test report.
2. Synthetic continual-learning results with at least 32 conflicting contexts.
3. The fixed random five-game suite definition.
4. Matched single-task references.
5. Two-game oracle-address results.
6. Two-game inferred-address results.
7. Five-game retention matrices after every game.
8. Routing confusion matrices.
9. Exact decoded-weight drift reports.
10. Per-layer memory rank and byte reports.
11. Throughput measurements.
12. Compression and rollback logs.
13. Plain PPO and current TFNS baselines.
14. A clear account of every failed experiment and why it was stopped.

---

# 29. Final research claim allowed by this design

The implementation may claim exact synaptic retention only in this precise form:

> Given a correctly retrieved protected canonical address and no accepted lossy compression beyond the stated tolerance, a dual-address write for another context leaves the decoded effective layer weights of the protected context unchanged up to floating-point error.

It may claim task-free continual learning only after content-based routing, unannounced switching, current-game learning, and worst-old-game retention all pass together.

---

# 30. Final build contract

Build one recurrent Atari agent whose effective weights are retrieved from a compact, content-addressed synaptic memory.

Use an internally learned history representation to retrieve a canonical memory address. Train new behavior in a temporary low-rank scratchpad. Commit the actual applied optimizer delta to the address dual so that the full new update is written while every unrelated protected address decodes exactly the same weights as before. Keep all memories revisable on recall. Allow shared weights and all memory factors to change through exact compensated consolidation and transactional recompression. Use sparse execution so cost depends on the recalled memory, not total lifetime memory. Use small raw episodic anchors only for routing, auditing, and compression—not action selection.

The system is not complete when it merely preserves the past. It is complete only when it simultaneously:

- Learns the current game near its matched single-task reference.
- Retains every prior game above the required threshold.
- Infers the correct memory from experience without a supplied identity.
- Preserves exact decoded-weight invariants.
- Uses compact bounded storage.
- Remains computationally practical as contexts accumulate.

That is the required path out of the current stability-plasticity local minimum.
