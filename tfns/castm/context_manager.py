"""Online, task-free context manager (architecture sections 1-2).

Discovers and recalls internal contexts from content alone — observation/action/
reward history surfaced as a normalized content query ``q`` from the shared
(W0-only, context-independent) encoder. **No game id, task id, curriculum index,
announced boundary, or evaluator label ever enters this module.** Internal context
ids are integers allocated online from a canonical address book; they are created
by this manager from experience, never derived from game names or curriculum
position.

The manager operates at *rollout granularity* (one decision per PPO rollout):
the trainer hands it the current batch of content queries and the raw frames that
produced them, and the manager returns the active internal context and its
canonical address for the next rollout, plus events (a context was discovered, a
revisit switch happened, a prototype refresh is due). Per-stream routing is
supported (each environment row routes independently); in the single-game-per-
envpool harness all rows share the regime, so a batch consensus drives the
discrete allocate/switch decision while per-stream nearest-match is exposed for
generality and tests.

Key robustness mechanism (section 2): the shared encoder stays plastic, so old
query embeddings drift. The manager keeps a bounded, label-free **raw-frame**
anchor buffer per context (train + held-out split) and rebuilds every context's
prototypes by *re-encoding its raw anchors under the current encoder* whenever the
trainer reports encoder drift (after each memory resolve). Prototypes therefore
track the live encoder without any labels.

Decision logic (adaptive, self-calibrating form of section 9.4/15.1):

* Raw shared-encoder features are mean-CENTERED before cosine (ReLU features live in
  the positive orthant, so uncentered cosine is ~1 for every regime — useless;
  centering removes the common component and exposes regime-specific structure).
* Each context keeps a running similarity baseline (mean, deviation) calibrated from
  its own real frames (warm-up grace + k-means centroid prototypes refreshed online).
* **KNOWN(c)**: the consensus similarity to the active context ``c`` is within a
  dev-scaled, capped band of ``c``'s own running mean.
* **revisit SWITCH**: a *different* known context matches at its own band and beats
  the active one for ``min_dwell`` rollouts.
* **NOVEL -> allocate**: no context matches its band for ``novel_persist`` rollouts;
  a new canonical address is allocated and seeded from the novel-evidence window.
* **UNCERTAIN**: otherwise — keep the current active context (no thrash).

A single anomalous rollout never allocates or switches: every transition requires
sustained persistence, so transient drift cannot split a context (test 9). The match
band is CAPPED so a clear regime change always breaks the match (no false merge),
and the baseline tracks the real within-regime similarity (no one-way ratchet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from tfns.castm import address as addr
from tfns.castm import router as rt


@dataclass
class ManagerConfig:
    d_q: int = 512
    d_k: int = 128
    n_max: int = 64
    max_contexts: int = 64
    proto_per_ctx: int = 12         # k-means centroids per context (coverage of regime diversity)
    anchor_cap: int = 2048          # raw frames stored per context
    heldout_frac: float = 0.25      # fraction of anchors reserved for held-out routing audit
    pending_cap: int = 4096         # recent frames/queries kept for novel-context seeding
    # Content-similarity thresholds (cosine in [-1,1]); calibrated per-encoder.
    # Adaptive, self-calibrating routing (architecture §9.4.3): a context is
    # "matched" when the consensus similarity is within `known_k` running deviations
    # of that context's own recent level; novelty fires when NO context matches and
    # the active context's match has dropped >= `novel_k` deviations below its level.
    # This tracks a drifting encoder without fixed thresholds (which a plastic encoder
    # makes fragile — cross-game sim on an untrained encoder is high).
    known_k: float = 4.0
    novel_k: float = 6.0
    known_cap: float = 0.12         # max similarity drop below a context's mean that still
                                    # counts as a match — caps the dev-scaled band so a clear
                                    # regime change (cross-game sim drop) always breaks the match
    known_floor: float = 0.10       # absolute floor so a context always matches itself early
                                    # (kept low so it never dominates and causes a false merge)
    sim_stat_ema: float = 0.9       # EMA on a context's running similarity mean / deviation
    switch_margin: float = 0.03     # a different ctx must beat the active ctx by this to switch
    novel_persist: int = 4          # consecutive novel rollouts before allocation
    min_dwell: int = 3              # min rollouts in a context before a switch/allocation
    warmup_rollouts: int = 6        # grace period after entering a context: force KNOWN so its
                                    # similarity baseline calibrates on real (not seeding) frames
    merge_thresh: float = 0.92      # prototype cosine above which a merge is *considered* (audited)
    novel_window_cap: int = 8       # rollouts of novel evidence kept for seeding a new context
    center: bool = True             # mean-center raw features before cosine (ReLU features live in
                                    # the positive orthant; uncentered cosine is ~1 for all regimes)
    center_ema: float = 0.99        # EMA factor for the running global feature mean
    # legacy fixed thresholds (retained for synthetic tests / fallback only)
    known_thresh: float = 0.55
    novel_thresh: float = 0.40
    sim_ema: float = 0.7
    seed: int = 0


@dataclass
class _AnchorBuffer:
    """Bounded raw-frame reservoir for one context (train + held-out split)."""

    cap: int
    heldout_frac: float
    rng: np.random.Generator
    train: list = field(default_factory=list)      # list[np.ndarray uint8 (H,W,C)]
    heldout: list = field(default_factory=list)
    _seen: int = 0

    def add_batch(self, frames: np.ndarray) -> None:
        # frames: (B, H, W, C) uint8. Reservoir-sample to keep a representative set.
        n_held = max(1, int(self.cap * self.heldout_frac))
        n_train = self.cap - n_held
        for f in frames:
            self._seen += 1
            # Route ~heldout_frac of samples into the held-out pool.
            if self.rng.random() < self.heldout_frac:
                self._reservoir(self.heldout, f, n_held)
            else:
                self._reservoir(self.train, f, n_train)

    def _reservoir(self, pool: list, item: np.ndarray, cap: int) -> None:
        if len(pool) < cap:
            pool.append(np.asarray(item, dtype=np.uint8))
        else:
            j = int(self.rng.integers(0, self._seen))
            if j < cap:
                pool[j] = np.asarray(item, dtype=np.uint8)

    def train_frames(self) -> np.ndarray:
        return np.asarray(self.train, dtype=np.uint8) if self.train else np.zeros((0,), np.uint8)

    def heldout_frames(self) -> np.ndarray:
        return np.asarray(self.heldout, dtype=np.uint8) if self.heldout else np.zeros((0,), np.uint8)


def _normalize_rows(q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    return q / (np.linalg.norm(q, axis=-1, keepdims=True) + eps)


def pooled_signature(obs: np.ndarray, grid: int = 8) -> np.ndarray:
    """Content signature from the RAW observation: spatial average-pool to ``grid×grid``
    per stacked frame, flattened (B, grid*grid*C).

    Why raw observations and not the shared encoder's features: the W0 encoder's
    penultimate features are empirically NON-discriminative across visually-distinct
    Atari games (cosine ~0.95+ even after mean-centering — both games map to the same
    control-feature region), so they cannot drive routing. The raw-pixel spatial layout
    separates regimes robustly, is shared-encoder-independent (so it does NOT drift as
    W0 trains — the architecture §2 drift machinery still runs but is trivially stable),
    and is domain-general (any agent has sensory input). This is the spec-allowed
    "observations" form of the content query; no game/task identity is used.
    """

    o = np.asarray(obs, dtype=np.float32)
    if o.ndim == 3:
        o = o[None]
    B, H, W, C = o.shape
    g = int(grid)
    s = (H // g) * g
    o = o[:, :s, :s, :]
    o = o.reshape(B, g, s // g, g, s // g, C).mean(axis=(2, 4))   # (B, g, g, C)
    return (o.reshape(B, g * g * C) / 255.0).astype(np.float32)


def farthest_point_prototypes(queries: np.ndarray, m_p: int) -> np.ndarray:
    """Greedy farthest-point selection (used only to seed k-means centroids)."""

    q = _normalize_rows(queries)
    n = q.shape[0]
    if n == 0:
        return np.zeros((0, q.shape[-1] if q.ndim == 2 else 0), np.float32)
    chosen = [0]
    while len(chosen) < min(int(m_p), n):
        sel = q[chosen]
        nearest = (q @ sel.T).max(axis=1)
        nearest[chosen] = 2.0
        chosen.append(int(np.argmin(nearest)))
    return q[chosen]


def kmeans_prototypes(queries: np.ndarray, m_p: int, iters: int = 8) -> np.ndarray:
    """Spherical k-means centroids of normalized queries (representative prototypes).

    Centroids (not farthest-point extremes) give HIGH within-context similarity —
    a typical frame is close to its nearest centroid — which is what routing needs
    to separate regimes. Farthest-point seeds the centroids for spread; assignment
    is by max cosine; each centroid is the renormalized mean of its members. Empty
    clusters are reseeded to the worst-covered point.
    """

    q = _normalize_rows(queries)
    n = q.shape[0]
    k = int(min(m_p, n))
    if n == 0:
        return np.zeros((0, q.shape[-1] if q.ndim == 2 else 0), np.float32)
    if k <= 1:
        c = q.mean(axis=0, keepdims=True)
        return _normalize_rows(c)
    cent = farthest_point_prototypes(q, k).copy()  # (k,d) seed
    for _ in range(int(iters)):
        sims = q @ cent.T                      # (n,k)
        assign = sims.argmax(axis=1)           # (n,)
        new = np.zeros_like(cent)
        for j in range(k):
            members = q[assign == j]
            if members.shape[0] == 0:
                # reseed empty cluster to the least-covered query
                worst = int(sims.max(axis=1).argmin())
                new[j] = q[worst]
            else:
                new[j] = members.mean(axis=0)
        new = _normalize_rows(new)
        if np.allclose(new, cent, atol=1e-5):
            cent = new
            break
        cent = new
    return cent.astype(np.float32)


class OnlineContextManager:
    """Stateful online discovery/recall of internal contexts from content."""

    def __init__(self, cfg: ManagerConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.book = addr.empty_address_book(d_k=cfg.d_k, n_max=cfg.n_max, seed=cfg.seed)
        self.index = rt.empty_prototype_index(cfg.max_contexts, cfg.proto_per_ctx, cfg.d_q)
        self.ctx_ids: list[int] = []                 # discovery order
        self.prototypes: dict[int, np.ndarray] = {}  # ctx -> (M_p, d_q) numpy
        self.anchors: dict[int, _AnchorBuffer] = {}
        self.active_ctx: int = -1
        self.dwell: int = 0
        self.novelty_streak: int = 0
        self._switch_cand: int = -1
        self._switch_streak: int = 0
        self.sim_ema: dict[int, float] = {}
        # Novel-evidence window: frames/queries from rollouts that look novel relative
        # to every known context. Reset whenever we are confidently in a known context,
        # so a new context is seeded ONLY from its own (uncontaminated) evidence.
        self.novel_q: list[np.ndarray] = []
        self.novel_frames: list[np.ndarray] = []
        # Recent in-hand queries per context (for cheap drift-tracking prototype
        # refresh on confident rollouts — no GPU re-encode). Keyed by ctx id.
        self.recent_q: dict[int, list[np.ndarray]] = {}
        self.recent_q_cap: int = 6                   # rollouts of recent queries kept
        # per-context running consensus-similarity baseline (adaptive routing)
        self.ctx_sim_mean: dict[int, float] = {}
        self.ctx_sim_dev: dict[int, float] = {}
        self.events: list[dict] = []                 # audit trail
        self.rollout: int = 0
        self.feat_mean: np.ndarray | None = None     # running global feature mean (for centering)

    def _center(self, raw: np.ndarray, update_mean: bool = False) -> np.ndarray:
        """Mean-center raw features (remove the common positive-orthant component),
        then L2-normalize. ``update_mean`` advances the running global mean (only the
        main stream does this; eval/audit use the current mean)."""

        raw = np.asarray(raw, dtype=np.float32)
        if not self.cfg.center:
            return _normalize_rows(raw)
        if self.feat_mean is None:
            self.feat_mean = raw.mean(axis=0)
        if update_mean:
            a = self.cfg.center_ema
            self.feat_mean = a * self.feat_mean + (1.0 - a) * raw.mean(axis=0)
        return _normalize_rows(raw - self.feat_mean[None, :])

    # --- introspection ---------------------------------------------------------
    @property
    def num_contexts(self) -> int:
        return len(self.ctx_ids)

    def address_of(self, ctx: int) -> np.ndarray:
        return np.asarray(addr.code(self.book, int(ctx)))

    def active_address(self) -> np.ndarray:
        if self.active_ctx < 0:
            # Neutral base route before any context exists: an unused column,
            # which decodes to pure W0 (no committed component selects it).
            return np.asarray(addr.code(self.book, 0))
        return self.address_of(self.active_ctx)

    # --- novel-evidence window -------------------------------------------------
    def _push_novel(self, queries: np.ndarray, frames: np.ndarray) -> None:
        self.novel_q.append(_normalize_rows(queries))
        self.novel_frames.append(np.asarray(frames, dtype=np.uint8))
        while len(self.novel_q) > self.cfg.novel_window_cap:
            self.novel_q.pop(0)
            self.novel_frames.pop(0)

    def _reset_novel(self) -> None:
        self.novel_q.clear()
        self.novel_frames.clear()

    def _novel_query_window(self) -> np.ndarray:
        if not self.novel_q:
            return np.zeros((0, self.cfg.d_q), np.float32)
        return np.concatenate(self.novel_q, axis=0)

    def _novel_frame_window(self) -> np.ndarray:
        if not self.novel_frames:
            return np.zeros((0,), np.uint8)
        return np.concatenate(self.novel_frames, axis=0)

    # --- similarity scoring ----------------------------------------------------
    def _consensus_sims(self, queries: np.ndarray) -> dict[int, float]:
        """Mean over the batch of each context's best-prototype cosine similarity."""

        q = _normalize_rows(queries)
        sims: dict[int, float] = {}
        for c in self.ctx_ids:
            protos = self.prototypes[c]
            if protos.shape[0] == 0:
                sims[c] = -1.0
            else:
                sims[c] = float((q @ protos.T).max(axis=1).mean())
        return sims

    def per_stream_route(self, queries: np.ndarray, raw: bool = True) -> np.ndarray:
        """Per-row nearest-context id (B,), -1 if no contexts. Pure content match.

        ``raw=True`` (default) mean-centers the input with the current global mean
        (the form callers/eval supply); pass ``raw=False`` for already-centered queries.
        """

        q = self._center(queries) if raw else _normalize_rows(queries)
        if not self.ctx_ids:
            return np.full((q.shape[0],), -1, np.int32)
        best_sim = np.full((q.shape[0],), -2.0, np.float32)
        best_ctx = np.full((q.shape[0],), self.ctx_ids[0], np.int32)
        for c in self.ctx_ids:
            protos = self.prototypes[c]
            if protos.shape[0] == 0:
                continue
            sim = (q @ protos.T).max(axis=1)
            upd = sim > best_sim
            best_sim = np.where(upd, sim, best_sim)
            best_ctx = np.where(upd, c, best_ctx)
        return best_ctx.astype(np.int32)

    # --- allocation ------------------------------------------------------------
    def _allocate(self, query_window: np.ndarray, frame_window: np.ndarray) -> int:
        self.book, ctx = addr.allocate_canonical(self.book)
        self.prototypes[ctx] = kmeans_prototypes(query_window, self.cfg.proto_per_ctx)
        self.ctx_ids.append(ctx)
        buf = _AnchorBuffer(self.cfg.anchor_cap, self.cfg.heldout_frac,
                            np.random.default_rng(self.cfg.seed + 100 + ctx))
        if frame_window.size:
            buf.add_batch(frame_window)
        self.anchors[ctx] = buf
        self.sim_ema[ctx] = float(self.cfg.known_thresh)
        self.recent_q[ctx] = [query_window]  # seed drift-tracking window from the evidence
        # Do NOT seed the running similarity baseline from the seeding within-sim:
        # k-means centroids trivially fit their own seeding frames (~1.0), but NEW
        # frames of the same regime score far lower under a barely-trained encoder.
        # The warm-up calibrates the baseline from real subsequent frames instead.
        return ctx

    def _known_level(self, c: int) -> float:
        """Adaptive similarity above which context ``c`` is considered matched.

        The drop below the running mean is dev-scaled but **capped** at ``known_cap``
        so a clear regime change (a cross-game similarity drop) always breaks the
        match even when the within-context variance is large.
        """
        m = self.ctx_sim_mean.get(c, self.cfg.known_floor)
        d = self.ctx_sim_dev.get(c, 0.05)
        drop = min(self.cfg.known_k * d, self.cfg.known_cap)
        return max(self.cfg.known_floor, m - drop)

    def _update_sim_stats(self, c: int, sim: float) -> None:
        a = self.cfg.sim_stat_ema
        m = self.ctx_sim_mean.get(c, sim)
        self.ctx_sim_mean[c] = a * m + (1.0 - a) * sim
        self.ctx_sim_dev[c] = a * self.ctx_sim_dev.get(c, 0.05) + (1.0 - a) * abs(sim - m)

    def _cheap_refresh_active(self, queries: np.ndarray) -> None:
        """Track encoder drift for the active context from in-hand confident queries.

        Rebuilds the active context's prototypes from a short window of its own recent
        queries (no GPU re-encode). Called only on KNOWN rollouts so foreign frames
        never contaminate a context's prototypes. This keeps within-context similarity
        high under a drifting encoder, preventing false splits (architecture §2).
        """

        c = self.active_ctx
        if c < 0:
            return
        ring = self.recent_q.setdefault(c, [])
        ring.append(_normalize_rows(queries))
        while len(ring) > self.recent_q_cap:
            ring.pop(0)
        window = np.concatenate(ring, axis=0)
        if window.shape[0] >= self.cfg.proto_per_ctx:
            self.prototypes[c] = kmeans_prototypes(window, self.cfg.proto_per_ctx)

    # --- main per-rollout update ----------------------------------------------
    def update(self, queries: np.ndarray, frames: np.ndarray) -> dict:
        """Route the current batch and decide allocate / switch / stay.

        ``queries``: (B, d_q) content queries from the shared encoder (this rollout).
        ``frames``:  (B, H, W, C) uint8 raw observations that produced them.
        Returns an event dict with ``active_ctx``, ``address`` (np d_k), ``decision``,
        ``allocated`` (bool), ``switched`` (bool), and diagnostics.
        """

        self.rollout += 1
        queries = self._center(queries, update_mean=True)

        # Bootstrap: discover the first context from its own initial evidence.
        if not self.ctx_ids:
            self._push_novel(queries, frames)
            window = self._novel_query_window()
            if window.shape[0] >= self.cfg.proto_per_ctx:
                ctx = self._allocate(window, self._novel_frame_window())
                self.active_ctx = ctx
                self.dwell = 0
                self.novelty_streak = 0
                self._reset_novel()
                ev = self._event("ALLOC", allocated=True, switched=False)
                self.events.append(ev)
                return ev
            return self._event("BOOTSTRAP", allocated=False, switched=False)

        sims = self._consensus_sims(queries)
        best_ctx = max(sims, key=lambda c: sims[c])
        best_sim = sims[best_ctx]
        active_sim = sims.get(self.active_ctx, -1.0)
        self.dwell += 1
        # instrumentation (logged by the trainer)
        self.last_active_sim = float(active_sim)
        self.last_best_sim = float(best_sim)
        self.last_active_level = float(self._known_level(self.active_ctx)) if self.active_ctx >= 0 else 0.0
        self.last_active_mean = float(self.ctx_sim_mean.get(self.active_ctx, 0.0))
        self.last_active_dev = float(self.ctx_sim_dev.get(self.active_ctx, 0.0))

        # Adaptive matching: which contexts match at their own (capped) running level?
        matched = [c for c in self.ctx_ids if sims[c] >= self._known_level(c)]
        in_active = self.active_ctx in matched
        # Warm-up grace: a freshly-entered context is trusted for a few rollouts so its
        # similarity baseline calibrates on real gameplay frames (the seeding baseline is
        # over-tight — prototypes scored on their own frames). Prevents instant false split.
        warming = (self.active_ctx >= 0) and (self.dwell <= self.cfg.warmup_rollouts)
        # best matched context other than the active one
        other = [c for c in matched if c != self.active_ctx]
        best_other = max(other, key=lambda c: sims[c]) if other else -1

        if in_active or warming:
            # Still confidently in the active context.
            self.novelty_streak = 0
            self._switch_cand = -1; self._switch_streak = 0
            self._reset_novel()
            # Fold frames into the context's baseline/prototypes only when they
            # genuinely match it: within `known_cap` of the running mean. This lets
            # the mean track real within-game similarity (NOT a one-way ratchet) while
            # rejecting clearly-foreign frames — including a regime change that lands
            # inside the warm-up window (review finding: warm-up pollution).
            ref_mean = self.ctx_sim_mean.get(self.active_ctx, self.cfg.known_floor)
            if warming or active_sim >= ref_mean - self.cfg.known_cap:
                self._update_sim_stats(self.active_ctx, active_sim)
                if self.active_ctx in self.anchors:
                    self.anchors[self.active_ctx].add_batch(frames)
                    self._cheap_refresh_active(queries)
            return self._event("KNOWN", allocated=False, switched=False,
                               best_sim=best_sim, best_ctx=self.active_ctx)

        # The active context no longer matches: revisit-switch or novel.
        if best_other >= 0 and sims[best_other] >= active_sim + self.cfg.switch_margin:
            # A different KNOWN context matches better -> candidate revisit.
            self.novelty_streak = 0
            self._reset_novel()
            if self._switch_cand == best_other:
                self._switch_streak += 1
            else:
                self._switch_cand = best_other; self._switch_streak = 1
            if self._switch_streak >= self.cfg.min_dwell:
                prev = self.active_ctx
                self.active_ctx = best_other
                self.dwell = 0
                self._switch_cand = -1; self._switch_streak = 0
                if best_other in self.anchors:
                    self.anchors[best_other].add_batch(frames)
                self.recent_q[best_other] = [queries]
                ev = self._event("SWITCH", allocated=False, switched=True,
                                 best_sim=sims[best_other], prev_ctx=prev, best_ctx=best_other)
                self.events.append(ev)
                return ev
            return self._event("UNCERTAIN", allocated=False, switched=False,
                               best_sim=best_sim, best_ctx=best_other)

        # No context matches well -> accumulate novel evidence.
        self._switch_cand = -1; self._switch_streak = 0
        self.novelty_streak += 1
        self._push_novel(queries, frames)
        if self.novelty_streak >= self.cfg.novel_persist and self.dwell >= self.cfg.min_dwell:
            prev = int(self.active_ctx)            # the outgoing context (for resolve attribution)
            ctx = self._allocate(self._novel_query_window(), self._novel_frame_window())
            self.active_ctx = ctx
            self.dwell = 0
            self.novelty_streak = 0
            self._reset_novel()
            ev = self._event("ALLOC", allocated=True, switched=False,
                             best_sim=best_sim, best_ctx=ctx, prev_ctx=prev)
            self.events.append(ev)
            return ev
        return self._event("UNCERTAIN", allocated=False, switched=False,
                           best_sim=best_sim, best_ctx=best_ctx)

    def _event(self, decision: str, **kw) -> dict:
        ev = {"rollout": self.rollout, "decision": decision, "active_ctx": int(self.active_ctx),
              "address": self.active_address(), "num_contexts": self.num_contexts,
              "dwell": self.dwell, "novelty_streak": self.novelty_streak}
        ev.update(kw)
        return ev

    # --- prototype refresh under encoder drift (section 2) ---------------------
    def refresh_prototypes(self, query_fn: Callable[[np.ndarray], np.ndarray]) -> dict:
        """Rebuild every context's prototypes by re-encoding its raw train anchors.

        ``query_fn`` maps raw frames (N,H,W,C uint8) -> normalized queries (N,d_q)
        under the *current* shared encoder. Returns an audit dict with held-out
        routing accuracy and the min cross-context prototype margin so the caller
        can detect a false merge (distinct contexts collapsing).
        """

        for c in self.ctx_ids:
            buf = self.anchors.get(c)
            if buf is None:
                continue
            frames = buf.train_frames()
            if frames.shape[0] >= self.cfg.proto_per_ctx:
                q = self._center(np.asarray(query_fn(frames)))   # re-encoded raw -> centered
                self.prototypes[c] = kmeans_prototypes(q, self.cfg.proto_per_ctx)
        audit = self.audit_routing(query_fn)
        audit["event"] = "REFRESH"
        self.events.append({"rollout": self.rollout, "decision": "REFRESH", **{k: v for k, v in audit.items() if k != "per_ctx"}})
        return audit

    def audit_routing(self, query_fn: Callable[[np.ndarray], np.ndarray]) -> dict:
        """Held-out routing accuracy + min inter-context prototype similarity.

        Re-encodes each context's *held-out* anchors and checks they route back to
        their own context. A high inter-context prototype similarity flags a
        potential false merge (audited before any merge is performed).
        """

        per_ctx = {}
        total_correct = 0
        total = 0
        for c in self.ctx_ids:
            buf = self.anchors.get(c)
            if buf is None:
                continue
            frames = buf.heldout_frames()
            if frames.shape[0] == 0:
                continue
            q = np.asarray(query_fn(frames))
            routed = self.per_stream_route(q)
            correct = int(np.sum(routed == c))
            per_ctx[c] = correct / max(len(routed), 1)
            total_correct += correct
            total += len(routed)
        # inter-context max prototype similarity (false-merge sentinel)
        max_inter = -1.0
        ids = list(self.ctx_ids)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                pi, pj = self.prototypes[ids[i]], self.prototypes[ids[j]]
                if pi.shape[0] and pj.shape[0]:
                    max_inter = max(max_inter, float((pi @ pj.T).max()))
        return {"overall": total_correct / max(total, 1), "per_ctx": per_ctx,
                "max_inter_sim": max_inter, "num_contexts": self.num_contexts}

    # --- serialization helpers -------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "ctx_ids": list(self.ctx_ids),
            "active_ctx": int(self.active_ctx),
            "dwell": int(self.dwell),
            "novelty_streak": int(self.novelty_streak),
            "rollout": int(self.rollout),
            "prototypes": {int(c): np.asarray(p) for c, p in self.prototypes.items()},
            "book_used": np.asarray(self.book.used),
            "book_K": np.asarray(self.book.K),
            "anchors": {int(c): {"train": buf.train_frames(), "heldout": buf.heldout_frames()}
                        for c, buf in self.anchors.items()},
            "sim_ema": {int(c): float(v) for c, v in self.sim_ema.items()},
            "ctx_sim_mean": {int(c): float(v) for c, v in self.ctx_sim_mean.items()},
            "ctx_sim_dev": {int(c): float(v) for c, v in self.ctx_sim_dev.items()},
            # in-progress transition state + drift/evidence windows (exact resume)
            "switch_cand": int(self._switch_cand),
            "switch_streak": int(self._switch_streak),
            "recent_q": {int(c): [np.asarray(a) for a in ring] for c, ring in self.recent_q.items()},
            "novel_q": [np.asarray(a) for a in self.novel_q],
            "novel_frames": [np.asarray(a) for a in self.novel_frames],
            "feat_mean": None if self.feat_mean is None else np.asarray(self.feat_mean),
        }

    def load_state_dict(self, sd: dict) -> None:
        import jax.numpy as jnp
        self.ctx_ids = list(sd["ctx_ids"])
        self.active_ctx = int(sd["active_ctx"])
        self.dwell = int(sd["dwell"])
        self.novelty_streak = int(sd["novelty_streak"])
        self.rollout = int(sd["rollout"])
        self.prototypes = {int(c): np.asarray(p, np.float32) for c, p in sd["prototypes"].items()}
        self.book = self.book.replace(K=jnp.asarray(sd["book_K"]), used=jnp.asarray(sd["book_used"]))
        self.sim_ema = {int(c): float(v) for c, v in sd.get("sim_ema", {}).items()}
        self.ctx_sim_mean = {int(c): float(v) for c, v in sd.get("ctx_sim_mean", {}).items()}
        self.ctx_sim_dev = {int(c): float(v) for c, v in sd.get("ctx_sim_dev", {}).items()}
        self._switch_cand = int(sd.get("switch_cand", -1))
        self._switch_streak = int(sd.get("switch_streak", 0))
        self.recent_q = {int(c): [np.asarray(a, np.float32) for a in ring]
                         for c, ring in sd.get("recent_q", {}).items()}
        self.novel_q = [np.asarray(a, np.float32) for a in sd.get("novel_q", [])]
        self.novel_frames = [np.asarray(a, np.uint8) for a in sd.get("novel_frames", [])]
        fm = sd.get("feat_mean", None)
        self.feat_mean = None if fm is None else np.asarray(fm, np.float32)
        self.anchors = {}
        for c, d in sd["anchors"].items():
            buf = _AnchorBuffer(self.cfg.anchor_cap, self.cfg.heldout_frac,
                                np.random.default_rng(self.cfg.seed + 100 + int(c)))
            buf.train = [np.asarray(f, np.uint8) for f in d["train"]]
            buf.heldout = [np.asarray(f, np.uint8) for f in d["heldout"]]
            self.anchors[int(c)] = buf


__all__ = ["ManagerConfig", "OnlineContextManager", "farthest_point_prototypes"]
