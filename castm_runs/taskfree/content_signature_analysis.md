# Content-signature discriminability analysis (pooled raw observations, 8×8)

Offline, random-policy frames (500/game), k-means(12) prototypes. Cross-game
best-prototype cosine similarity (diagonal = within-game ≈ 1.0). A pair is
**separable** when its cross-sim is well below the within-sim; a high cross-sim
(≳ a context's match band) causes a FALSE MERGE in the online manager.

## RAW pooled cosine
```
                 BK    Pong   SI    Sea   BR
Breakout-v5    +1.00 +0.73 +0.50 +0.53 +0.79
Pong-v5        +0.70 +1.00 +0.77 +0.94 +0.77   <- Pong~Seaquest 0.94 (collide)
SpaceInvaders  +0.49 +0.77 +1.00 +0.77 +0.64
Seaquest-v5    +0.53 +0.93 +0.77 +1.00 +0.71   <- Seaquest~Pong 0.93 (collide)
BeamRider-v5   +0.71 +0.74 +0.62 +0.66 +0.98
```

## Centered by the (oracle) all-5 mean
```
                 BK    Pong   SI    Sea   BR
Breakout-v5    +1.00 -0.12 -0.06 -0.79 +0.09
Pong-v5        -0.25 +1.00 -0.91 +0.42 +0.46
SpaceInvaders  -0.07 -0.88 +1.00 -0.22 +0.94   <- SI~BeamRider 0.94 (collide)
Seaquest-v5    -0.81 +0.41 -0.24 +1.00 +0.16
BeamRider-v5   +0.08 -0.80 +0.84 -0.32 +0.99   <- BeamRider~SI 0.84 (collide)
```

## Conclusion
- **2–3 visually-distinct regimes separate cleanly** (e.g. SpaceInvaders/Seaquest/
  Breakout in Stage 2: inter-context prototype sim 0.098, router top-1 = 1.0).
- **The 5-game set is not separable by pooled pixels under any single centering**:
  RAW collides Pong↔Seaquest; centered collides SpaceInvaders↔BeamRider (both are
  vertical shooters with near-identical spatial layout). The online run's *running*-
  mean centering (dominated by the very-different Pong) additionally collides
  SpaceInvaders↔Breakout, which is what the Stage-3 run hit.
- **The continual-learning MECHANISM is not the bottleneck** (it is proven at 2 and
  3 contexts). The **content representation** is. A learned, discriminative content
  encoder (contrastive/predictive, or earlier conv features re-exercising the §2
  drift-refresh machinery) is required to scale task-free discovery to visually-
  overlapping regimes — next-experiment #2.
