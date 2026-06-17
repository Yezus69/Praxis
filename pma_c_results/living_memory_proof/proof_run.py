"""Robust proof harness: run one continual_living_memory ablation, stream results to JSON."""
import os, time, json
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
import numpy as np, jax
from pmac.agents.ppo_living_memory_fast import FastLMConfig
from pmac.agents.continual_living_memory import continual_living_memory

ABL   = os.environ.get("ABLATION","full")
GAMES = os.environ.get("GAMES","SpaceInvaders-v5,Breakout-v5,BeamRider-v5,Asterix-v5,Qbert-v5").split(",")
PG    = int(os.environ.get("PER_GAME","1500000"))
NB    = int(os.environ.get("NBLOCKS","1"))
NE    = int(os.environ.get("NENVS","256"))
RP    = os.environ.get("RESULT_PATH", f"/root/proof_{ABL}.json")
print(f"devices={jax.devices()} ablation={ABL} games={GAMES} per_game={PG} n_blocks={NB} envs={NE}", flush=True)
cfg = FastLMConfig(num_envs=NE, num_steps=128, n_blocks=NB, hot_capacity=4096)
t0 = time.time()
res = continual_living_memory(GAMES, len(GAMES), cfg, seed=0, ablation=ABL, per_game_steps=PG, result_path=RP)
wall = time.time()-t0
print(f"DONE ablation={ABL} wall={wall:.0f}s", flush=True)
print("retention:", {k: round(float(v),3) for k,v in res.get("retention",{}).items()}, flush=True)
print(f"mean_ret={res.get('mean_retention'):.3f} worst_ret={res.get('worst_retention'):.3f} rejections={res.get('gate_rejections')}", flush=True)
