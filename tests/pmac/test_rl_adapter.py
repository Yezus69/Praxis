import numpy as np
import jax
import jax.numpy as jnp
import optax

from pmac.adapters.rl import RLAdapter, init_actor_critic
from pmac.continual import clip_global
from pmac.envs.gridworld import GridWorld


def _zero_policy(params, obs):
    del params
    shape = obs.shape[:-1]
    return jnp.zeros((*shape, 4), dtype=jnp.float32), jnp.zeros(shape, dtype=jnp.float32)


def test_gridworld_step_reward_and_rollout_shapes():
    env = GridWorld(grid_size=3, horizon=6, goal_cells=[1])
    state = env.state_from_position(
        jnp.array([0, 2], dtype=jnp.int32),
        jnp.array([0, 0], dtype=jnp.int32),
    )
    obs = env.observe(state)
    next_state, reward = env.step(state, jnp.array([3, 2], dtype=jnp.int32))

    assert obs.shape == (2, 10)
    assert np.allclose(np.asarray(reward), np.array([1.0, 1.0], dtype=np.float32))
    assert np.all(np.asarray(next_state.done))
    assert np.all(np.asarray(next_state.reached))

    traj = env.rollout(jax.random.PRNGKey(0), _zero_policy, None, 0, batch_size=5)

    assert traj.obs.shape == (6, 5, 10)
    assert traj.actions.shape == (6, 5)
    assert traj.rewards.shape == (6, 5)
    assert traj.mask.shape == (6, 5)


def test_rl_adapter_distance_is_nonnegative_and_zero_at_equality():
    env = GridWorld(grid_size=3, horizon=6, goal_cells=[0])
    adapter = RLAdapter(env, value_distance_coef=0.25)
    behavior = {
        "policy_logits": jnp.array([[2.0, 0.0, -1.0, 0.5], [0.1, 0.2, 0.3, 0.4]]),
        "value": jnp.array([1.0, -0.5]),
    }
    same = adapter.distance(behavior, behavior)
    shifted = {
        "policy_logits": behavior["policy_logits"] + jnp.array([[0.0, 0.1, 0.0, 0.0]]),
        "value": behavior["value"] + jnp.array([0.0, 0.2]),
    }
    dist = adapter.distance(shifted, behavior)

    assert np.allclose(np.asarray(same), 0.0, atol=1e-6)
    assert dist.shape == (2,)
    assert np.all(np.asarray(dist) >= -1e-7)


def test_a2c_steps_increase_one_goal_success_on_tiny_grid():
    env = GridWorld(grid_size=3, horizon=8, goal_cells=[0])
    adapter = RLAdapter(env, entropy_coef=0.02, eval_episodes=64)
    params = init_actor_critic(jax.random.PRNGKey(0), env.obs_dim, hidden_sizes=(32,))
    opt = optax.adam(0.02)
    opt_state = opt.init(params)

    eval_set = {
        "goal_id": 0,
        "num_episodes": 64,
        "key": jax.random.PRNGKey(123),
        "greedy": True,
    }
    initial = adapter.evaluate_skill(params, eval_set)

    for step in range(80):
        key = jax.random.PRNGKey(1_000 + step)
        batch = adapter.rollout_batch(params, key, goal_id=0, batch_size=64)
        grads = jax.grad(adapter.current_loss)(params, batch)
        grads = clip_global(grads, 5.0)
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    final = adapter.evaluate_skill(params, eval_set)

    assert final >= initial
    assert final >= 0.4
