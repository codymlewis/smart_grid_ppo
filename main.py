from typing import Tuple, NamedTuple
import functools
import math
import random
import itertools
import grid2op
import grid2op.Converter
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state  # , orbax_utils
import optax
import chex
import distrax
from tqdm import trange


class ActorCritic(nn.Module):
    n_actions: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        actor_mean = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.n_actions, kernel_init=nn.initializers.orthogonal(0.01), bias_init=nn.initializers.constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(x)
        critic = nn.relu(critic)
        critic = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.constant(0.0)
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class TransitionBatch(NamedTuple):
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array
    rng: jax.random.PRNGKey

    def init(num_timesteps, num_actors, obs_shape, act_shape, seed):
        n = num_timesteps * num_actors
        return TransitionBatch(
            np.zeros((n,) + obs_shape, dtype=np.float32),
            np.zeros((n,) + act_shape, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros((n,) + act_shape, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            jax.random.PRNGKey(seed),
        )

    def sample(self, batch_size: int = 128):
        _rng, rng = jax.random.split(self.rng)
        idx = jax.random.choice(_rng, jnp.arange(self.obs.shape[0]), shape=(batch_size,), replace=False)
        return TransitionBatch(
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.values[idx],
            self.log_probs[idx],
            self.dones[idx],
            rng,
        )


@jax.jit
def learner_step(
    state: train_state.TrainState,
    transitions: TransitionBatch,
    gamma: float = 1.0,
    lamb: float = 1.0,
    eps: float = 0.2,
    coef1: float = 0.1,
    coef2: float = 0.1,
):
    # Calculate advantage
    _, last_val = state.apply_fn(state.params, transitions.obs[-1])

    def calc_advantages(last_advantage, done_and_delta):
        done, delta = done_and_delta
        advantage = delta + gamma * lamb * done * last_advantage
        return advantage, advantage

    next_values = jnp.concatenate((transitions.values[1:], last_val.reshape(1)), axis=0)
    deltas = transitions.rewards + gamma * next_values * transitions.dones - transitions.values
    _, advantages = jax.lax.scan(calc_advantages, 0.0, (transitions.dones, deltas))

    def loss_fn(params):
        pis, values = jax.vmap(functools.partial(state.apply_fn, params))(transitions.obs)
        log_probs = jax.vmap(lambda pi, a: pi.log_prob(a))(pis, transitions.actions)
        # Value loss
        targets = advantages + transitions.values
        value_losses = jnp.mean((values - targets)**2)
        # Actor loss
        ratio = jnp.exp(log_probs - transitions.log_probs)
        norm_advantages = (advantages - advantages.mean(-1)) / (advantages.std(-1) + 1e-8)
        actor_loss1 = (ratio.T * norm_advantages).T
        actor_loss2 = (jnp.clip(ratio, 1 - eps, 1 + eps).T * norm_advantages).T
        actor_loss = jnp.mean(jnp.minimum(actor_loss1, actor_loss2))
        # Entropy loss
        entropy = pi.entropy().mean()
        # Then the full loss
        loss = actor_loss - coef1 * value_losses + coef2 * entropy
        return -loss

    loss_fn(state.params)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


if __name__ == "__main__":
    seed = 63
    batch_size = 128
    num_episodes = 100
    num_timesteps = 100
    num_actors = 10
    num_steps = 10
    random.seed(seed)
    env = grid2op.make("rte_case14_realistic")  # Change to "lrpn_idf_2023"
    obs = env.reset()
    model = ActorCritic(env.action_space.n)
    rngkey = jax.random.PRNGKey(seed)
    params_rngkey, rngkey = jax.random.split(rngkey)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(params_rngkey, obs.to_vect()),
        tx=optax.adam(1e-4),
    )
    converter = grid2op.Converter.ToVect(env.action_space)

    for _ in (pbar := trange(num_episodes)):
        rngkeys = jax.random.split(rngkey, num_actors * num_timesteps + 1)
        rngkey = rngkeys[0]
        rngkeys = iter(rngkeys[1:])
        transitions = TransitionBatch.init(num_timesteps, num_actors, obs.to_vect().shape, (env.action_space.n,), seed)
        counter = itertools.count()
        for a in range(num_actors):
            last_obs = env.reset().to_vect()
            for t in range(num_timesteps):
                i = next(counter)
                pi, transitions.values[i] = state.apply_fn(state.params, last_obs)
                transitions.actions[i] = pi.sample(seed=next(rngkeys))
                transitions.log_probs[i] = pi.log_prob(transitions.actions[i])
                obs, transitions.rewards[i], transitions.dones[i], info = env.step(
                    converter.convert_act(transitions.actions[i])
                )
                transitions.obs[i] = last_obs = obs.to_vect()
                if transitions.dones[i]:
                    last_obs = env.reset().to_vect()

        for i in range(num_steps):
            trans_batch = transitions.sample(batch_size)
            loss, state = learner_step(state, trans_batch)
        pbar.set_postfix_str(f"Loss: {loss:.5f}")
