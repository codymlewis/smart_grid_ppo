"""
A demonstration of proximal policy optimisation http://arxiv.org/abs/1707.06347 on the L2RPN (https://l2rpn.chalearn.org/)
task. The implementation of PPO in https://github.com/luchris429/purejaxrl was very helpful for this.
"""

from __future__ import annotations
import os
from typing import Sequence, NamedTuple, Tuple
import functools
import math
import itertools
import grid2op
import grid2op.Converter
from lightsim2grid import LightSimBackend
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
import chex
import distrax
from tqdm import trange


class ActorCritic(nn.Module):
    "An Actor Critic neural network model."
    n_actions: int

    @nn.compact
    def __call__(self, x: chex.Array) -> chex.Array:
        actor_mean = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(math.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(math.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.n_actions, kernel_init=nn.initializers.orthogonal(0.01), bias_init=nn.initializers.constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.n_actions,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(math.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(x)
        critic = nn.relu(critic)
        critic = nn.Dense(
            64, kernel_init=nn.initializers.orthogonal(math.sqrt(2)), bias_init=nn.initializers.constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            1, kernel_init=nn.initializers.orthogonal(1.0), bias_init=nn.initializers.constant(0.0)
        )(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class TransitionBatch(NamedTuple):
    "Class to store the current batch data, and produce minibatchs from it."
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array
    rng: np.random.Generator

    def init(
        num_timesteps: int, num_actors: int, obs_shape: Sequence[int], act_shape: Sequence[int], seed: int = 0
    ) -> TransitionBatch:
        n = num_timesteps * num_actors
        return TransitionBatch(
            np.zeros((n,) + obs_shape, dtype=np.float32),
            np.zeros((n,) + act_shape, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.zeros(n, dtype=np.float32),
            np.random.default_rng(seed),
        )

    def sample(self, batch_size: int = 128) -> TransitionMinibatch:
        idx_start = self.rng.choice(self.obs.shape[0] - batch_size)
        idx = np.arange(idx_start, idx_start + batch_size)
        return TransitionMinibatch(
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.values[idx],
            self.log_probs[idx],
            self.dones[idx],
        )


class TransitionMinibatch(NamedTuple):
    "Class to store a minibatch of the transition data."
    obs: chex.Array
    actions: chex.Array
    rewards: chex.Array
    values: chex.Array
    log_probs: chex.Array
    dones: chex.Array


@jax.jit
def learner_step(
    state: train_state.TrainState,
    transitions: TransitionMinibatch,
    gamma: float = 0.99,
    lamb: float = 0.95,
    eps: float = 0.2,
    coef1: float = 1.0,
    coef2: float = 0.01,
) -> Tuple[float, train_state.TrainState]:
    """
    This is the last two lines of Algorithm 1 in http://arxiv.org/abs/1707.06347, also including the loss function
    calculation.
    """
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
        entropy = jax.vmap(lambda pi: pi.entropy())(pis).mean()
        # Then the full loss
        loss = actor_loss - coef1 * value_losses + coef2 * entropy
        return -loss  # Flip the sign to maximise the loss

    # With that we can calculate a standard gradient descent update
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return loss, state


if __name__ == "__main__":
    seed = 63
    batch_size = 128
    num_episodes = 100
    num_timesteps = 100
    num_actors = 15
    num_steps = 10
    env_name = "rte_case14_realistic"
    env = grid2op.make(env_name)
    if not os.path.exists(grid2op.get_current_local_dir() + f"/{env_name}_test"):
        env.train_val_split_random(pct_val=0.0, add_for_test="test", pct_test=10.0)
    train_env = grid2op.make(env_name + "_train", backend=LightSimBackend())
    obs = train_env.reset()
    model = ActorCritic(env.action_space.n)
    rngkey = jax.random.PRNGKey(seed)
    params_rngkey, rngkey = jax.random.split(rngkey)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=model.init(params_rngkey, obs.to_vect()),
        # We use AMSGrad instead of Adam, due to greater stability in noise https://arxiv.org/abs/1904.09237
        tx=optax.amsgrad(1e-4),
    )
    converter = grid2op.Converter.ToVect(env.action_space)
    obs_shape = obs.to_vect().shape
    act_shape = (env.action_space.n,)

    for e in (pbar := trange(num_episodes)):
        # We generate all of the random generation keys that we will need pre-emptively
        rngkeys = jax.random.split(rngkey, num_actors * num_timesteps + 1)
        rngkey = rngkeys[0]
        rngkeys = iter(rngkeys[1:])
        # Allocate the memory for our data batch and the index where each sample is stored
        transitions = TransitionBatch.init(num_timesteps, num_actors, obs_shape, act_shape, seed + e)
        counter = itertools.count()
        # Now we perform the actor loop from Algorithm 1 in http://arxiv.org/abs/1707.06347
        for a in range(num_actors):
            last_obs = train_env.reset().to_vect()
            for t in range(num_timesteps):
                i = next(counter)
                pi, transitions.values[i] = state.apply_fn(state.params, last_obs)
                transitions.actions[i] = pi.sample(seed=next(rngkeys))
                transitions.log_probs[i] = pi.log_prob(transitions.actions[i])
                obs, transitions.rewards[i], transitions.dones[i], info = train_env.step(
                    converter.convert_act(transitions.actions[i])
                )
                transitions.obs[i] = last_obs = obs.to_vect()
                if transitions.dones[i]:
                    last_obs = train_env.reset().to_vect()

        # Then we peform the updates with our newly formed batch of data
        for _ in range(num_steps):
            trans_batch = transitions.sample(batch_size)
            loss, state = learner_step(state, trans_batch)
        pbar.set_postfix_str(f"Loss: {loss:.5f}")

    print("Now, let's see how long the trained model can run the power network.")
    test_env = grid2op.make(env_name + "_test", backend=LightSimBackend())
    obs = test_env.reset().to_vect()
    for i in itertools.count():
        rngkey, _rngkey = jax.random.split(rngkey)
        pi, _ = state.apply_fn(state.params, obs)
        action = pi.sample(seed=_rngkey)
        obs, reward, done, info = test_env.step(converter.convert_act(action))
        obs = obs.to_vect()
        if done:
            print()
            print(f"Ran the network for {i + 1} time steps")
            break
        if i % 100 == 0 and i > 0:
            print("." * (i // 100), end="\r")
