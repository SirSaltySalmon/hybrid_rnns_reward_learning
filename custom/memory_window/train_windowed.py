"""Windowed training loop.

Each step predicts ONE random target trial per block using only the
preceding N trials (sliding window). Per-step cost ~= batch x N cell-evals,
so it stays at or below the baseline full-sequence loop. Optimiser, config,
and data handling mirror custom/train.py.
"""

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import pandas as pd

from hybrid_rnns_reward_learning import hyb_rnn_utilities

from custom import metrics
from custom.memory_window import window


def load_data(config):
  hum_dat = pd.read_csv(config.dataset_path)
  return hyb_rnn_utilities.format_data_for_model_training(hum_dat)


def train(config, data=None, log_every=500):
  """Fit one window-limited Memory-ANN. Returns fitted params."""
  n = int(config.memory_window_N)
  batch_size = int(config.batch_size)
  if data is None:
    data = load_data(config)
  train_dat = data['train_dat']
  valid_dat = data['valid_dat']
  t_len = int(train_dat.shape[1])

  forward = window.make_forward(config)
  optimizer = optax.adamw(
      learning_rate=config.learning_rate, weight_decay=config.weight_decay
  )

  @jax.jit
  def sample_windows(dat, key):
    k_blocks, k_targets = jax.random.split(key)
    batch = hyb_rnn_utilities.get_batch(dat, batch_size, k_blocks)
    targets = jax.random.randint(k_targets, (batch_size,), 1, t_len)

    def one(seq, t):
      padded = window.pad_front(seq, n)
      win = jax.lax.dynamic_slice_in_dim(padded, t, n, axis=0)
      return win, seq[t, :4]

    return jax.vmap(one)(batch, targets)

  @jax.jit
  def loss_fn(params, key, windows, targets):
    probs, _ = forward.apply(params, key, windows)
    nll, mask = window.last_step_nll(probs, targets)
    return jnp.sum(nll * mask) / jnp.maximum(jnp.sum(mask), 1.0)

  @jax.jit
  def update(params, key, opt_state, windows, targets):
    loss, grads = jax.value_and_grad(loss_fn)(params, key, windows, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, loss

  print(f'Training Memory-ANN with memory window N={n} '
        f'(seed={config.random_seed}, steps={config.n_training_steps}).')
  print(f'  train blocks: {train_dat.shape[0]}  batch: {batch_size}')

  rng = hk.PRNGSequence(config.random_seed)
  w0, _ = sample_windows(train_dat, next(rng))
  params = forward.init(next(rng), w0)
  opt_state = jax.jit(optimizer.init)(params)

  for step in range(int(config.n_training_steps)):
    windows, targets = sample_windows(train_dat, next(rng))
    params, opt_state, loss = update(
        params, next(rng), opt_state, windows, targets
    )
    if step % log_every == 0:
      vw, vt = sample_windows(valid_dat, next(rng))
      valid_loss = loss_fn(params, next(rng), vw, vt)
      # Loss is already per-trial NLL, so n_trials=1 reuses the paper-acc
      # formatting from custom/metrics.py unchanged.
      scalars = {'train_loss': [float(loss)], 'valid_loss': [float(valid_loss)]}
      print(metrics.format_step_report(step, scalars, n_trials=1))

  return params
