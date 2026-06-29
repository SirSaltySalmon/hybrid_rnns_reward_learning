"""Windowed training loop.

Each step predicts ONE random target trial per block using only the
preceding N trials (sliding window). Per-step cost ~= batch x N cell-evals,
so it stays at or below the baseline full-sequence loop. Optimiser, config,
and data handling mirror custom/train.py.

Optimiser steps run inside a jitted ``jax.lax.scan`` chunk (``chunk`` ==
``log_every``) so the GPU executes many steps per dispatch instead of one
Python iteration per step. This removes per-step host/dispatch overhead that
otherwise starves the GPU on this small model (benchmarked ~3.6x at N=5,
~1.5x at N=50, ~1.1x at N=150). Host syncs happen only once per chunk, at
logging time.
"""

import functools

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
  """Fit one window-limited Memory-ANN.

  Returns ``(params, final_losses)`` where ``final_losses`` is a dict with
  ``train_loss`` and ``valid_loss`` (per-trial NLL) measured after the final
  optimisation step.
  """
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

  def step_fn(carry, _):
    params, opt_state, key = carry
    key, k_sample, k_loss = jax.random.split(key, 3)
    windows, targets = sample_windows(train_dat, k_sample)
    loss, grads = jax.value_and_grad(loss_fn)(params, k_loss, windows, targets)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return (params, opt_state, key), loss

  @functools.partial(jax.jit, static_argnums=(3,))
  def run_chunk(params, opt_state, key, length):
    """Run ``length`` optimiser steps in one dispatch; return final loss."""
    (params, opt_state, key), losses = jax.lax.scan(
        step_fn, (params, opt_state, key), None, length=length
    )
    return params, opt_state, key, losses[-1]

  print(f'Training Memory-ANN with memory window N={n} '
        f'(seed={config.random_seed}, steps={config.n_training_steps}).')
  print(f'  train blocks: {train_dat.shape[0]}  batch: {batch_size}')

  key = jax.random.PRNGKey(int(config.random_seed))
  key, k_init_sample, k_init = jax.random.split(key, 3)
  w0, _ = sample_windows(train_dat, k_init_sample)
  params = forward.init(k_init, w0)
  opt_state = jax.jit(optimizer.init)(params)

  total_steps = int(config.n_training_steps)
  chunk = max(1, min(int(log_every), total_steps)) if total_steps else 1

  loss = jnp.asarray(float('nan'))
  step = 0
  remaining = total_steps
  while remaining > 0:
    length = min(chunk, remaining)
    params, opt_state, key, loss = run_chunk(params, opt_state, key, length)
    step += length
    remaining -= length
    # Sync to host only here (once per chunk). Loss is already per-trial NLL,
    # so n_trials=1 reuses the paper-acc formatting from custom/metrics.py.
    k_valid = jax.random.fold_in(key, step)
    k_vsample, k_vloss = jax.random.split(k_valid)
    vw, vt = sample_windows(valid_dat, k_vsample)
    valid_loss = loss_fn(params, k_vloss, vw, vt)
    scalars = {
        'train_loss': [jax.device_get(loss)],
        'valid_loss': [jax.device_get(valid_loss)],
    }
    print(metrics.format_step_report(step, scalars, n_trials=1))

  final_train_loss = float(loss)
  k_final = jax.random.fold_in(key, step + 1)
  k_fsample, k_floss = jax.random.split(k_final)
  vw, vt = sample_windows(valid_dat, k_fsample)
  final_valid_loss = float(loss_fn(params, k_floss, vw, vt))
  final_losses = {'train_loss': final_train_loss, 'valid_loss': final_valid_loss}
  return params, final_losses
