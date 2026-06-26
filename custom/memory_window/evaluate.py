"""Full held-out windowed accuracy.

Accuracy follows the repo convention (custom/metrics.py): the geometric-mean
probability the model assigns to the actual human choice,
``exp(-mean_per_trial_nll) * 100`` (chance = 25%). Every predictable target
(trials 1..T-1) in every held-out block is scored with its own N-window.
Missed trials (all-zero action one-hot) are masked out.
"""

import jax
import jax.numpy as jnp

from custom.metrics import loss_to_paper_accuracy_pct
from custom.memory_window import window


def evaluate(config, params, eval_dat, batch_blocks=64, key=None):
  """Return held-out accuracy (%) for window size config.memory_window_N."""
  n = int(config.memory_window_N)
  forward = window.make_forward(config)
  if key is None:
    key = jax.random.PRNGKey(0)

  @jax.jit
  def chunk_nll(params, key, blocks):
    windows, targets = jax.vmap(lambda s: window.all_windows(s, n))(blocks)
    b, k, _, f = windows.shape
    windows = windows.reshape(b * k, n, f)
    targets = targets.reshape(b * k, targets.shape[-1])
    probs, _ = forward.apply(params, key, windows)
    nll, mask = window.last_step_nll(probs, targets)
    return jnp.sum(nll * mask), jnp.sum(mask)

  total_nll = 0.0
  total_n = 0.0
  n_blocks = int(eval_dat.shape[0])
  for i in range(0, n_blocks, batch_blocks):
    chunk = eval_dat[i:i + batch_blocks]
    key, sub = jax.random.split(key)
    s_nll, s_n = chunk_nll(params, sub, chunk)
    total_nll += float(s_nll)
    total_n += float(s_n)

  # Per-trial mean NLL, so n_trials=1 reuses the paper-acc map from metrics.
  mean_nll = total_nll / max(total_n, 1.0)
  return loss_to_paper_accuracy_pct(mean_nll, n_trials=1)
