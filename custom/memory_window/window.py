"""Sliding-window memory limit for Memory-ANN (BiRNN).

To predict the action at trial ``t`` the model sees only trials
``[t-N, t-1]`` fed through a fresh-initialised BiRNN. Front zero-padding
covers early trials (``t < N``) where fewer than N trials of history exist.
The upstream BiRNN is reused unchanged.

Alignment matches custom/train.py: reading trials up to ``t-1`` predicts the
action at trial ``t`` (i.e. action_probs at the window's last step).
"""

import haiku as hk
import jax
import jax.numpy as jnp

from hybrid_rnns_reward_learning.bi_rnn import BiRNN

# Same probability floor as custom/train.py loss.
_PROB_SCALE = 1 - 1e-5
_PROB_SHIFT = 5e-4


def make_forward(config):
  """hk.transform of a BiRNN whose batch size is taken from the input."""

  def _fn(input_seq):
    model = BiRNN(config.rnn_rl_params, config.network_params)
    init = model.initial_state(input_seq.shape[0])
    return hk.dynamic_unroll(model, input_seq, init, time_major=False)

  return hk.transform(_fn)


def pad_front(seq, n):
  """Prepend ``n`` zero trials so window slices for early targets are valid."""
  pad = jnp.zeros((n, seq.shape[-1]), seq.dtype)
  return jnp.concatenate([pad, seq], axis=0)


def windows_at(seq, target_idxs, n):
  """Length-``n`` windows ending just before each target index.

  Args:
    seq: (T, F) single block.
    target_idxs: (K,) ints in [1, T-1]; window for t = seq[t-n:t].
    n: window length.

  Returns:
    windows (K, n, F), targets (K, 4) one-hot human action at each target
    (all-zero rows mark missed trials).
  """
  padded = pad_front(seq, n)  # seq[j] -> padded[n + j]; seq[t-n:t] = padded[t:t+n]
  windows = jax.vmap(
      lambda t: jax.lax.dynamic_slice_in_dim(padded, t, n, axis=0)
  )(target_idxs)
  targets = seq[target_idxs, :4]
  return windows, targets


def all_windows(seq, n):
  """Every predictable target (trials 1..T-1) for one block."""
  targets = jnp.arange(1, seq.shape[0])
  return windows_at(seq, targets, n)


def last_step_nll(action_probs_seq, targets):
  """NLL of the human choice at each window's final step, plus valid mask.

  Args:
    action_probs_seq: (K, n, 4) probabilities from the windowed forward.
    targets: (K, 4) one-hot actions (all-zero = missed trial).

  Returns:
    nll (K,), mask (K,) bool of valid (non-missed) targets.
  """
  probs = action_probs_seq[:, -1, :]
  probs = _PROB_SCALE * probs + _PROB_SHIFT
  nll = -jnp.sum(jnp.log(probs) * targets, axis=-1)
  mask = jnp.sum(targets, axis=-1) > 0
  return nll, mask
