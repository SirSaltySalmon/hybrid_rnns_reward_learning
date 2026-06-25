"""Training loop with paper-style accuracy logging.

Vendored from hybrid_rnns_reward_learning.fit_hyb_rnn.train so upstream
files stay untouched. Only the logging block differs.
"""

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import pandas as pd

from hybrid_rnns_reward_learning import hyb_rnn_utilities
from hybrid_rnns_reward_learning.bi_rnn import BiRNN
from hybrid_rnns_reward_learning.cogmod import CogMod
from hybrid_rnns_reward_learning.fit_hyb_rnn import rl_param_names
from hybrid_rnns_reward_learning.rnn import RNN

from custom.metrics import enrich_scalars, format_step_report


def default_log_every(config):
  return 10 if config.debug else 500


def train(config, log_every=None, n_trials=None):
  """Fit a model and print loss plus paper-style accuracy."""
  if log_every is None:
    log_every = default_log_every(config)
  if n_trials is None:
    n_trials = config.n_trials

  def _cogmod_fn(input_seq, return_all_states=False):
    model = CogMod(config.rnn_rl_params, config.network_params)
    initial_state = model.initial_state(config.batch_size)
    return hk.dynamic_unroll(
        model,
        input_seq,
        initial_state,
        time_major=False,
        return_all_states=return_all_states)

  def _birnn_fn(input_seq, return_all_states=False):
    bi_rnn = BiRNN(config.rnn_rl_params, config.network_params)
    initial_state = bi_rnn.initial_state(config.batch_size)
    return hk.dynamic_unroll(
        bi_rnn,
        input_seq,
        initial_state,
        time_major=False,
        return_all_states=return_all_states)

  def _rnn_fn(input_seq, return_all_states=False):
    rnn = RNN(config.rnn_rl_params, config.network_params)
    initial_state = rnn.initial_state(config.batch_size)
    return hk.dynamic_unroll(
        rnn,
        input_seq,
        initial_state,
        time_major=False,
        return_all_states=return_all_states)

  @jax.jit
  def loss_fn(params, key, batch_dat):
    batch_size = batch_dat.shape[0]
    action_probs_seq, _ = forward.apply(params, key, batch_dat)
    action_probs_seq = (1 - 1e-5) * action_probs_seq + 5e-4
    loss = -jnp.sum(jnp.log(action_probs_seq[:, :-1]) * batch_dat[:, 1:, :4]
                    ) / batch_size
    return loss

  @jax.jit
  def update_func(params, key, opt_state, batch_dat):
    loss, grads = jax.value_and_grad(loss_fn)(params, key, batch_dat)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    scalars = {'train_loss': [loss]}
    return new_params, new_opt_state, scalars

  print('Loading data from {}'.format(config.dataset_path))
  hum_dat = pd.read_csv(config.dataset_path)
  hum_dat_tensor = hyb_rnn_utilities.format_data_for_model_training(hum_dat)

  train_dat = hum_dat_tensor['train_dat']
  valid_dat = hum_dat_tensor['valid_dat']
  test_dat = hum_dat_tensor['test_dat']
  print('Size of training data: {} blocks.'.format(len(train_dat)))

  if config.model_name == 'cogmod':
    print('Using cogmod to fit data.')
    forward = hk.transform(_cogmod_fn)
  elif config.model_name == 'birnn':
    print('Using BiRNN to fit data.')
    forward = hk.transform(_birnn_fn)
  elif config.model_name == 'rnn':
    print('Using RNN to fit data.')
    forward = hk.transform(_rnn_fn)
  else:
    raise ValueError('Unknown model name: %s' % config.model_name)

  rng_seq = hk.PRNGSequence(config.random_seed)
  train_batch = hyb_rnn_utilities.get_batch(
      train_dat, config.batch_size, next(rng_seq)
  )
  params = forward.init(next(rng_seq), train_batch)

  optimizer = optax.adamw(
      learning_rate=config.learning_rate,
      weight_decay=config.weight_decay
  )
  init_opt = jax.jit(optimizer.init)
  opt_state = init_opt(params)

  print('Start fitting the model')
  scalars = None
  for current_step in range(config.n_training_steps):
    train_batch = hyb_rnn_utilities.get_batch(
        train_dat, config.batch_size, next(rng_seq)
    )
    params, opt_state, scalars = update_func(
        params, next(rng_seq), opt_state, train_batch
    )

    if current_step % log_every == 0:
      test_batch = hyb_rnn_utilities.get_batch(
          test_dat, config.batch_size, next(rng_seq)
      )
      test_loss = loss_fn(params, next(rng_seq), test_batch)

      valid_batch = hyb_rnn_utilities.get_batch(
          valid_dat, config.batch_size, next(rng_seq)
      )
      valid_loss = loss_fn(params, next(rng_seq), valid_batch)

      scalars.update({
          'step': [current_step],
          'test_loss': [jax.device_get(test_loss)],
          'valid_loss': [jax.device_get(valid_loss)],
      })

      for key, value in scalars.items():
        if key in ['train_loss', 'w'] + rl_param_names:
          scalars[key] = jax.device_get(value)

      scalars = enrich_scalars(scalars, n_trials=n_trials)
      print(format_step_report(current_step, scalars, n_trials=n_trials))

  return scalars, params
