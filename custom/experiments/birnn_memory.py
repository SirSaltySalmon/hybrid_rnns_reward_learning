"""BiRNN with memory-style recurrence (matches train_models.ipynb sketch)."""

NAME = 'birnn_memory'
DESCRIPTION = 'BiRNN with state recurrence, forgetting, and zero-value updates.'
LOG_EVERY = 50


def configure(config):
  config.debug = True
  config.model_name = 'birnn'
  config.n_training_steps = 1001
  config.batch_size = 32

  config.rnn_rl_params.w_v = 1
  config.rnn_rl_params.w_h = 1
  config.rnn_rl_params.fit_forget = True
  config.rnn_rl_params.o = False
  config.rnn_rl_params.s = True
  config.rnn_rl_params.zero_values = True
  config.rnn_rl_params.fit_init_v = True
  config.rnn_rl_params.fit_init_h = True
  return config
