"""Fast local smoke test."""

NAME = 'debug'
DESCRIPTION = 'BiRNN, 100 steps, batch 2 — quick sanity check.'
LOG_EVERY = 10


def configure(config):
  config.debug = True
  config.model_name = 'birnn'
  config.n_training_steps = 100
  config.batch_size = 2
  return config
