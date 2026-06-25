"""Copy this file to create a new experiment: custom/experiments/my_run.py

Run with:
  python custom/run.py my_run

For a paper-faithful starting point, copy instead:
  custom/experiments/paper_replication.py
"""

NAME = 'template'
DESCRIPTION = 'Describe what this experiment tests.'
LOG_EVERY = 10  # print interval; omit to use 10 (debug) or 500 (full)


def configure(config):
  # Local dataset path is set automatically in custom/config.py.

  # --- quick local test ---
  config.debug = True
  config.model_name = 'birnn'  # 'rnn' | 'birnn' | 'cogmod'
  config.n_training_steps = 100
  config.batch_size = 2

  # --- paper-scale training (see paper_replication.py) ---
  # from custom.experiments._paper_training import (
  #     apply_paper_training,
  #     apply_memory_ann,  # or apply_vanilla_rnn / apply_best_rl
  # )
  # apply_paper_training(config)
  # apply_memory_ann(config)

  return config
