"""Shared settings from Eckstein et al. Methods (Model training).

Paper: Hybrid neural-cognitive models reveal how memory shapes human
reward learning. Nat Hum Behav (2026). doi:10.1038/s41562-025-02324-0

Reported held-out trial-wise prediction accuracy (chance = 25%):
  - Memory-ANN / Vanilla RNN: 68.3%
  - Context-ANN: 65.4%
  - RL-ANN: 60.8%
  - Best RL: 60.6%

Training grid in Methods: lr {1e-3, 1e-4, 1e-5}, weight decay {1e-3, 1e-4,
1e-5}, hidden units {16, 32, 64}, batch size {32, 64, 128}, up to 1e6 steps
with best combo chosen on validation (Supplementary Table 1).

This module applies the repo's non-debug defaults (lr=1e-4, wd=1e-5,
hidden=16, batch=32, 1e6 steps), which sit inside the paper's sweep grid.
"""

# Held-out test accuracies reported in the paper (percent).
PAPER_TARGET_ACC_PCT = {
    'memory_ann': 68.3,
    'vanilla_rnn': 68.3,
    'context_ann': 65.4,
    'rl_ann': 60.8,
    'best_rl': 60.6,
}


def apply_paper_training(config):
  """Paper-scale optimizer and schedule (non-debug)."""
  config.debug = False
  config.n_training_steps = int(1e6)
  config.batch_size = 32
  config.learning_rate = 1e-4
  config.weight_decay = 1e-5
  config.network_params.hidden_size = 16
  return config


def apply_memory_ann(config):
  """Memory-ANN — winning hybrid model; maps to repo ``birnn``."""
  config.model_name = 'birnn'
  config.rnn_rl_params.w_v = 1
  config.rnn_rl_params.w_h = 1
  config.rnn_rl_params.fit_forget = True
  config.rnn_rl_params.o = False
  config.rnn_rl_params.s = True
  config.rnn_rl_params.zero_values = True
  config.rnn_rl_params.fit_init_v = True
  config.rnn_rl_params.fit_init_h = True
  return config


def apply_vanilla_rnn(config):
  """Vanilla RNN baseline from the paper; maps to repo ``rnn``."""
  config.model_name = 'rnn'
  config.rnn_rl_params.s = True
  config.rnn_rl_params.o = False
  return config


def apply_best_rl(config):
  """Best RL cognitive model; maps to repo ``cogmod``."""
  config.model_name = 'cogmod'
  config.rnn_rl_params.fit_alpha = True
  config.rnn_rl_params.fit_beta = True
  config.rnn_rl_params.fit_bias = True
  config.rnn_rl_params.fit_forget = True
  config.rnn_rl_params.fit_init_v = True
  config.rnn_rl_params.fit_persev_t = True
  config.rnn_rl_params.fit_init_h = False
  config.rnn_rl_params.fit_persev_p = False
  config.rnn_rl_params.fit_w = False
  return config
