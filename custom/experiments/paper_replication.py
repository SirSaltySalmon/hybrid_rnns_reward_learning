"""Paper-faithful Memory-ANN replication (~68.3% held-out accuracy).

This is the winning model from Eckstein et al. (Nat Hum Behav, 2026).
It uses the paper's training scale (1M steps, batch 128, hidden 32) and the hybrid
architecture implemented as ``birnn`` in this repo (see train_models.ipynb
"Fit winning hybRNN model").

Run:
  python custom/run.py paper_replication

Expect several hours on CPU; use a GPU if available. The printed
``paper_acc`` on test batches should trend toward ~68% when converged.
Full paper evaluation averages over all held-out participant blocks;
this trainer logs random test minibatches, so single-batch numbers will
vary.

To replicate other paper models, copy this file and swap the architecture
helper in ``configure()``:
  - apply_vanilla_rnn  → target ~68.3%
  - apply_best_rl      → target ~60.6%
"""

from custom.experiments._paper_training import (
    PAPER_TARGET_ACC_PCT,
    apply_memory_ann,
    apply_paper_training,
    apply_best_rl,
    apply_vanilla_rnn,
)

NAME = 'paper_replication'
DESCRIPTION = (
    'Memory-ANN (BiRNN), paper Methods training; target ~68.3% test accuracy.'
)
LOG_EVERY = 500
TARGET_ACC_PCT = PAPER_TARGET_ACC_PCT['memory_ann']

# Re-export architecture helpers for copy-paste experiments.
__all__ = [
    'apply_memory_ann',
    'apply_vanilla_rnn',
    'apply_best_rl',
    'apply_paper_training',
    'TARGET_ACC_PCT',
]


def configure(config):
  apply_paper_training(config)
  apply_memory_ann(config)
  return config
