"""Metrics helpers for interpreting hybrid RNN training loss.

Paper reference (Nature Human Behaviour):
  acc = exp(-L / (batch_size * n_trials))

This repo reports loss = L / batch_size, so:
  paper_acc_pct = exp(-loss / n_trials) * 100
"""

import numpy as np

CHANCE_ACCURACY_PCT = 25.0  # uniform policy over 4 bandit arms


def to_scalar(value):
  return float(np.asarray(value).item())


def loss_to_paper_accuracy_pct(loss, n_trials=150):
  """Map training loss to the paper's trial-wise prediction accuracy (%)."""
  return float(np.exp(-to_scalar(loss) / n_trials) * 100.0)


def enrich_scalars(scalars, n_trials=150):
  """Add *_acc_pct fields alongside *_loss fields."""
  enriched = dict(scalars)
  for loss_key in ('train_loss', 'valid_loss', 'test_loss'):
    if loss_key not in scalars or not scalars[loss_key]:
      continue
    acc_key = loss_key.replace('_loss', '_acc_pct')
    enriched[acc_key] = [
        loss_to_paper_accuracy_pct(scalars[loss_key][0], n_trials=n_trials)
    ]
  return enriched


def format_step_report(step, scalars, n_trials=150):
  """Human-readable multi-line report for one training checkpoint."""
  enriched = enrich_scalars(scalars, n_trials=n_trials)
  lines = [f'Step: {step}']
  for loss_key in ('train_loss', 'valid_loss', 'test_loss'):
    if loss_key not in enriched:
      continue
    split = loss_key.replace('_loss', '')
    loss = to_scalar(enriched[loss_key][0])
    acc = to_scalar(enriched[f'{split}_acc_pct'][0])
    lines.append(f'  {split:5s}: loss={loss:10.4f}  paper_acc={acc:6.2f}%')
  lines.append(f'  chance baseline: {CHANCE_ACCURACY_PCT:.1f}%')
  return '\n'.join(lines)
