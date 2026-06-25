"""Run a named experiment: apply custom config, train, save results.

Usage (from repo root, venv activated):
  python custom/run.py                  # default: debug
  python custom/run.py birnn_memory
  python custom/run.py --list
  python custom/run.py debug --no-save
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from custom.config import build_config, list_experiments
from custom.io import save_run
from custom.train import default_log_every, train


def main():
  parser = argparse.ArgumentParser(
      description='Run a custom hybrid RNN experiment.'
  )
  parser.add_argument(
      'experiment',
      nargs='?',
      default='debug',
      help='Experiment module name under custom/experiments/ (default: debug)',
  )
  parser.add_argument(
      '--list',
      action='store_true',
      help='List available experiments and exit.',
  )
  parser.add_argument(
      '--no-save',
      action='store_true',
      help='Skip writing summary to custom/outputs/.',
  )
  parser.add_argument(
      '--log-every',
      type=int,
      default=None,
      help='Override logging interval in training steps.',
  )
  args = parser.parse_args()

  if args.list:
    print('Available experiments:')
    for name, description in list_experiments():
      suffix = f' — {description}' if description else ''
      print(f'  {name}{suffix}')
    return

  config, experiment = build_config(args.experiment)
  log_every = args.log_every
  if log_every is None:
    log_every = getattr(experiment, 'LOG_EVERY', default_log_every(config))

  print(f'Running experiment: {args.experiment}')
  print(f'  model: {config.model_name}')
  print(f'  steps: {config.n_training_steps}  batch: {config.batch_size}')
  print(f'  dataset: {config.dataset_path}')
  target_acc = getattr(experiment, 'TARGET_ACC_PCT', None)
  if target_acc is not None:
    print(f'  paper target (held-out): ~{target_acc}% trial-wise accuracy')

  scalars, params = train(config, log_every=log_every)

  if not args.no_save:
    save_run(args.experiment, config, scalars, params=params)


if __name__ == '__main__':
  main()
