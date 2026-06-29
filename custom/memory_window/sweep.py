"""Drive the memory-window sweep: train + held-out eval over N x seeds.

Presets (see also custom/train_models.ipynb):
  smoke      N={5,50},     seed 0,     500 steps
  one        N={20},       seed 0,   2,000 steps
  all_smoke  full N grid,  seed 0,     500 steps
  all        full N grid,  seed 0,  10,000 steps
  full       full N grid,  seeds 42-46, 1M steps
  full_i     N_GRID[i] only, seeds 42-46, 1M steps

Usage (repo root, venv active):
  python custom/memory_window/sweep.py --preset smoke
  python custom/memory_window/sweep.py --preset full
  python custom/memory_window/sweep.py --preset full_0   # N=3 only, 5 seeds, 1M steps
  python custom/memory_window/sweep.py --preset one
  python custom/memory_window/sweep.py --preset full --seed 42  # full grid, this seed only

--seed overrides the preset's seed list and runs that single seed (e.g. for
splitting a full sweep across machines / runs).

Per-N full presets: full_i trains N_GRID[i] with 5 seeds, 1M steps, log_every=2000.
  full_0 -> N=3, full_1 -> N=5, ... full_9 -> N=150

Plot afterwards (separate helper):
  python custom/memory_window/plot.py custom/outputs/memory_window/<ts>/results.csv

WSL GPU:
  wsl -d Ubuntu-24.04 bash custom/memory_window/run_sweep.sh full
  wsl -d Ubuntu-24.04 bash custom/memory_window/run_sweep.sh full_3
"""

import argparse
import csv
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from custom.config import base_config
from custom.experiments._paper_training import (
    apply_memory_ann,
    apply_paper_training,
)
from custom.memory_window import evaluate as evaluate_mod
from custom.memory_window import train_windowed
from custom.paths import OUTPUTS_DIR

# Memory windows from EXPERIMENT.md.
N_GRID = [3, 5, 8, 12, 20, 35, 50, 75, 100, 150]
#N_GRID = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
SEEDS = [42, 43, 44, 45, 46]
_FULL_TRAIN = dict(seeds=SEEDS, n_training_steps=int(1e6), log_every=2000)

PRESETS = {
    'full': dict(n_grid=N_GRID, **_FULL_TRAIN),
    'all_smoke': dict(n_grid=N_GRID, seeds=[0],
                      n_training_steps=500, log_every=100),
    'all': dict(n_grid=N_GRID, seeds=[0],
                n_training_steps=10000, log_every=200),
    'smoke': dict(n_grid=[5, 50], seeds=[0],
                  n_training_steps=500, log_every=100),
    'one': dict(n_grid=[20], seeds=[0],
                n_training_steps=2000, log_every=200),
}
for i, n in enumerate(N_GRID):
  PRESETS[f'full_{i}'] = dict(n_grid=[n], **_FULL_TRAIN)


def make_config(n, seed, n_training_steps):
  """Memory-ANN at paper scale, with a sliding memory window of size n."""
  config = base_config()
  apply_paper_training(config)
  apply_memory_ann(config)
  config.debug = False
  config.random_seed = int(seed)
  config.n_training_steps = int(n_training_steps)
  config.memory_window_N = int(n)
  return config


def run(preset='smoke', out_dir=None, data=None, seed=None):
  cfg = PRESETS[preset]
  seeds = [int(seed)] if seed is not None else list(cfg['seeds'])
  started = datetime.now(timezone.utc)
  started_local = started.astimezone()
  ts = started.strftime('%Y%m%dT%H%M%SZ')
  out_dir = Path(out_dir) if out_dir else (OUTPUTS_DIR / 'memory_window' / ts)
  out_dir.mkdir(parents=True, exist_ok=True)
  results_path = out_dir / 'results.csv'
  params_dir = out_dir / 'params'
  params_dir.mkdir(parents=True, exist_ok=True)

  if data is None:
    seed0_cfg = make_config(cfg['n_grid'][0], seeds[0], 1)
    data = train_windowed.load_data(seed0_cfg)

  rows = []
  with results_path.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(
        ['N', 'seed', 'n_training_steps', 'test_acc_pct', 'valid_acc_pct',
         'final_train_loss', 'final_valid_loss', 'params_path']
    )
    for n in cfg['n_grid']:
      for seed_val in seeds:
        config = make_config(n, seed_val, cfg['n_training_steps'])
        print(f'\n=== N={n} seed={seed_val} steps={config.n_training_steps} ===')
        params, final_losses = train_windowed.train(
            config, data=data, log_every=cfg['log_every']
        )
        test_acc = evaluate_mod.evaluate(config, params, data['test_dat'])
        valid_acc = evaluate_mod.evaluate(config, params, data['valid_dat'])
        train_loss = final_losses['train_loss']
        valid_loss = final_losses['valid_loss']
        print(f'  -> held-out test acc {test_acc:.2f}%  '
              f'valid acc {valid_acc:.2f}%  '
              f'final train/valid loss {train_loss:.4f}/{valid_loss:.4f}')

        params_path = params_dir / f'params_N{n}_seed{seed_val}.pkl'
        with params_path.open('wb') as pf:
          pickle.dump(params, pf)

        rel_params_path = params_path.relative_to(out_dir).as_posix()
        writer.writerow(
            [n, seed_val, config.n_training_steps, test_acc, valid_acc,
             train_loss, valid_loss, rel_params_path]
        )
        f.flush()
        rows.append(dict(N=n, seed=seed_val, test_acc_pct=test_acc,
                         valid_acc_pct=valid_acc,
                         final_train_loss=train_loss,
                         final_valid_loss=valid_loss,
                         params_path=str(params_path)))

  # Representative resolved config for the sweep. memory_window_N and
  # random_seed vary per run (see "sweep" below); everything else is shared.
  rep_config = make_config(
      cfg['n_grid'][0], seeds[0], cfg['n_training_steps']
  )

  summary = {
      'experiment': 'memory_window',
      'preset': preset,
      'timestamp': ts,
      'started_at': started_local.strftime('%A, %d %B %Y, %H:%M:%S %Z (UTC%z)'),
      'started_at_utc': started.strftime('%A, %d %B %Y, %H:%M:%S UTC'),
      'sweep': {
          'n_grid': cfg['n_grid'],
          'seeds': seeds,
          'n_training_steps': cfg['n_training_steps'],
          'log_every': cfg['log_every'],
      },
      'config': rep_config.to_dict(),
  }
  (out_dir / 'summary.json').write_text(
      json.dumps(summary, indent=2, default=str), encoding='utf-8'
  )
  print(f'\nSaved results to {results_path}')
  return results_path, rows


def main():
  parser = argparse.ArgumentParser(description='Memory-window sweep.')
  parser.add_argument('--preset', default='smoke', choices=sorted(PRESETS))
  parser.add_argument('--out-dir', default=None)
  parser.add_argument(
      '--seed', type=int, default=None,
      help='Run only this seed (overrides the preset seed list).')
  args = parser.parse_args()

  results_path, _ = run(preset=args.preset, out_dir=args.out_dir,
                        seed=args.seed)
  print('Plot separately with: '
        f'python custom/memory_window/plot.py "{results_path}"')


if __name__ == '__main__':
  main()
