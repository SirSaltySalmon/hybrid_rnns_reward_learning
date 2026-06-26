"""Plot held-out accuracy vs memory window N and mark the elbow.

Elbow = point of diminishing returns on accuracy as N grows, found with a
dependency-free Kneedle-style detector on (log N, accuracy).
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from custom.metrics import CHANCE_ACCURACY_PCT


def load_results(results_path, acc_col='test_acc_pct'):
  by_n = defaultdict(list)
  with open(results_path, newline='') as f:
    for row in csv.DictReader(f):
      by_n[int(row['N'])].append(float(row[acc_col]))
  ns = np.array(sorted(by_n))
  mean = np.array([np.mean(by_n[n]) for n in ns])
  std = np.array([np.std(by_n[n]) for n in ns])
  return ns, mean, std, by_n


def find_elbow(ns, acc):
  """Largest vertical gap above the chord on (log N, acc): the elbow N."""
  if len(ns) < 3:
    return int(ns[0])
  x = np.log(ns.astype(float))
  y = acc.astype(float)
  xn = (x - x[0]) / (x[-1] - x[0] + 1e-12)
  yn = (y - y[0]) / (y[-1] - y[0] + 1e-12)
  return int(ns[int(np.argmax(yn - xn))])


def plot(results_path, out_png=None, acc_col='test_acc_pct'):
  ns, mean, std, by_n = load_results(results_path, acc_col=acc_col)
  elbow = find_elbow(ns, mean)

  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(7, 5))
  for n in ns:
    ax.scatter([n] * len(by_n[n]), by_n[n], color='0.7', s=18, zorder=1)
  ax.errorbar(ns, mean, yerr=std, marker='o', capsize=3, zorder=2,
              label='mean +/- sd')
  ax.axvline(elbow, ls='--', color='red', label=f'elbow N={elbow}')
  ax.axhline(CHANCE_ACCURACY_PCT, ls=':', color='0.5',
             label=f'chance {CHANCE_ACCURACY_PCT:.0f}%')
  ax.set_xscale('log')
  ax.set_xticks(ns)
  ax.set_xticklabels([str(n) for n in ns])
  ax.set_xlabel('Memory window N (trials, log scale)')
  ax.set_ylabel('Held-out accuracy (%)')
  ax.set_title('How far back does memory reach?')
  ax.legend()
  fig.tight_layout()

  out_png = out_png or str(results_path).replace('.csv', '.png')
  fig.savefig(out_png, dpi=150)
  print(f'Saved plot to {out_png}  (elbow N={elbow})')
  return out_png, elbow


if __name__ == '__main__':
  import argparse
  matplotlib.use('Agg')
  parser = argparse.ArgumentParser(description='Plot memory-window elbow.')
  parser.add_argument('results_csv')
  parser.add_argument('--out', default=None)
  args = parser.parse_args()
  plot(args.results_csv, out_png=args.out)
