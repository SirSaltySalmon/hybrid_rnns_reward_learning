"""Persist run metadata for comparing experiments over time."""

import json
from datetime import datetime, timezone

from custom.metrics import enrich_scalars, to_scalar
from custom.paths import OUTPUTS_DIR


def _config_snapshot(config):
  """JSON-serializable view of the ml_collections config."""
  snapshot = config.to_dict()
  snapshot['network_params']['final_activation_fn'] = 'softmax'
  return snapshot


def _scalars_snapshot(scalars, n_trials):
  enriched = enrich_scalars(scalars, n_trials=n_trials)
  out = {}
  for key, value in enriched.items():
    if isinstance(value, list) and value:
      out[key] = to_scalar(value[0])
    else:
      out[key] = value
  return out


def save_run(experiment_name, config, scalars, params=None):
  """Write config + final metrics under custom/outputs/<experiment>/<timestamp>/."""
  timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  run_dir = OUTPUTS_DIR / experiment_name / timestamp
  run_dir.mkdir(parents=True, exist_ok=True)

  summary = {
      'experiment': experiment_name,
      'timestamp': timestamp,
      'config': _config_snapshot(config),
      'final_scalars': _scalars_snapshot(scalars, config.n_trials),
  }
  summary_path = run_dir / 'summary.json'
  summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

  if params is not None:
    import pickle

    with (run_dir / 'params.pkl').open('wb') as f:
      pickle.dump(params, f)

  print(f'Saved run summary to {summary_path}')
  return run_dir
