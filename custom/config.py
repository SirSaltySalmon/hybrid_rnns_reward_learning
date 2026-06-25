"""Build runnable configs from upstream defaults + experiment overrides."""

import importlib
import pkgutil

from hybrid_rnns_reward_learning import rnn_config

from custom.paths import default_dataset_path


def base_config():
  """Upstream defaults with local dataset path applied."""
  config = rnn_config.get_config()
  config.dataset_path = str(default_dataset_path())
  return config


def apply_experiment(config, experiment_module):
  """Apply an experiment module's configure() hook."""
  if not hasattr(experiment_module, 'configure'):
    raise ValueError(
        f'Experiment {experiment_module.__name__!r} must define configure(config).'
    )
  return experiment_module.configure(config)


def load_experiment(name):
  """Import custom.experiments.<name>."""
  if name.startswith('.'):
    raise ValueError(f'Invalid experiment name: {name!r}')
  return importlib.import_module(f'custom.experiments.{name}')


def list_experiments():
  """Yield (name, description) for each experiment module."""
  import custom.experiments as experiments_pkg

  for module_info in pkgutil.iter_modules(experiments_pkg.__path__):
    if module_info.name.startswith('_'):
      continue
    module = importlib.import_module(f'custom.experiments.{module_info.name}')
    description = getattr(module, 'DESCRIPTION', '')
    yield module_info.name, description


def build_config(experiment_name):
  """Base config + experiment overrides."""
  experiment = load_experiment(experiment_name)
  config = base_config()
  return apply_experiment(config, experiment), experiment
