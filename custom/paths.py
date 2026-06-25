"""Local paths for custom experiments (never edit upstream config)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / 'data'
OUTPUTS_DIR = Path(__file__).resolve().parent / 'outputs'


def default_dataset_path():
  return DATA_DIR / 'openSourceHumDataset.csv'
