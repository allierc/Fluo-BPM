"""Parameterized sweeps: one spec file per experiment instead of one YAML per arm.

A spec names a base configuration, the parameter to sweep, and the values:

    name: RI_sweep
    base: config/base/reference_optics.yaml
    overrides:                       # applied to every arm
      volume.nz: 768
      phantom.n_cells: 2000
    sweep:
      parameter: phantom.dn_cell
      values:  [0.0, 0.0025, 0.005, 0.010, 0.020]
      labels:  [dn0000, dn0025, dn0050, dn0100, dn0200]

Arms land in log/<name>/<label>/. Overrides use dotted paths into the config tree, so
anything in FluoBPMConfig can be swept without adding code.
"""
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import yaml

from .config import FluoBPMConfig


def set_dotted(tree: Dict[str, Any], path: str, value: Any) -> None:
    """tree['a']['b'] = value for path 'a.b', creating nothing: a wrong path raises."""
    keys = path.split('.')
    node = tree
    for key in keys[:-1]:
        if key not in node:
            raise KeyError(f'no such config section: {path} (at {key!r})')
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f'no such config field: {path} (at {keys[-1]!r})')
    node[keys[-1]] = value


class Sweep:
    def __init__(self, spec_path: str):
        self.path = Path(spec_path)
        spec = yaml.safe_load(self.path.read_text())
        self.name: str = spec['name']
        self.description: str = spec.get('description', self.name)
        self.base_path = Path(spec['base'])
        self.overrides: Dict[str, Any] = spec.get('overrides', {}) or {}
        sweep = spec['sweep']
        self.parameter: str = sweep['parameter']
        self.values: List[Any] = sweep['values']
        self.labels: List[str] = sweep.get('labels') or [
            f'{self.parameter.split(".")[-1]}_{v}' for v in self.values]
        if len(self.labels) != len(self.values):
            raise ValueError('sweep.labels and sweep.values must be the same length')

    def arms(self) -> Iterator[Tuple[str, Any, FluoBPMConfig]]:
        """(label, value, config) per arm, with the base and overrides applied."""
        base = yaml.safe_load(self.base_path.read_text())
        for label, value in zip(self.labels, self.values):
            tree = yaml.safe_load(yaml.safe_dump(base))     # deep copy
            for path, override in self.overrides.items():
                set_dotted(tree, path, override)
            set_dotted(tree, self.parameter, value)
            tree['description'] = (
                f'{self.description}\n\nArm {label}: {self.parameter} = {value}. '
                f'Swept from {self.base_path} by {self.path}.')
            config = FluoBPMConfig(**tree)
            config.config_file = str(self.path)
            yield label, value, config

    def out_dir(self, label: str, log_dir: str = './log') -> Path:
        return Path(log_dir) / self.name / label
