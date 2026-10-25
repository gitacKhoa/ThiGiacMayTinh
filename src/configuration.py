"""Versioned JSON inspection settings shared by CLI and future UI."""

from dataclasses import asdict
import json
from pathlib import Path

from src.inspection import InspectionConfig
from src.measurement.measure import MeasurementOptions
from src.measurement.uncertainty import UncertaintyBudget
from src.qc.tolerance_check import ToleranceSpec


def config_to_dict(config):
    config.validate()
    return {'version': 1, **asdict(config)}


def config_from_dict(data):
    allowed = {'version', 'roi', 'segmentation', 'edges', 'measurement', 'uncertainty', 'tolerances'}
    if not isinstance(data, dict) or data.get('version') != 1:
        raise ValueError('Inspection configuration must have version=1')
    if set(data) - allowed:
        raise ValueError(f'Unknown configuration fields: {sorted(set(data) - allowed)}')
    try:
        config = InspectionConfig(
            roi=tuple(data['roi']) if data.get('roi') is not None else None,
            segmentation=dict(data.get('segmentation', {})), edges=dict(data.get('edges', {})),
            measurement=MeasurementOptions(**data.get('measurement', {})),
            uncertainty=UncertaintyBudget(**data.get('uncertainty', {})),
            tolerances={name: ToleranceSpec(**spec) for name, spec in data.get('tolerances', {}).items()},
        )
    except (TypeError, AttributeError) as exc:
        raise ValueError(f'Invalid inspection configuration: {exc}') from exc
    config.validate()
    return config


def load_config(path):
    try:
        return config_from_dict(json.loads(Path(path).read_text(encoding='utf-8-sig')))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Invalid configuration JSON: {path}') from exc


def save_config(path, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_to_dict(config), indent=2, allow_nan=False), encoding='utf-8')
