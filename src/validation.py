"""Summarize repeated measurement reports against user-provided physical references.

This does not certify the system or automatically mark an uncertainty budget validated.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.measurement.uncertainty import repeatability


def evaluate_reports(reports, references):
    """Pure API: report dictionaries + {dimension: reference_mm} -> summary."""
    reports = list(reports)
    if len(reports) < 2 or not references:
        raise ValueError('Need at least two independent runs and one reference dimension')
    kinds = set()
    values = {name: [] for name in references}
    for report in reports:
        if report.get('version') != 1 or report.get('units') != 'mm' or not report.get('measurement'):
            raise ValueError('Expected version 1 measurement reports in mm')
        measurement = report['measurement']
        kinds.add(measurement['kind'])
        for name in references:
            if name not in measurement['dimensions']:
                raise ValueError(f'Missing measured dimension: {name}')
            values[name].append(float(measurement['dimensions'][name]['value_mm']))
    if len(kinds) != 1:
        raise ValueError('Cannot combine different measurement models')
    summary = {}
    for name, reference in references.items():
        reference = float(reference)
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError('Reference dimensions must be finite and positive')
        sample = np.asarray(values[name])
        stats = repeatability(sample)
        errors = sample - reference
        summary[name] = dict(**stats, reference_mm=reference, bias_mm=float(errors.mean()),
                             rmse_mm=float(np.sqrt(np.mean(errors ** 2))),
                             max_absolute_error_mm=float(np.max(np.abs(errors))))
    return dict(version=1, measurement_kind=kinds.pop(), dimensions=summary,
                note='Use independent captures of a physical standard. Reference uncertainty and systematic '
                     'effects are not inferred here; this report does not validate the entire measurement system.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', nargs='+', required=True, help='measurements.json from independent runs')
    parser.add_argument('--reference', nargs=2, action='append', required=True, metavar=('DIMENSION', 'MM'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    try:
        paths = [Path(p).resolve() for p in args.reports]
        if len(set(paths)) != len(paths):
            raise ValueError('Duplicate report paths are not independent measurements')
        references = {}
        for name, value in args.reference:
            if name in references:
                raise ValueError(f'Duplicate reference dimension: {name}')
            references[name] = float(value)
        result = evaluate_reports([json.loads(p.read_text(encoding='utf-8')) for p in paths], references)
        output = Path(args.output)
        if output.resolve() in paths:
            raise ValueError('Validation output must not overwrite an input report')
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f'Validation error: {exc}\n')
    for name, stats in result['dimensions'].items():
        print(f'{name}: bias={stats["bias_mm"]:+.4f} mm; RMSE={stats["rmse_mm"]:.4f} mm; '
              f'max error={stats["max_absolute_error_mm"]:.4f} mm')
    return result


if __name__ == '__main__':
    main()
