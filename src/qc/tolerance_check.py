"""Explicit guard-band decisions; no nominal/tolerance or validated uncertainty is invented."""
from dataclasses import dataclass

import numpy as np

from src.measurement.uncertainty import UncertaintyResult


@dataclass(frozen=True)
class ToleranceSpec:
    nominal_mm: float
    lower_deviation_mm: float
    upper_deviation_mm: float

    def __post_init__(self):
        values = (self.nominal_mm, self.lower_deviation_mm, self.upper_deviation_mm)
        if any(isinstance(v, bool) or not isinstance(v, (int, float, np.number)) for v in values):
            raise ValueError('Tolerance values must be numbers')
        if (not np.isfinite(values).all() or self.nominal_mm <= 0
                or self.lower_deviation_mm > self.upper_deviation_mm
                or self.nominal_mm + self.lower_deviation_mm < 0
                or not np.isfinite([self.nominal_mm + self.lower_deviation_mm,
                                   self.nominal_mm + self.upper_deviation_mm]).all()):
            raise ValueError('Invalid nominal dimension or signed tolerance deviations')


@dataclass(frozen=True)
class ToleranceResult:
    name: str
    value_mm: float
    nominal_mm: float
    lower_limit_mm: float
    upper_limit_mm: float
    deviation_mm: float
    expanded_uncertainty_mm: float | None
    status: str
    reason: str
    inside_nominal_limits: bool


def check_tolerance(name, value_mm, spec, uncertainty):
    if not isinstance(name, str) or not name:
        raise ValueError('Dimension name must be nonempty')
    if not isinstance(spec, ToleranceSpec) or not isinstance(uncertainty, UncertaintyResult):
        raise ValueError('ToleranceSpec and UncertaintyResult are required')
    if (not np.isfinite(value_mm) or value_mm <= 0 or not np.isfinite(uncertainty.expanded_mm)
            or uncertainty.expanded_mm < 0 or not isinstance(uncertainty.validated, bool)):
        raise ValueError('Invalid dimension or expanded uncertainty')
    lower = float(spec.nominal_mm + spec.lower_deviation_mm)
    upper = float(spec.nominal_mm + spec.upper_deviation_mm)
    interval_low = value_mm - uncertainty.expanded_mm
    interval_high = value_mm + uncertainty.expanded_mm
    if not np.isfinite([interval_low, interval_high]).all():
        raise ValueError('Measurement uncertainty interval overflow')
    if not uncertainty.validated or not uncertainty.available:
        status, reason = 'NOT_VALIDATED', 'Characterize the full uncertainty budget before a conformity decision'
    elif interval_low >= lower and interval_high <= upper:
        status, reason = 'PASS', 'Entire expanded-uncertainty interval lies inside specification limits'
    elif interval_high < lower or interval_low > upper:
        status, reason = 'FAIL', 'Entire expanded-uncertainty interval lies outside specification limits'
    else:
        status, reason = 'INDETERMINATE', 'Expanded-uncertainty interval overlaps a specification limit'
    return ToleranceResult(name, float(value_mm), spec.nominal_mm, lower, upper,
                           float(value_mm - spec.nominal_mm),
                           uncertainty.expanded_mm if uncertainty.available else None, status, reason,
                           bool(lower <= value_mm <= upper))
