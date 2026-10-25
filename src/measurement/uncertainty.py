"""Conditional fit uncertainty plus a user-characterized external budget.

No board-fit RMSE substitution, automatic accuracy claim or certification.
External terms must be independent standard uncertainties (one sigma).
"""
from dataclasses import dataclass
import math

import numpy as np


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f'{name} must be a finite nonnegative number')
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


@dataclass(frozen=True)
class UncertaintyBudget:
    standard_mm: float = 0.0
    relative_standard: float = 0.0
    coverage_factor: float = 2.0
    validated: bool = False

    def __post_init__(self):
        _nonnegative(self.standard_mm, 'standard_mm')
        _nonnegative(self.relative_standard, 'relative_standard')
        _nonnegative(self.coverage_factor, 'coverage_factor')
        if self.coverage_factor == 0:
            raise ValueError('coverage_factor must be positive')
        if not isinstance(self.validated, bool):
            raise ValueError('validated must be a boolean')
        if self.validated and self.standard_mm == 0 and self.relative_standard == 0:
            raise ValueError('A validated budget must include a characterized external uncertainty term')


@dataclass(frozen=True)
class UncertaintyResult:
    standard_mm: float
    expanded_mm: float
    coverage_factor: float
    validated: bool
    components: dict
    note: str
    available: bool = True


def estimate_uncertainty(value_mm, fit_standard_mm, budget=None):
    _nonnegative(value_mm, 'value_mm')
    _nonnegative(fit_standard_mm, 'fit_standard_mm')
    if value_mm == 0:
        raise ValueError('Measured dimension must be positive')
    budget = budget or UncertaintyBudget()
    if not isinstance(budget, UncertaintyBudget):
        raise ValueError('budget must be an UncertaintyBudget')
    scale = float(value_mm * budget.relative_standard)
    standard = math.hypot(float(fit_standard_mm), float(budget.standard_mm), scale)
    expanded = standard * budget.coverage_factor
    if not np.isfinite([standard, expanded]).all():
        raise ValueError('Uncertainty calculation overflow')
    return UncertaintyResult(standard, expanded, budget.coverage_factor, budget.validated,
                             dict(fit_standard_mm=float(fit_standard_mm), external_standard_mm=budget.standard_mm,
                                  scale_standard_mm=scale),
                             'RSS assumes independent components. Fit covariance assumes iid edge residuals and exact H. '
                             'Coverage factor is user-defined, not an automatic confidence guarantee. '
                             + ('External budget asserted validated by user.' if budget.validated
                                else 'PROVISIONAL: does not quantify all calibration, correlation, bias and plane effects.'))


def repeatability(values_mm):
    values = np.asarray(values_mm, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError('Need >= 2 finite positive measurements from independent runs')
    standard_deviation = float(np.std(values, ddof=1))
    return dict(n=len(values), mean_mm=float(values.mean()), standard_deviation_mm=standard_deviation,
                standard_uncertainty_mean_mm=standard_deviation / np.sqrt(len(values)))
