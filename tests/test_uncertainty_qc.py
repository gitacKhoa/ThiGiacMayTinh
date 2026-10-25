import math
import unittest

import numpy as np

from src.measurement.uncertainty import UncertaintyBudget, estimate_uncertainty, repeatability
from src.qc.tolerance_check import ToleranceSpec, check_tolerance
from src.validation import evaluate_reports


class UncertaintyQCTests(unittest.TestCase):
    def test_rss_and_repeatability(self):
        result = estimate_uncertainty(100, .03, UncertaintyBudget(.04, .001, 2, True))
        self.assertAlmostEqual(result.standard_mm, math.sqrt(.03**2 + .04**2 + .1**2))
        self.assertAlmostEqual(result.expanded_mm, 2 * result.standard_mm)
        self.assertFalse(estimate_uncertainty(100, .03).validated)
        stats = repeatability([10., 11., 12.])
        self.assertEqual(stats['n'], 3)
        self.assertEqual(stats['standard_deviation_mm'], 1)
        self.assertAlmostEqual(stats['standard_uncertainty_mean_mm'], 1 / np.sqrt(3))

    def test_qc_guardband_and_asymmetric_limits(self):
        spec = ToleranceSpec(10, -.25, .5)
        for value, expected in ((10.0, 'PASS'), (10.75, 'FAIL'), (10.5, 'INDETERMINATE'),
                                (9.75, 'INDETERMINATE')):
            u = estimate_uncertainty(value, 0, UncertaintyBudget(.05, 0, 2, True))
            with self.subTest(value=value):
                result = check_tolerance('diameter', value, spec, u)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.lower_limit_mm, 9.75)
                self.assertEqual(result.upper_limit_mm, 10.5)
        u = estimate_uncertainty(15, .01)
        self.assertEqual(check_tolerance('diameter', 15, spec, u).status, 'NOT_VALIDATED')

    def test_invalid_inputs(self):
        for options in ({'standard_mm': -1}, {'relative_standard': float('nan')},
                        {'coverage_factor': 0}, {'validated': 'yes'}, {'validated': True}):
            with self.assertRaises(ValueError):
                UncertaintyBudget(**options)
        for values in ([1], [1, float('nan')], [1, -2]):
            with self.assertRaises(ValueError):
                repeatability(values)
        for values in ((0, -.1, .1), (10, 1, -1), (10, -11, 1), (10, 0, float('inf'))):
            with self.assertRaises(ValueError):
                ToleranceSpec(*values)
        with self.assertRaises(ValueError):
            estimate_uncertainty(-1, .01)

    def test_physical_reference_summary(self):
        reports = [dict(version=1, units='mm', measurement=dict(kind='rectangle',
                   dimensions={'length': {'value_mm': value}})) for value in (80.1, 80.2, 80.3)]
        result = evaluate_reports(reports, {'length': 80})['dimensions']['length']
        self.assertAlmostEqual(result['bias_mm'], .2)
        self.assertAlmostEqual(result['standard_deviation_mm'], .1)
        self.assertAlmostEqual(result['max_absolute_error_mm'], .3)
        with self.assertRaises(ValueError):
            evaluate_reports(reports[:1], {'length': 80})


if __name__ == '__main__':
    unittest.main()
