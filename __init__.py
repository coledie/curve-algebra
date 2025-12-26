"""
Curve Algebra - A functional, composable system for mathematical curves.

Usage:
    from curve_algebra import Curve, Finance, CurveAnalytics

    # Create curves
    a = Curve.linear(2, 100)
    b = Curve.exponential(1000, 0.05)
    
    # Arithmetic
    c = a + b * 2
    
    # Time period operations
    d = c.period(0, 12).scale(1.5)
    
    # Point operations
    e = d.at_point(6).add(500)
    
    # Evaluate
    value = e.at(10)
    
    # Finance helpers
    loan = Finance.loan(200000, 0.065, 1500)
"""

from curve import (
    Curve,
    CurvePeriod,
    PointOperation,
    Evaluable,
    evaluate,
)

from finance import (
    Finance,
    CurveAnalytics,
    monthly_payment,
    compound_growth,
    present_value,
    future_value,
)

__all__ = [
    'Curve',
    'CurvePeriod', 
    'PointOperation',
    'Evaluable',
    'evaluate',
    'Finance',
    'CurveAnalytics',
    'monthly_payment',
    'compound_growth',
    'present_value',
    'future_value',
]

__version__ = '0.1.0'
