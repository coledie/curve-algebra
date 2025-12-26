"""
Curve Algebra - A functional, composable system for mathematical curves.

Core design principles:
1. Curves are lazy - they define computation, not data
2. Curves are composable - parameters can be curves themselves
3. Time periods return functional views for transformation
4. All operations are immutable - transforms return new curves
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Union, Optional, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
import math


# Type alias for things that can be evaluated to a number at time t
Evaluable = Union[float, int, 'Curve']


def evaluate(x: Evaluable, t: float) -> float:
    """Evaluate something that might be a curve or a constant."""
    if isinstance(x, Curve):
        return x.at(t)
    return float(x)


class Curve(ABC):
    """
    Base class for all curves. A curve is a function from time to value.
    
    Curves support:
    - Arithmetic: +, -, *, /, **, negation
    - Composition: curves as parameters to other curves
    - Time slicing: get functional views of time periods
    - Evaluation: get concrete values at specific times
    """
    
    @abstractmethod
    def at(self, t: float) -> float:
        """Evaluate the curve at time t."""
        pass
    
    def __call__(self, t: float) -> float:
        """Allow curve(t) syntax."""
        return self.at(t)
    
    # ==================== Arithmetic Operations ====================
    
    def __add__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(self, other, lambda a, b: a + b, '+')
    
    def __radd__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(other, self, lambda a, b: a + b, '+')
    
    def __sub__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(self, other, lambda a, b: a - b, '-')
    
    def __rsub__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(other, self, lambda a, b: a - b, '-')
    
    def __mul__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(self, other, lambda a, b: a * b, '*')
    
    def __rmul__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(other, self, lambda a, b: a * b, '*')
    
    def __truediv__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(self, other, lambda a, b: a / b if b != 0 else float('inf'), '/')
    
    def __rtruediv__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(other, self, lambda a, b: a / b if b != 0 else float('inf'), '/')
    
    def __pow__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(self, other, lambda a, b: a ** b, '**')
    
    def __rpow__(self, other: Evaluable) -> Curve:
        return ArithmeticCurve(other, self, lambda a, b: a ** b, '**')
    
    def __neg__(self) -> Curve:
        return TransformCurve(self, lambda v, t: -v, 'neg')
    
    def __abs__(self) -> Curve:
        return TransformCurve(self, lambda v, t: abs(v), 'abs')
    
    # ==================== Time Period Operations ====================
    
    def period(self, t1: float, t2: float) -> CurvePeriod:
        """
        Get a functional view of this curve over [t1, t2].
        
        The returned CurvePeriod allows you to apply transformations
        that only affect this time range.
        
        Example:
            curve.period(0, 12).scale(1.5)  # Scale by 1.5x for first year
            curve.period(6, 18).add(Curve.constant(100))  # Add bonus mid-period
        """
        return CurvePeriod(self, t1, t2)
        
    def at_point(self, t: float) -> PointOperation:
        """
        Get a point operation view for applying discrete changes at time t.
        
        Example:
            curve.at_point(12).add(1000)  # One-time bonus at t=12
            curve.at_point(6).split(0.2, other_curve)  # Split 20% to other curve
        """
        return PointOperation(self, t)
    
    # ==================== Functional Transforms ====================
    
    def map(self, fn: Callable[[float, float], float]) -> Curve:
        """
        Apply a function to transform this curve.
        
        fn(value, t) -> new_value
        
        Example:
            curve.map(lambda v, t: max(v, 0))  # Floor at zero
            curve.map(lambda v, t: v * (1 + 0.01 * t))  # Time-dependent scaling
        """
        return TransformCurve(self, fn, f'map({fn.__name__ if hasattr(fn, "__name__") else "λ"})')
    
    # ==================== Evaluation Helpers ====================
    
    def sample(self, t_start: float, t_end: float, steps: int = 100) -> List[Tuple[float, float]]:
        """Sample the curve at regular intervals."""
        dt = (t_end - t_start) / steps
        return [(t_start + i * dt, self.at(t_start + i * dt)) for i in range(steps + 1)]
    
    def sum_over(self, t_start: float, t_end: float, dt: float = 1.0) -> float:
        """Sum discrete samples over a period (useful for periodic payments)."""
        total = 0.0
        t = t_start
        while t <= t_end:
            total += self.at(t)
            t += dt
        return total
    
    def integrate_over(self, t_start: float, t_end: float, steps: int = 1000) -> float:
        """Numerical integration over a period."""
        dt = (t_end - t_start) / steps
        total = 0.0
        for i in range(steps):
            t = t_start + i * dt
            total += self.at(t) * dt
        return total
    
    def find_zero(self, t_start: float, t_end: float, tolerance: float = 1e-6) -> Optional[float]:
        """Find where the curve crosses zero using bisection."""
        a, b = t_start, t_end
        fa, fb = self.at(a), self.at(b)
        
        if fa * fb > 0:
            return None  # No sign change
        
        while b - a > tolerance:
            mid = (a + b) / 2
            fmid = self.at(mid)
            if fa * fmid <= 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
        
        return (a + b) / 2
    
    def find_value(self, target: float, t_start: float, t_end: float, tolerance: float = 1e-6) -> Optional[float]:
        """Find where the curve equals target."""
        shifted = self - target
        return shifted.find_zero(t_start, t_end, tolerance)
    
    # ==================== Static Factory Methods ====================
    
    @staticmethod
    def constant(value: Evaluable) -> Curve:
        """Create a constant curve: f(t) = value"""
        return ConstantCurve(value)
    
    @staticmethod
    def linear(slope: Evaluable, intercept: Evaluable = 0) -> Curve:
        """Create a linear curve: f(t) = slope * t + intercept"""
        return LinearCurve(slope, intercept)
    
    @staticmethod
    def exponential(base: Evaluable, rate: Evaluable) -> Curve:
        """Create an exponential curve: f(t) = base * e^(rate * t)"""
        return ExponentialCurve(base, rate)
    
    @staticmethod
    def compound(principal: Evaluable, rate: Evaluable, n: int = 1) -> Curve:
        """Create a compound interest curve: f(t) = principal * (1 + rate/n)^(n*t)"""
        return CompoundCurve(principal, rate, n)
    
    @staticmethod
    def power(base: Evaluable, exponent: Evaluable) -> Curve:
        """Create a power curve: f(t) = base * t^exponent"""
        return PowerCurve(base, exponent)
    
    @staticmethod
    def polynomial(coefficients: List[Evaluable]) -> Curve:
        """Create a polynomial curve from coefficients [c0, c1, c2, ...]: f(t) = c0 + c1*t + c2*t^2 + ..."""
        return PolynomialCurve(coefficients)
    
    @staticmethod
    def step(transitions: List[Tuple[float, Evaluable]]) -> Curve:
        """Create a step curve with transitions at specific times."""
        return StepCurve(transitions)
    
    @staticmethod
    def periodic(inner: Curve, period: float) -> Curve:
        """Create a periodic curve that repeats every 'period' time units."""
        return PeriodicCurve(inner, period)
    
    @staticmethod
    def sine(amplitude: Evaluable = 1, frequency: Evaluable = 1, phase: Evaluable = 0, offset: Evaluable = 0) -> Curve:
        """Create a sine curve: f(t) = amplitude * sin(2π * frequency * t + phase) + offset"""
        return SineCurve(amplitude, frequency, phase, offset)
    
    @staticmethod
    def piecewise(pieces: List[Tuple[float, float, Curve]]) -> Curve:
        """Create a piecewise curve from [(t_start, t_end, curve), ...]"""
        return PiecewiseCurve(pieces)
    
    @staticmethod
    def function(fn: Callable[[float], float]) -> Curve:
        """Create a curve from a Python function."""
        return FunctionCurve(fn)
    
    @staticmethod
    def interpolate(points: List[Tuple[float, float]], method: str = 'linear') -> Curve:
        """Create an interpolated curve from data points."""
        return InterpolatedCurve(points, method)
    
    @staticmethod
    def recurrence(initial: float, step_fn: Callable[[float, float, int], float], dt: float = 1.0) -> Curve:
        """
        Create a curve defined by a recurrence relation.
        
        Args:
            initial: Initial value at t=0
            step_fn: Function (prev_value, current_time, step_index) -> new_value
            dt: Time step between iterations
        
        Example:
            # Loan balance
            def loan_step(prev, t, i):
                return prev * 1.005 - 1000  # 0.5% interest, $1000 payment
            loan = Curve.recurrence(10000, loan_step, dt=1.0)
        """
        return RecurrenceCurve(initial, step_fn, dt)


class ConstantCurve(Curve):
    def __init__(self, value: Evaluable):
        self.value = value
    
    def at(self, t: float) -> float:
        return evaluate(self.value, t)
    
    def __repr__(self):
        return f'Constant({self.value})'


class LinearCurve(Curve):
    def __init__(self, slope: Evaluable, intercept: Evaluable = 0):
        self.slope = slope
        self.intercept = intercept
    
    def at(self, t: float) -> float:
        m = evaluate(self.slope, t)
        b = evaluate(self.intercept, t)
        return m * t + b
    
    def __repr__(self):
        return f'Linear({self.slope}*t + {self.intercept})'


class ExponentialCurve(Curve):
    def __init__(self, base: Evaluable, rate: Evaluable):
        self.base = base
        self.rate = rate
    
    def at(self, t: float) -> float:
        b = evaluate(self.base, t)
        r = evaluate(self.rate, t)
        return b * math.exp(r * t)
    
    def __repr__(self):
        return f'Exponential({self.base} * e^({self.rate}*t))'


class CompoundCurve(Curve):
    def __init__(self, principal: Evaluable, rate: Evaluable, periods_per_unit: float = 1):
        self.principal = principal
        self.rate = rate
        self.n = periods_per_unit
    
    def at(self, t: float) -> float:
        p = evaluate(self.principal, t)
        r = evaluate(self.rate, t)
        return p * (1 + r / self.n) ** (self.n * t)
    
    def __repr__(self):
        return f'Compound({self.principal}, {self.rate}, n={self.n})'


class PowerCurve(Curve):
    def __init__(self, base: Evaluable, exponent: Evaluable):
        self.base = base
        self.exponent = exponent
    
    def at(self, t: float) -> float:
        b = evaluate(self.base, t)
        e = evaluate(self.exponent, t)
        return b ** (e * t) if isinstance(self.exponent, Curve) else b ** e
    
    def __repr__(self):
        return f'Power({self.base}^{self.exponent})'


class PolynomialCurve(Curve):
    def __init__(self, coefficients: List[Evaluable]):
        self.coefficients = coefficients
    
    def at(self, t: float) -> float:
        result = 0.0
        for i, c in enumerate(self.coefficients):
            result += evaluate(c, t) * (t ** i)
        return result
    
    def __repr__(self):
        return f'Polynomial({self.coefficients})'


class StepCurve(Curve):
    def __init__(self, values: List[Tuple[float, Evaluable]]):
        # Sort by time
        self.values = sorted(values, key=lambda x: x[0])
    
    def at(self, t: float) -> float:
        result = 0.0
        for breakpoint, value in self.values:
            if t >= breakpoint:
                result = evaluate(value, t)
            else:
                break
        return result
    
    def __repr__(self):
        return f'Step({self.values})'


class PeriodicCurve(Curve):
    """Returns amount at each period interval, 0 otherwise."""
    def __init__(self, amount: Evaluable, period: float, phase: float = 0):
        self.amount = amount
        self.period = period
        self.phase = phase
    
    def at(self, t: float) -> float:
        adjusted = t - self.phase
        if adjusted >= 0 and abs(adjusted % self.period) < 1e-9:
            return evaluate(self.amount, t)
        return 0.0
    
    def __repr__(self):
        return f'Periodic({self.amount} every {self.period})'


class SineCurve(Curve):
    def __init__(self, amplitude: Evaluable = 1, frequency: Evaluable = 1, phase: Evaluable = 0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
    
    def at(self, t: float) -> float:
        a = evaluate(self.amplitude, t)
        f = evaluate(self.frequency, t)
        p = evaluate(self.phase, t)
        return a * math.sin(2 * math.pi * f * t + p)
    
    def __repr__(self):
        return f'Sine(A={self.amplitude}, f={self.frequency}, φ={self.phase})'


class PiecewiseCurve(Curve):
    def __init__(self, pieces: List[Tuple[float, float, Curve]]):
        self.pieces = sorted(pieces, key=lambda x: x[0])
    
    def at(self, t: float) -> float:
        for t_start, t_end, curve in self.pieces:
            if t_start <= t < t_end:
                return curve.at(t)
        return 0.0
    
    def __repr__(self):
        return f'Piecewise({len(self.pieces)} pieces)'


class FunctionCurve(Curve):
    def __init__(self, fn: Callable[[float], float], name: str = 'custom'):
        self.fn = fn
        self.name = name
    
    def at(self, t: float) -> float:
        return self.fn(t)
    
    def __repr__(self):
        return f'Function({self.name})'


class InterpolatedCurve(Curve):
    def __init__(self, points: List[Tuple[float, float]], interpolation: str = 'linear'):
        self.points = sorted(points, key=lambda x: x[0])
        self.interpolation = interpolation
    
    def at(self, t: float) -> float:
        if not self.points:
            return 0.0
        
        # Before first point
        if t <= self.points[0][0]:
            return self.points[0][1]
        
        # After last point
        if t >= self.points[-1][0]:
            return self.points[-1][1]
        
        # Find surrounding points
        for i in range(len(self.points) - 1):
            t0, v0 = self.points[i]
            t1, v1 = self.points[i + 1]
            if t0 <= t <= t1:
                if self.interpolation == 'step':
                    return v0
                else:  # linear
                    alpha = (t - t0) / (t1 - t0)
                    return v0 + alpha * (v1 - v0)
        
        return 0.0
    
    def __repr__(self):
        return f'Interpolated({len(self.points)} points, {self.interpolation})'


class RecurrenceCurve(Curve):
    """A curve defined by a recurrence relation with memoization."""
    def __init__(self, initial: float, step_fn: Callable[[float, float, int], float], dt: float = 1.0):
        self.initial = initial
        self.step_fn = step_fn
        self.dt = dt
        self._cache: dict[int, float] = {0: initial}
    
    def at(self, t: float) -> float:
        if t < 0:
            return self.initial
        
        step = int(t / self.dt)
        
        # Build up cache if needed
        if step not in self._cache:
            # Find highest cached step
            max_cached = max(self._cache.keys())
            for i in range(max_cached + 1, step + 1):
                prev = self._cache[i - 1]
                self._cache[i] = self.step_fn(prev, i * self.dt, i)
        
        # Interpolate if t is between steps
        step_t = step * self.dt
        if abs(t - step_t) < 1e-9:
            return self._cache[step]
        else:
            # Linear interpolation between steps
            if step + 1 not in self._cache:
                prev = self._cache[step]
                self._cache[step + 1] = self.step_fn(prev, (step + 1) * self.dt, step + 1)
            
            alpha = (t - step_t) / self.dt
            return self._cache[step] + alpha * (self._cache[step + 1] - self._cache[step])
    
    def __repr__(self):
        return f'Recurrence(initial={self.initial}, dt={self.dt})'


# ==================== Operation Curves ====================


class ArithmeticCurve(Curve):
    def __init__(self, left: Evaluable, right: Evaluable, op: Callable[[float, float], float], op_name: str):
        self.left = left
        self.right = right
        self.op = op
        self.op_name = op_name
    
    def at(self, t: float) -> float:
        l = evaluate(self.left, t)
        r = evaluate(self.right, t)
        return self.op(l, r)
    
    def __repr__(self):
        return f'({self.left} {self.op_name} {self.right})'


class TransformCurve(Curve):
    def __init__(self, source: Curve, fn: Callable[[float, float], float], name: str):
        self.source = source
        self.fn = fn
        self.name = name
    
    def at(self, t: float) -> float:
        return self.fn(self.source.at(t), t)
    
    def __repr__(self):
        return f'{self.name}({self.source})'


class TimeShiftCurve(Curve):
    def __init__(self, source: Curve, offset: float):
        self.source = source
        self.offset = offset
    
    def at(self, t: float) -> float:
        return self.source.at(t - self.offset)
    
    def __repr__(self):
        return f'Delay({self.source}, {self.offset})'


class TimeScaleCurve(Curve):
    def __init__(self, source: Curve, factor: float):
        self.source = source
        self.factor = factor
    
    def at(self, t: float) -> float:
        return self.source.at(t / self.factor)
    
    def __repr__(self):
        return f'TimeScale({self.source}, {self.factor})'


class DerivativeCurve(Curve):
    def __init__(self, source: Curve, dt: float = 0.0001):
        self.source = source
        self.dt = dt
    
    def at(self, t: float) -> float:
        return (self.source.at(t + self.dt) - self.source.at(t - self.dt)) / (2 * self.dt)
    
    def __repr__(self):
        return f'd/dt({self.source})'


class IntegralCurve(Curve):
    def __init__(self, source: Curve, t0: float = 0, initial: float = 0):
        self.source = source
        self.t0 = t0
        self.initial = initial
    
    def at(self, t: float) -> float:
        # Simple trapezoidal integration
        steps = max(100, int(abs(t - self.t0) * 10))
        if steps == 0:
            return self.initial
        
        dt = (t - self.t0) / steps
        total = self.initial
        for i in range(steps):
            t_i = self.t0 + i * dt
            total += (self.source.at(t_i) + self.source.at(t_i + dt)) / 2 * dt
        return total
    
    def __repr__(self):
        return f'∫({self.source})'


# ==================== Time Period Operations ====================


class CurvePeriod:
    """
    A functional view into a specific time period of a curve.
    
    Operations on a CurvePeriod return new Curves where the transformation
    is only applied within the specified time range.
    """
    
    def __init__(self, source: Curve, t1: float, t2: float):
        self.source = source
        self.t1 = t1
        self.t2 = t2
    
    def _in_range(self, t: float) -> bool:
        return self.t1 <= t < self.t2
    
    def transform(self, fn: Callable[[float, float], float]) -> Curve:
        """Apply a transformation function only within this period."""
        return PeriodTransformCurve(self.source, self.t1, self.t2, fn)
    
    def scale(self, factor: Evaluable) -> Curve:
        """Scale values in this period by factor."""
        return self.transform(lambda v, t: v * evaluate(factor, t))
    
    def add(self, amount: Evaluable) -> Curve:
        """Add amount to values in this period."""
        return self.transform(lambda v, t: v + evaluate(amount, t))
    
    def replace(self, new_curve: Curve) -> Curve:
        """Replace this period with a different curve entirely."""
        return PeriodReplaceCurve(self.source, self.t1, self.t2, new_curve)
    
    def multiply(self, curve: Curve) -> Curve:
        """Multiply by another curve in this period."""
        return self.transform(lambda v, t: v * curve.at(t))
    
    def apply(self, curve_fn: Callable[[Curve], Curve]) -> Curve:
        """
        Apply a curve transformation function to just this period.
        
        Example:
            curve.period(0, 12).apply(lambda c: c ** 2)
        """
        transformed = curve_fn(self.source)
        return PeriodReplaceCurve(self.source, self.t1, self.t2, transformed)
    
    def compound(self, rate: Evaluable) -> Curve:
        """Apply compound growth within this period."""
        def compounder(v: float, t: float) -> float:
            r = evaluate(rate, t)
            t_in_period = t - self.t1
            return v * (1 + r) ** t_in_period
        return self.transform(compounder)
    
    def decay(self, rate: Evaluable) -> Curve:
        """Apply exponential decay within this period."""
        def decayer(v: float, t: float) -> float:
            r = evaluate(rate, t)
            t_in_period = t - self.t1
            return v * math.exp(-r * t_in_period)
        return self.transform(decayer)
    
    def __repr__(self):
        return f'Period[{self.t1}, {self.t2}] of {self.source}'


class PeriodTransformCurve(Curve):
    """A curve with a transformation applied only within a time period."""
    
    def __init__(self, source: Curve, t1: float, t2: float, fn: Callable[[float, float], float]):
        self.source = source
        self.t1 = t1
        self.t2 = t2
        self.fn = fn
    
    def at(self, t: float) -> float:
        v = self.source.at(t)
        if self.t1 <= t < self.t2:
            return self.fn(v, t)
        return v
    
    def __repr__(self):
        return f'PeriodTransform[{self.t1}, {self.t2}]({self.source})'


class PeriodReplaceCurve(Curve):
    """A curve with a time period replaced by another curve."""
    
    def __init__(self, source: Curve, t1: float, t2: float, replacement: Curve):
        self.source = source
        self.t1 = t1
        self.t2 = t2
        self.replacement = replacement
    
    def at(self, t: float) -> float:
        if self.t1 <= t < self.t2:
            return self.replacement.at(t)
        return self.source.at(t)
    
    def __repr__(self):
        return f'PeriodReplace[{self.t1}, {self.t2}]({self.source} → {self.replacement})'


# ==================== Point Operations ====================


class PointOperation:
    """Operations at a specific point in time."""
    
    def __init__(self, source: Curve, t: float):
        self.source = source
        self.t = t
    
    def add(self, amount: Evaluable) -> Curve:
        """Add a discrete amount at this point."""
        return PointAddCurve(self.source, self.t, amount)
    
    def multiply(self, factor: Evaluable) -> Curve:
        """Multiply by factor at this point (affects all future values)."""
        return PointMultiplyCurve(self.source, self.t, factor)
    
    def split(self, fraction: float) -> Tuple[Curve, Curve]:
        """
        Split the curve at this point.
        Returns (main_curve * (1-fraction), split_off_curve * fraction)
        """
        main = PointMultiplyCurve(self.source, self.t, 1 - fraction)
        split_off = PointSplitCurve(self.source, self.t, fraction)
        return main, split_off
    
    def reset(self, value: Evaluable) -> Curve:
        """Reset the curve to a specific value at this point."""
        return PointResetCurve(self.source, self.t, value)
    
    def __repr__(self):
        return f'PointOp@{self.t}({self.source})'


class PointAddCurve(Curve):
    """A curve with a discrete addition at a specific point."""
    
    def __init__(self, source: Curve, t_point: float, amount: Evaluable):
        self.source = source
        self.t_point = t_point
        self.amount = amount
        self._added = False
    
    def at(self, t: float) -> float:
        v = self.source.at(t)
        if t >= self.t_point:
            v += evaluate(self.amount, t)
        return v
    
    def __repr__(self):
        return f'PointAdd@{self.t_point}({self.source}, +{self.amount})'


class PointMultiplyCurve(Curve):
    """A curve multiplied by a factor from a specific point onward."""
    
    def __init__(self, source: Curve, t_point: float, factor: Evaluable):
        self.source = source
        self.t_point = t_point
        self.factor = factor
    
    def at(self, t: float) -> float:
        v = self.source.at(t)
        if t >= self.t_point:
            v *= evaluate(self.factor, t)
        return v
    
    def __repr__(self):
        return f'PointMultiply@{self.t_point}({self.source}, *{self.factor})'


class PointSplitCurve(Curve):
    """The split-off portion of a curve from a point."""
    
    def __init__(self, source: Curve, t_point: float, fraction: float):
        self.source = source
        self.t_point = t_point
        self.fraction = fraction
    
    def at(self, t: float) -> float:
        if t < self.t_point:
            return 0.0
        return self.source.at(t) * self.fraction
    
    def __repr__(self):
        return f'Split@{self.t_point}({self.source}, {self.fraction})'


class PointResetCurve(Curve):
    """A curve that resets to a value at a specific point."""
    
    def __init__(self, source: Curve, t_point: float, value: Evaluable):
        self.source = source
        self.t_point = t_point
        self.value = value
    
    def at(self, t: float) -> float:
        if t < self.t_point:
            return self.source.at(t)
        return evaluate(self.value, t)
    
    def __repr__(self):
        return f'Reset@{self.t_point}({self.source} → {self.value})'
