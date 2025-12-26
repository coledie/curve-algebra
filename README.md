# Curve Algebra

I noticed I keep calculating debt wrong so I made a calculator to help.

A functional, composable Python library for mathematical curves with a focus on financial calculations.

## Core Design Principles

1. **Curves are lazy** - They define computation, not data. Values are computed on demand.
2. **Curves are composable** - Parameters can themselves be curves (nested curves).
3. **Time periods return functional views** - Transform specific portions of curves immutably.
4. **All operations are immutable** - Transforms return new curves, never mutate.

## Installation

In this project directory,

```bash
python -m pip install -e .
```

## Quick Start

```python
from curve import Curve

# Create basic curves
linear = Curve.linear(slope=2, intercept=100)      # 2t + 100
exp = Curve.exponential(base=1000, rate=0.05)      # 1000 * e^(0.05t)
compound = Curve.compound(principal=10000, rate=0.07)  # Compound interest

# Evaluate at any time
value = compound.at(10)  # Value at t=10

# Arithmetic on curves
combined = linear + exp * 2 - 500

# Nested curves (parameters can be curves!)
growing_rate = Curve.linear(0.01, 0.05)  # Rate that grows over time
dynamic_exp = Curve.exponential(1000, growing_rate)
```

## Time Period Operations

The key feature: get functional views into time ranges and transform them.

```python
base = Curve.compound(10000, 0.07)

# Scale values in a specific period
scaled = base.period(5, 15).scale(1.5)  # 1.5x during [5, 15]

# Add a bonus during a period
boosted = base.period(10, 20).add(5000)

# Replace a period with a different curve
replacement = Curve.linear(100, 500)
modified = base.period(5, 10).replace(replacement)

# Apply compound growth to just a period
growing = base.period(0, 12).compound(0.10)  # 10% growth for first year

# Chain operations
result = (base
    .period(0, 6).scale(0.9)       # 90% for first 6 months
    .period(12, 24).add(1000)      # Bonus year 2
    .period(24, float('inf')).compound(0.05))  # Different rate after year 2
```

## Point Operations

Apply discrete changes at specific moments:

```python
curve = Curve.compound(50000, 0.07)

# Add lump sum at a point
with_bonus = curve.at_point(12).add(10000)

# Split curve at a point
main, split_off = curve.at_point(24).split(0.3)  # 30% goes to split_off

# Reset to a value
reset = curve.at_point(36).reset(40000)
```

## Recurrence Relations

For complex scenarios that depend on previous values:

```python
# Loan with monthly payment and compound interest
def loan_step(prev_balance, t, month):
    interest = prev_balance * 0.005  # 0.5% monthly
    payment = 500
    return max(0, prev_balance + interest - payment)

loan = Curve.recurrence(initial=10000, step_fn=loan_step, dt=1.0)
```

## Built-in Curve Types

| Constructor | Formula | Use Case |
|-------------|---------|----------|
| `Curve.constant(v)` | v | Fixed values |
| `Curve.linear(m, b)` | m*t + b | Linear growth/decay |
| `Curve.exponential(base, rate)` | base * e^(rate*t) | Continuous growth |
| `Curve.compound(P, r, n)` | P * (1+r/n)^(nt) | Compound interest |
| `Curve.polynomial(*coeffs)` | c₀ + c₁t + c₂t² + ... | Polynomial curves |
| `Curve.step([(t, v), ...])` | Piecewise constant | Payment schedules |
| `Curve.sine(A, f, φ)` | A * sin(2πft + φ) | Periodic patterns |
| `Curve.from_function(fn)` | fn(t) | Custom functions |
| `Curve.from_points(pts)` | Interpolated | Data-driven curves |
| `Curve.recurrence(init, step)` | Iterative | Complex dependencies |

## Finance Module

Pre-built financial curve constructors:

```python
from finance import Finance, CurveAnalytics

# Loan with fixed payments
loan = Finance.loan(
    principal=200000,
    annual_rate=0.065,
    monthly_payment=1500
)

# Amortizing loan (calculates payment for you)
balance, payment = Finance.amortization(
    principal=200000,
    annual_rate=0.065,
    term_months=360
)

# Investment with contributions
portfolio = Finance.investment(
    initial=10000,
    annual_return=0.08,
    monthly_contribution=500,
    contribution_growth=0.02  # Contributions grow 2%/year
)

# Retirement drawdown with inflation
retirement = Finance.retirement_drawdown(
    portfolio=1000000,
    monthly_withdrawal=4000,
    annual_return=0.05,
    inflation=0.025
)

# Find when loan pays off
payoff_month = CurveAnalytics.find_payoff_time(loan)
```

## Analysis Tools

```python
curve = Curve.linear(-10, 100)

# Find zero crossing
zero = curve.find_zero(0, 20)

# Find when curve reaches target value
target_time = curve.find_value(50, 0, 20)

# Numerical integration
total = curve.integrate_over(0, 10)

# Sample at regular intervals
points = curve.sample(0, 10, steps=100)

# Derivative and integral curves
deriv = curve.derivative()
integral = curve.integral()
```

## Your Use Case Example

"Curve A decreases by fixed amount monthly, remainder compounds"

```python
principal = 10000
monthly_decrease = 200
monthly_rate = 0.005

def decrease_then_compound(prev, t, month):
    after_decrease = prev - monthly_decrease
    if after_decrease <= 0:
        return 0
    return after_decrease * (1 + monthly_rate)

curve_a = Curve.recurrence(principal, decrease_then_compound, dt=1.0)

# Split at month 12, with split going to different rate
balance_at_12 = curve_a.at(12)
split_amount = balance_at_12 * 0.3

# Split-off curve compounds at 8% annual
split_curve = Curve.compound(split_amount, 0.08/12).delay(12)
```

## API Reference

### Curve Methods

**Evaluation:**
- `curve.at(t)` - Get value at time t
- `curve(t)` - Same as `.at(t)`
- `curve.sample(t1, t2, steps)` - Sample at regular intervals

**Arithmetic:** `+`, `-`, `*`, `/`, `**`, `-curve`, `abs(curve)`

**Time Periods:**
- `curve.period(t1, t2)` - Get view of [t1, t2)
- `curve.before(t)` - Get view of (-∞, t)
- `curve.after(t)` - Get view of [t, ∞)

**CurvePeriod Methods:**
- `.scale(factor)` - Multiply values in period
- `.add(amount)` - Add to values in period
- `.replace(new_curve)` - Replace with different curve
- `.transform(fn)` - Apply custom function
- `.compound(rate)` - Apply compound growth
- `.decay(rate)` - Apply exponential decay

**Point Operations:**
- `curve.at_point(t).add(amount)` - One-time addition
- `curve.at_point(t).multiply(factor)` - Scale from point onward
- `curve.at_point(t).split(fraction)` - Split into two curves
- `curve.at_point(t).reset(value)` - Reset to value

**Transforms:**
- `curve.map(fn)` - Apply fn(value, t) → new_value
- `curve.clamp(min, max)` - Clamp values
- `curve.delay(offset)` - Shift forward in time
- `curve.scale_time(factor)` - Stretch/compress time
- `curve.derivative()` - Numerical derivative
- `curve.integral()` - Numerical integral

**Analysis:**
- `curve.find_zero(t1, t2)` - Find zero crossing
- `curve.find_value(target, t1, t2)` - Find when curve = target
- `curve.integrate_over(t1, t2)` - Numerical integration
- `curve.sum_over(t1, t2, dt)` - Sum discrete samples

## License

MIT
