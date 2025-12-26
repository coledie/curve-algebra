"""
Curve Algebra Examples - Core Curve Operations

This file demonstrates the power and flexibility of the curve algebra system
for general mathematical operations.
"""

from curve import (Curve, ConstantCurve, LinearCurve, ExponentialCurve, 
                   CompoundCurve, PolynomialCurve, SineCurve, 
                   RecurrenceCurve, DerivativeCurve)


def example_basic_curves():
    """Basic curve creation and evaluation."""
    print("=" * 60)
    print("BASIC CURVES")
    print("=" * 60)
    
    # Constant curve
    c = ConstantCurve(100)
    print(f"Constant(100) at t=5: {c.at(5)}")
    
    # Linear curve: y = 2t + 10
    linear = LinearCurve(slope=2, intercept=10)
    print(f"Linear(2t + 10) at t=5: {linear.at(5)}")  # 20
    
    # Exponential: 100 * e^(0.05t)
    exp = ExponentialCurve(base=100, rate=0.05)
    print(f"Exponential(100, 0.05) at t=10: {exp.at(10):.2f}")
    
    # Compound interest: 1000 * (1 + 0.05)^t
    compound = CompoundCurve(principal=1000, rate=0.05)
    print(f"Compound(1000, 5%) at t=10: {compound.at(10):.2f}")
    print()


def example_nested_curves():
    """Curves as parameters to other curves."""
    print("=" * 60)
    print("NESTED CURVES (Curves as Parameters)")
    print("=" * 60)
    
    # Exponential with linearly growing rate
    # base * e^(rate(t) * t) where rate(t) = 0.01 * t
    growing_rate = LinearCurve(slope=0.01, intercept=0)
    exp_growing = ExponentialCurve(base=100, rate=growing_rate)
    
    print("Exponential with linearly growing rate:")
    for t in [0, 5, 10, 15, 20]:
        rate_at_t = growing_rate.at(t)
        value = exp_growing.at(t)
        print(f"  t={t}: rate={rate_at_t:.2f}, value={value:.2f}")
    
    # Polynomial with curve coefficients
    # c0(t) + c1(t)*t + c2(t)*t^2
    # where c0 = 10, c1 = Linear(0.1, 1), c2 = 0.5
    c1 = LinearCurve(0.1, 1)  # Coefficient grows over time
    poly = PolynomialCurve([10, c1, 0.5])
    
    print("\nPolynomial with dynamic coefficient c1:")
    for t in [0, 2, 4, 6]:
        c1_val = c1.at(t)
        poly_val = poly.at(t)
        print(f"  t={t}: c1={c1_val:.2f}, polynomial={poly_val:.2f}")
    print()


def example_arithmetic():
    """Arithmetic operations on curves."""
    print("=" * 60)
    print("ARITHMETIC ON CURVES")
    print("=" * 60)
    
    a = LinearCurve(2, 100)      # 2t + 100
    b = ExponentialCurve(50, 0.1)  # 50 * e^(0.1t)
    
    # Add curves
    sum_curve = a + b
    print(f"(2t + 100) + (50*e^0.1t) at t=5: {sum_curve.at(5):.2f}")
    
    # Subtract
    diff = a - b
    print(f"(2t + 100) - (50*e^0.1t) at t=5: {diff.at(5):.2f}")
    
    # Multiply
    product = a * b
    print(f"(2t + 100) * (50*e^0.1t) at t=5: {product.at(5):.2f}")
    
    # Divide
    quotient = a / b
    print(f"(2t + 100) / (50*e^0.1t) at t=5: {quotient.at(5):.2f}")
    
    # Power
    powered = a ** 2
    print(f"(2t + 100)^2 at t=5: {powered.at(5):.2f}")
    
    # Mix with scalars
    scaled = a * 1.5 + 200
    print(f"1.5*(2t + 100) + 200 at t=5: {scaled.at(5):.2f}")
    print()


def example_time_periods():
    """Time period operations - the key feature!"""
    print("=" * 60)
    print("TIME PERIOD OPERATIONS")
    print("=" * 60)
    
    # Start with a linear curve
    base = LinearCurve(10, 1000)  # 10t + 1000
    
    print("Base curve (10t + 1000):")
    for t in [0, 5, 10, 15, 20]:
        print(f"  t={t}: {base.at(t):.2f}")
    
    # Scale the period [5, 15] by 2x
    scaled = base.period(5, 15).scale(2)
    
    print("\nAfter scaling [5, 15] by 2x:")
    for t in [0, 5, 10, 15, 20]:
        print(f"  t={t}: {scaled.at(t):.2f}")
    
    # Add 500 during period [10, 20]
    boosted = base.period(10, 20).add(500)
    
    print("\nAfter adding 500 during [10, 20]:")
    for t in [0, 5, 10, 15, 20]:
        print(f"  t={t}: {boosted.at(t):.2f}")
    
    # Replace a period with a different curve entirely
    replacement = ConstantCurve(2000)
    replaced = base.period(8, 12).replace(replacement)
    
    print("\nAfter replacing [8, 12] with constant 2000:")
    for t in [0, 5, 8, 10, 12, 15]:
        print(f"  t={t}: {replaced.at(t):.2f}")
    
    # Apply compound growth to a specific period
    compounding = base.period(5, 15).compound(0.1)  # 10% growth
    
    print("\nAfter applying 10% compound growth during [5, 15]:")
    for t in [0, 5, 10, 15, 20]:
        print(f"  t={t}: {compounding.at(t):.2f}")
    print()


def example_point_operations():
    """Discrete operations at specific time points."""
    print("=" * 60)
    print("POINT OPERATIONS")
    print("=" * 60)
    
    base = CompoundCurve(10000, 0.05)  # 10000 growing at 5%
    
    print("Base compound curve:")
    for t in [0, 5, 10, 15]:
        print(f"  t={t}: {base.at(t):.2f}")
    
    # Add a lump sum at t=5
    with_bonus = base.at_point(5).add(5000)
    
    print("\nAfter adding $5000 at t=5:")
    for t in [0, 5, 10, 15]:
        print(f"  t={t}: {with_bonus.at(t):.2f}")
    
    # Split the curve at t=10 - 30% goes to new curve
    main, split_off = base.at_point(10).split(0.3)
    
    print("\nAfter 30% split at t=10:")
    print("Main curve (70%):")
    for t in [0, 5, 10, 15]:
        print(f"  t={t}: {main.at(t):.2f}")
    
    print("Split-off curve (30%):")
    for t in [0, 5, 10, 15]:
        print(f"  t={t}: {split_off.at(t):.2f}")
    print()


def example_chained_operations():
    """Chain multiple operations together."""
    print("=" * 60)
    print("CHAINED OPERATIONS")
    print("=" * 60)
    
    # Investment that:
    # 1. Grows at 7% annually
    # 2. Gets a $10,000 bonus at year 3
    # 3. Has 20% split off at year 5 for a different purpose
    # 4. Grows at 5% (more conservative) after year 7
    
    investment = CompoundCurve(50000, 0.07)  # Start with $50k at 7%
    
    # Chain the operations
    result = (investment
        .at_point(3).add(10000)                    # Bonus at year 3
    )
    
    # For the split, we need to capture both curves
    main, split_off = result.at_point(5).split(0.2)
    
    # Continue working with main
    main = main.period(7, float('inf')).transform(
        lambda v, t: v * (1.05 ** (t - 7)) / (1.07 ** (t - 7))  # Switch growth rate
    )
    
    print("Complex investment scenario:")
    print("(Starting $50k, 7% growth, $10k bonus at yr 3, 20% split at yr 5, 5% after yr 7)")
    print("\nMain portfolio:")
    for t in [0, 3, 5, 7, 10]:
        print(f"  Year {t}: ${main.at(t):,.2f}")
    
    print("\nSplit-off fund (starts at year 5):")
    for t in [0, 5, 7, 10]:
        print(f"  Year {t}: ${split_off.at(t):,.2f}")
    print()


def example_recurrence_relation():
    """Custom recurrence relations for complex scenarios."""
    print("=" * 60)
    print("RECURRENCE RELATIONS")
    print("=" * 60)
    
    # Custom scenario with recurrence
    principal = 10000
    monthly_decrease = 200
    monthly_rate = 0.005  # 0.5% monthly
    
    def decrease_then_compound(prev: float, t: float, month: int) -> float:
        after_decrease = prev - monthly_decrease
        if after_decrease <= 0:
            return 0
        return after_decrease * (1 + monthly_rate)
    
    curve = RecurrenceCurve(principal, decrease_then_compound, dt=1.0)
    
    print(f"Recurrence: Start ${principal:,}, decrease ${monthly_decrease}/month, compound {monthly_rate*100}%")
    for month in [0, 6, 12, 18, 24, 30]:
        balance = curve.at(month)
        print(f"  Month {month}: ${balance:,.2f}")
        if balance <= 0:
            break
    print()


def example_functional_transforms():
    """Functional transformations on curves."""
    print("=" * 60)
    print("FUNCTIONAL TRANSFORMS")
    print("=" * 60)
    
    # Sine wave
    wave = SineCurve(amplitude=100, frequency=0.1)
    
    print("Sine wave (A=100, f=0.1):")
    for t in range(0, 25, 5):
        print(f"  t={t}: {wave.at(t):,.2f}")
    
    # Apply custom map function
    # e.g., apply a "tax" that takes 20% of positive values
    taxed = wave.map(lambda v, t: v * 0.8 if v > 0 else v)
    
    print("\nWith 20% tax on positive values:")
    for t in range(0, 25, 5):
        print(f"  t={t}: {taxed.at(t):,.2f}")
    
    # Derivative
    deriv = DerivativeCurve(wave)
    
    print("\nDerivative of sine wave:")
    for t in range(0, 25, 5):
        print(f"  t={t}: {deriv.at(t):,.2f}")
    print()


def example_analysis():
    """Curve analysis tools."""
    print("=" * 60)
    print("CURVE ANALYSIS")
    print("=" * 60)
    
    # Create a curve that crosses zero
    curve = LinearCurve(-10, 100)  # 100 - 10t, crosses zero at t=10
    
    zero_crossing = curve.find_zero(0, 20)
    print(f"Linear curve (100 - 10t) crosses zero at t={zero_crossing:.2f}")
    
    # Find when compound curve reaches a target
    growth = CompoundCurve(1000, 0.07)
    double_time = growth.find_value(2000, 0, 20)
    print(f"$1000 at 7% doubles at t={double_time:.2f} years")
    
    # Numerical integration
    total = growth.integrate_over(0, 10)
    print(f"Integral of compound curve from 0 to 10: {total:,.2f}")
    
    # Sample the curve
    samples = growth.sample(0, 5, steps=5)
    print(f"\nSampled compound curve (0 to 5):")
    for t, v in samples:
        print(f"  t={t:.1f}: ${v:,.2f}")
    print()


if __name__ == '__main__':
    example_basic_curves()
    example_nested_curves()
    example_arithmetic()
    example_time_periods()
    example_point_operations()
    example_chained_operations()
    example_recurrence_relation()
    example_functional_transforms()
    example_analysis()
