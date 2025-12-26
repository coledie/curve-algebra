"""
Curve Algebra Examples - Visualization Demos

This file demonstrates visualization capabilities for curves, including
comparative plots and complex scenarios.
"""

from curve import (Curve, ConstantCurve, LinearCurve, ExponentialCurve,
                   CompoundCurve, RecurrenceCurve, TimeShiftCurve)
from visualize import plot_curves, show


def visualize_basic_curves():
    """Visualize basic curve types."""
    print("=" * 60)
    print("VISUALIZING BASIC CURVES")
    print("=" * 60)
    
    constant = ConstantCurve(100)
    linear = LinearCurve(5, 50)
    exponential = ExponentialCurve(50, 0.05)
    compound = CompoundCurve(50, 0.05)
    
    curves_to_plot = [
        (constant, "Constant (100)"),
        (linear, "Linear (5t + 50)"),
        (exponential, "Exponential (50*e^0.05t)"),
        (compound, "Compound (50*(1.05)^t)")
    ]
    
    plot_curves(
        curves_to_plot,
        t_range=(0, 20),
        title="Basic Curve Types Comparison",
        xlabel="Time (t)",
        ylabel="Value"
    )
    
    show()
    print("Plot displayed.\n")


def visualize_period_operations():
    """Visualize time period transformations."""
    print("=" * 60)
    print("VISUALIZING PERIOD OPERATIONS")
    print("=" * 60)
    
    base = LinearCurve(10, 1000)
    scaled = base.period(5, 15).scale(2)
    boosted = base.period(10, 20).add(500)
    
    curves_to_plot = [
        (base, "Base Curve (10t + 1000)"),
        (scaled, "Scaled 2x in [5, 15]"),
        (boosted, "Add 500 in [10, 20]")
    ]
    
    plot_curves(
        curves_to_plot,
        t_range=(0, 25),
        title="Period Operations on Curves",
        xlabel="Time",
        ylabel="Value"
    )
    
    show()
    print("Plot displayed.\n")


def visualize_curve_splitting():
    """Visualize the curve splitting example."""
    print("=" * 60)
    print("VISUALIZING CURVE SPLITTING")
    print("=" * 60)
    
    # Monthly decrease + compounding remainder scenario
    principal = 10000
    monthly_decrease = 200
    monthly_rate = 0.005
    
    def decrease_then_compound(prev: float, t: float, month: int) -> float:
        after_decrease = prev - monthly_decrease
        if after_decrease <= 0:
            return 0
        return after_decrease * (1 + monthly_rate)
    
    curve_a = RecurrenceCurve(principal, decrease_then_compound, dt=1.0)
    
    # Split at month 12
    balance_at_12 = curve_a.at(12)
    split_fraction = 0.3
    split_amount = balance_at_12 * split_fraction
    
    # Create split curve with different growth
    split_curve = CompoundCurve(split_amount, 0.08/12)
    split_curve = TimeShiftCurve(split_curve, -12)
    
    # Main curve after split
    def main_after_split(prev: float, t: float, month: int) -> float:
        if month == 12:
            prev = prev * (1 - split_fraction)
        after_decrease = prev - monthly_decrease
        if after_decrease <= 0:
            return 0
        return after_decrease * (1 + monthly_rate)
    
    main_curve = RecurrenceCurve(principal, main_after_split, dt=1.0)
    
    combined_curve = main_curve + split_curve
    
    curves_to_plot = [
        (main_curve, "Main Curve (70% after split)"),
        (split_curve, "Split Curve (30% at 8% growth)"),
        (combined_curve, "Combined Total")
    ]
    
    plot_curves(
        curves_to_plot,
        t_range=(0, 48),
        title="Curve Splitting Example: Main vs Split vs Combined",
        xlabel="Month",
        ylabel="Balance ($)"
    )
    
    show()
    print("Plot displayed.\n")


def visualize_investment_comparison():
    """Compare different investment strategies."""
    print("=" * 60)
    print("VISUALIZING INVESTMENT STRATEGIES")
    print("=" * 60)
    
    # Three different strategies
    conservative = CompoundCurve(10000, 0.04)  # 4% annual
    moderate = CompoundCurve(10000, 0.07)      # 7% annual
    aggressive = CompoundCurve(10000, 0.10)    # 10% annual
    
    curves_to_plot = [
        (conservative, "Conservative (4%)"),
        (moderate, "Moderate (7%)"),
        (aggressive, "Aggressive (10%)")
    ]
    
    plot_curves(
        curves_to_plot,
        t_range=(0, 30),
        title="Investment Strategy Comparison (30 years)",
        xlabel="Years",
        ylabel="Portfolio Value ($)"
    )
    
    show()
    print("Plot displayed.\n")


def visualize_loan_comparison():
    """Compare loan payoff scenarios."""
    print("=" * 60)
    print("VISUALIZING LOAN SCENARIOS")
    print("=" * 60)
    
    # Same loan with different payment amounts
    principal = 100000
    monthly_rate = 0.065 / 12
    
    def create_loan_curve(payment: float):
        def loan_recurrence(prev: float, t: float, month: int) -> float:
            if prev <= 0:
                return 0
            new_balance = prev * (1 + monthly_rate) - payment
            return max(0, new_balance)
        return RecurrenceCurve(principal, loan_recurrence, dt=1.0)
    
    loan_1000 = create_loan_curve(1000)
    loan_1500 = create_loan_curve(1500)
    loan_2000 = create_loan_curve(2000)
    
    curves_to_plot = [
        (loan_1000, "$1000/month payment"),
        (loan_1500, "$1500/month payment"),
        (loan_2000, "$2000/month payment")
    ]
    
    plot_curves(
        curves_to_plot,
        t_range=(0, 240),
        title="Loan Balance - Effect of Payment Amount",
        xlabel="Months",
        ylabel="Balance ($)"
    )
    
    show()
    print("Plot displayed.\n")


if __name__ == '__main__':
    print("Running visualization examples...")
    print("Close each plot window to proceed to the next example.\n")
    
    visualize_basic_curves()
    visualize_period_operations()
    visualize_curve_splitting()
    visualize_investment_comparison()
    visualize_loan_comparison()
    
    print("All visualizations complete!")
