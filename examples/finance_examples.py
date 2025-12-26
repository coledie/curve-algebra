"""
Curve Algebra Examples - Financial Applications

This file demonstrates financial scenarios using the curve algebra system,
including loans, investments, retirement planning, and complex curve splitting.
"""

from curve import (Curve, ConstantCurve, LinearCurve, CompoundCurve, 
                   RecurrenceCurve, TimeShiftCurve)
from finance import Finance, CurveAnalytics, monthly_payment


def example_loan_scenarios():
    """Financial loan scenarios."""
    print("=" * 60)
    print("LOAN SCENARIOS")
    print("=" * 60)
    
    # Basic loan
    loan_balance = Finance.loan(
        principal=200000,
        annual_rate=0.065,
        monthly_payment=1500
    )
    
    print("$200,000 loan at 6.5%, $1500/month payments:")
    for month in [0, 60, 120, 180, 240, 300]:
        print(f"  Month {month}: ${loan_balance.at(month):,.2f}")
    
    # Find payoff time
    payoff = CurveAnalytics.find_payoff_time(loan_balance)
    print(f"  Payoff month: {payoff:.1f} ({payoff/12:.1f} years)")
    
    # Amortizing loan with calculated payment
    amort_balance, payment = Finance.amortization(
        principal=200000,
        annual_rate=0.065,
        term_months=360  # 30 years
    )
    
    print(f"\nSame loan, 30-year amortization:")
    print(f"  Monthly payment: ${payment:,.2f}")
    for month in [0, 60, 120, 180, 240, 300, 360]:
        print(f"  Month {month}: ${amort_balance.at(month):,.2f}")
    
    # Mortgage with extra payments
    extra_payments = [(12, 5000), (24, 5000), (36, 5000)]  # $5k extra years 1-3
    extra_balance, base_pmt = Finance.mortgage_with_extra_payments(
        principal=200000,
        annual_rate=0.065,
        term_months=360,
        extra_payments=extra_payments
    )
    
    print(f"\nWith $5000 extra payment at months 12, 24, 36:")
    for month in [0, 12, 24, 36, 60, 120]:
        print(f"  Month {month}: ${extra_balance.at(month):,.2f}")
    
    payoff_early = CurveAnalytics.find_payoff_time(extra_balance)
    savings = (payoff - payoff_early) if payoff_early else 0
    print(f"  New payoff: {payoff_early:.1f} months (saved {savings:.1f} months)")
    print()


def example_investment_scenarios():
    """Investment and retirement scenarios."""
    print("=" * 60)
    print("INVESTMENT SCENARIOS")
    print("=" * 60)
    
    # Regular investment with contributions
    portfolio = Finance.investment(
        initial=10000,
        annual_return=0.08,
        monthly_contribution=500,
        contribution_growth=0.02  # Increase contributions 2%/year
    )
    
    print("$10,000 initial + $500/month (growing 2%/yr), 8% return:")
    for year in [0, 5, 10, 20, 30]:
        month = year * 12
        print(f"  Year {year}: ${portfolio.at(month):,.2f}")
    
    # Retirement drawdown
    retirement = Finance.retirement_drawdown(
        portfolio=1000000,
        monthly_withdrawal=4000,  # $48k/year
        annual_return=0.05,
        inflation=0.025
    )
    
    print("\n$1M portfolio, $4000/month withdrawal (2.5% inflation), 5% return:")
    for year in [0, 5, 10, 15, 20, 25, 30]:
        month = year * 12
        balance = retirement.at(month)
        print(f"  Year {year}: ${balance:,.2f}")
        if balance <= 0:
            print(f"  Portfolio exhausted around year {year}")
            break
    
    depletion = CurveAnalytics.find_payoff_time(retirement, max_time=600)
    if depletion:
        print(f"  Funds last: {depletion/12:.1f} years")
    print()


def example_curve_splitting_and_merging():
    """Complex curve splitting with different growth rates."""
    print("=" * 60)
    print("SPLITTING CURVES WITH DIFFERENT BEHAVIORS")
    print("=" * 60)
    
    # Main income stream
    income = LinearCurve(100, 5000)  # Starts at 5000, grows by 100/month
    
    print("Income curve (5000 + 100t):")
    for t in [0, 6, 12, 18, 24]:
        print(f"  Month {t}: ${income.at(t):,.2f}")
    
    # At month 6, split 30% into savings that compounds
    # The remaining 70% continues as income
    
    main_income, savings_seed = income.at_point(6).split(0.3)
    
    # Now make the savings grow with compound interest
    # We need to capture the value at the split point and grow from there
    savings_start_value = income.at(6) * 0.3
    savings_growth = CompoundCurve(savings_start_value, rate=0.05/12)
    # Apply time shift to start at month 6
    savings_growth = TimeShiftCurve(savings_growth, -6)
    
    print(f"\nAfter 30% split at month 6 (split amount: ${savings_start_value:,.2f}):")
    print("Main income (70%):")
    for t in [0, 6, 12, 18, 24]:
        print(f"  Month {t}: ${main_income.at(t):,.2f}")
    
    print("Savings fund (30% starting at month 6, compounding at 5%):")
    for t in [0, 6, 12, 18, 24]:
        print(f"  Month {t}: ${savings_growth.at(t):,.2f}")
    
    print("\nTotal (main + savings):")
    total = main_income + savings_growth
    for t in [0, 6, 12, 18, 24]:
        print(f"  Month {t}: ${total.at(t):,.2f}")
    print()


def example_complex_recurrence():
    """Custom recurrence relations for complex loan scenarios."""
    print("=" * 60)
    print("COMPLEX RECURRENCE - INTEREST ONLY THEN AMORTIZE")
    print("=" * 60)
    
    # Loan where you pay interest only for first 12 months, then amortize
    principal = 100000
    annual_rate = 0.06
    monthly_rate = annual_rate / 12
    
    # Calculate amortizing payment for months 13+
    remaining_term = 348  # 29 years
    amort_payment = monthly_payment(principal, annual_rate, remaining_term)
    
    def complex_loan(prev: float, t: float, month: int) -> float:
        if prev <= 0:
            return 0
        
        interest = prev * monthly_rate
        
        if month < 12:
            # Interest-only period
            return prev  # Balance unchanged
        else:
            # Amortizing period
            new_balance = prev * (1 + monthly_rate) - amort_payment
            return max(0, new_balance)
    
    loan = RecurrenceCurve(principal, complex_loan, dt=1.0)
    
    print(f"$100k loan: Interest-only for 12 months, then amortize")
    print(f"Amortizing payment: ${amort_payment:,.2f}")
    for month in [0, 6, 12, 24, 60, 120]:
        print(f"  Month {month}: ${loan.at(month):,.2f}")
    print()


def example_your_use_case():
    """
    Specific use case:
    "Curve A decreases by fixed amount monthly, remainder compounds"
    """
    print("=" * 60)
    print("USE CASE: Monthly decrease + compounding remainder")
    print("=" * 60)
    
    # Using recurrence (most accurate)
    principal = 10000
    monthly_decrease = 200
    monthly_rate = 0.005  # 0.5% monthly = ~6% annual
    
    def decrease_then_compound(prev: float, t: float, month: int) -> float:
        after_decrease = prev - monthly_decrease
        if after_decrease <= 0:
            return 0
        return after_decrease * (1 + monthly_rate)
    
    curve_a = RecurrenceCurve(principal, decrease_then_compound, dt=1.0)
    
    print(f"Starting: ${principal:,}")
    print(f"Monthly decrease: ${monthly_decrease}")
    print(f"Monthly compound rate: {monthly_rate*100}%")
    print()
    
    print("Balance over time:")
    for month in [0, 6, 12, 18, 24, 30, 36, 42, 48]:
        balance = curve_a.at(month)
        print(f"  Month {month}: ${balance:,.2f}")
        if balance <= 0:
            break
    
    # Now split this at month 12, with split going to different rate
    print("\n--- Splitting at month 12 ---")
    
    balance_at_12 = curve_a.at(12)
    split_fraction = 0.3
    split_amount = balance_at_12 * split_fraction
    
    print(f"Balance at month 12: ${balance_at_12:,.2f}")
    print(f"Split off {split_fraction*100}%: ${split_amount:,.2f}")
    
    # Create the split portion with different growth
    split_curve = CompoundCurve(split_amount, 0.08/12)  # 8% annual
    # Apply time shift to start at month 12
    split_curve = TimeShiftCurve(split_curve, -12)
    
    # Main curve continues but scaled down
    def main_after_split(prev: float, t: float, month: int) -> float:
        if month == 12:
            # At split point, reduce by split amount
            prev = prev * (1 - split_fraction)
        after_decrease = prev - monthly_decrease
        if after_decrease <= 0:
            return 0
        return after_decrease * (1 + monthly_rate)
    
    main_curve = RecurrenceCurve(principal, main_after_split, dt=1.0)
    
    print("\nMain curve (70% after split):")
    for month in [0, 12, 18, 24, 30]:
        print(f"  Month {month}: ${main_curve.at(month):,.2f}")
    
    print("\nSplit curve (30%, compounding at 8%):")
    for month in [0, 12, 18, 24, 30]:
        print(f"  Month {month}: ${split_curve.at(month):,.2f}")
    
    print("\nTotal:")
    for month in [0, 12, 18, 24, 30]:
        total = main_curve.at(month) + split_curve.at(month)
        print(f"  Month {month}: ${total:,.2f}")
    print()


if __name__ == '__main__':
    example_loan_scenarios()
    example_investment_scenarios()
    example_curve_splitting_and_merging()
    example_complex_recurrence()
    example_your_use_case()
