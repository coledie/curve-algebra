"""
Finance-specific curve builders and utilities.

This module provides convenient constructors for common financial curves
and scenarios like loans, investments, amortization, etc.
"""

from typing import List, Tuple, Optional, Callable
from curve import Curve, Evaluable, evaluate
import math


class Finance:
    """Factory for common financial curves."""
    
    @staticmethod
    def loan(
        principal: float,
        annual_rate: float,
        monthly_payment: float,
        start_month: int = 0
    ) -> Curve:
        """
        Create a loan balance curve with monthly payments and compounding.
        
        The balance compounds monthly, then payment is subtracted.
        
        Args:
            principal: Initial loan amount
            annual_rate: Annual interest rate (e.g., 0.05 for 5%)
            monthly_payment: Fixed monthly payment amount
            start_month: When payments begin (default: 0)
        
        Returns:
            Curve showing balance over time (months)
        """
        monthly_rate = annual_rate / 12
        
        def step(prev: float, t: float, i: int) -> float:
            if i < start_month:
                return prev * (1 + monthly_rate)  # Interest accrues
            # Add interest, subtract payment
            new_balance = prev * (1 + monthly_rate) - monthly_payment
            return max(0, new_balance)  # Can't go negative
        
        return Curve.recurrence(principal, step, dt=1.0)
    
    @staticmethod
    def amortization(
        principal: float,
        annual_rate: float,
        term_months: int
    ) -> Tuple[Curve, float]:
        """
        Create an amortizing loan curve with calculated payment.
        
        Returns:
            Tuple of (balance_curve, monthly_payment)
        """
        monthly_rate = annual_rate / 12
        
        # Standard amortization formula
        if monthly_rate == 0:
            payment = principal / term_months
        else:
            payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
                     ((1 + monthly_rate)**term_months - 1)
        
        return Finance.loan(principal, annual_rate, payment), payment
    
    @staticmethod
    def investment(
        initial: float,
        annual_return: float,
        monthly_contribution: float = 0,
        contribution_growth: float = 0
    ) -> Curve:
        """
        Create an investment curve with optional contributions.
        
        Args:
            initial: Starting investment
            annual_return: Expected annual return (e.g., 0.07 for 7%)
            monthly_contribution: Monthly addition (can be negative for withdrawals)
            contribution_growth: Annual growth rate of contributions
        
        Returns:
            Curve showing portfolio value over time (months)
        """
        monthly_return = (1 + annual_return) ** (1/12) - 1
        monthly_contrib_growth = (1 + contribution_growth) ** (1/12) - 1
        
        def step(prev: float, t: float, i: int) -> float:
            contribution = monthly_contribution * (1 + monthly_contrib_growth) ** i
            return prev * (1 + monthly_return) + contribution
        
        return Curve.recurrence(initial, step, dt=1.0)
    
    @staticmethod
    def savings(
        initial: float,
        annual_rate: float,
        monthly_deposit: float = 0
    ) -> Curve:
        """Simple savings account with deposits."""
        return Finance.investment(initial, annual_rate, monthly_deposit)
    
    @staticmethod
    def depreciation_straight_line(
        initial_value: float,
        salvage_value: float,
        useful_life: float
    ) -> Curve:
        """Straight-line depreciation over useful life."""
        annual_depreciation = (initial_value - salvage_value) / useful_life
        return Curve.linear(-annual_depreciation, initial_value).clamp(minimum=salvage_value)
    
    @staticmethod
    def depreciation_declining_balance(
        initial_value: float,
        rate: float,
        salvage_value: float = 0
    ) -> Curve:
        """Declining balance (accelerated) depreciation."""
        return Curve.exponential(initial_value, -rate).clamp(minimum=salvage_value)
    
    @staticmethod
    def bond_value(
        face_value: float,
        coupon_rate: float,
        market_rate: float,
        years_to_maturity: float,
        payments_per_year: int = 2
    ) -> Curve:
        """
        Bond value curve as it approaches maturity.
        
        This creates a curve showing the bond's present value over time
        as it moves toward maturity.
        """
        coupon_payment = face_value * coupon_rate / payments_per_year
        periods = int(years_to_maturity * payments_per_year)
        r = market_rate / payments_per_year
        
        def bond_pv_at(t: float) -> float:
            # t is in years
            remaining_periods = max(0, periods - int(t * payments_per_year))
            if remaining_periods == 0:
                return face_value
            
            # PV of remaining coupons + PV of face value
            if r == 0:
                coupon_pv = coupon_payment * remaining_periods
            else:
                coupon_pv = coupon_payment * (1 - (1 + r)**(-remaining_periods)) / r
            
            face_pv = face_value / (1 + r)**remaining_periods
            return coupon_pv + face_pv
        
        return Curve.from_function(bond_pv_at, f'Bond({face_value}, {coupon_rate})')
    
    @staticmethod
    def annuity_pv(
        payment: float,
        annual_rate: float,
        periods: int,
        periods_per_year: int = 12
    ) -> float:
        """Calculate present value of an annuity."""
        r = annual_rate / periods_per_year
        if r == 0:
            return payment * periods
        return payment * (1 - (1 + r)**(-periods)) / r
    
    @staticmethod
    def annuity_fv(
        payment: float,
        annual_rate: float,
        periods: int,
        periods_per_year: int = 12
    ) -> float:
        """Calculate future value of an annuity."""
        r = annual_rate / periods_per_year
        if r == 0:
            return payment * periods
        return payment * ((1 + r)**periods - 1) / r
    
    @staticmethod
    def mortgage_with_extra_payments(
        principal: float,
        annual_rate: float,
        term_months: int,
        extra_payments: List[Tuple[int, float]] = None
    ) -> Tuple[Curve, float]:
        """
        Mortgage with optional extra principal payments.
        
        Args:
            principal: Loan amount
            annual_rate: Annual interest rate
            term_months: Original term in months
            extra_payments: List of (month, amount) for extra payments
        
        Returns:
            Tuple of (balance_curve, base_monthly_payment)
        """
        monthly_rate = annual_rate / 12
        
        # Calculate base payment
        if monthly_rate == 0:
            base_payment = principal / term_months
        else:
            base_payment = principal * (monthly_rate * (1 + monthly_rate)**term_months) / \
                          ((1 + monthly_rate)**term_months - 1)
        
        extra_dict = dict(extra_payments) if extra_payments else {}
        
        def step(prev: float, t: float, i: int) -> float:
            if prev <= 0:
                return 0
            
            interest = prev * monthly_rate
            principal_payment = base_payment - interest
            extra = extra_dict.get(i, 0)
            
            new_balance = prev - principal_payment - extra
            return max(0, new_balance)
        
        return Curve.recurrence(principal, step, dt=1.0), base_payment
    
    @staticmethod
    def income_stream(
        base_income: float,
        annual_raise: float = 0,
        start_time: float = 0
    ) -> Curve:
        """
        Income stream with annual raises.
        
        Args:
            base_income: Starting monthly income
            annual_raise: Annual raise percentage (e.g., 0.03 for 3%)
            start_time: When income starts
        
        Returns:
            Monthly income curve
        """
        monthly_growth = (1 + annual_raise) ** (1/12) - 1
        return Curve.exponential(base_income, math.log(1 + monthly_growth)).delay(start_time)
    
    @staticmethod
    def retirement_drawdown(
        portfolio: float,
        monthly_withdrawal: float,
        annual_return: float,
        inflation: float = 0
    ) -> Curve:
        """
        Retirement portfolio with inflation-adjusted withdrawals.
        
        Args:
            portfolio: Starting portfolio value
            monthly_withdrawal: Initial monthly withdrawal
            annual_return: Expected annual return
            inflation: Annual inflation rate for withdrawal increases
        
        Returns:
            Portfolio balance curve
        """
        monthly_return = (1 + annual_return) ** (1/12) - 1
        monthly_inflation = (1 + inflation) ** (1/12) - 1
        
        def step(prev: float, t: float, i: int) -> float:
            withdrawal = monthly_withdrawal * (1 + monthly_inflation) ** i
            new_balance = prev * (1 + monthly_return) - withdrawal
            return max(0, new_balance)
        
        return Curve.recurrence(portfolio, step, dt=1.0)


class CurveAnalytics:
    """Analytical tools for curves."""
    
    @staticmethod
    def find_payoff_time(balance_curve: Curve, max_time: float = 360) -> Optional[float]:
        """Find when a loan balance reaches zero."""
        return balance_curve.find_value(0, 0, max_time, tolerance=0.01)
    
    @staticmethod
    def total_paid(
        balance_curve: Curve, 
        payment: float, 
        periods: int
    ) -> Tuple[float, float, float]:
        """
        Calculate total payments, principal, and interest.
        
        Returns:
            Tuple of (total_paid, principal_paid, interest_paid)
        """
        total = 0
        principal_paid = 0
        
        for i in range(periods):
            if balance_curve.at(i) <= 0:
                break
            total += payment
        
        initial_balance = balance_curve.at(0)
        final_balance = balance_curve.at(periods)
        principal_paid = initial_balance - max(0, final_balance)
        interest_paid = total - principal_paid
        
        return total, principal_paid, interest_paid
    
    @staticmethod
    def irr(cash_flows: List[Tuple[float, float]], guess: float = 0.1) -> float:
        """
        Calculate Internal Rate of Return for cash flows.
        
        Args:
            cash_flows: List of (time, amount) pairs
            guess: Initial guess for rate
        
        Returns:
            IRR as a decimal
        """
        def npv(rate: float) -> float:
            return sum(cf / (1 + rate)**t for t, cf in cash_flows)
        
        # Newton-Raphson iteration
        rate = guess
        for _ in range(100):
            npv_val = npv(rate)
            if abs(npv_val) < 0.0001:
                return rate
            
            # Numerical derivative
            delta = 0.0001
            npv_deriv = (npv(rate + delta) - npv_val) / delta
            
            if abs(npv_deriv) < 1e-10:
                break
            
            rate = rate - npv_val / npv_deriv
        
        return rate
    
    @staticmethod
    def compare_scenarios(
        curves: List[Tuple[str, Curve]],
        time_points: List[float]
    ) -> List[dict]:
        """Compare multiple curves at specific time points."""
        results = []
        for t in time_points:
            row = {'time': t}
            for name, curve in curves:
                row[name] = curve.at(t)
            results.append(row)
        return results
    
    @staticmethod
    def sensitivity_analysis(
        curve_factory: Callable[[float], Curve],
        param_range: List[float],
        eval_time: float
    ) -> List[Tuple[float, float]]:
        """
        Analyze how curve values change with a parameter.
        
        Args:
            curve_factory: Function that creates a curve given a parameter
            param_range: List of parameter values to test
            eval_time: Time at which to evaluate
        
        Returns:
            List of (parameter, value) pairs
        """
        return [(p, curve_factory(p).at(eval_time)) for p in param_range]


# Convenience functions for quick calculations

def monthly_payment(principal: float, annual_rate: float, months: int) -> float:
    """Calculate monthly payment for a loan."""
    r = annual_rate / 12
    if r == 0:
        return principal / months
    return principal * (r * (1 + r)**months) / ((1 + r)**months - 1)


def compound_growth(principal: float, rate: float, time: float, n: float = 1) -> float:
    """Calculate compound growth: P(1 + r/n)^(nt)."""
    return principal * (1 + rate / n) ** (n * time)


def present_value(future_value: float, rate: float, time: float) -> float:
    """Calculate present value."""
    return future_value / (1 + rate) ** time


def future_value(present_value: float, rate: float, time: float) -> float:
    """Calculate future value."""
    return present_value * (1 + rate) ** time
