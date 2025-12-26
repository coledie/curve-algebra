"""
Curve Visualization Module

Provides utilities to visualize curve objects using matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np
from curve import Curve
from typing import List, Optional, Tuple, Union


def plot_curve(
    curve: Curve,
    t_range: Tuple[float, float],
    num_points: int = 200,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Axes:
    """
    Plot a single curve over a time range.
    
    Args:
        curve: The curve object to plot
        t_range: Tuple of (t_start, t_end)
        num_points: Number of points to sample
        label: Label for the curve in the legend
        ax: Matplotlib axes to plot on (creates new if None)
        **kwargs: Additional arguments passed to plt.plot
        
    Returns:
        The matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    t_start, t_end = t_range
    t_values = np.linspace(t_start, t_end, num_points)
    y_values = [curve.at(t) for t in t_values]
    
    ax.plot(t_values, y_values, label=label, **kwargs)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    if label:
        ax.legend()
    
    return ax


def plot_curves(
    curves: List[Tuple[Curve, str]],
    t_range: Tuple[float, float],
    num_points: int = 200,
    title: Optional[str] = None,
    xlabel: str = 'Time (t)',
    ylabel: str = 'Value',
    figsize: Tuple[float, float] = (12, 7)
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot multiple curves on the same axes.
    
    Args:
        curves: List of (curve, label) tuples
        t_range: Tuple of (t_start, t_end)
        num_points: Number of points to sample
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    t_start, t_end = t_range
    t_values = np.linspace(t_start, t_end, num_points)
    
    for curve, label in curves:
        y_values = [curve.at(t) for t in t_values]
        ax.plot(t_values, y_values, label=label, linewidth=2)
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def plot_with_markers(
    curve: Curve,
    t_range: Tuple[float, float],
    markers: Optional[List[float]] = None,
    num_points: int = 200,
    title: Optional[str] = None,
    label: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a curve with special markers at specific time points.
    
    Args:
        curve: The curve to plot
        t_range: Tuple of (t_start, t_end)
        markers: List of t values to mark with dots
        num_points: Number of points to sample
        title: Plot title
        label: Curve label
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    t_start, t_end = t_range
    t_values = np.linspace(t_start, t_end, num_points)
    y_values = [curve.at(t) for t in t_values]
    
    ax.plot(t_values, y_values, label=label, linewidth=2)
    
    if markers:
        marker_values = [curve.at(t) for t in markers]
        ax.scatter(markers, marker_values, color='red', s=100, zorder=5, 
                  label='Marked Points')
        
        # Add annotations for marked points
        for t, y in zip(markers, marker_values):
            ax.annotate(f't={t}\ny={y:.2f}', 
                       xy=(t, y), 
                       xytext=(10, 10),
                       textcoords='offset points',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow', 
                                alpha=0.7),
                       arrowprops=dict(arrowstyle='->', 
                                     connectionstyle='arc3,rad=0'))
    
    ax.set_xlabel('Time (t)', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if title:
        ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


def compare_curves_side_by_side(
    curves: List[Tuple[Curve, str, Tuple[float, float]]],
    num_points: int = 200,
    figsize: Tuple[float, float] = (15, 5)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot multiple curves in separate subplots side by side.
    
    Args:
        curves: List of (curve, title, t_range) tuples
        num_points: Number of points to sample
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, list of axes)
    """
    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for ax, (curve, title, t_range) in zip(axes, curves):
        t_start, t_end = t_range
        t_values = np.linspace(t_start, t_end, num_points)
        y_values = [curve.at(t) for t in t_values]
        
        ax.plot(t_values, y_values, linewidth=2)
        ax.set_xlabel('Time (t)')
        ax.set_ylabel('Value')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def show():
    """Display all created plots."""
    plt.show()


def savefig(filename: str, dpi: int = 300):
    """Save the current figure to a file."""
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
