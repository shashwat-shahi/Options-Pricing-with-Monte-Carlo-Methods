"""
Monte Carlo Options Pricing Library

A comprehensive library for pricing exotic options using Monte Carlo simulation
with variance reduction techniques, Greeks calculation, and volatility modeling.
"""

__version__ = "0.1.0"
__author__ = "Shashwat Shahi"

from .core.monte_carlo_engine import MonteCarloEngine
from .pricing.options import Option, EuropeanOption, AsianOption, BarrierOption
from .greeks.greeks_calculator import GreeksCalculator
from .volatility.volatility_surface import VolatilitySurface
from .variance_reduction.techniques import (
    AntitheticVariates,
    ControlVariates,
    ImportanceSampling,
)

__all__ = [
    "MonteCarloEngine",
    "Option",
    "EuropeanOption", 
    "AsianOption",
    "BarrierOption",
    "GreeksCalculator",
    "VolatilitySurface",
    "AntitheticVariates",
    "ControlVariates", 
    "ImportanceSampling",
]