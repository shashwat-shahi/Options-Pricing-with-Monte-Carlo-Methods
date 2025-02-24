"""
Option classes for various types of options.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Optional, List
from enum import Enum


class OptionType(Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class BarrierType(Enum):
    """Barrier option type enumeration."""
    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"


class Option(ABC):
    """Abstract base class for all option types."""
    
    def __init__(self, 
                 spot_price: float,
                 strike_price: float,
                 time_to_expiry: float,
                 risk_free_rate: float,
                 dividend_yield: float = 0.0,
                 option_type: OptionType = OptionType.CALL):
        """
        Initialize option parameters.
        
        Args:
            spot_price: Current price of underlying asset
            strike_price: Strike price of option
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            option_type: Type of option (call or put)
        """
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.time_to_expiry = time_to_expiry
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.option_type = option_type
    
    @abstractmethod
    def calculate_payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate option payoff from simulated paths.
        
        Args:
            paths: Array of simulated price paths
            
        Returns:
            Array of payoffs for each path
        """
        pass
    
    def _vanilla_payoff(self, final_prices: np.ndarray) -> np.ndarray:
        """Calculate vanilla option payoff."""
        if self.option_type == OptionType.CALL:
            return np.maximum(final_prices - self.strike_price, 0)
        else:
            return np.maximum(self.strike_price - final_prices, 0)


class EuropeanOption(Option):
    """European option that can only be exercised at expiration."""
    
    def calculate_payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculate European option payoff."""
        final_prices = paths[:, -1]
        return self._vanilla_payoff(final_prices)


class AmericanOption(Option):
    """American option that can be exercised any time before expiration."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # American options require more sophisticated pricing methods
        # This is a simplified implementation
    
    def calculate_payoff(self, paths: np.ndarray) -> np.ndarray:
        """
        Calculate American option payoff using optimal early exercise.
        Simplified implementation - uses maximum intrinsic value along path.
        """
        n_paths, n_steps = paths.shape
        payoffs = np.zeros(n_paths)
        
        for i in range(n_paths):
            path = paths[i, :]
            intrinsic_values = self._vanilla_payoff(path)
            payoffs[i] = np.max(intrinsic_values)
        
        return payoffs


class AsianOption(Option):
    """Asian option with payoff dependent on average price."""
    
    def __init__(self, *args, averaging_method: str = "arithmetic", **kwargs):
        """
        Initialize Asian option.
        
        Args:
            averaging_method: "arithmetic" or "geometric" averaging
        """
        super().__init__(*args, **kwargs)
        self.averaging_method = averaging_method
    
    def calculate_payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculate Asian option payoff based on average price."""
        if self.averaging_method == "arithmetic":
            average_prices = np.mean(paths, axis=1)
        elif self.averaging_method == "geometric":
            average_prices = np.exp(np.mean(np.log(paths), axis=1))
        else:
            raise ValueError("averaging_method must be 'arithmetic' or 'geometric'")
        
        return self._vanilla_payoff(average_prices)


class BarrierOption(Option):
    """Barrier option with knock-in or knock-out features."""
    
    def __init__(self, *args, 
                 barrier_level: float,
                 barrier_type: BarrierType,
                 **kwargs):
        """
        Initialize barrier option.
        
        Args:
            barrier_level: Barrier price level
            barrier_type: Type of barrier (up/down, in/out)
        """
        super().__init__(*args, **kwargs)
        self.barrier_level = barrier_level
        self.barrier_type = barrier_type
    
    def calculate_payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculate barrier option payoff."""
        final_prices = paths[:, -1]
        vanilla_payoffs = self._vanilla_payoff(final_prices)
        
        # Check barrier conditions
        if self.barrier_type == BarrierType.UP_AND_OUT:
            barrier_hit = np.any(paths >= self.barrier_level, axis=1)
            return vanilla_payoffs * (~barrier_hit)
        
        elif self.barrier_type == BarrierType.UP_AND_IN:
            barrier_hit = np.any(paths >= self.barrier_level, axis=1)
            return vanilla_payoffs * barrier_hit
        
        elif self.barrier_type == BarrierType.DOWN_AND_OUT:
            barrier_hit = np.any(paths <= self.barrier_level, axis=1)
            return vanilla_payoffs * (~barrier_hit)
        
        elif self.barrier_type == BarrierType.DOWN_AND_IN:
            barrier_hit = np.any(paths <= self.barrier_level, axis=1)
            return vanilla_payoffs * barrier_hit
        
        else:
            raise ValueError(f"Unknown barrier type: {self.barrier_type}")


class LookbackOption(Option):
    """Lookback option with payoff based on extreme values."""
    
    def __init__(self, *args, lookback_type: str = "floating", **kwargs):
        """
        Initialize lookback option.
        
        Args:
            lookback_type: "floating" or "fixed" lookback
        """
        super().__init__(*args, **kwargs)
        self.lookback_type = lookback_type
    
    def calculate_payoff(self, paths: np.ndarray) -> np.ndarray:
        """Calculate lookback option payoff."""
        if self.lookback_type == "floating":
            if self.option_type == OptionType.CALL:
                # Call pays (S_T - min(S))
                final_prices = paths[:, -1]
                min_prices = np.min(paths, axis=1)
                return np.maximum(final_prices - min_prices, 0)
            else:
                # Put pays (max(S) - S_T)
                final_prices = paths[:, -1]
                max_prices = np.max(paths, axis=1)
                return np.maximum(max_prices - final_prices, 0)
        
        elif self.lookback_type == "fixed":
            if self.option_type == OptionType.CALL:
                # Call pays (max(S) - K)
                max_prices = np.max(paths, axis=1)
                return np.maximum(max_prices - self.strike_price, 0)
            else:
                # Put pays (K - min(S))
                min_prices = np.min(paths, axis=1)
                return np.maximum(self.strike_price - min_prices, 0)
        
        else:
            raise ValueError("lookback_type must be 'floating' or 'fixed'")


class BasketOption(Option):
    """Multi-asset basket option."""
    
    def __init__(self, 
                 spot_prices: List[float],
                 weights: List[float],
                 strike_price: float,
                 time_to_expiry: float,
                 risk_free_rate: float,
                 dividend_yields: Optional[List[float]] = None,
                 option_type: OptionType = OptionType.CALL):
        """
        Initialize basket option.
        
        Args:
            spot_prices: List of current prices for each asset
            weights: List of weights for each asset in basket
            strike_price: Strike price of option
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            dividend_yields: List of dividend yields for each asset
            option_type: Type of option (call or put)
        """
        if len(spot_prices) != len(weights):
            raise ValueError("spot_prices and weights must have same length")
        
        # Use weighted average spot price for base class
        weighted_spot = np.sum(np.array(spot_prices) * np.array(weights))
        
        super().__init__(
            spot_price=weighted_spot,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            dividend_yield=0.0,  # Will be handled separately
            option_type=option_type
        )
        
        self.spot_prices = spot_prices
        self.weights = weights
        self.dividend_yields = dividend_yields or [0.0] * len(spot_prices)
    
    def calculate_payoff(self, paths_list: List[np.ndarray]) -> np.ndarray:
        """
        Calculate basket option payoff.
        
        Args:
            paths_list: List of price paths for each asset in basket
            
        Returns:
            Array of payoffs
        """
        if len(paths_list) != len(self.weights):
            raise ValueError("Number of paths must match number of assets")
        
        # Calculate basket values at each time step
        basket_values = np.zeros_like(paths_list[0])
        for i, (paths, weight) in enumerate(zip(paths_list, self.weights)):
            basket_values += weight * paths
        
        # Calculate payoff based on final basket value
        final_values = basket_values[:, -1]
        return self._vanilla_payoff(final_values)