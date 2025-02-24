"""
Variance reduction techniques for Monte Carlo simulation.
"""

import numpy as np
from typing import Optional, Callable, Any
from abc import ABC, abstractmethod


class VarianceReduction(ABC):
    """Abstract base class for variance reduction techniques."""
    
    @abstractmethod
    def apply(self, payoffs: np.ndarray, paths: np.ndarray, 
             option: 'Option') -> np.ndarray:
        """Apply variance reduction technique to payoffs."""
        pass


class AntitheticVariates(VarianceReduction):
    """Antithetic variates variance reduction technique."""
    
    def apply(self, payoffs: np.ndarray, paths: np.ndarray, 
             option: 'Option') -> np.ndarray:
        """
        Apply antithetic variates to reduce variance.
        
        Args:
            payoffs: Original payoffs
            paths: Simulated paths
            option: Option object
            
        Returns:
            Variance-reduced payoffs
        """
        # For antithetic variates, we need to generate complementary paths
        # This should ideally be done during path generation
        # For now, we'll return original payoffs with a note
        # In practice, this would be integrated with the Monte Carlo engine
        return payoffs
    
    @staticmethod
    def generate_antithetic_paths(engine: 'MonteCarloEngine', 
                                 option: 'Option',
                                 process: 'StochasticProcess',
                                 n_paths: int, n_steps: int) -> tuple:
        """
        Generate paths using antithetic variates.
        
        Returns:
            Tuple of (original_payoffs, antithetic_payoffs)
        """
        # Generate random numbers
        T = option.time_to_expiry
        dt = T / n_steps
        Z = np.random.normal(0, 1, (n_paths, n_steps))
        
        # Generate original paths
        original_paths = process.simulate_paths_with_randoms(
            Z, n_paths, n_steps, dt, option.spot_price
        )
        
        # Generate antithetic paths using -Z
        antithetic_paths = process.simulate_paths_with_randoms(
            -Z, n_paths, n_steps, dt, option.spot_price
        )
        
        # Calculate payoffs
        original_payoffs = option.calculate_payoff(original_paths)
        antithetic_payoffs = option.calculate_payoff(antithetic_paths)
        
        # Return average of original and antithetic payoffs
        return (original_payoffs + antithetic_payoffs) / 2


class ControlVariates(VarianceReduction):
    """Control variates variance reduction technique."""
    
    def __init__(self, control_option: Optional['Option'] = None,
                 control_price: Optional[float] = None):
        """
        Initialize control variates.
        
        Args:
            control_option: Option to use as control variate
            control_price: Known analytical price of control option
        """
        self.control_option = control_option
        self.control_price = control_price
        self.beta = None  # Optimal coefficient
    
    def apply(self, payoffs: np.ndarray, paths: np.ndarray, 
             option: 'Option') -> np.ndarray:
        """
        Apply control variates to reduce variance.
        
        Args:
            payoffs: Original payoffs (target option)
            paths: Simulated paths
            option: Target option object
            
        Returns:
            Variance-reduced payoffs
        """
        if self.control_option is None:
            # Use geometric mean as control variate for Asian options
            return self._geometric_control_variate(payoffs, paths, option)
        
        # Calculate control variate payoffs
        control_payoffs = self.control_option.calculate_payoff(paths)
        
        # Estimate optimal beta coefficient
        if self.beta is None:
            covariance = np.cov(payoffs, control_payoffs)[0, 1]
            variance_control = np.var(control_payoffs)
            self.beta = covariance / variance_control if variance_control > 0 else 0
        
        # Apply control variate adjustment
        control_mean = self.control_price if self.control_price else np.mean(control_payoffs)
        adjusted_payoffs = payoffs - self.beta * (control_payoffs - control_mean)
        
        return adjusted_payoffs
    
    def _geometric_control_variate(self, payoffs: np.ndarray, 
                                  paths: np.ndarray, 
                                  option: 'Option') -> np.ndarray:
        """Use geometric average as control variate for arithmetic Asian options."""
        from ..pricing.options import AsianOption
        
        if not isinstance(option, AsianOption) or option.averaging_method != "arithmetic":
            return payoffs
        
        # Calculate geometric average payoffs
        geometric_averages = np.exp(np.mean(np.log(paths), axis=1))
        geometric_payoffs = option._vanilla_payoff(geometric_averages)
        
        # Estimate optimal beta
        covariance = np.cov(payoffs, geometric_payoffs)[0, 1]
        variance_geometric = np.var(geometric_payoffs)
        beta = covariance / variance_geometric if variance_geometric > 0 else 0
        
        # Apply control variate (using known analytical formula for geometric Asian)
        # This would require implementing the analytical formula
        # For now, use sample mean as approximation
        geometric_mean = np.mean(geometric_payoffs)
        adjusted_payoffs = payoffs - beta * (geometric_payoffs - geometric_mean)
        
        return adjusted_payoffs


class ImportanceSampling(VarianceReduction):
    """Importance sampling variance reduction technique."""
    
    def __init__(self, drift_adjustment: float = 0.0):
        """
        Initialize importance sampling.
        
        Args:
            drift_adjustment: Adjustment to drift for importance sampling
        """
        self.drift_adjustment = drift_adjustment
    
    def apply(self, payoffs: np.ndarray, paths: np.ndarray, 
             option: 'Option') -> np.ndarray:
        """
        Apply importance sampling to reduce variance.
        
        Args:
            payoffs: Original payoffs
            paths: Simulated paths
            option: Option object
            
        Returns:
            Importance-weighted payoffs
        """
        # Calculate likelihood ratios for drift adjustment
        if abs(self.drift_adjustment) < 1e-10:
            return payoffs
        
        n_paths, n_steps = paths.shape
        T = option.time_to_expiry
        dt = T / n_steps
        
        # Calculate log-likelihood ratio for each path
        log_likelihood_ratios = np.zeros(n_paths)
        
        for i in range(n_paths):
            path = paths[i, :]
            log_returns = np.diff(np.log(path))
            
            # Likelihood ratio for drift change
            # L = exp(θ * sum(ΔW) - 0.5 * θ^2 * T)
            brownian_increments = log_returns / np.sqrt(dt)  # Approximate
            log_likelihood_ratios[i] = (
                self.drift_adjustment * np.sum(brownian_increments) * np.sqrt(dt) -
                0.5 * self.drift_adjustment**2 * T
            )
        
        likelihood_ratios = np.exp(log_likelihood_ratios)
        
        # Apply importance weights
        weighted_payoffs = payoffs * likelihood_ratios
        
        return weighted_payoffs


class StratifiedSampling(VarianceReduction):
    """Stratified sampling variance reduction technique."""
    
    def __init__(self, n_strata: int = 10):
        """
        Initialize stratified sampling.
        
        Args:
            n_strata: Number of strata to use
        """
        self.n_strata = n_strata
    
    def apply(self, payoffs: np.ndarray, paths: np.ndarray, 
             option: 'Option') -> np.ndarray:
        """
        Apply stratified sampling (post-processing approach).
        
        Note: True stratified sampling should be applied during random number generation.
        This is a simplified post-processing approach.
        """
        # Sort payoffs and create strata
        n_paths = len(payoffs)
        sorted_indices = np.argsort(payoffs)
        stratum_size = n_paths // self.n_strata
        
        stratified_payoffs = np.zeros_like(payoffs)
        
        for i in range(self.n_strata):
            start_idx = i * stratum_size
            end_idx = (i + 1) * stratum_size if i < self.n_strata - 1 else n_paths
            
            stratum_indices = sorted_indices[start_idx:end_idx]
            stratum_mean = np.mean(payoffs[stratum_indices])
            
            # Replace with stratum mean (reduces variance within strata)
            stratified_payoffs[stratum_indices] = stratum_mean
        
        return stratified_payoffs


class MomentMatching(VarianceReduction):
    """Moment matching variance reduction technique."""
    
    def __init__(self, target_moments: Optional[dict] = None):
        """
        Initialize moment matching.
        
        Args:
            target_moments: Dictionary with target moments {'mean': float, 'var': float}
        """
        self.target_moments = target_moments or {}
    
    def apply(self, payoffs: np.ndarray, paths: np.ndarray, 
             option: 'Option') -> np.ndarray:
        """
        Apply moment matching to adjust final payoffs.
        
        This adjusts the final stock prices to match theoretical moments.
        """
        final_prices = paths[:, -1]
        
        # Calculate current moments
        current_mean = np.mean(final_prices)
        current_var = np.var(final_prices)
        
        # Target moments (use theoretical values if not provided)
        S0 = option.spot_price
        r = option.risk_free_rate
        T = option.time_to_expiry
        
        target_mean = self.target_moments.get('mean', S0 * np.exp(r * T))
        target_var = self.target_moments.get('var', None)
        
        # Adjust final prices to match target mean
        adjusted_final_prices = final_prices - current_mean + target_mean
        
        # Adjust variance if target provided
        if target_var is not None and current_var > 0:
            scale_factor = np.sqrt(target_var / np.var(adjusted_final_prices))
            adjusted_final_prices = target_mean + scale_factor * (adjusted_final_prices - target_mean)
        
        # Recalculate payoffs with adjusted final prices
        adjusted_payoffs = option._vanilla_payoff(adjusted_final_prices)
        
        return adjusted_payoffs


def combine_techniques(*techniques: VarianceReduction) -> Callable:
    """
    Combine multiple variance reduction techniques.
    
    Args:
        techniques: Variable number of variance reduction techniques
        
    Returns:
        Combined variance reduction function
    """
    def combined_reduction(payoffs: np.ndarray, paths: np.ndarray, 
                          option: 'Option') -> np.ndarray:
        result = payoffs.copy()
        for technique in techniques:
            result = technique.apply(result, paths, option)
        return result
    
    return combined_reduction