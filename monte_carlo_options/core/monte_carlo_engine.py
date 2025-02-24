"""
Core Monte Carlo simulation engine for options pricing.
"""

import numpy as np
from typing import Optional, Callable, Dict, Any, Tuple
from abc import ABC, abstractmethod
import time


class StochasticProcess(ABC):
    """Abstract base class for stochastic processes."""
    
    @abstractmethod
    def simulate_paths(self, n_paths: int, n_steps: int, dt: float, 
                      initial_value: float, **kwargs) -> np.ndarray:
        """Simulate paths for the stochastic process."""
        pass


class GeometricBrownianMotion(StochasticProcess):
    """Geometric Brownian Motion process for stock price simulation."""
    
    def __init__(self, mu: float, sigma: float):
        """
        Initialize GBM process.
        
        Args:
            mu: Drift parameter (risk-free rate - dividend yield)
            sigma: Volatility parameter
        """
        self.mu = mu
        self.sigma = sigma
    
    def simulate_paths(self, n_paths: int, n_steps: int, dt: float, 
                      initial_value: float, **kwargs) -> np.ndarray:
        """
        Simulate stock price paths using GBM.
        
        Args:
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size
            initial_value: Initial stock price
            
        Returns:
            Array of shape (n_paths, n_steps+1) with simulated paths
        """
        # Generate random increments
        Z = np.random.normal(0, 1, (n_paths, n_steps))
        
        # Pre-compute constants
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        # Initialize paths array
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = initial_value
        
        # Generate log-normal paths
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
        
        return paths
    
    def simulate_paths_with_randoms(self, Z: np.ndarray, n_paths: int, n_steps: int, 
                                   dt: float, initial_value: float) -> np.ndarray:
        """
        Simulate paths using provided random numbers (for antithetic variates).
        
        Args:
            Z: Random numbers array of shape (n_paths, n_steps)
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            dt: Time step size
            initial_value: Initial stock price
            
        Returns:
            Array of shape (n_paths, n_steps+1) with simulated paths
        """
        # Pre-compute constants
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)
        
        # Initialize paths array
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = initial_value
        
        # Generate log-normal paths
        for i in range(n_steps):
            paths[:, i + 1] = paths[:, i] * np.exp(drift + diffusion * Z[:, i])
        
        return paths


class MonteCarloEngine:
    """
    Monte Carlo simulation engine for options pricing with variance reduction.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo engine.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        self.last_simulation_time = None
        self.last_convergence_data = None
    
    def price_option(self, 
                    option: 'Option',
                    process: StochasticProcess,
                    n_paths: int = 100000,
                    n_steps: int = 252,
                    variance_reduction: Optional[Callable] = None,
                    convergence_analysis: bool = False) -> Dict[str, Any]:
        """
        Price an option using Monte Carlo simulation.
        
        Args:
            option: Option object to price
            process: Stochastic process for underlying asset
            n_paths: Number of simulation paths
            n_steps: Number of time steps
            variance_reduction: Variance reduction technique
            convergence_analysis: Whether to perform convergence analysis
            
        Returns:
            Dictionary with pricing results and statistics
        """
        start_time = time.time()
        
        # Time to expiration
        T = option.time_to_expiry
        dt = T / n_steps
        
        # Simulate underlying asset paths
        paths = process.simulate_paths(
            n_paths=n_paths, 
            n_steps=n_steps, 
            dt=dt, 
            initial_value=option.spot_price
        )
        
        # Calculate payoffs
        payoffs = option.calculate_payoff(paths)
        
        # Apply variance reduction if provided
        if variance_reduction is not None:
            payoffs = variance_reduction(payoffs, paths, option)
        
        # Discount payoffs
        discounted_payoffs = payoffs * np.exp(-option.risk_free_rate * T)
        
        # Calculate statistics
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_paths)
        
        simulation_time = time.time() - start_time
        self.last_simulation_time = simulation_time
        
        results = {
            'price': option_price,
            'standard_error': std_error,
            'confidence_interval_95': (
                option_price - 1.96 * std_error,
                option_price + 1.96 * std_error
            ),
            'n_paths': n_paths,
            'simulation_time': simulation_time,
            'payoffs': discounted_payoffs if convergence_analysis else None
        }
        
        # Convergence analysis
        if convergence_analysis:
            convergence_data = self._analyze_convergence(discounted_payoffs)
            results['convergence'] = convergence_data
            self.last_convergence_data = convergence_data
        
        return results
    
    def _analyze_convergence(self, payoffs: np.ndarray, 
                           step_size: int = 1000) -> Dict[str, np.ndarray]:
        """
        Analyze convergence of Monte Carlo estimates.
        
        Args:
            payoffs: Array of discounted payoffs
            step_size: Step size for convergence analysis
            
        Returns:
            Dictionary with convergence data
        """
        n_paths = len(payoffs)
        steps = np.arange(step_size, n_paths + 1, step_size)
        
        estimates = []
        errors = []
        
        for n in steps:
            subset = payoffs[:n]
            estimate = np.mean(subset)
            error = np.std(subset) / np.sqrt(n)
            
            estimates.append(estimate)
            errors.append(error)
        
        return {
            'n_paths': steps,
            'estimates': np.array(estimates),
            'standard_errors': np.array(errors)
        }
    
    def estimate_required_paths(self, option: 'Option', 
                               process: StochasticProcess,
                               target_error: float = 0.01,
                               initial_paths: int = 10000) -> int:
        """
        Estimate number of paths required for target standard error.
        
        Args:
            option: Option to price
            process: Stochastic process
            target_error: Target standard error
            initial_paths: Initial number of paths for estimation
            
        Returns:
            Estimated required number of paths
        """
        # Run initial simulation
        result = self.price_option(option, process, n_paths=initial_paths)
        current_error = result['standard_error']
        
        # Estimate required paths (error scales as 1/sqrt(n))
        required_paths = int((current_error / target_error)**2 * initial_paths)
        
        return max(required_paths, initial_paths)