"""
Greeks calculation system for real-time risk management using finite difference methods.
"""

import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class GreekType(Enum):
    """Greek type enumeration."""
    DELTA = "delta"      # ∂V/∂S
    GAMMA = "gamma"      # ∂²V/∂S²
    THETA = "theta"      # ∂V/∂t
    VEGA = "vega"        # ∂V/∂σ
    RHO = "rho"          # ∂V/∂r
    VOLGA = "volga"      # ∂²V/∂σ²
    VANNA = "vanna"      # ∂²V/∂S∂σ


@dataclass
class GreeksResult:
    """Container for Greeks calculation results."""
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    volga: Optional[float] = None
    vanna: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class GreeksCalculator:
    """
    Greeks calculation system using finite difference methods.
    """
    
    def __init__(self, 
                 finite_difference_step: Dict[str, float] = None,
                 use_central_difference: bool = True):
        """
        Initialize Greeks calculator.
        
        Args:
            finite_difference_step: Step sizes for finite differences
            use_central_difference: Whether to use central differences (more accurate)
        """
        self.fd_steps = finite_difference_step or {
            'spot': 0.01,      # 1% of spot price
            'time': 1/365,     # 1 day
            'volatility': 0.01, # 1% volatility
            'rate': 0.0001,    # 1 basis point
        }
        self.use_central_difference = use_central_difference
    
    def calculate_greeks(self, 
                        option: 'Option',
                        pricing_engine: 'MonteCarloEngine',
                        process: 'StochasticProcess',
                        greeks_to_calculate: Optional[list] = None,
                        n_paths: int = 100000,
                        **kwargs) -> GreeksResult:
        """
        Calculate Greeks for an option using finite differences.
        
        Args:
            option: Option to calculate Greeks for
            pricing_engine: Monte Carlo pricing engine
            process: Stochastic process for underlying
            greeks_to_calculate: List of Greeks to calculate (default: all)
            n_paths: Number of paths for Monte Carlo simulation
            **kwargs: Additional arguments for pricing engine
            
        Returns:
            GreeksResult object with calculated Greeks
        """
        if greeks_to_calculate is None:
            greeks_to_calculate = ['delta', 'gamma', 'theta', 'vega', 'rho']
        
        result = GreeksResult()
        
        # Base option price
        base_price = self._get_option_price(option, pricing_engine, process, n_paths, **kwargs)
        
        for greek in greeks_to_calculate:
            if greek.lower() == 'delta':
                result.delta = self._calculate_delta(
                    option, pricing_engine, process, base_price, n_paths, **kwargs
                )
            elif greek.lower() == 'gamma':
                result.gamma = self._calculate_gamma(
                    option, pricing_engine, process, base_price, n_paths, **kwargs
                )
            elif greek.lower() == 'theta':
                result.theta = self._calculate_theta(
                    option, pricing_engine, process, base_price, n_paths, **kwargs
                )
            elif greek.lower() == 'vega':
                result.vega = self._calculate_vega(
                    option, pricing_engine, process, base_price, n_paths, **kwargs
                )
            elif greek.lower() == 'rho':
                result.rho = self._calculate_rho(
                    option, pricing_engine, process, base_price, n_paths, **kwargs
                )
            elif greek.lower() == 'volga':
                result.volga = self._calculate_volga(
                    option, pricing_engine, process, base_price, n_paths, **kwargs
                )
            elif greek.lower() == 'vanna':
                result.vanna = self._calculate_vanna(
                    option, pricing_engine, process, base_price, n_paths, **kwargs
                )
        
        return result
    
    def _get_option_price(self, option, pricing_engine, process, n_paths, **kwargs):
        """Get option price from Monte Carlo simulation."""
        result = pricing_engine.price_option(
            option=option,
            process=process,
            n_paths=n_paths,
            **kwargs
        )
        return result['price']
    
    def _calculate_delta(self, option, pricing_engine, process, base_price, n_paths, **kwargs):
        """Calculate Delta: ∂V/∂S"""
        h = self.fd_steps['spot'] * option.spot_price
        
        if self.use_central_difference:
            # Central difference: (V(S+h) - V(S-h)) / (2h)
            option_up = self._create_shifted_option(option, spot_shift=h)
            option_down = self._create_shifted_option(option, spot_shift=-h)
            
            price_up = self._get_option_price(option_up, pricing_engine, process, n_paths, **kwargs)
            price_down = self._get_option_price(option_down, pricing_engine, process, n_paths, **kwargs)
            
            delta = (price_up - price_down) / (2 * h)
        else:
            # Forward difference: (V(S+h) - V(S)) / h
            option_up = self._create_shifted_option(option, spot_shift=h)
            price_up = self._get_option_price(option_up, pricing_engine, process, n_paths, **kwargs)
            
            delta = (price_up - base_price) / h
        
        return delta
    
    def _calculate_gamma(self, option, pricing_engine, process, base_price, n_paths, **kwargs):
        """Calculate Gamma: ∂²V/∂S²"""
        h = self.fd_steps['spot'] * option.spot_price
        
        # Central difference: (V(S+h) - 2V(S) + V(S-h)) / h²
        option_up = self._create_shifted_option(option, spot_shift=h)
        option_down = self._create_shifted_option(option, spot_shift=-h)
        
        price_up = self._get_option_price(option_up, pricing_engine, process, n_paths, **kwargs)
        price_down = self._get_option_price(option_down, pricing_engine, process, n_paths, **kwargs)
        
        gamma = (price_up - 2 * base_price + price_down) / (h ** 2)
        
        return gamma
    
    def _calculate_theta(self, option, pricing_engine, process, base_price, n_paths, **kwargs):
        """Calculate Theta: ∂V/∂t"""
        h = self.fd_steps['time']
        
        if option.time_to_expiry <= h:
            # Option is near expiry, use forward difference
            option_future = self._create_shifted_option(option, time_shift=-h/2)
            price_future = self._get_option_price(option_future, pricing_engine, process, n_paths, **kwargs)
            theta = (price_future - base_price) / (h/2)
        else:
            if self.use_central_difference:
                # Central difference: (V(t+h) - V(t-h)) / (2h)
                option_future = self._create_shifted_option(option, time_shift=-h)
                option_past = self._create_shifted_option(option, time_shift=h)
                
                price_future = self._get_option_price(option_future, pricing_engine, process, n_paths, **kwargs)
                price_past = self._get_option_price(option_past, pricing_engine, process, n_paths, **kwargs)
                
                theta = (price_future - price_past) / (2 * h)
            else:
                # Forward difference: (V(t+h) - V(t)) / h
                option_future = self._create_shifted_option(option, time_shift=-h)
                price_future = self._get_option_price(option_future, pricing_engine, process, n_paths, **kwargs)
                
                theta = (price_future - base_price) / h
        
        return theta
    
    def _calculate_vega(self, option, pricing_engine, process, base_price, n_paths, **kwargs):
        """Calculate Vega: ∂V/∂σ"""
        h = self.fd_steps['volatility']
        
        if self.use_central_difference:
            # Central difference: (V(σ+h) - V(σ-h)) / (2h)
            process_up = self._create_shifted_process(process, volatility_shift=h)
            process_down = self._create_shifted_process(process, volatility_shift=-h)
            
            price_up = self._get_option_price(option, pricing_engine, process_up, n_paths, **kwargs)
            price_down = self._get_option_price(option, pricing_engine, process_down, n_paths, **kwargs)
            
            vega = (price_up - price_down) / (2 * h)
        else:
            # Forward difference: (V(σ+h) - V(σ)) / h
            process_up = self._create_shifted_process(process, volatility_shift=h)
            price_up = self._get_option_price(option, pricing_engine, process_up, n_paths, **kwargs)
            
            vega = (price_up - base_price) / h
        
        return vega
    
    def _calculate_rho(self, option, pricing_engine, process, base_price, n_paths, **kwargs):
        """Calculate Rho: ∂V/∂r"""
        h = self.fd_steps['rate']
        
        if self.use_central_difference:
            # Central difference: (V(r+h) - V(r-h)) / (2h)
            option_up = self._create_shifted_option(option, rate_shift=h)
            option_down = self._create_shifted_option(option, rate_shift=-h)
            
            process_up = self._create_shifted_process(process, rate_shift=h)
            process_down = self._create_shifted_process(process, rate_shift=-h)
            
            price_up = self._get_option_price(option_up, pricing_engine, process_up, n_paths, **kwargs)
            price_down = self._get_option_price(option_down, pricing_engine, process_down, n_paths, **kwargs)
            
            rho = (price_up - price_down) / (2 * h)
        else:
            # Forward difference: (V(r+h) - V(r)) / h
            option_up = self._create_shifted_option(option, rate_shift=h)
            process_up = self._create_shifted_process(process, rate_shift=h)
            
            price_up = self._get_option_price(option_up, pricing_engine, process_up, n_paths, **kwargs)
            
            rho = (price_up - base_price) / h
        
        return rho
    
    def _calculate_volga(self, option, pricing_engine, process, base_price, n_paths, **kwargs):
        """Calculate Volga: ∂²V/∂σ²"""
        h = self.fd_steps['volatility']
        
        # Central difference: (V(σ+h) - 2V(σ) + V(σ-h)) / h²
        process_up = self._create_shifted_process(process, volatility_shift=h)
        process_down = self._create_shifted_process(process, volatility_shift=-h)
        
        price_up = self._get_option_price(option, pricing_engine, process_up, n_paths, **kwargs)
        price_down = self._get_option_price(option, pricing_engine, process_down, n_paths, **kwargs)
        
        volga = (price_up - 2 * base_price + price_down) / (h ** 2)
        
        return volga
    
    def _calculate_vanna(self, option, pricing_engine, process, base_price, n_paths, **kwargs):
        """Calculate Vanna: ∂²V/∂S∂σ"""
        h_spot = self.fd_steps['spot'] * option.spot_price
        h_vol = self.fd_steps['volatility']
        
        # Mixed partial derivative: (V(S+h,σ+h) - V(S+h,σ-h) - V(S-h,σ+h) + V(S-h,σ-h)) / (4*h_s*h_v)
        option_up_up = self._create_shifted_option(option, spot_shift=h_spot)
        process_up_up = self._create_shifted_process(process, volatility_shift=h_vol)
        
        option_up_down = self._create_shifted_option(option, spot_shift=h_spot)
        process_up_down = self._create_shifted_process(process, volatility_shift=-h_vol)
        
        option_down_up = self._create_shifted_option(option, spot_shift=-h_spot)
        process_down_up = self._create_shifted_process(process, volatility_shift=h_vol)
        
        option_down_down = self._create_shifted_option(option, spot_shift=-h_spot)
        process_down_down = self._create_shifted_process(process, volatility_shift=-h_vol)
        
        price_up_up = self._get_option_price(option_up_up, pricing_engine, process_up_up, n_paths, **kwargs)
        price_up_down = self._get_option_price(option_up_down, pricing_engine, process_up_down, n_paths, **kwargs)
        price_down_up = self._get_option_price(option_down_up, pricing_engine, process_down_up, n_paths, **kwargs)
        price_down_down = self._get_option_price(option_down_down, pricing_engine, process_down_down, n_paths, **kwargs)
        
        vanna = (price_up_up - price_up_down - price_down_up + price_down_down) / (4 * h_spot * h_vol)
        
        return vanna
    
    def _create_shifted_option(self, option, spot_shift=0, time_shift=0, rate_shift=0):
        """Create a new option with shifted parameters."""
        from ..pricing.options import Option
        import copy
        
        new_option = copy.deepcopy(option)
        new_option.spot_price += spot_shift
        new_option.time_to_expiry = max(0.001, new_option.time_to_expiry + time_shift)
        new_option.risk_free_rate += rate_shift
        
        return new_option
    
    def _create_shifted_process(self, process, volatility_shift=0, rate_shift=0):
        """Create a new process with shifted parameters."""
        from ..core.monte_carlo_engine import GeometricBrownianMotion
        import copy
        
        new_process = copy.deepcopy(process)
        if hasattr(new_process, 'sigma'):
            new_process.sigma += volatility_shift
        if hasattr(new_process, 'mu'):
            new_process.mu += rate_shift
        
        return new_process
    
    def calculate_portfolio_greeks(self, 
                                  positions: list,
                                  pricing_engine: 'MonteCarloEngine',
                                  **kwargs) -> GreeksResult:
        """
        Calculate portfolio-level Greeks.
        
        Args:
            positions: List of (option, quantity, process) tuples
            pricing_engine: Monte Carlo pricing engine
            **kwargs: Additional arguments for Greeks calculation
            
        Returns:
            Portfolio-level Greeks
        """
        portfolio_greeks = GreeksResult()
        
        for option, quantity, process in positions:
            option_greeks = self.calculate_greeks(
                option=option,
                pricing_engine=pricing_engine,
                process=process,
                **kwargs
            )
            
            # Add weighted Greeks to portfolio
            for greek_name, greek_value in option_greeks.to_dict().items():
                if greek_value is not None:
                    current_value = getattr(portfolio_greeks, greek_name) or 0
                    setattr(portfolio_greeks, greek_name, current_value + quantity * greek_value)
        
        return portfolio_greeks