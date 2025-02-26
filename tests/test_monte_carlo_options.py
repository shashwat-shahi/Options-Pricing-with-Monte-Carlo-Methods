"""
Test suite for Monte Carlo options pricing library.
"""

import numpy as np
import pytest
from monte_carlo_options import (
    MonteCarloEngine, 
    EuropeanOption, 
    AsianOption, 
    BarrierOption,
    GreeksCalculator,
    VolatilitySurface,
    AntitheticVariates,
    ControlVariates
)
from monte_carlo_options.core.monte_carlo_engine import GeometricBrownianMotion
from monte_carlo_options.pricing.options import OptionType, BarrierType


class TestMonteCarloEngine:
    """Test Monte Carlo simulation engine."""
    
    def test_gbm_simulation(self):
        """Test Geometric Brownian Motion simulation."""
        np.random.seed(42)
        
        gbm = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        paths = gbm.simulate_paths(
            n_paths=1000,
            n_steps=252,
            dt=1/252,
            initial_value=100.0
        )
        
        assert paths.shape == (1000, 253)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(paths > 0)
        
        # Test that final prices follow log-normal distribution approximately
        final_prices = paths[:, -1]
        log_returns = np.log(final_prices / 100.0)
        
        # Should have mean close to (mu - 0.5*sigma^2)*T
        expected_mean = (0.05 - 0.5 * 0.2**2) * 1.0
        assert abs(np.mean(log_returns) - expected_mean) < 0.05
    
    def test_monte_carlo_pricing(self):
        """Test Monte Carlo option pricing."""
        np.random.seed(42)
        
        # Create European call option
        option = EuropeanOption(
            spot_price=100.0,
            strike_price=105.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            option_type=OptionType.CALL
        )
        
        # Create process
        process = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        
        # Create engine and price option
        engine = MonteCarloEngine(random_seed=42)
        result = engine.price_option(
            option=option,
            process=process,
            n_paths=50000,
            convergence_analysis=True
        )
        
        assert 'price' in result
        assert 'standard_error' in result
        assert 'confidence_interval_95' in result
        assert result['price'] > 0
        assert result['standard_error'] > 0
        
        # Price should be reasonable for a call option
        assert 1.0 < result['price'] < 20.0


class TestOptions:
    """Test option implementations."""
    
    def test_european_option(self):
        """Test European option payoff calculation."""
        option = EuropeanOption(
            spot_price=100.0,
            strike_price=105.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            option_type=OptionType.CALL
        )
        
        # Test payoffs
        final_prices = np.array([90, 100, 110, 120])
        paths = np.column_stack([
            [100, 100, 100, 100],  # Initial prices
            final_prices           # Final prices
        ])
        
        payoffs = option.calculate_payoff(paths)
        expected = np.array([0, 0, 5, 15])  # max(S-K, 0)
        
        np.testing.assert_array_equal(payoffs, expected)
    
    def test_asian_option(self):
        """Test Asian option payoff calculation."""
        option = AsianOption(
            spot_price=100.0,
            strike_price=105.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            averaging_method="arithmetic"
        )
        
        # Create simple paths
        paths = np.array([
            [100, 110, 120],  # Path 1: average = 110
            [100, 90, 80],    # Path 2: average = 90
        ])
        
        payoffs = option.calculate_payoff(paths)
        expected = np.array([5, 0])  # max(avg-K, 0) = max(110-105, 0), max(90-105, 0)
        
        np.testing.assert_array_equal(payoffs, expected)
    
    def test_barrier_option(self):
        """Test barrier option payoff calculation."""
        option = BarrierOption(
            spot_price=100.0,
            strike_price=105.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            barrier_level=115.0,
            barrier_type=BarrierType.UP_AND_OUT
        )
        
        # Create paths - one hits barrier, one doesn't
        paths = np.array([
            [100, 110, 120],  # Path 1: hits barrier (120 > 115)
            [100, 105, 110],  # Path 2: doesn't hit barrier
        ])
        
        payoffs = option.calculate_payoff(paths)
        # Path 1: hits barrier -> payoff = 0
        # Path 2: doesn't hit barrier -> payoff = max(110-105, 0) = 5
        expected = np.array([0, 5])
        
        np.testing.assert_array_equal(payoffs, expected)


class TestGreeksCalculator:
    """Test Greeks calculation system."""
    
    def test_greeks_calculation(self):
        """Test basic Greeks calculation."""
        np.random.seed(42)
        
        option = EuropeanOption(
            spot_price=100.0,
            strike_price=100.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            option_type=OptionType.CALL
        )
        
        process = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        engine = MonteCarloEngine(random_seed=42)
        
        calculator = GreeksCalculator()
        greeks = calculator.calculate_greeks(
            option=option,
            pricing_engine=engine,
            process=process,
            greeks_to_calculate=['delta', 'gamma'],
            n_paths=20000
        )
        
        assert greeks.delta is not None
        assert greeks.gamma is not None
        
        # Delta should be positive for ATM call
        assert 0 < greeks.delta < 1
        
        # Gamma should be positive
        assert greeks.gamma > 0


class TestVolatilitySurface:
    """Test volatility surface modeling."""
    
    def test_volatility_surface_creation(self):
        """Test volatility surface creation and calibration."""
        surface = VolatilitySurface(
            spot_price=100.0,
            risk_free_rate=0.05
        )
        
        # Add some volatility points
        vol_data = [
            {'strike': 90, 'time_to_expiry': 0.25, 'implied_volatility': 0.25},
            {'strike': 100, 'time_to_expiry': 0.25, 'implied_volatility': 0.20},
            {'strike': 110, 'time_to_expiry': 0.25, 'implied_volatility': 0.22},
            {'strike': 90, 'time_to_expiry': 0.5, 'implied_volatility': 0.23},
            {'strike': 100, 'time_to_expiry': 0.5, 'implied_volatility': 0.19},
            {'strike': 110, 'time_to_expiry': 0.5, 'implied_volatility': 0.21},
        ]
        
        surface.add_volatility_data(vol_data)
        surface.calibrate()
        
        assert surface.calibrated
        assert surface.volatility_grid is not None
        
        # Test interpolation
        vol = surface.get_volatility(strike=100, time_to_expiry=0.25)
        assert 0.15 < vol < 0.25
    
    def test_arbitrage_checks(self):
        """Test arbitrage violation checks."""
        surface = VolatilitySurface(
            spot_price=100.0,
            risk_free_rate=0.05,
            interpolation_method='linear'  # Use linear for small datasets
        )
        
        # Add more volatility points for stable interpolation
        vol_data = [
            {'strike': 90, 'time_to_expiry': 0.25, 'implied_volatility': 0.22},
            {'strike': 95, 'time_to_expiry': 0.25, 'implied_volatility': 0.20},
            {'strike': 100, 'time_to_expiry': 0.25, 'implied_volatility': 0.18},
            {'strike': 105, 'time_to_expiry': 0.25, 'implied_volatility': 0.19},
            {'strike': 110, 'time_to_expiry': 0.25, 'implied_volatility': 0.21},
            {'strike': 90, 'time_to_expiry': 0.5, 'implied_volatility': 0.21},
            {'strike': 95, 'time_to_expiry': 0.5, 'implied_volatility': 0.19},
            {'strike': 100, 'time_to_expiry': 0.5, 'implied_volatility': 0.17},
            {'strike': 105, 'time_to_expiry': 0.5, 'implied_volatility': 0.18},
            {'strike': 110, 'time_to_expiry': 0.5, 'implied_volatility': 0.20},
        ]
        
        surface.add_volatility_data(vol_data)
        surface.calibrate()
        
        violations = surface.check_arbitrage_violations()
        
        assert isinstance(violations, dict)
        assert 'calendar_arbitrage' in violations
        assert 'butterfly_arbitrage' in violations
        assert 'negative_volatility' in violations


class TestVarianceReduction:
    """Test variance reduction techniques."""
    
    def test_control_variates(self):
        """Test control variates technique."""
        np.random.seed(42)
        
        # Create sample payoffs and paths
        n_paths = 1000
        paths = np.random.lognormal(mean=0, sigma=0.2, size=(n_paths, 100))
        payoffs = np.maximum(paths[:, -1] - 100, 0)  # Call payoffs
        
        cv = ControlVariates()
        adjusted_payoffs = cv.apply(payoffs, paths, None)
        
        assert len(adjusted_payoffs) == len(payoffs)
        assert isinstance(adjusted_payoffs, np.ndarray)
    
    def test_antithetic_variates(self):
        """Test antithetic variates technique."""
        av = AntitheticVariates()
        
        # Create sample data
        payoffs = np.array([1, 2, 3, 4, 5])
        paths = np.random.random((5, 10))
        
        result = av.apply(payoffs, paths, None)
        
        assert len(result) == len(payoffs)
        assert isinstance(result, np.ndarray)


class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_complete_pricing_workflow(self):
        """Test complete workflow from option creation to Greeks calculation."""
        np.random.seed(42)
        
        # Create option
        option = EuropeanOption(
            spot_price=100.0,
            strike_price=105.0,
            time_to_expiry=0.25,
            risk_free_rate=0.05,
            option_type=OptionType.CALL
        )
        
        # Create process
        process = GeometricBrownianMotion(mu=0.05, sigma=0.2)
        
        # Create engine
        engine = MonteCarloEngine(random_seed=42)
        
        # Price option
        pricing_result = engine.price_option(
            option=option,
            process=process,
            n_paths=10000
        )
        
        # Calculate Greeks
        calculator = GreeksCalculator()
        greeks = calculator.calculate_greeks(
            option=option,
            pricing_engine=engine,
            process=process,
            greeks_to_calculate=['delta'],
            n_paths=5000
        )
        
        assert pricing_result['price'] > 0
        assert greeks.delta is not None
        
        # Create volatility surface
        surface = VolatilitySurface(spot_price=100.0, interpolation_method='linear')
        vol_data = [
            {'strike': 95, 'time_to_expiry': 0.25, 'implied_volatility': 0.22},
            {'strike': 100, 'time_to_expiry': 0.25, 'implied_volatility': 0.20},
            {'strike': 105, 'time_to_expiry': 0.25, 'implied_volatility': 0.21},
            {'strike': 95, 'time_to_expiry': 0.5, 'implied_volatility': 0.21},
            {'strike': 100, 'time_to_expiry': 0.5, 'implied_volatility': 0.19},
            {'strike': 105, 'time_to_expiry': 0.5, 'implied_volatility': 0.20},
        ]
        surface.add_volatility_data(vol_data)
        surface.calibrate()
        
        vol = surface.get_volatility(strike=102, time_to_expiry=0.25)
        assert 0.19 < vol < 0.22