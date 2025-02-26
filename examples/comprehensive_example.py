"""
Example demonstrating Monte Carlo options pricing with variance reduction and Greeks.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from monte_carlo_options import (
    MonteCarloEngine,
    EuropeanOption,
    AsianOption, 
    BarrierOption,
    GreeksCalculator,
    VolatilitySurface,
    ControlVariates,
    AntitheticVariates
)
from monte_carlo_options.core.monte_carlo_engine import GeometricBrownianMotion
from monte_carlo_options.pricing.options import OptionType, BarrierType


def main():
    """Run examples demonstrating the Monte Carlo options pricing system."""
    
    print("=" * 60)
    print("Monte Carlo Options Pricing with Variance Reduction")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Example 1: European Option Pricing
    print("\n1. European Option Pricing")
    print("-" * 30)
    
    # Create European call option
    european_option = EuropeanOption(
        spot_price=100.0,
        strike_price=105.0,
        time_to_expiry=0.25,  # 3 months
        risk_free_rate=0.05,
        option_type=OptionType.CALL
    )
    
    # Create Geometric Brownian Motion process
    process = GeometricBrownianMotion(mu=0.05, sigma=0.2)
    
    # Create Monte Carlo engine
    engine = MonteCarloEngine(random_seed=42)
    
    # Price the option
    result = engine.price_option(
        option=european_option,
        process=process,
        n_paths=100000,
        convergence_analysis=True
    )
    
    print(f"European Call Option Price: ${result['price']:.4f}")
    print(f"Standard Error: ${result['standard_error']:.4f}")
    print(f"95% Confidence Interval: ${result['confidence_interval_95'][0]:.4f} - ${result['confidence_interval_95'][1]:.4f}")
    print(f"Simulation Time: {result['simulation_time']:.2f} seconds")
    
    # Example 2: Asian Option Pricing
    print("\n2. Asian Option Pricing") 
    print("-" * 30)
    
    asian_option = AsianOption(
        spot_price=100.0,
        strike_price=100.0,
        time_to_expiry=0.5,
        risk_free_rate=0.05,
        option_type=OptionType.CALL,
        averaging_method="arithmetic"
    )
    
    asian_result = engine.price_option(
        option=asian_option,
        process=process,
        n_paths=100000
    )
    
    print(f"Asian Call Option Price: ${asian_result['price']:.4f}")
    print(f"Standard Error: ${asian_result['standard_error']:.4f}")
    
    # Example 3: Barrier Option Pricing
    print("\n3. Barrier Option Pricing")
    print("-" * 30)
    
    barrier_option = BarrierOption(
        spot_price=100.0,
        strike_price=100.0,
        time_to_expiry=0.25,
        risk_free_rate=0.05,
        option_type=OptionType.CALL,
        barrier_level=120.0,
        barrier_type=BarrierType.UP_AND_OUT
    )
    
    barrier_result = engine.price_option(
        option=barrier_option,
        process=process,
        n_paths=100000
    )
    
    print(f"Up-and-Out Barrier Call Price: ${barrier_result['price']:.4f}")
    print(f"Standard Error: ${barrier_result['standard_error']:.4f}")
    
    # Example 4: Greeks Calculation
    print("\n4. Greeks Calculation")
    print("-" * 30)
    
    calculator = GreeksCalculator()
    greeks = calculator.calculate_greeks(
        option=european_option,
        pricing_engine=engine,
        process=process,
        greeks_to_calculate=['delta', 'gamma', 'theta', 'vega', 'rho'],
        n_paths=50000
    )
    
    print(f"Delta: {greeks.delta:.4f}")
    print(f"Gamma: {greeks.gamma:.6f}")
    print(f"Theta: {greeks.theta:.4f}")
    print(f"Vega: {greeks.vega:.4f}")
    print(f"Rho: {greeks.rho:.4f}")
    
    # Example 5: Volatility Surface
    print("\n5. Volatility Surface Modeling")
    print("-" * 30)
    
    # Create volatility surface
    vol_surface = VolatilitySurface(
        spot_price=100.0,
        risk_free_rate=0.05,
        interpolation_method='rbf',
        arbitrage_free=True
    )
    
    # Add market volatility data
    vol_data = []
    strikes = [85, 90, 95, 100, 105, 110, 115]
    times = [0.25, 0.5, 1.0]
    
    for T in times:
        for K in strikes:
            # Generate synthetic volatility smile
            moneyness = K / 100.0
            vol = 0.2 + 0.05 * ((moneyness - 1.0) ** 2) + 0.02 * (T - 0.5)
            vol += np.random.normal(0, 0.005)  # Add some noise
            
            vol_data.append({
                'strike': K,
                'time_to_expiry': T,
                'implied_volatility': max(0.1, vol)
            })
    
    vol_surface.add_volatility_data(vol_data)
    vol_surface.calibrate()
    
    print(f"Volatility surface calibrated with {len(vol_data)} data points")
    print(f"ATM 3-month volatility: {vol_surface.get_volatility(100, 0.25):.2%}")
    print(f"ATM 1-year volatility: {vol_surface.get_volatility(100, 1.0):.2%}")
    
    # Check for arbitrage violations
    violations = vol_surface.check_arbitrage_violations()
    print(f"Arbitrage violations: {any(violations.values())}")
    
    # Example 6: Variance Reduction Techniques
    print("\n6. Variance Reduction Techniques")
    print("-" * 30)
    
    # Compare pricing with and without control variates
    cv = ControlVariates()
    
    # Price with control variates
    cv_result = engine.price_option(
        option=european_option,
        process=process,
        n_paths=50000,
        variance_reduction=cv.apply
    )
    
    # Compare results
    print(f"Standard Monte Carlo Error: ${result['standard_error']:.4f}")
    print(f"Control Variates Error: ${cv_result['standard_error']:.4f}")
    
    variance_reduction = (1 - (cv_result['standard_error'] / result['standard_error'])**2) * 100
    print(f"Variance Reduction: {variance_reduction:.1f}%")
    
    # Example 7: Convergence Analysis
    print("\n7. Convergence Analysis")
    print("-" * 30)
    
    if 'convergence' in result:
        convergence = result['convergence']
        print(f"Convergence analyzed over {len(convergence['estimates'])} intervals")
        print(f"Final estimate: ${convergence['estimates'][-1]:.4f}")
        print(f"Final standard error: ${convergence['standard_errors'][-1]:.4f}")
    
    # Example 8: Portfolio Greeks
    print("\n8. Portfolio Greeks")
    print("-" * 30)
    
    # Create portfolio positions
    positions = [
        (european_option, 100, process),    # Long 100 European calls
        (asian_option, -50, process),       # Short 50 Asian calls
        (barrier_option, 75, process)       # Long 75 barrier calls
    ]
    
    portfolio_greeks = calculator.calculate_portfolio_greeks(
        positions=positions,
        pricing_engine=engine,
        n_paths=20000
    )
    
    print(f"Portfolio Delta: {portfolio_greeks.delta:.2f}")
    print(f"Portfolio Gamma: {portfolio_greeks.gamma:.4f}")
    print(f"Portfolio Vega: {portfolio_greeks.vega:.2f}")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


def plot_convergence(result):
    """Plot convergence analysis if available."""
    if 'convergence' not in result:
        return
    
    convergence = result['convergence']
    
    plt.figure(figsize=(12, 4))
    
    # Plot price convergence
    plt.subplot(1, 2, 1)
    plt.plot(convergence['n_paths'], convergence['estimates'])
    plt.axhline(y=result['price'], color='r', linestyle='--', label='Final Estimate')
    plt.xlabel('Number of Paths')
    plt.ylabel('Option Price')
    plt.title('Price Convergence')
    plt.legend()
    plt.grid(True)
    
    # Plot standard error convergence
    plt.subplot(1, 2, 2)
    plt.loglog(convergence['n_paths'], convergence['standard_errors'])
    plt.xlabel('Number of Paths')
    plt.ylabel('Standard Error')
    plt.title('Standard Error Convergence')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def demonstrate_volatility_surface():
    """Demonstrate volatility surface functionality with plots."""
    
    # Create and calibrate volatility surface
    vol_surface = VolatilitySurface(
        spot_price=100.0,
        risk_free_rate=0.05,
        interpolation_method='cubic'
    )
    
    # Generate market data
    strikes = np.linspace(80, 120, 9)
    times = [0.25, 0.5, 0.75, 1.0]
    
    vol_data = []
    for T in times:
        for K in strikes:
            # Volatility smile: higher vol for OTM options
            moneyness = K / 100.0
            atm_vol = 0.2 + 0.01 * (T - 0.5)
            smile = 0.03 * ((moneyness - 1.0) ** 2)
            vol = atm_vol + smile
            
            vol_data.append({
                'strike': K,
                'time_to_expiry': T,
                'implied_volatility': vol
            })
    
    vol_surface.add_volatility_data(vol_data)
    vol_surface.calibrate()
    
    # Create plots
    fig = vol_surface.plot_surface(figsize=(15, 6))
    
    # Additional analysis plots
    plt.figure(figsize=(15, 5))
    
    # Term structure plot
    plt.subplot(1, 3, 1)
    times_ts, vols_ts = vol_surface.get_term_structure(moneyness=1.0)
    plt.plot(times_ts, vols_ts, 'b-', linewidth=2)
    plt.xlabel('Time to Expiry')
    plt.ylabel('Implied Volatility')
    plt.title('ATM Volatility Term Structure')
    plt.grid(True)
    
    # Volatility smile plot
    plt.subplot(1, 3, 2)
    strikes_smile, vols_smile = vol_surface.get_volatility_smile(time_to_expiry=0.25)
    plt.plot(strikes_smile, vols_smile, 'r-', linewidth=2)
    plt.xlabel('Strike')
    plt.ylabel('Implied Volatility')
    plt.title('3-Month Volatility Smile')
    plt.grid(True)
    
    # Moneyness smile
    plt.subplot(1, 3, 3)
    moneyness = strikes_smile / 100.0
    plt.plot(moneyness, vols_smile, 'g-', linewidth=2)
    plt.xlabel('Moneyness (K/S)')
    plt.ylabel('Implied Volatility')
    plt.title('3-Month Volatility Smile (Moneyness)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    
    # Uncomment to show plots (requires display)
    # plot_convergence(result)
    # demonstrate_volatility_surface()