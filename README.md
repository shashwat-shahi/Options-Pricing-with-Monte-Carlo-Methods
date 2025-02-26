# Options Pricing with Monte Carlo Methods

A comprehensive Python library for pricing exotic options using Monte Carlo simulation with variance reduction techniques, Greeks calculation, and volatility surface modeling.

## Features

### 🎯 Monte Carlo Simulation Engine
- **Geometric Brownian Motion** for stock price simulation
- **Multiple exotic options** support:
  - European Options
  - Asian Options (arithmetic/geometric averaging)
  - Barrier Options (up/down, in/out)
  - Lookback Options (floating/fixed)
  - American Options
  - Basket Options (multi-asset)
- **Convergence analysis** and performance monitoring
- **Configurable random number generation** with reproducible results

### 📉 Variance Reduction Techniques
- **Antithetic Variates** - reduce variance by using negatively correlated paths
- **Control Variates** - leverage known analytical solutions
- **Importance Sampling** - shift probability distributions for better sampling
- **Stratified Sampling** - divide sample space into strata
- **Moment Matching** - adjust samples to match theoretical moments
- **Combinable techniques** for maximum variance reduction

### 📊 Greeks Calculation System
Real-time risk management with finite difference methods:
- **Delta** (∂V/∂S) - price sensitivity to underlying
- **Gamma** (∂²V/∂S²) - delta sensitivity to underlying  
- **Theta** (∂V/∂t) - time decay
- **Vega** (∂V/∂σ) - volatility sensitivity
- **Rho** (∂V/∂r) - interest rate sensitivity
- **Volga** (∂²V/∂σ²) - vega sensitivity to volatility
- **Vanna** (∂²V/∂S∂σ) - delta sensitivity to volatility
- **Portfolio-level Greeks** calculation

### 🌊 Volatility Surface Modeling
- **Spline interpolation** methods (RBF, cubic, linear)
- **Arbitrage-free constraints** enforcement
- **Calendar arbitrage** detection and correction
- **Butterfly arbitrage** detection and correction
- **Market data integration** with bid/ask spreads
- **Term structure** and **volatility smile** extraction
- **Interactive plotting** and visualization

## Installation

```bash
# Clone the repository
git clone https://github.com/shashwat-shahi/Options-Pricing-with-Monte-Carlo-Methods.git
cd Options-Pricing-with-Monte-Carlo-Methods

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## Quick Start

### Basic European Option Pricing

```python
import numpy as np
from monte_carlo_options import (
    MonteCarloEngine, EuropeanOption, 
    GeometricBrownianMotion, OptionType
)

# Set up the option
option = EuropeanOption(
    spot_price=100.0,
    strike_price=105.0,
    time_to_expiry=0.25,  # 3 months
    risk_free_rate=0.05,
    option_type=OptionType.CALL
)

# Create the underlying process
process = GeometricBrownianMotion(mu=0.05, sigma=0.2)

# Set up Monte Carlo engine
engine = MonteCarloEngine(random_seed=42)

# Price the option
result = engine.price_option(
    option=option,
    process=process,
    n_paths=100000,
    convergence_analysis=True
)

print(f"Option Price: ${result['price']:.4f}")
print(f"Standard Error: ${result['standard_error']:.4f}")
print(f"95% Confidence Interval: ${result['confidence_interval_95'][0]:.4f} - ${result['confidence_interval_95'][1]:.4f}")
```

### Asian Option with Variance Reduction

```python
from monte_carlo_options import AsianOption, ControlVariates

# Create Asian option
asian_option = AsianOption(
    spot_price=100.0,
    strike_price=100.0,
    time_to_expiry=0.5,
    risk_free_rate=0.05,
    option_type=OptionType.CALL,
    averaging_method="arithmetic"
)

# Use control variates for variance reduction
cv = ControlVariates()

result = engine.price_option(
    option=asian_option,
    process=process,
    n_paths=50000,
    variance_reduction=cv.apply
)

print(f"Asian Option Price: ${result['price']:.4f}")
```

### Barrier Option Pricing

```python
from monte_carlo_options import BarrierOption, BarrierType

# Create barrier option
barrier_option = BarrierOption(
    spot_price=100.0,
    strike_price=100.0,
    time_to_expiry=0.25,
    risk_free_rate=0.05,
    option_type=OptionType.CALL,
    barrier_level=120.0,
    barrier_type=BarrierType.UP_AND_OUT
)

result = engine.price_option(
    option=barrier_option,
    process=process,
    n_paths=100000
)

print(f"Barrier Option Price: ${result['price']:.4f}")
```

### Greeks Calculation

```python
from monte_carlo_options import GreeksCalculator

calculator = GreeksCalculator()
greeks = calculator.calculate_greeks(
    option=option,
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
```

### Volatility Surface Modeling

```python
from monte_carlo_options import VolatilitySurface

# Create volatility surface
vol_surface = VolatilitySurface(
    spot_price=100.0,
    risk_free_rate=0.05,
    interpolation_method='rbf',
    arbitrage_free=True
)

# Add market volatility data
vol_data = [
    {'strike': 90, 'time_to_expiry': 0.25, 'implied_volatility': 0.25},
    {'strike': 100, 'time_to_expiry': 0.25, 'implied_volatility': 0.20},
    {'strike': 110, 'time_to_expiry': 0.25, 'implied_volatility': 0.22},
    {'strike': 90, 'time_to_expiry': 0.5, 'implied_volatility': 0.23},
    {'strike': 100, 'time_to_expiry': 0.5, 'implied_volatility': 0.19},
    {'strike': 110, 'time_to_expiry': 0.5, 'implied_volatility': 0.21},
]

vol_surface.add_volatility_data(vol_data)
vol_surface.calibrate()

# Query interpolated volatility
vol = vol_surface.get_volatility(strike=105, time_to_expiry=0.375)
print(f"Interpolated volatility: {vol:.2%}")

# Check for arbitrage violations
violations = vol_surface.check_arbitrage_violations()
print(f"Arbitrage violations: {any(violations.values())}")

# Plot the surface
fig = vol_surface.plot_surface()
```

### Portfolio Greeks

```python
# Define portfolio positions: (option, quantity, process)
positions = [
    (option, 100, process),          # Long 100 European calls
    (asian_option, -50, process),    # Short 50 Asian calls
    (barrier_option, 75, process)    # Long 75 barrier calls
]

portfolio_greeks = calculator.calculate_portfolio_greeks(
    positions=positions,
    pricing_engine=engine,
    n_paths=20000
)

print(f"Portfolio Delta: {portfolio_greeks.delta:.2f}")
print(f"Portfolio Gamma: {portfolio_greeks.gamma:.4f}")
print(f"Portfolio Vega: {portfolio_greeks.vega:.2f}")
```

## Advanced Usage

### Custom Variance Reduction

```python
from monte_carlo_options.variance_reduction import combine_techniques
from monte_carlo_options import AntitheticVariates, ControlVariates

# Combine multiple variance reduction techniques
combined_vr = combine_techniques(
    AntitheticVariates(),
    ControlVariates(),
)

result = engine.price_option(
    option=option,
    process=process,
    n_paths=50000,
    variance_reduction=combined_vr
)
```

### Custom Greeks Configuration

```python
# Configure finite difference steps
calculator = GreeksCalculator(
    finite_difference_step={
        'spot': 0.01,      # 1% of spot price
        'time': 1/365,     # 1 day
        'volatility': 0.01, # 1% volatility
        'rate': 0.0001,    # 1 basis point
    },
    use_central_difference=True
)
```

### Convergence Analysis

```python
result = engine.price_option(
    option=option,
    process=process,
    n_paths=100000,
    convergence_analysis=True
)

# Plot convergence
import matplotlib.pyplot as plt

convergence = result['convergence']
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(convergence['n_paths'], convergence['estimates'])
plt.xlabel('Number of Paths')
plt.ylabel('Option Price')
plt.title('Price Convergence')

plt.subplot(1, 2, 2)
plt.loglog(convergence['n_paths'], convergence['standard_errors'])
plt.xlabel('Number of Paths')
plt.ylabel('Standard Error')
plt.title('Error Convergence')

plt.show()
```

## Architecture

The library is organized into several key modules:

- **`core/`** - Monte Carlo simulation engine and stochastic processes
- **`pricing/`** - Option definitions and payoff calculations
- **`greeks/`** - Greeks calculation using finite differences
- **`volatility/`** - Volatility surface modeling and interpolation
- **`variance_reduction/`** - Variance reduction techniques

## Performance Considerations

- **Vectorized Operations**: All calculations use NumPy for optimal performance
- **Memory Management**: Efficient handling of large simulation arrays
- **Parallel Processing**: Ready for multi-threading implementations
- **Caching**: Interpolators and calculations cache results when appropriate

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=monte_carlo_options --cov-report=html

# Run specific test categories
pytest tests/test_monte_carlo_options.py::TestMonteCarloEngine -v
```

## Examples

See the `examples/` directory for comprehensive demonstrations:

- **`comprehensive_example.py`** - Complete workflow demonstration
- **Jupyter notebooks** (coming soon) - Interactive tutorials

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Dependencies

- **NumPy** ≥ 1.21.0 - Numerical computing
- **SciPy** ≥ 1.7.0 - Scientific computing and interpolation
- **Pandas** ≥ 1.3.0 - Data manipulation
- **Matplotlib** ≥ 3.4.0 - Plotting and visualization
- **Seaborn** ≥ 0.11.0 - Statistical plotting

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Black-Scholes-Merton option pricing model
- Monte Carlo methods in finance literature
- Variance reduction techniques from numerical analysis
- Financial risk management practices