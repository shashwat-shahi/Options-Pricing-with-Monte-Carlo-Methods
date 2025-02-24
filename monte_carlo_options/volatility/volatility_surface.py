"""
Volatility surface modeling using spline interpolation and arbitrage-free constraints.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import RBFInterpolator, griddata, interp2d
from scipy.optimize import minimize, LinearConstraint
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings


@dataclass
class VolatilityPoint:
    """A single volatility observation."""
    strike: float
    time_to_expiry: float
    implied_volatility: float
    bid_vol: Optional[float] = None
    ask_vol: Optional[float] = None
    mid_vol: Optional[float] = None
    weight: float = 1.0


class VolatilitySurface:
    """
    Volatility surface modeling with spline interpolation and arbitrage-free constraints.
    """
    
    def __init__(self, 
                 spot_price: float,
                 risk_free_rate: float = 0.05,
                 dividend_yield: float = 0.0,
                 interpolation_method: str = 'rbf',
                 arbitrage_free: bool = True):
        """
        Initialize volatility surface.
        
        Args:
            spot_price: Current price of underlying asset
            risk_free_rate: Risk-free interest rate
            dividend_yield: Dividend yield
            interpolation_method: Interpolation method ('rbf', 'cubic', 'linear')
            arbitrage_free: Whether to enforce arbitrage-free constraints
        """
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.interpolation_method = interpolation_method
        self.arbitrage_free = arbitrage_free
        
        self.volatility_points: List[VolatilityPoint] = []
        self.interpolator = None
        self.calibrated = False
        
        # Surface grid for evaluation
        self.strike_grid = None
        self.time_grid = None
        self.volatility_grid = None
        
    def add_volatility_point(self, 
                           strike: float, 
                           time_to_expiry: float, 
                           implied_volatility: float,
                           **kwargs):
        """Add a volatility observation to the surface."""
        point = VolatilityPoint(
            strike=strike,
            time_to_expiry=time_to_expiry,
            implied_volatility=implied_volatility,
            **kwargs
        )
        self.volatility_points.append(point)
        self.calibrated = False
    
    def add_volatility_data(self, data: Union[pd.DataFrame, List[Dict]]):
        """
        Add multiple volatility points from DataFrame or list of dictionaries.
        
        Args:
            data: DataFrame with columns ['strike', 'time_to_expiry', 'implied_volatility']
                  or list of dictionaries with same keys
        """
        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                self.add_volatility_point(
                    strike=row['strike'],
                    time_to_expiry=row['time_to_expiry'],
                    implied_volatility=row['implied_volatility'],
                    **{k: v for k, v in row.items() 
                       if k not in ['strike', 'time_to_expiry', 'implied_volatility']}
                )
        else:
            for point_data in data:
                self.add_volatility_point(**point_data)
        
        self.calibrated = False
    
    def calibrate(self, 
                 strike_range: Optional[Tuple[float, float]] = None,
                 time_range: Optional[Tuple[float, float]] = None,
                 grid_size: Tuple[int, int] = (50, 20)):
        """
        Calibrate the volatility surface using interpolation.
        
        Args:
            strike_range: Range of strikes for surface grid
            time_range: Range of times for surface grid
            grid_size: Size of interpolation grid (n_strikes, n_times)
        """
        if len(self.volatility_points) < 3:
            raise ValueError("Need at least 3 volatility points for calibration")
        
        # Extract data points
        strikes = np.array([p.strike for p in self.volatility_points])
        times = np.array([p.time_to_expiry for p in self.volatility_points])
        vols = np.array([p.implied_volatility for p in self.volatility_points])
        weights = np.array([p.weight for p in self.volatility_points])
        
        # Set grid ranges
        if strike_range is None:
            strike_min, strike_max = strikes.min(), strikes.max()
            strike_buffer = (strike_max - strike_min) * 0.1
            strike_range = (strike_min - strike_buffer, strike_max + strike_buffer)
        
        if time_range is None:
            time_min, time_max = times.min(), times.max()
            time_buffer = (time_max - time_min) * 0.1
            time_range = (max(0.01, time_min - time_buffer), time_max + time_buffer)
        
        # Create interpolation grid
        self.strike_grid = np.linspace(strike_range[0], strike_range[1], grid_size[0])
        self.time_grid = np.linspace(time_range[0], time_range[1], grid_size[1])
        
        # Convert to moneyness for better interpolation
        moneyness = strikes / self.spot_price
        grid_moneyness = self.strike_grid / self.spot_price
        
        # Perform interpolation
        if self.interpolation_method == 'rbf':
            self._calibrate_rbf(moneyness, times, vols, weights, grid_moneyness)
        elif self.interpolation_method == 'cubic':
            self._calibrate_cubic(moneyness, times, vols, grid_moneyness)
        elif self.interpolation_method == 'linear':
            self._calibrate_linear(moneyness, times, vols, grid_moneyness)
        else:
            raise ValueError(f"Unknown interpolation method: {self.interpolation_method}")
        
        # Apply arbitrage-free constraints if requested
        if self.arbitrage_free:
            self._enforce_arbitrage_free_constraints()
        
        self.calibrated = True
    
    def _calibrate_rbf(self, moneyness, times, vols, weights, grid_moneyness):
        """Calibrate using Radial Basis Function interpolation."""
        # Prepare data points
        points = np.column_stack([moneyness, times])
        
        # Check if we have enough unique points for RBF
        unique_points = np.unique(points, axis=0)
        if len(unique_points) < 4:
            # Fall back to linear interpolation for insufficient data
            self._calibrate_linear(moneyness, times, vols, grid_moneyness)
            return
        
        try:
            # Create RBF interpolator with reduced smoothing for small datasets
            smoothing = min(0.01, 0.001 * len(points))
            self.interpolator = RBFInterpolator(
                points, 
                vols, 
                kernel='linear',  # Use linear kernel for stability
                smoothing=smoothing
            )
            
            # Generate surface
            M, T = np.meshgrid(grid_moneyness, self.time_grid, indexing='ij')
            grid_points = np.column_stack([M.ravel(), T.ravel()])
            
            interpolated_vols = self.interpolator(grid_points)
            self.volatility_grid = interpolated_vols.reshape(M.shape)
            
        except np.linalg.LinAlgError:
            # Fall back to linear interpolation if RBF fails
            warnings.warn("RBF interpolation failed, falling back to linear interpolation")
            self._calibrate_linear(moneyness, times, vols, grid_moneyness)
    
    def _calibrate_cubic(self, moneyness, times, vols, grid_moneyness):
        """Calibrate using cubic interpolation."""
        # Use griddata for cubic interpolation
        points = np.column_stack([moneyness, times])
        M, T = np.meshgrid(grid_moneyness, self.time_grid, indexing='ij')
        grid_points = np.column_stack([M.ravel(), T.ravel()])
        
        interpolated_vols = griddata(
            points, vols, grid_points, method='cubic', fill_value=np.nan
        )
        self.volatility_grid = interpolated_vols.reshape(M.shape)
        
        # Fill NaN values with linear interpolation
        if np.any(np.isnan(self.volatility_grid)):
            linear_vols = griddata(
                points, vols, grid_points, method='linear', fill_value='extrapolate'
            )
            linear_grid = linear_vols.reshape(M.shape)
            self.volatility_grid = np.where(
                np.isnan(self.volatility_grid), linear_grid, self.volatility_grid
            )
    
    def _calibrate_linear(self, moneyness, times, vols, grid_moneyness):
        """Calibrate using linear interpolation."""
        points = np.column_stack([moneyness, times])
        M, T = np.meshgrid(grid_moneyness, self.time_grid, indexing='ij')
        grid_points = np.column_stack([M.ravel(), T.ravel()])
        
        interpolated_vols = griddata(
            points, vols, grid_points, method='linear', fill_value=np.nan
        )
        self.volatility_grid = interpolated_vols.reshape(M.shape)
        
        # Fill NaN values with nearest neighbor interpolation
        if np.any(np.isnan(self.volatility_grid)):
            nearest_vols = griddata(
                points, vols, grid_points, method='nearest'
            )
            nearest_grid = nearest_vols.reshape(M.shape)
            self.volatility_grid = np.where(
                np.isnan(self.volatility_grid), nearest_grid, self.volatility_grid
            )
    
    def _enforce_arbitrage_free_constraints(self):
        """Enforce arbitrage-free constraints on the volatility surface."""
        if self.volatility_grid is None:
            return
        
        # Apply smoothing to reduce arbitrage violations
        self._apply_smoothing_constraints()
        
        # Ensure positive volatilities
        self.volatility_grid = np.maximum(self.volatility_grid, 0.01)
        
        # Check and correct calendar arbitrage (vol should not decrease too rapidly with time)
        self._correct_calendar_arbitrage()
        
        # Check and correct butterfly arbitrage (convexity constraints)
        self._correct_butterfly_arbitrage()
    
    def _apply_smoothing_constraints(self):
        """Apply smoothing to reduce arbitrage violations."""
        from scipy.ndimage import gaussian_filter
        
        # Apply mild Gaussian smoothing
        self.volatility_grid = gaussian_filter(self.volatility_grid, sigma=0.5)
    
    def _correct_calendar_arbitrage(self):
        """Correct calendar arbitrage violations."""
        # Total variance should be non-decreasing in time
        for i in range(len(self.strike_grid)):
            total_variances = self.volatility_grid[i, :] ** 2 * self.time_grid
            
            # Ensure non-decreasing total variance
            for j in range(1, len(self.time_grid)):
                if total_variances[j] < total_variances[j-1]:
                    # Adjust volatility to maintain non-decreasing total variance
                    min_variance = total_variances[j-1]
                    self.volatility_grid[i, j] = np.sqrt(min_variance / self.time_grid[j])
                    total_variances[j] = min_variance
    
    def _correct_butterfly_arbitrage(self):
        """Correct butterfly arbitrage violations."""
        # Check convexity constraints in strike dimension
        for j in range(len(self.time_grid)):
            vol_slice = self.volatility_grid[:, j]
            
            # Apply convexity correction
            for i in range(1, len(self.strike_grid) - 1):
                # Second derivative should not be too negative
                second_deriv = (vol_slice[i+1] - 2*vol_slice[i] + vol_slice[i-1])
                if second_deriv < -0.1:  # Threshold for correction
                    # Smooth the local region
                    vol_slice[i] = (vol_slice[i-1] + vol_slice[i+1]) / 2
            
            self.volatility_grid[:, j] = vol_slice
    
    def get_volatility(self, strike: float, time_to_expiry: float) -> float:
        """
        Get interpolated volatility for given strike and time.
        
        Args:
            strike: Strike price
            time_to_expiry: Time to expiration
            
        Returns:
            Interpolated implied volatility
        """
        if not self.calibrated:
            raise ValueError("Surface must be calibrated before querying volatilities")
        
        moneyness = strike / self.spot_price
        
        if self.interpolation_method == 'rbf' and self.interpolator is not None:
            point = np.array([[moneyness, time_to_expiry]])
            return float(self.interpolator(point)[0])
        else:
            # Use grid interpolation
            return self._interpolate_from_grid(moneyness, time_to_expiry)
    
    def _interpolate_from_grid(self, moneyness: float, time_to_expiry: float) -> float:
        """Interpolate volatility from precomputed grid."""
        grid_moneyness = self.strike_grid / self.spot_price
        
        # Find indices for interpolation
        if moneyness < grid_moneyness.min() or moneyness > grid_moneyness.max():
            # Extrapolation
            if moneyness < grid_moneyness.min():
                m_idx = 0
                m_weight = 1.0
            else:
                m_idx = len(grid_moneyness) - 2
                m_weight = 0.0
        else:
            m_idx = np.searchsorted(grid_moneyness, moneyness) - 1
            m_idx = max(0, min(m_idx, len(grid_moneyness) - 2))
            m_weight = (grid_moneyness[m_idx + 1] - moneyness) / (grid_moneyness[m_idx + 1] - grid_moneyness[m_idx])
        
        if time_to_expiry < self.time_grid.min() or time_to_expiry > self.time_grid.max():
            # Extrapolation
            if time_to_expiry < self.time_grid.min():
                t_idx = 0
                t_weight = 1.0
            else:
                t_idx = len(self.time_grid) - 2
                t_weight = 0.0
        else:
            t_idx = np.searchsorted(self.time_grid, time_to_expiry) - 1
            t_idx = max(0, min(t_idx, len(self.time_grid) - 2))
            t_weight = (self.time_grid[t_idx + 1] - time_to_expiry) / (self.time_grid[t_idx + 1] - self.time_grid[t_idx])
        
        # Bilinear interpolation
        vol_00 = self.volatility_grid[m_idx, t_idx]
        vol_01 = self.volatility_grid[m_idx, t_idx + 1]
        vol_10 = self.volatility_grid[m_idx + 1, t_idx]
        vol_11 = self.volatility_grid[m_idx + 1, t_idx + 1]
        
        vol_0 = m_weight * vol_00 + (1 - m_weight) * vol_10
        vol_1 = m_weight * vol_01 + (1 - m_weight) * vol_11
        
        return t_weight * vol_0 + (1 - t_weight) * vol_1
    
    def plot_surface(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot the volatility surface."""
        if not self.calibrated:
            raise ValueError("Surface must be calibrated before plotting")
        
        fig = plt.figure(figsize=figsize)
        
        # 3D surface plot
        ax1 = fig.add_subplot(121, projection='3d')
        M, T = np.meshgrid(self.strike_grid, self.time_grid, indexing='ij')
        
        surf = ax1.plot_surface(M, T, self.volatility_grid, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('Strike')
        ax1.set_ylabel('Time to Expiry')
        ax1.set_zlabel('Implied Volatility')
        ax1.set_title('Volatility Surface')
        
        # Contour plot
        ax2 = fig.add_subplot(122)
        contour = ax2.contour(M, T, self.volatility_grid, levels=20)
        ax2.clabel(contour, inline=True, fontsize=8)
        ax2.set_xlabel('Strike')
        ax2.set_ylabel('Time to Expiry')
        ax2.set_title('Volatility Surface Contours')
        
        # Plot original data points
        strikes = [p.strike for p in self.volatility_points]
        times = [p.time_to_expiry for p in self.volatility_points]
        ax2.scatter(strikes, times, c='red', s=50, alpha=0.7, label='Market Data')
        ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def get_term_structure(self, moneyness: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get volatility term structure for a given moneyness level.
        
        Args:
            moneyness: Moneyness level (strike/spot)
            
        Returns:
            Tuple of (times, volatilities)
        """
        if not self.calibrated:
            raise ValueError("Surface must be calibrated before querying term structure")
        
        strike = moneyness * self.spot_price
        volatilities = np.array([
            self.get_volatility(strike, t) for t in self.time_grid
        ])
        
        return self.time_grid, volatilities
    
    def get_volatility_smile(self, time_to_expiry: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get volatility smile for a given time to expiry.
        
        Args:
            time_to_expiry: Time to expiration
            
        Returns:
            Tuple of (strikes, volatilities)
        """
        if not self.calibrated:
            raise ValueError("Surface must be calibrated before querying smile")
        
        volatilities = np.array([
            self.get_volatility(strike, time_to_expiry) for strike in self.strike_grid
        ])
        
        return self.strike_grid, volatilities
    
    def check_arbitrage_violations(self) -> Dict[str, bool]:
        """
        Check for common arbitrage violations in the surface.
        
        Returns:
            Dictionary indicating types of arbitrage violations found
        """
        violations = {
            'calendar_arbitrage': False,
            'butterfly_arbitrage': False,
            'negative_volatility': False
        }
        
        if not self.calibrated:
            return violations
        
        # Check for negative volatilities
        if np.any(self.volatility_grid < 0):
            violations['negative_volatility'] = True
        
        # Check calendar arbitrage (total variance decreasing with time)
        for i in range(len(self.strike_grid)):
            total_vars = self.volatility_grid[i, :] ** 2 * self.time_grid
            if np.any(np.diff(total_vars) < -1e-6):
                violations['calendar_arbitrage'] = True
                break
        
        # Check butterfly arbitrage (negative convexity)
        for j in range(len(self.time_grid)):
            vol_slice = self.volatility_grid[:, j]
            second_derivs = np.diff(vol_slice, 2)
            if np.any(second_derivs < -0.1):
                violations['butterfly_arbitrage'] = True
                break
        
        return violations