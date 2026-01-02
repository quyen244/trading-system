"""
Adaptive Multi-Strategy Ensemble System

Combines multiple strategies with dynamic weight allocation based on:
1. Market regime detection (trending, ranging, volatile)
2. Strategy performance tracking
3. Adaptive weight optimization

This is the crown jewel of the trading system - intelligently switching
between strategies based on current market conditions.

Author: Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

from trading_system.strategies.base_strategy import BaseStrategy
from trading_system.utils.logger import setup_logger
from trading_system.data.storage import StorageEngine 

logger = setup_logger('EnsembleStrategy')


class MarketRegimeDetector:
    """
    Detect current market regime (trending, ranging, volatile).
    """
    
    def __init__(self, lookback: int = 50):
        """
        Args:
            lookback: Lookback period for regime detection
        """
        self.lookback = lookback
    
    def detect_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect market regime for each timestamp.
        
        Returns:
            Series with regime labels: 'trending_up', 'trending_down', 'ranging', 'volatile'
        """
        import ta
        
        regime = pd.Series(index=df.index, dtype='object')
        
        for i in range(self.lookback, len(df)):
            window = df.iloc[i-self.lookback:i]
            
            # Calculate indicators
            # 1. ADX for trend strength
            adx_indicator = ta.trend.ADXIndicator(window['high'], window['low'], window['close'], window=14)
            adx = adx_indicator.adx().iloc[-1]
            
            # 2. Price direction
            sma_fast = window['close'].rolling(10).mean().iloc[-1]
            sma_slow = window['close'].rolling(30).mean().iloc[-1]
            
            # 3. Volatility (ATR)
            atr_indicator = ta.volatility.AverageTrueRange(window['high'], window['low'], window['close'], window=14)
            atr = atr_indicator.average_true_range().iloc[-1]
            avg_atr = atr_indicator.average_true_range().mean()
            
            # 4. Bollinger Band width
            bb = ta.volatility.BollingerBands(window['close'], window=20)
            bb_width = (bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / bb.bollinger_mavg().iloc[-1]
            
            # Regime classification
            if adx > 25:  # Strong trend
                if sma_fast > sma_slow:
                    regime.iloc[i] = 'trending_up'
                else:
                    regime.iloc[i] = 'trending_down'
            elif atr > avg_atr * 1.5:  # High volatility
                regime.iloc[i] = 'volatile'
            else:  # Ranging market
                regime.iloc[i] = 'ranging'
        
        return regime


class AdaptiveEnsembleStrategy(BaseStrategy):
    """
    Adaptive ensemble that combines multiple strategies with dynamic weights.
    
    Key Features:
    - Automatic market regime detection
    - Strategy weight allocation based on regime
    - Performance-based weight adjustment
    - Dynamic strategy switching
    
    Example usage:
        # Define strategies with their optimal regimes
        strategies = {
            'trend_following': (TrendFollowingStrategy({'fast_ma': 50, 'slow_ma': 200}), 
                               ['trending_up', 'trending_down']),
            'mean_reversion': (MeanReversionStrategy({'bb_length': 20}), 
                              ['ranging']),
            'breakout': (BreakoutStrategy({'lookback': 20}), 
                        ['volatile'])
        }
        
        ensemble = AdaptiveEnsembleStrategy(strategies=strategies)
        signals = ensemble.generate_signals(df)
    """
    
    def __init__(
        self,
        strategies: Dict[str, Tuple[BaseStrategy, List[str]]],
        params: dict = None
    ):
        """
        Args:
            strategies: Dict of {name: (strategy_instance, optimal_regimes)}
            params:
                - regime_lookback: Lookback for regime detection (default: 50)
                - weight_adjustment: Enable performance-based weights (default: True)
                - rebalance_period: Periods between weight rebalancing (default: 20)
        """
        super().__init__("AdaptiveEnsemble", params)
        self.strategies = strategies
        self.regime_lookback = self.params.get('regime_lookback', 50)
        self.weight_adjustment = self.params.get('weight_adjustment', True)
        self.rebalance_period = self.params.get('rebalance_period', 20)
        
        # Initialize regime detector
        self.regime_detector = MarketRegimeDetector(self.regime_lookback)
        
        # Initialize weights (equal weight to start)
        self.weights = {name: 1.0 / len(strategies) for name in strategies.keys()}
        
        # Track strategy performance
        self.strategy_returns = {name: [] for name in strategies.keys()}
    
    def _calculate_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Calculate strategy weights based on current regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Dictionary of strategy weights
        """
        weights = {}
        total_weight = 0
        
        for name, (strategy, optimal_regimes) in self.strategies.items():
            if regime in optimal_regimes:
                # Strategy is optimal for this regime
                weights[name] = 1.0
            else:
                # Strategy is not optimal, reduce weight
                weights[name] = 0.2
            
            total_weight += weights[name]
        
        # Normalize weights to sum to 1
        if total_weight > 0:
            weights = {name: w / total_weight for name, w in weights.items()}
        else:
            # Fallback to equal weights
            weights = {name: 1.0 / len(self.strategies) for name in self.strategies.keys()}
        
        return weights
    
    def _adjust_weights_by_performance(
        self,
        current_weights: Dict[str, float],
        lookback: int = 20
    ) -> Dict[str, float]:
        """
        Adjust weights based on recent strategy performance.
        
        Args:
            current_weights: Current strategy weights
            lookback: Lookback period for performance calculation
            
        Returns:
            Adjusted weights
        """
        if not self.weight_adjustment:
            return current_weights
        
        # Calculate recent Sharpe ratios for each strategy
        sharpe_ratios = {}
        
        for name, returns in self.strategy_returns.items():
            if len(returns) >= lookback:
                recent_returns = returns[-lookback:]
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                
                if std_return > 0:
                    sharpe = mean_return / std_return
                else:
                    sharpe = 0
                
                # Ensure non-negative
                sharpe_ratios[name] = max(sharpe, 0)
            else:
                sharpe_ratios[name] = 1.0  # Neutral weight for new strategies
        
        # Combine regime weights with performance weights
        adjusted_weights = {}
        total_weight = 0
        
        for name in current_weights.keys():
            # 70% regime-based, 30% performance-based
            regime_weight = current_weights[name]
            performance_weight = sharpe_ratios.get(name, 1.0)
            
            combined = 0.7 * regime_weight + 0.3 * performance_weight
            adjusted_weights[name] = combined
            total_weight += combined
        
        # Normalize
        if total_weight > 0:
            adjusted_weights = {name: w / total_weight for name, w in adjusted_weights.items()}
        
        return adjusted_weights
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate ensemble signals by combining multiple strategies.
        
        Args:
            data: OHLCV dataframe
            
        Returns:
            DataFrame with ensemble signals and regime information
        """
        data = self.validate_data(data)
        df = data.copy()
        
        # Detect market regime
        logger.info("Detecting market regimes...")
        df['regime'] = self.regime_detector.detect_regime(df)
        
        # Generate signals for each strategy
        logger.info("Generating signals for each strategy...")
        strategy_signals = {}
        
        for name, (strategy, _) in self.strategies.items():
            try:
                strategy_df = strategy.generate_signals(df.copy())
                strategy_signals[name] = strategy_df['signal']
                logger.info(f"Generated signals for {name}")
            except Exception as e:
                logger.error(f"Error generating signals for {name}: {e}")
                strategy_signals[name] = pd.Series(0, index=df.index)
        # Combine signals with adaptive weights
        logger.info("Combining signals with adaptive weights...")
        df['ensemble_signal'] = 0.0
        df['active_strategies'] = ''
        
        for i in range(self.regime_lookback, len(df) - self.regime_lookback):
            current_regime = df['regime'].iloc[i]
            
            # Calculate regime-based weights
            regime_weights = self._calculate_regime_weights(current_regime)
            
            # Adjust weights by performance (every rebalance_period)
            if i % self.rebalance_period == 0:
                regime_weights = self._adjust_weights_by_performance(regime_weights)
                self.weights = regime_weights
            
            # Combine signals
            combined_signal = 0.0
            active_strats = []
            
            for name, weight in self.weights.items():
                try:
                    signal = strategy_signals[name].iloc[i]
                except Exception as e:
                    logger.error(f"Error getting signal for {name}: {e} {i}")
                    signal = 0

                combined_signal += weight * signal
                
                if abs(signal) > 0 and weight > 0.1:
                    active_strats.append(f"{name}({weight:.2f})")
            
            # Convert to discrete signal (-1, 0, 1)
            if combined_signal > 0.3:
                df.loc[df.index[i], 'signal'] = 1
            elif combined_signal < -0.3:
                df.loc[df.index[i], 'signal'] = -1
            else:
                df.loc[df.index[i], 'signal'] = 0
            
            df.loc[df.index[i], 'ensemble_signal'] = combined_signal
            df.loc[df.index[i], 'active_strategies'] = ', '.join(active_strats)
        
        # Add weight columns for analysis
        for name in self.strategies.keys():
            df[f'weight_{name}'] = self.weights[name]
        
        return df
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current strategy weights."""
        return self.weights.copy()
    
    def get_regime_distribution(self, df: pd.DataFrame) -> pd.Series:
        """Get distribution of market regimes."""
        if 'regime' not in df.columns:
            df['regime'] = self.regime_detector.detect_regime(df)
        
        return df['regime'].value_counts(normalize=True)


def create_default_ensemble() -> AdaptiveEnsembleStrategy:
    """
    Create a default ensemble with common strategies.
    
    Returns:
        Configured AdaptiveEnsembleStrategy
    """
    from trading_system.strategies.trend import TrendFollowingStrategy, MomentumStrategy
    from trading_system.strategies.mean_reversion import MeanReversionStrategy
    from trading_system.strategies.breakout import BreakoutStrategy
    from trading_system.strategies.macd_strategy import MACDStrategy
    
    strategies = {
        'trend_following': (
            TrendFollowingStrategy({'fast_ma': 50, 'slow_ma': 200}),
            ['trending_up', 'trending_down']
        ),
        'mean_reversion': (
            MeanReversionStrategy({'bb_length': 20, 'bb_std': 2.0}),
            ['ranging']
        ),
        'breakout': (
            BreakoutStrategy({'lookback': 20}),
            ['volatile', 'trending_up']
        ),
        'macd': (
            MACDStrategy({'fast_period': 12, 'slow_period': 26}),
            ['trending_up', 'trending_down']
        ),
        'momentum': (
            MomentumStrategy({'period': 14}),
            ['ranging', 'volatile']
        )
    }
    
    ensemble = AdaptiveEnsembleStrategy(
        strategies=strategies,
        params={
            'regime_lookback': 50,
            'weight_adjustment': True,
            'rebalance_period': 20
        }
    )
    
    return ensemble


if __name__ == "__main__":
    print("Adaptive Multi-Strategy Ensemble System")
    print("=" * 50)
    print("\nFeatures:")
    print("✓ Automatic market regime detection")
    print("✓ Dynamic strategy weight allocation")
    print("✓ Performance-based weight adjustment")
    print("✓ Intelligent strategy switching")
    print("\nExample usage:")
    
    # Load data
    storage = StorageEngine()
    df = storage.load_market_data('BTC/USDT', '1h', '2025-01-01'  , '2025-12-31')
    # Create ensemble
    ensemble = create_default_ensemble()
    
    # Generate signals
    signals = ensemble.generate_signals(df)
    
    # Check current weights
    print("Current weights:", ensemble.get_current_weights())
    
    # Check regime distribution
    print("Regime distribution:", ensemble.get_regime_distribution(signals))

# python -m trading_system.strategies.ensemble