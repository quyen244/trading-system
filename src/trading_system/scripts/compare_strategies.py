"""
Multi-Strategy Comparison System

Run multiple strategies and compare their performance side-by-side.

Author: Trading System
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from datetime import datetime
import logging

from trading_system.backtesting.engine import BacktestEngine
from trading_system.utils.logger import setup_logger

# Import all strategies
from trading_system.strategies.mean_reversion import MeanReversionStrategy
from trading_system.strategies.trend import TrendFollowingStrategy, MomentumStrategy
from trading_system.strategies.breakout import BreakoutStrategy
from trading_system.strategies.ma_strategy import MovingAverageCrossoverStrategy
from trading_system.strategies.macd_strategy import MACDStrategy, MACDDivergenceStrategy

logger = setup_logger('StrategyComparison')


class StrategyComparator:
    """
    Compare multiple trading strategies on the same dataset.
    
    Example usage:
        comparator = StrategyComparator(data=df)
        
        strategies = [
            ('MeanReversion', MeanReversionStrategy, {'bb_length': 20, 'bb_std': 2.0}),
            ('TrendFollowing', TrendFollowingStrategy, {'fast_ma': 50, 'slow_ma': 200}),
            ('MACD', MACDStrategy, {})
        ]
        
        results = comparator.compare_strategies(strategies)
        comparator.print_comparison_table(results)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 10000,
        commission: float = 0.001
    ):
        """
        Initialize the comparator.
        
        Args:
            data: Historical price data
            initial_capital: Starting capital
            commission: Commission rate
        """
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.backtest_engine = BacktestEngine(initial_capital, commission)
        
    def compare_strategies(
        self,
        strategies: List[tuple]
    ) -> pd.DataFrame:
        """
        Run and compare multiple strategies.
        
        Args:
            strategies: List of (name, strategy_class, params) tuples
            
        Returns:
            comparison_df: DataFrame with comparison metrics
        """
        logger.info(f"Comparing {len(strategies)} strategies")
        
        results = []
        equity_curves = {}
        all_trades = {}
        
        for name, strategy_class, params in strategies:
            logger.info(f"Running backtest for: {name}")
            
            try:
                # Create strategy instance
                strategy = strategy_class(params=params)
                
                # Run backtest
                metrics, equity_curve, trades = self.backtest_engine.run_backtest(
                    strategy, self.data
                )
                
                # Store results
                result = {
                    'Strategy': name,
                    **metrics,
                    'Num_Trades': len(trades)
                }
                results.append(result)
                
                equity_curves[name] = equity_curve
                all_trades[name] = trades
                
                logger.info(f"{name} - Sharpe: {metrics.get('sharpe_ratio', 0):.2f}, "
                          f"Return: {metrics.get('total_return', 0):.2%}")
                
            except Exception as e:
                logger.error(f"Error running {name}: {e}")
                results.append({
                    'Strategy': name,
                    'Error': str(e)
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Store for later use
        self.equity_curves = equity_curves
        self.all_trades = all_trades
        self.comparison_df = comparison_df
        
        return comparison_df
    
    def print_comparison_table(self, comparison_df: pd.DataFrame = None):
        """Print a formatted comparison table."""
        if comparison_df is None:
            comparison_df = self.comparison_df
        
        print("\n" + "="*100)
        print("STRATEGY COMPARISON RESULTS")
        print("="*100)
        
        # Select key metrics to display
        display_cols = [
            'Strategy', 'total_return', 'sharpe_ratio', 'max_drawdown',
            'win_rate', 'profit_factor', 'Num_Trades'
        ]
        
        # Filter available columns
        available_cols = [col for col in display_cols if col in comparison_df.columns]
        display_df = comparison_df[available_cols].copy()
        
        # Format percentages
        if 'total_return' in display_df.columns:
            display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x:.2%}")
        if 'max_drawdown' in display_df.columns:
            display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
        if 'win_rate' in display_df.columns:
            display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2%}")
        
        # Format floats
        if 'sharpe_ratio' in display_df.columns:
            display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
        if 'profit_factor' in display_df.columns:
            display_df['profit_factor'] = display_df['profit_factor'].apply(lambda x: f"{x:.2f}")
        
        print(display_df.to_string(index=False))
        print("="*100)
        
        # Highlight best performers
        if 'sharpe_ratio' in comparison_df.columns:
            best_sharpe = comparison_df.loc[comparison_df['sharpe_ratio'].idxmax(), 'Strategy']
            print(f"\n🏆 Best Sharpe Ratio: {best_sharpe}")
        
        if 'total_return' in comparison_df.columns:
            best_return = comparison_df.loc[comparison_df['total_return'].idxmax(), 'Strategy']
            print(f"💰 Best Total Return: {best_return}")
        
        if 'max_drawdown' in comparison_df.columns:
            best_dd = comparison_df.loc[comparison_df['max_drawdown'].idxmin(), 'Strategy']
            print(f"🛡️  Lowest Drawdown: {best_dd}")
        
        print()
    
    def export_results(self, filename: str = None):
        """Export comparison results to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_comparison_{timestamp}.csv"
        
        self.comparison_df.to_csv(filename, index=False)
        logger.info(f"Results exported to: {filename}")
        
        return filename
    
    def get_best_strategy(self, metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Get the best performing strategy based on a metric.
        
        Args:
            metric: Metric to use for ranking
            
        Returns:
            dict with strategy name and metrics
        """
        if metric not in self.comparison_df.columns:
            raise ValueError(f"Metric '{metric}' not found in results")
        
        best_idx = self.comparison_df[metric].idxmax()
        best_strategy = self.comparison_df.iloc[best_idx].to_dict()
        
        return best_strategy


def run_all_strategies(data: pd.DataFrame, initial_capital: float = 10000):
    """
    Convenience function to run all available strategies with default parameters.
    
    Args:
        data: Historical price data
        initial_capital: Starting capital
        
    Returns:
        comparison_df: Results DataFrame
    """
    strategies = [
        ('Mean Reversion', MeanReversionStrategy, {'bb_length': 20, 'bb_std': 2.0}),
        ('Trend Following', TrendFollowingStrategy, {'fast_ma': 50, 'slow_ma': 200}),
        ('Momentum (RSI)', MomentumStrategy, {'period': 14}),
        ('Breakout', BreakoutStrategy, {'lookback': 20}),
        ('MA Crossover (EMA)', MovingAverageCrossoverStrategy, {
            'fast_period': 10, 'slow_period': 30, 'ma_type': 'EMA'
        }),
        ('MA Crossover (SMA)', MovingAverageCrossoverStrategy, {
            'fast_period': 10, 'slow_period': 30, 'ma_type': 'SMA'
        }),
        ('MACD', MACDStrategy, {}),
        ('MACD Divergence', MACDDivergenceStrategy, {}),
    ]
    
    comparator = StrategyComparator(data, initial_capital)
    results = comparator.compare_strategies(strategies)
    comparator.print_comparison_table(results)
    
    return comparator, results


if __name__ == "__main__":
    print("Multi-Strategy Comparison System")
    print("=" * 50)
    print("\nExample usage:")
    print("""
    from trading_system.data.ingestion import DataIngestion
    from trading_system.scripts.compare_strategies import run_all_strategies
    
    # Load data
    data_loader = DataIngestion()
    df = data_loader.fetch_binance_data('BTCUSDT', '1h', days=365)
    
    # Run all strategies
    comparator, results = run_all_strategies(df, initial_capital=10000)
    
    # Export results
    comparator.export_results()
    
    # Get best strategy
    best = comparator.get_best_strategy('sharpe_ratio')
    print(f"Best strategy: {best['Strategy']}")
    """)
