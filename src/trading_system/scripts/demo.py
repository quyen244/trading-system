"""
Complete Trading System Demo

This script demonstrates all the new features:
1. Strategy Parameter Fine-Tuning
2. Multi-Strategy Comparison
3. Beautiful Visualizations
4. Advanced Strategies
5. Adaptive Ensemble System

Author: Trading System
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Import data loading
from trading_system.data.storage import StorageEngine

# Import optimization
from trading_system.scripts.finetune import StrategyOptimizer

# Import comparison
from trading_system.scripts.compare_strategies import run_all_strategies, StrategyComparator

# Import visualization
from trading_system.visualization.charts import TradingVisualizer

# Import strategies
from trading_system.strategies.mean_reversion import MeanReversionStrategy
from trading_system.strategies.trend import TrendFollowingStrategy
from trading_system.strategies.advanced_strategies import (
    VolumeProfileStrategy,
    StatisticalArbitrageStrategy,
    MLMomentumStrategy
)
from trading_system.strategies.ensemble import create_default_ensemble

# Import backtesting
from trading_system.backtesting.engine import BacktestEngine


def demo_1_strategy_optimization():
    """Demo 1: Strategy Parameter Fine-Tuning"""
    print("\n" + "="*80)
    print("DEMO 1: STRATEGY PARAMETER FINE-TUNING")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    data_loader = StorageEngine()
    df = data_loader.load_market_data('BTCUSDT', '1h', start_date=datetime(2025, 1, 1), end_date=datetime(2025, 12, 31))
    
    # Create optimizer
    print("\n🔧 Creating optimizer for Mean Reversion Strategy...")
    optimizer = StrategyOptimizer(
        strategy_class=MeanReversionStrategy,
        data=df,
        optimization_metric='sharpe_ratio'
    )
    
    # Define parameter grid
    param_grid = {
        'bb_length': [15, 20, 25, 30],
        'bb_std': [1.5, 2.0, 2.5]
    }
    
    print(f"\n🔍 Running grid search with {len(param_grid['bb_length']) * len(param_grid['bb_std'])} combinations...")
    best_params, results = optimizer.grid_search(param_grid, verbose=True)
    
    print(f"\n✅ Best parameters found:")
    print(f"   BB Length: {best_params['bb_length']}")
    print(f"   BB Std: {best_params['bb_std']}")
    print(f"   Sharpe Ratio: {results.iloc[0]['sharpe_ratio']:.2f}")
    
    # Show top 5 results
    print("\n📈 Top 5 parameter combinations:")
    print(results.head())
    
    return best_params, results


def demo_2_strategy_comparison():
    """Demo 2: Multi-Strategy Comparison"""
    print("\n" + "="*80)
    print("DEMO 2: MULTI-STRATEGY COMPARISON")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    data_loader = StorageEngine()
    df = data_loader.load_market_data('BTCUSDT', '1h', start_date=datetime(2025, 1, 1), end_date=datetime(2025, 12, 31))
    
    # Run all strategies
    print("\n🚀 Running all strategies...")
    comparator, results = run_all_strategies(df, initial_capital=10000)
    
    # Export results
    filename = comparator.export_results()
    print(f"\n💾 Results exported to: {filename}")
    
    # Get best strategy
    best = comparator.get_best_strategy('sharpe_ratio')
    print(f"\n🏆 Best strategy by Sharpe Ratio: {best['Strategy']}")
    
    return comparator, results


def demo_3_visualizations(comparator):
    """Demo 3: Beautiful Visualizations"""
    print("\n" + "="*80)
    print("DEMO 3: BEAUTIFUL VISUALIZATIONS")
    print("="*80)
    
    print("\n🎨 Creating visualizations...")
    
    # Create visualizer
    viz = TradingVisualizer(style='dark')
    
    # Get equity curves from comparator
    equity_curves = comparator.equity_curves
    
    # Plot 1: Strategy Comparison
    print("   📊 Chart 1: Strategy Comparison")
    viz.plot_strategy_comparison(equity_curves, title="Strategy Performance Comparison")
    
    # Plot 2: Best Strategy Equity Curve
    best_strategy_name = list(equity_curves.keys())[0]
    best_equity = equity_curves[best_strategy_name]
    
    print("   📈 Chart 2: Equity Curve")
    viz.plot_equity_curve(best_equity, title=f"{best_strategy_name} - Equity Curve")
    
    # Plot 3: Drawdown Analysis
    print("   📉 Chart 3: Drawdown Analysis")
    viz.plot_drawdown(best_equity, title=f"{best_strategy_name} - Drawdown")
    
    # Plot 4: Trade Distribution
    best_trades = comparator.all_trades[best_strategy_name]
    print("   📊 Chart 4: Trade Distribution")
    viz.plot_trade_distribution(best_trades, title=f"{best_strategy_name} - Trade Analysis")
    
    # Plot 5: Monthly Returns Heatmap
    print("   🔥 Chart 5: Monthly Returns Heatmap")
    viz.plot_monthly_returns(best_equity, title=f"{best_strategy_name} - Monthly Returns")
    
    print("\n✅ All visualizations created!")
    print("   Close the chart windows to continue...")
    
    # Show all charts
    viz.show()
    
    # Save charts
    print("\n💾 Saving charts...")
    viz.save_all(prefix="demo_charts", dpi=300)
    
    return viz


def demo_4_advanced_strategies():
    """Demo 4: Advanced Trading Strategies"""
    print("\n" + "="*80)
    print("DEMO 4: ADVANCED TRADING STRATEGIES")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    data_loader = StorageEngine()
    df = data_loader.load_market_data('BTCUSDT', '1h', start_date=datetime(2025, 1, 1), end_date=datetime(2025, 12, 31))
    
    # Test advanced strategies
    advanced_strategies = [
        ('Volume Profile', VolumeProfileStrategy, {'lookback': 50, 'num_bins': 20}),
        ('Statistical Arbitrage', StatisticalArbitrageStrategy, {'window': 20, 'entry_threshold': 2.0}),
        ('ML Momentum', MLMomentumStrategy, {'lookback': 20, 'prediction_threshold': 0.6})
    ]
    
    print("\n🚀 Testing advanced strategies...")
    comparator = StrategyComparator(df, initial_capital=10000)
    results = comparator.compare_strategies(advanced_strategies)
    comparator.print_comparison_table(results)
    
    return comparator, results


def demo_5_adaptive_ensemble():
    """Demo 5: Adaptive Multi-Strategy Ensemble"""
    print("\n" + "="*80)
    print("DEMO 5: ADAPTIVE MULTI-STRATEGY ENSEMBLE")
    print("="*80)
    
    # Load data
    print("\n📊 Loading data...")
    data_loader = StorageEngine()
    df = data_loader.load_market_data('BTCUSDT', '1h', start_date=datetime(2025, 1, 1), end_date=datetime(2025, 12, 31))
    
    # Create ensemble
    print("\n🤖 Creating adaptive ensemble...")
    ensemble = create_default_ensemble()
    
    # Generate signals
    print("\n🔮 Generating ensemble signals...")
    signals = ensemble.generate_signals(df)
    
    # Show current weights
    print("\n⚖️  Current Strategy Weights:")
    weights = ensemble.get_current_weights()
    for name, weight in weights.items():
        print(f"   {name}: {weight:.2%}")
    
    # Show regime distribution
    print("\n🌍 Market Regime Distribution:")
    regime_dist = ensemble.get_regime_distribution(signals)
    for regime, pct in regime_dist.items():
        print(f"   {regime}: {pct:.1%}")
    
    # Run backtest
    print("\n📊 Running ensemble backtest...")
    backtest_engine = BacktestEngine(initial_capital=10000)
    metrics, equity_curve, trades = backtest_engine.run_backtest(ensemble, df)
    
    print("\n📈 Ensemble Performance:")
    print(f"   Total Return: {metrics.get('total_return', 0):.2%}")
    print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"   Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    print(f"   Win Rate: {metrics.get('win_rate', 0):.2%}")
    print(f"   Number of Trades: {len(trades)}")
    
    # Visualize
    print("\n🎨 Creating ensemble visualization...")
    viz = TradingVisualizer(style='dark')
    viz.plot_equity_curve(equity_curve, title="Adaptive Ensemble - Equity Curve")
    viz.plot_drawdown(equity_curve, title="Adaptive Ensemble - Drawdown")
    viz.show()
    
    return ensemble, signals, metrics


def run_full_demo():
    """Run the complete demo of all features"""
    print("\n" + "="*80)
    print("🚀 COMPLETE TRADING SYSTEM DEMO")
    print("="*80)
    print("\nThis demo showcases all new features:")
    print("1. Strategy Parameter Fine-Tuning")
    print("2. Multi-Strategy Comparison")
    print("3. Beautiful Visualizations (5 charts)")
    print("4. Advanced Trading Strategies")
    print("5. Adaptive Multi-Strategy Ensemble")
    print("\n" + "="*80)
    
    input("\nPress Enter to start Demo 1: Strategy Optimization...")
    best_params, opt_results = demo_1_strategy_optimization()
    
    input("\nPress Enter to start Demo 2: Strategy Comparison...")
    comparator, comp_results = demo_2_strategy_comparison()
    
    input("\nPress Enter to start Demo 3: Visualizations...")
    viz = demo_3_visualizations(comparator)
    
    input("\nPress Enter to start Demo 4: Advanced Strategies...")
    adv_comparator, adv_results = demo_4_advanced_strategies()
    
    input("\nPress Enter to start Demo 5: Adaptive Ensemble...")
    ensemble, signals, metrics = demo_5_adaptive_ensemble()
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("="*80)
    print("\n🎉 All features demonstrated successfully!")
    print("\nNext steps:")
    print("1. Review the exported CSV files")
    print("2. Check the saved chart images")
    print("3. Experiment with different parameters")
    print("4. Deploy the ensemble system for live trading")
    print("\n" + "="*80)


if __name__ == "__main__":
    # Run individual demos or full demo
    import sys
    
    if len(sys.argv) > 1:
        demo_num = sys.argv[1]
        
        if demo_num == '1':
            demo_1_strategy_optimization()
        elif demo_num == '2':
            demo_2_strategy_comparison()
        elif demo_num == '4':
            demo_4_advanced_strategies()
        elif demo_num == '5':
            demo_5_adaptive_ensemble()
        else:
            print("Usage: python demo.py [1|2|4|5]")
    else:
        # Run full demo
        run_full_demo()
