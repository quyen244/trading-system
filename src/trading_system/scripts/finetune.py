"""
Strategy Parameter Fine-Tuning System

This module provides tools for optimizing trading strategy parameters using:
- Grid Search
- Random Search  
- Bayesian Optimization

Author: Trading System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from itertools import product
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from trading_system.backtesting.engine import BacktestEngine
from trading_system.utils.logger import setup_logger

logger = setup_logger('StrategyOptimizer')


class StrategyOptimizer:
    """
    User-friendly strategy parameter optimization class.
    
    Example usage:
        optimizer = StrategyOptimizer(strategy_class=MeanReversionStrategy, data=df)
        
        # Define parameter grid
        param_grid = {
            'bb_length': [10, 20, 30],
            'bb_std': [1.5, 2.0, 2.5]
        }
        
        # Run optimization
        best_params, results = optimizer.grid_search(param_grid)
    """
    
    def __init__(
        self, 
        strategy_class, 
        data: pd.DataFrame,
        initial_capital: float = 10000,
        commission: float = 0.001,
        optimization_metric: str = 'sharpe_ratio'
    ):
        """
        Initialize the optimizer.
        
        Args:
            strategy_class: Strategy class to optimize (not instance)
            data: Historical price data for backtesting
            initial_capital: Starting capital for backtests
            commission: Commission rate
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'win_rate', etc.)
        """
        self.strategy_class = strategy_class
        self.data = data
        self.initial_capital = initial_capital
        self.commission = commission
        self.optimization_metric = optimization_metric
        self.backtest_engine = BacktestEngine(initial_capital, commission)
        
    def _run_single_backtest(self, params: Dict[str, Any]) -> Tuple[Dict, float]:
        """Run a single backtest with given parameters."""
        try:
            # Create strategy instance with params
            strategy = self.strategy_class(params=params)
            
            # Run backtest
            metrics, equity_curve, trades = self.backtest_engine.run_backtest(strategy, self.data)
            
            # Extract optimization metric
            metric_value = metrics.get(self.optimization_metric, 0)
            
            return params, metric_value, metrics
        except Exception as e:
            logger.error(f"Error in backtest with params {params}: {e}")
            return params, -np.inf, {}
    
    def grid_search(
        self, 
        param_grid: Dict[str, List],
        n_jobs: int = 1,
        verbose: bool = True
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Perform grid search optimization.
        
        Args:
            param_grid: Dictionary of parameter names to lists of values
                Example: {'bb_length': [10, 20, 30], 'bb_std': [1.5, 2.0]}
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            verbose: Show progress bar
            
        Returns:
            best_params: Best parameter combination
            results_df: DataFrame with all results
        """
        logger.info(f"Starting grid search with {len(param_grid)} parameters")
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))
        
        logger.info(f"Total combinations to test: {len(param_combinations)}")
        
        results = []
        
        if verbose:
            pbar = tqdm(total=len(param_combinations), desc="Grid Search")
        
        # Run backtests
        for combo in param_combinations:
            params = dict(zip(param_names, combo))
            _, metric_value, metrics = self._run_single_backtest(params)
            
            result = {**params, self.optimization_metric: metric_value, **metrics}
            results.append(result)
            
            if verbose:
                pbar.update(1)
                pbar.set_postfix({self.optimization_metric: f"{metric_value:.4f}"})
        
        if verbose:
            pbar.close()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(self.optimization_metric, ascending=False)
        
        # Get best parameters
        best_params = results_df.iloc[0][param_names].to_dict()
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {self.optimization_metric}: {results_df.iloc[0][self.optimization_metric]:.4f}")
        
        return best_params, results_df
    
    def random_search(
        self,
        param_distributions: Dict[str, Tuple],
        n_iter: int = 50,
        random_state: Optional[int] = None,
        verbose: bool = True
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Perform random search optimization.
        
        Args:
            param_distributions: Dictionary of parameter names to (min, max) tuples
                Example: {'bb_length': (10, 50), 'bb_std': (1.0, 3.0)}
            n_iter: Number of random combinations to try
            random_state: Random seed for reproducibility
            verbose: Show progress bar
            
        Returns:
            best_params: Best parameter combination
            results_df: DataFrame with all results
        """
        logger.info(f"Starting random search with {n_iter} iterations")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        results = []
        
        if verbose:
            pbar = tqdm(total=n_iter, desc="Random Search")
        
        for i in range(n_iter):
            # Generate random parameters
            params = {}
            for param_name, (min_val, max_val) in param_distributions.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)
            
            # Run backtest
            _, metric_value, metrics = self._run_single_backtest(params)
            
            result = {**params, self.optimization_metric: metric_value, **metrics}
            results.append(result)
            
            if verbose:
                pbar.update(1)
                pbar.set_postfix({self.optimization_metric: f"{metric_value:.4f}"})
        
        if verbose:
            pbar.close()
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(self.optimization_metric, ascending=False)
        
        # Get best parameters
        param_names = list(param_distributions.keys())
        best_params = results_df.iloc[0][param_names].to_dict()
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {self.optimization_metric}: {results_df.iloc[0][self.optimization_metric]:.4f}")
        
        return best_params, results_df
    
    def bayesian_optimization(
        self,
        param_bounds: Dict[str, Tuple],
        n_iter: int = 30,
        init_points: int = 5,
        verbose: bool = True
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Perform Bayesian optimization (requires bayesian-optimization package).
        
        Args:
            param_bounds: Dictionary of parameter names to (min, max) tuples
            n_iter: Number of optimization iterations
            init_points: Number of random exploration points
            verbose: Show progress
            
        Returns:
            best_params: Best parameter combination
            results_df: DataFrame with all results
        """
        try:
            from bayes_opt import BayesianOptimization
        except ImportError:
            logger.error("bayesian-optimization package not installed. Run: pip install bayesian-optimization")
            raise
        
        logger.info(f"Starting Bayesian optimization with {n_iter} iterations")
        
        # Define objective function
        def objective(**params):
            _, metric_value, _ = self._run_single_backtest(params)
            return metric_value
        
        # Create optimizer
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_bounds,
            random_state=42,
            verbose=2 if verbose else 0
        )
        
        # Run optimization
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        
        # Get best parameters
        best_params = optimizer.max['params']
        
        # Convert results to DataFrame
        results = []
        for i, res in enumerate(optimizer.res):
            result = {**res['params'], self.optimization_metric: res['target']}
            results.append(result)
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(self.optimization_metric, ascending=False)
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best {self.optimization_metric}: {optimizer.max['target']:.4f}")
        
        return best_params, results_df
    
    def walk_forward_optimization(
        self,
        param_grid: Dict[str, List],
        train_size: int = 252,  # 1 year
        test_size: int = 63,    # 3 months
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Perform walk-forward optimization to avoid overfitting.
        
        Args:
            param_grid: Parameter grid for optimization
            train_size: Number of periods for training
            test_size: Number of periods for testing
            verbose: Show progress
            
        Returns:
            results_df: DataFrame with walk-forward results
        """
        logger.info("Starting walk-forward optimization")
        
        results = []
        n_splits = (len(self.data) - train_size) // test_size
        
        for i in range(n_splits):
            train_start = i * test_size
            train_end = train_start + train_size
            test_start = train_end
            test_end = test_start + test_size
            
            if test_end > len(self.data):
                break
            
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            # Optimize on training data
            temp_optimizer = StrategyOptimizer(
                self.strategy_class,
                train_data,
                self.initial_capital,
                self.commission,
                self.optimization_metric
            )
            
            best_params, _ = temp_optimizer.grid_search(param_grid, verbose=False)
            
            # Test on out-of-sample data
            temp_optimizer.data = test_data
            _, test_metric, test_metrics = temp_optimizer._run_single_backtest(best_params)
            
            results.append({
                'fold': i + 1,
                'train_period': f"{train_start}-{train_end}",
                'test_period': f"{test_start}-{test_end}",
                **best_params,
                f'test_{self.optimization_metric}': test_metric,
                **{f'test_{k}': v for k, v in test_metrics.items()}
            })
            
            if verbose:
                logger.info(f"Fold {i+1}/{n_splits}: Test {self.optimization_metric} = {test_metric:.4f}")
        
        results_df = pd.DataFrame(results)
        
        return results_df


if __name__ == "__main__":
    # Example usage
    print("Strategy Optimizer - Example Usage")
    print("=" * 50)
    
    # This is just a template - actual usage would be:
    # from trading_system.strategies.mean_reversion import MeanReversionStrategy
    # from trading_system.data.ingestion import DataIngestion
    # 
    # data_loader = DataIngestion()
    # df = data_loader.fetch_binance_data('BTCUSDT', '1h', days=365)
    # 
    # optimizer = StrategyOptimizer(
    #     strategy_class=MeanReversionStrategy,
    #     data=df,
    #     optimization_metric='sharpe_ratio'
    # )
    # 
    # param_grid = {
    #     'bb_length': [10, 20, 30, 40],
    #     'bb_std': [1.5, 2.0, 2.5, 3.0]
    # }
    # 
    # best_params, results = optimizer.grid_search(param_grid)
    # print(f"Best parameters: {best_params}")
    # print(results.head())
