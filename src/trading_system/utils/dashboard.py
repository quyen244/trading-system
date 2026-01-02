from trading_system.strategies import *
from typing import Dict, List
import pandas as pd


def get_strategy(name):
    strats = {
        "Mean Reversion": MeanReversionStrategy,
        "RSI": RsiStrategy,
        "MACD": MACDStrategy,
        "Break out": BreakoutStrategy,
        "MA Crossover": MovingAverageCrossoverStrategy,
        "MACD Divergence": MACDDivergenceStrategy
    }
    return strats.get(name)

def backtest_data_format(strategy_id: int, symbol: str, metrics: Dict, equity_series: pd.Series , params: Dict = None):
    """
    return to dict format for backtest data.
    
    Args:
        strategy_id (int): ID of the strategy.
        symbol (str): Trading symbol.
        metrics (Dict): Backtest metrics.
        equity_series (pd.Series): Equity curve.
    """

    backtest_data = {
        "strategy_id": strategy_id,
        "symbol": symbol,
        "start_date": equity_series.index[0],
        "end_date": equity_series.index[-1],
        "initial_capital": metrics["initial_capital"],
        "total_return": metrics["total_return"],
        "cagr": metrics["cagr"],
        "max_drawdown": metrics["max_drawdown"],
        "sharpe_ratio": metrics["sharpe_ratio"],
        "sortino_ratio": metrics["sortino_ratio"],
        "win_rate_daily": metrics["win_rate_daily"],
        "num_trades": metrics["num_trades"],
        "params": params
    }

    return backtest_data

def get_id_strategy(name: str) -> int:
    strats = {
        "RSI": 1,
        "MACD": 2,
        "Break out": 3,
        "Mean Reversion": 4,
        "MA Crossover": 5,
        "MACD Divergence": 6
    }

    return strats.get(name)