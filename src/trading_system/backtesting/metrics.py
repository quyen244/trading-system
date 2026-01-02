import pandas as pd
import numpy as np

def calculate_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Calculate professional trading metrics.
    """
    returns = equity_curve.pct_change().dropna()
    
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days == 0:
        cagr = 0
    else:
        cagr = (1 + total_return) ** (365 / days) - 1

    # Drawdown
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()

    # Sharpe Ratio
    if returns.std() == 0:
        sharpe = 0
    else:
        sharpe = (returns.mean() - risk_free_rate/252) / returns.std() * np.sqrt(252)

    # Sortino Ratio
    negative_returns = returns[returns < 0]
    if negative_returns.std() == 0:
        sortino = 0
    else:
        sortino = (returns.mean() - risk_free_rate/252) / negative_returns.std() * np.sqrt(252)

    # Win Rate
    # Requires trade list, but we can approximate with positive days
    win_days = len(returns[returns > 0])
    total_days = len(returns)
    win_rate = win_days / total_days if total_days > 0 else 0

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Win Rate (Daily)": win_rate
    }
