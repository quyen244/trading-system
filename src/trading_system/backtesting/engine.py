import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from trading_system.risk.manager import RiskManager
from trading_system.features.engineering import FeatureEngineer
from trading_system.backtesting.metrics import calculate_metrics
from trading_system.backtesting.trade import Trade
from trading_system.data.storage import StorageEngine

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, initial_capital: float = 10000, commission: float = 0.0002, slippage: float = 0.0001):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: The initial capital to use for the backtest.
            commission: Trading fee (default 0.02% = 0.0002).
            slippage: Expected slippage per trade.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.feature_engineer = FeatureEngineer()
        self.storage = StorageEngine()

    def run_backtest(self, strategy, raw_data: pd.DataFrame) -> Tuple[Dict, pd.Series, List[Trade]]:
        """
        Run backtest for a given strategy and data.
        """
        logger.info(f"Starting backtest for {strategy.name}")
        
        # 1. Prepare Data
        data = self.feature_engineer.add_all_features(raw_data)
        
        # 2. Generate Signals
        signaled_data = strategy.generate_signals(data)
        
        # 3. Simulate Execution
        risk_manager = RiskManager(self.initial_capital)
        current_capital = self.initial_capital
        equity_series = pd.Series(index=signaled_data.index, dtype='float64')
        equity_series.iloc[0] = current_capital
        
        trades = []
        active_trade: Optional[Trade] = None
        
        for i in range(1, len(signaled_data)):
            timestamp = signaled_data.index[i]
            prev_row = signaled_data.iloc[i-1]
            curr_row = signaled_data.iloc[i]
            
            close_price = curr_row['close']
            high_price = curr_row['high']
            low_price = curr_row['low']
            atr = curr_row.get('ATR_14', 0)
            signal = prev_row.get('signal', 0)
            
            # 3.1 Handle Active Position (Exit Logic)
            if active_trade:
                exit_price = None
                exit_reason = None
                
                if active_trade.side == "long":
                    # Check SL
                    if low_price <= active_trade.stop_loss:
                        exit_price = active_trade.stop_loss
                        exit_reason = "Stop Loss"
                    # Check TP
                    elif high_price >= active_trade.take_profit:
                        exit_price = active_trade.take_profit
                        exit_reason = "Take Profit"
                    # Check Exit Signal (Short signal)
                    elif signal == -1:
                        exit_price = close_price
                        exit_reason = "Signal"
                
                elif active_trade.side == "short":
                    # Check SL
                    if high_price >= active_trade.stop_loss:
                        exit_price = active_trade.stop_loss
                        exit_reason = "Stop Loss"
                    # Check TP
                    elif low_price <= active_trade.take_profit:
                        exit_price = active_trade.take_profit
                        exit_reason = "Take Profit"
                    # Check Exit Signal (Long signal)
                    elif signal == 1:
                        exit_price = close_price
                        exit_reason = "Signal"
                
                if exit_price is not None:
                    # Apply Slippage and Comission
                    adjusted_exit = exit_price * (1 - self.slippage) if active_trade.side == "long" else exit_price * (1 + self.slippage)
                    
                    active_trade.exit_time = timestamp
                    active_trade.exit_price = adjusted_exit
                    
                    if active_trade.side == "long":
                        active_trade.gross_pnl = (active_trade.exit_price - active_trade.entry_price) * active_trade.position_size
                    else:
                        active_trade.gross_pnl = (active_trade.entry_price - active_trade.exit_price) * active_trade.position_size
                    
                    # Net PnL = Gross PnL - Fees (on entry and exit volume)
                    entry_v = active_trade.entry_price * active_trade.position_size
                    exit_v = active_trade.exit_price * active_trade.position_size
                    fees = (entry_v + exit_v) * self.commission
                    active_trade.net_pnl = active_trade.gross_pnl - fees
                    
                    current_capital += active_trade.net_pnl
                    trades.append(active_trade)
                    active_trade = None

            # 3.2 Open New Position (Entry Logic)
            if not active_trade and signal != 0:
                side = "long" if signal == 1 else "short"
                entry_price = close_price * (1 + self.slippage) if side == "long" else close_price * (1 - self.slippage)
                
                # Risk Rules: 1.5x ATR for SL, 3x ATR for TP
                sl_dist = 1.5 * atr if atr > 0 else entry_price * 0.02
                tp_dist = 3.0 * atr if atr > 0 else entry_price * 0.04
                
                if side == "long":
                    stop_loss = entry_price - sl_dist
                    take_profit = entry_price + tp_dist
                else:
                    stop_loss = entry_price + sl_dist
                    take_profit = entry_price - tp_dist
                
                # Position Sizing (Risk 1% of capital per 1.5x ATR)
                risk_amount = current_capital * 0.01
                position_size = risk_amount / sl_dist if sl_dist > 0 else 0
                
                if position_size > 0:
                    active_trade = Trade(
                        symbol=signaled_data.get('symbol', 'UNKNOWN'),
                        entry_time=timestamp,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        position_size=position_size,
                        side=side
                    )

            # 3.3 Update Equity Series
            unrealized_pnl = 0
            if active_trade:
                if active_trade.side == "long":
                    unrealized_pnl = (close_price - active_trade.entry_price) * active_trade.position_size
                else:
                    unrealized_pnl = (active_trade.entry_price - close_price) * active_trade.position_size
                
                # Subtract entry fees from unrealized
                entry_v = active_trade.entry_price * active_trade.position_size
                unrealized_pnl -= entry_v * self.commission
            
            equity_series.iloc[i] = current_capital + unrealized_pnl

        # 4. Calculate Final Metrics
        final_metrics = calculate_metrics(equity_series, trades)
        
        return final_metrics, equity_series, trades

    def save_results(self, strategy_name: str, symbol: str, metrics: Dict, trades: List[Trade], equity_series: pd.Series):
        """
        Persist backtest results to database.
        
        Args:
            strategy_name (str): Name of the strategy.
            symbol (str): Trading symbol.
            metrics (Dict): Backtest metrics.
            trades (List[Trade]): List of trades.
            equity_series (pd.Series): Equity curve.
        """
        backtest_data = {
            "strategy_id": strategy_name,
            "symbol": symbol,
            "start_date": equity_series.index[0],
            "end_date": equity_series.index[-1],
            "initial_capital": self.initial_capital,
            "total_return": metrics["total_return"],
            "cagr": metrics["cagr"],
            "max_drawdown": metrics["max_drawdown"],
            "sharpe_ratio": metrics["sharpe_ratio"],
            "sortino_ratio": metrics["sortino_ratio"],
            "win_rate_daily": metrics["win_rate_daily"],
            "num_trades": metrics["num_trades"]
        }
        
        try:
            bt_id = self.storage.save_backtest(backtest_data, trades)
            return bt_id
        except Exception as e:
            logger.error(f"Failed to save backtest: {e}")
            return None

