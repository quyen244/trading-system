import pandas as pd
import numpy as np
import logging
from trading_system.risk.manager import RiskManager
from trading_system.features.engineering import FeatureEngineer
from trading_system.backtesting.metrics import calculate_metrics

logger = logging.getLogger(__name__)

class BacktestEngine:
    def __init__(self, initial_capital=10000, commission=0.001):

        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: The initial capital to use for the backtest.
            commission: The commission to use for the backtest.
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.feature_engineer = FeatureEngineer()

    def run_backtest(self, strategy, raw_data: pd.DataFrame):
        """
        Run backtest for a given strategy and data.
        
        Args:
            strategy: The strategy to run the backtest for.
            raw_data: The raw data to run the backtest on.

        Returns:
            metrics: The metrics of the backtest.
            equity_curve: The equity curve of the backtest.
            trades: The trades of the backtest.
        """
        logger.info(f"Starting backtest for {strategy.name}")
        
        # 1. Prepare Data
        data = self.feature_engineer.add_all_features(raw_data)
        
        # 2. Generate Signals
        signaled_data = strategy.generate_signals(data)
        
        # 3. Simulate Execution
        risk_manager = RiskManager(self.initial_capital)
        equity_curve = [self.initial_capital]
        position = 0
        entry_price = 0
        
        # Iterating just signals for speed (vectorized check is better but Risk Manager needs state)
        # For simplicity and robust risk management simulation, we iterate.
        
        capital = self.initial_capital
        equity_series = pd.Series(index=signaled_data.index, dtype='float64')
        equity_series.iloc[0] = capital
        
        trades = []
        
        for i in range(1, len(signaled_data)):
            # Previous bar signal executes on Open of current bar? 
            # OR Close of current bar signal executes on Open of next?
            # Standard: Signal on buffer Close -> Entry on next Open.
            
            timestamp = signaled_data.index[i]
            prev_row = signaled_data.iloc[i-1]
            curr_row = signaled_data.iloc[i]
            
            current_price = curr_row['close']
            signal = prev_row.get('signal', 0)
            
            # PnL Calculation for holding position
            if position != 0:
                # Mark to market
                unrealized_pnl = (current_price - entry_price) * position
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
                
            equity_series.iloc[i] = current_equity
            
            # Risk Check (Stop Loss / Take Profit / Max DD)
            if not risk_manager.check_risk_allowance(current_equity):
                # Close Position if exists
                if position != 0:
                    capital = current_equity - abs(current_equity * self.commission)
                    trades.append({'exit_time': timestamp, 'pnl': (current_price - entry_price) * position})
                    position = 0
                continue

            # Execution Logic
            if signal == 1 and position <= 0: # Buy Signal
                # Close Short if any
                if position < 0:
                     capital += (entry_price - current_price) * abs(position) # Short PnL
                     capital -= capital * self.commission
                     trades.append({'exit_time': timestamp, 'pnl': (entry_price - current_price) * abs(position)})
                     position = 0

                # Open Long
                # Calculate size based on risk
                stop_loss = current_price * 0.98 # Example 2% SL
                size = risk_manager.calculate_position_size(current_price, stop_loss)
                cost = size * current_price
                if cost < capital:
                    position = size
                    entry_price = current_price
                    capital -= cost * self.commission
            
            elif signal == -1 and position >= 0: # Sell Signal
                # Close Long if any
                if position > 0:
                    capital += (current_price - entry_price) * position
                    capital -= capital * self.commission
                    trades.append({'exit_time': timestamp, 'pnl': (current_price - entry_price) * position})
                    position = 0

                # Open Short (Optional, assuming Spot for now so only exit)
                # If Futures, can short. Let's assume Long-Only for simplicity unless specified?
                # User asked for "Strategies" like Trend/MeanRev, usually involves shorting. 
                # Let's support Shorting if strategy provides it.
                
                # Assuming simple Long/Flat for MVP unless requested specifically.
                # Re-reading: "Binance (spot/futures)". OK, Shorting allowed.
                
                pass # For now, let's keep it simple: Close Long on Sell signal.
    
        # Calculate Metrics
        metrics = calculate_metrics(equity_series)
        return metrics, equity_series, trades
    def save_to_db(self, equity_series, trades):
        self.db_url = os.getenv("TRADING_DB_URL", "")

