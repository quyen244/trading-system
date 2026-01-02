import pandas as pd
import numpy as np  

class RiskManager:
    def __init__(self, initial_capital=10000, risk_per_trade=0.01, max_daily_loss=0.03, max_drawdown=0.10):
        """
        Initialize the RiskManager with default values.
        Args:
            initial_capital: The initial capital to use for the backtest.
            risk_per_trade: The risk per trade as a percentage of the initial capital.
            max_daily_loss: The maximum daily loss as a percentage of the initial capital.
            max_drawdown: The maximum drawdown as a percentage of the initial capital.
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade # 1% per trade
        self.max_daily_loss = max_daily_loss # 3% daily
        self.max_drawdown = max_drawdown # 10% total
        
        self.daily_pnl = 0
        self.peak_capital = initial_capital
        self.kill_switch = False

    def check_risk_allowance(self, portfolio_value: float = None) -> bool:
        """
        Check if we are allowed to trade based on risk constraints.
        """
        if self.kill_switch:
            return False

        if portfolio_value:
             # Update Peak Capital
            if portfolio_value > self.peak_capital:
                self.peak_capital = portfolio_value
            
            # Check Drawdown
            drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
            if drawdown >= self.max_drawdown:
                print(f"KILL SWITCH ACTIVATED: Max Drawdown hit ({drawdown*100:.2f}%)")
                self.kill_switch = True
                return False

        # Check Daily Loss
        # Assuming daily_pnl is reset externally or tracked
        if self.daily_pnl <= -(self.initial_capital * self.max_daily_loss):
             print(f"Trading Halted: Max Daily Loss hit")
             return False

        return True

    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate position size based on risk amount and stop loss distance.
        Risk Amount = Capital * Risk Per Trade
        Position Size = Risk Amount / (Entry - SL)
        
        Args:
            entry_price: The entry price of the position.
            stop_loss: The stop loss price of the position.

        Returns:
            The position size based on risk amount and stop loss distance.
        """
        if not self.check_risk_allowance(self.current_capital):
            return 0.0

        risk_amount = self.current_capital * self.risk_per_trade
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0.0
            
        position_size = risk_amount / price_diff
        return position_size

    def calculate_volatility_adjusted_size(self, entry_price: float, atr: float, multiplier: float = 2.0) -> dict:
        """
        Calculate position size based on Volatility (ATR).
        SL Distance = ATR * Multiplier.
        
        Args:
            entry_price: The entry price of the position.
            atr: The ATR of the position.
            multiplier: The multiplier for the ATR.

        Returns:
            The position size based on Volatility (ATR).
        """
        if not self.check_risk_allowance(self.current_capital):
            return {"quantity": 0, "stop_loss": 0, "take_profit": 0}

        sl_distance = atr * multiplier
        stop_loss = entry_price - sl_distance # Assuming Long for calculation
        
        quantity = self.calculate_position_size(entry_price, stop_loss)
        take_profit = entry_price + (sl_distance * 2.0) # 1:2 Risk Reward
        
        return {
            "quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit
        }

    def update_pnl(self, pnl: float):
        self.daily_pnl += pnl
        self.current_capital += pnl
