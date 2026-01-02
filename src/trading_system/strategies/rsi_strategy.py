# src/strategies/rsi_momentum.py

import pandas as pd
import ta 
from trading_system.strategies.base_strategy import BaseStrategy
from trading_system.risk.manager import RiskManager

class RsiStrategy(BaseStrategy):
    def __init__(self, params: dict, risk_manager = RiskManager):
        super().__init__(params, risk_manager)
        self.name = "RsiMomentum_v1"
        
        # Load params (có giá trị mặc định để tránh lỗi)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_lower = params.get('rsi_lower', 30)
        self.rsi_upper = params.get('rsi_upper', 70)
        self.ema_trend = params.get('ema_trend', 200)
        self.atr_period = params.get('atr_period', 14)

    def get_mlflow_params(self) -> dict:
        return {
            "strategy": self.name,
            "rsi_period": self.rsi_period,
            "rsi_lower": self.rsi_lower,
            "ema_trend": self.ema_trend,
            "atr_period": self.atr_period
        }

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.validate_data(df.copy())

        # 1. Feature Engineering (Tính chỉ báo) - FIXED TA USAGE
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=self.rsi_period).rsi()
        df['EMA_Trend'] = ta.trend.EMAIndicator(df['close'], window=self.ema_trend).ema_indicator()
        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=self.atr_period)
        df['ATR'] = atr_indicator.average_true_range()

  
        df['signal'] = 0        
        df['entry_price'] = 0.0
        df['quantity'] = 0.0
        df['stop_loss'] = 0.0
        df['take_profit'] = 0.0


        trend_up = df['close'] > df['EMA_Trend']
        
   
        rsi_cross_up = (df['RSI'].shift(1) < self.rsi_lower) & (df['RSI'] >= self.rsi_lower)

        entry_condition = trend_up & rsi_cross_up

        for i in df.index[df['close'] > 0]: 
             if entry_condition[i]:
                current_price = df.loc[i, 'close']
                current_atr = df.loc[i, 'ATR']
                
              
                trade_setup = self.risk_manager.calculate_position_size(
                    entry_price=current_price,
                    atr_value=current_atr,
                    atr_multiplier=2.0 
                )
                
                if trade_setup:
                    df.loc[i, 'signal'] = 1
                    df.loc[i, 'entry_price'] = current_price
                    df.loc[i, 'quantity'] = trade_setup['quantity']
                    df.loc[i, 'stop_loss'] = trade_setup['stop_loss']
                    df.loc[i, 'take_profit'] = trade_setup['take_profit']

        # Logic bán (Exit): Bán khi RSI chạm Overbought hoặc chạm SL/TP (được xử lý ở Backtest Engine)
        # ở đây ta chỉ đưa ra tín hiệu thoát chủ động theo chỉ báo
        exit_condition = df['RSI'] > self.rsi_upper
        df.loc[exit_condition, 'signal'] = -1
        
        return df
