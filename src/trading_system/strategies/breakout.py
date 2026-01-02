import pandas as pd
import ta
from trading_system.strategies.base_strategy import BaseStrategy

class BreakoutStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        super().__init__("Breakout", params)
        self.lookback = self.params.get('lookback', 20)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.validate_data(data)
        df = data.copy()
        
        # Donchian Channels (Max High / Min Low of last N periods)
        df['rolling_high'] = df['high'].rolling(window=self.lookback).max().shift(1)
        df['rolling_low'] = df['low'].rolling(window=self.lookback).min().shift(1)
        
        df['signal'] = 0
        
        # Breakout High -> Buy
        df.loc[df['close'] > df['rolling_high'], 'signal'] = 1
        
        # Breakdown Low -> Sell
        df.loc[df['close'] < df['rolling_low'], 'signal'] = -1
        
        return df
