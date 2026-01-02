import pandas as pd
import ta
from trading_system.strategies.base_strategy import BaseStrategy

class TrendFollowingStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        super().__init__("TrendFollowing", params)
        self.fast_ma = self.params.get('fast_ma', 50)
        self.slow_ma = self.params.get('slow_ma', 200)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.validate_data(data)
        df = data.copy()
        
        if f'EMA_{self.fast_ma}' not in df.columns:
            df[f'EMA_{self.fast_ma}'] = ta.trend.EMAIndicator(df['close'], window=self.fast_ma).ema_indicator()
        if f'EMA_{self.slow_ma}' not in df.columns:
            df[f'EMA_{self.slow_ma}'] = ta.trend.EMAIndicator(df['close'], window=self.slow_ma).ema_indicator()
            
  
        df['signal'] = 0
        
        long_condition = (df[f'EMA_{self.fast_ma}'] > df[f'EMA_{self.slow_ma}'])
        
        short_condition = (df[f'EMA_{self.fast_ma}'] < df[f'EMA_{self.slow_ma}'])
        
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        return df

class MomentumStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        super().__init__("Momentum", params)
        self.period = self.params.get('period', 14)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.validate_data(data)
        df = data.copy()
        
    
        if f'RSI_{self.period}' not in df.columns:
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=self.period).rsi()
        else:
            df['rsi'] = df[f'RSI_{self.period}']
            
        df['signal'] = 0
        df.loc[df['rsi'] > 70, 'signal'] = -1 # Overbought -> Sell
        df.loc[df['rsi'] < 30, 'signal'] = 1  # Oversold -> Buy
        
        return df
