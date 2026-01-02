import pandas as pd
import ta
from trading_system.strategies.base_strategy import BaseStrategy

class MovingAverageCrossoverStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        """
        Moving Average Crossover Strategy
        
        Args:
            params (dict): Strategy parameters
                - fast_period: Fast MA period (default: 10)
                - slow_period: Slow MA period (default: 30)
                - ma_type: Type of MA - 'SMA' or 'EMA' (default: 'EMA')
        """
        super().__init__("MA_Crossover", params)
        self.fast_period = self.params.get('fast_period', 10)
        self.slow_period = self.params.get('slow_period', 30)
        self.ma_type = self.params.get('ma_type', 'EMA')

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MA crossover
        
        Args:
            data (pd.DataFrame): OHLCV dataframe
            
        Returns:
            pd.DataFrame: DataFrame with 'signal' column
        """
        data = self.validate_data(data)
        df = data.copy()
        
        # Calculate moving averages
        if self.ma_type == 'EMA':
            df['MA_fast'] = ta.trend.EMAIndicator(df['close'], window=self.fast_period).ema_indicator()
            df['MA_slow'] = ta.trend.EMAIndicator(df['close'], window=self.slow_period).ema_indicator()
        else:  # SMA
            df['MA_fast'] = ta.trend.SMAIndicator(df['close'], window=self.fast_period).sma_indicator()
            df['MA_slow'] = ta.trend.SMAIndicator(df['close'], window=self.slow_period).sma_indicator()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Generate signals
        # Buy when fast MA crosses above slow MA
        df.loc[(df['MA_fast'] > df['MA_slow']) & 
               (df['MA_fast'].shift(1) <= df['MA_slow'].shift(1)), 'signal'] = 1
        
        # Sell when fast MA crosses below slow MA
        df.loc[(df['MA_fast'] < df['MA_slow']) & 
               (df['MA_fast'].shift(1) >= df['MA_slow'].shift(1)), 'signal'] = -1
        
        return df
