import pandas as pd
import ta
from trading_system.strategies.base_strategy import BaseStrategy

class MACDStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        """
        MACD (Moving Average Convergence Divergence) Strategy
        
        Args:
            params (dict): Strategy parameters
                - fast_period: Fast EMA period (default: 12)
                - slow_period: Slow EMA period (default: 26)
                - signal_period: Signal line period (default: 9)
                - histogram_threshold: Minimum histogram value for signal (default: 0)
        """
        super().__init__("MACD", params)
        self.fast_period = self.params.get('fast_period', 5)
        self.slow_period = self.params.get('slow_period', 30)
        self.signal_period = self.params.get('signal_period', 9)
        self.histogram_threshold = self.params.get('histogram_threshold', 0)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MACD indicator
        
        Args:
            data (pd.DataFrame): OHLCV dataframe
            
        Returns:
            pd.DataFrame: DataFrame with 'signal' column
        """
        data = self.validate_data(data)
        df = data.copy()

        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['ATR_14'] = atr_indicator.average_true_range()
        
        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            df['close'], 
            window_slow=self.slow_period, 
            window_fast=self.fast_period, 
            window_sign=self.signal_period
        )
        
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        df['MACD_histogram'] = macd_indicator.macd_diff()
        df.dropna(inplace=True)
        # Initialize signal column
        df['signal'] = 0
        
        # Strategy 1: MACD Line crosses Signal Line
        # Buy when MACD crosses above signal line
        df.loc[(df['MACD'] > df['MACD_signal']) & 
               (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'signal'] = 1
        
        # Sell when MACD crosses below signal line
        df.loc[(df['MACD'] < df['MACD_signal']) & 
               (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'signal'] = -1
        
        # Strategy 2: Histogram crosses zero line (alternative, can be enabled)
        # Uncomment below to use histogram strategy instead
        # df.loc[(df['MACD_histogram'] > self.histogram_threshold) & 
        #        (df['MACD_histogram'].shift(1) <= self.histogram_threshold), 'signal'] = 1
        # df.loc[(df['MACD_histogram'] < -self.histogram_threshold) & 
        #        (df['MACD_histogram'].shift(1) >= -self.histogram_threshold), 'signal'] = -1
        
        return df


class MACDDivergenceStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        """
        MACD Divergence Strategy - Detects bullish/bearish divergences
        
        Args:
            params (dict): Strategy parameters
                - fast_period: Fast EMA period (default: 12)
                - slow_period: Slow EMA period (default: 26)
                - signal_period: Signal line period (default: 9)
                - lookback: Lookback period for divergence detection (default: 14)
        """
        super().__init__("MACD_Divergence", params)
        self.fast_period = self.params.get('fast_period', 12)
        self.slow_period = self.params.get('slow_period', 26)
        self.signal_period = self.params.get('signal_period', 9)
        self.lookback = self.params.get('lookback', 14)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on MACD divergence
        
        Args:
            data (pd.DataFrame): OHLCV dataframe
            
        Returns:
            pd.DataFrame: DataFrame with 'signal' column
        """
        data = self.validate_data(data)
        df = data.copy()
        
        # Calculate MACD
        macd_indicator = ta.trend.MACD(
            df['close'], 
            window_slow=self.slow_period, 
            window_fast=self.fast_period, 
            window_sign=self.signal_period
        )
        
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        
        # Find local highs and lows
        df['price_high'] = df['close'].rolling(window=self.lookback, center=True).max()
        df['price_low'] = df['close'].rolling(window=self.lookback, center=True).min()
        df['macd_high'] = df['MACD'].rolling(window=self.lookback, center=True).max()
        df['macd_low'] = df['MACD'].rolling(window=self.lookback, center=True).min()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Bullish divergence: Price makes lower low, but MACD makes higher low
        bullish_div = (
            (df['close'] == df['price_low']) & 
            (df['close'] < df['close'].shift(self.lookback)) &
            (df['MACD'] == df['macd_low']) &
            (df['MACD'] > df['MACD'].shift(self.lookback))
        )
        
        # Bearish divergence: Price makes higher high, but MACD makes lower high
        bearish_div = (
            (df['close'] == df['price_high']) & 
            (df['close'] > df['close'].shift(self.lookback)) &
            (df['MACD'] == df['macd_high']) &
            (df['MACD'] < df['MACD'].shift(self.lookback))
        )
        
        df.loc[bullish_div, 'signal'] = 1
        df.loc[bearish_div, 'signal'] = -1
        
        return df
