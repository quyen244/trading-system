import pandas as pd
import ta 
from trading_system.utils.logger import setup_logger

logger = setup_logger('FeatureEngineer')

class FeatureEngineer:
    def __init__(self):
        pass

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and features to the dataframe.
        """
        df = df.copy()
        
        # Ensure we have required columns
        # pandas_ta usually expects open, high, low, close, volume (lowercase)
        
        # 1. Momentum Indicators
        df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD - Extract individual series from MACD indicator
        macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        df['MACD_diff'] = macd_indicator.macd_diff() 

        # 2. Volatility Indicators
        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['ATR_14'] = atr_indicator.average_true_range()
        
        # Bollinger Bands - Extract individual series
        bbands = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_high'] = bbands.bollinger_hband()
        df['BB_low'] = bbands.bollinger_lband()
        df['BB_mid'] = bbands.bollinger_mavg()
        df['BB_width'] = bbands.bollinger_wband()
        df['BB_pband'] = bbands.bollinger_pband()
        
        # 3. Trend Indicators
        df['EMA_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['EMA_100'] = ta.trend.EMAIndicator(df['close'], window=100).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        # ADX - Extract individual series
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['ADX'] = adx_indicator.adx()
        df['ADX_pos'] = adx_indicator.adx_pos()
        df['ADX_neg'] = adx_indicator.adx_neg()

        # 4. Volume Indicators
        vwap_indicator = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'])
        df['VWAP'] = vwap_indicator.volume_weighted_average_price()
        
        # 5. Lag Features (Returns)
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

        # 6. Target Variable (Forward Return for ML)
        df['target_return'] = df['close'].shift(-1).pct_change(1) # Next candle return
        
        # Drop NaNs created by lagging/rolling
        df.dropna(inplace=True)
        
        return df

    def categorize_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Market Regime features (Trend vs Range).
        """
        # Example logic: ADX > 25 implies Trend, else Range
        if 'ADX' in df.columns:
            df['is_trending'] = (df['ADX'] > 25).astype(int)
        
        # Volatility Regime
        if 'ATR_14' in df.columns:
            df['volatility_regime'] = pd.qcut(df['ATR_14'], q=3, labels=[0, 1, 2]).astype(int) # Low, Mid, High
            
        return df
