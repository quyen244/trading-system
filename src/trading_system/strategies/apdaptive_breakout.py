import pandas as pd
from trading_system.strategies.base_strategy import BaseStrategy
from trading_system.utils.indicators import calculate_atr, calculate_ema

class AdaptiveBreakoutStrategy(BaseStrategy):
    def __init__(self, params=None):
        """
        Description: Adaptive Breakout Strategy
        
        """
        self.lookback_period = params.get('lookback', 20)  
        self.atr_period = params.get('atr_period', 14)
        self.trend_filter_period = params.get('trend_ema', 200)
        self.stop_loss_atr_mult = params.get('sl_mult', 2.0) 
        self.risk_per_trade = params.get('risk_per_trade', 0.02)

    def generate_signals(self, df: pd.DataFrame):
        """
        Args:
            df (pd.DataFrame): Dataframe OHLCV

        Returns:
            pd.DataFrame: 'signal', 'position_size', 'stop_loss_price'
        """
        # 1. TÃ­nh toÃ¡n chá»‰ bÃ¡o
        df['ATR'] = calculate_atr(df, self.atr_period)
        df['Trend_EMA'] = calculate_ema(df, self.trend_filter_period)
        
        # Donchian Channel (Äá»‰nh cao nháº¥t vÃ  ÄÃ¡y tháº¥p nháº¥t trong N phiÃªn)
        df['Upper_Channel'] = df['high'].rolling(self.lookback_period).max().shift(1)
        df['Lower_Channel'] = df['low'].rolling(self.lookback_period).min().shift(1)

        # 2. Logic vÃ o lá»‡nh (Entry Logic)
        # Mua khi GiÃ¡ > Upper Channel VÃ€ GiÃ¡ > EMA 200 (Chá»‰ Ä‘Ã¡nh thuáº­n xu hÆ°á»›ng dÃ i)
        long_condition = (df['close'] > df['Upper_Channel']) & (df['close'] > df['Trend_EMA'])
        
        # BÃ¡n khá»‘ng (Short) khi GiÃ¡ < Lower Channel VÃ€ GiÃ¡ < EMA 200
        short_condition = (df['close'] < df['Lower_Channel']) & (df['close'] < df['Trend_EMA'])

        # 3. Quáº£n lÃ½ vá»‘n (Position Sizing - Quan trá»ng nháº¥t cho Production)
        # CÃ´ng thá»©c: Volume = (Account_Balance * Risk%) / (ATR * Multiplier)
        # Biáº¿n Ä‘á»™ng cÃ ng lá»›n (ATR cao) -> ÄÃ¡nh volume cÃ ng nhá» Ä‘á»ƒ giá»¯ rá»§i ro cá»‘ Ä‘á»‹nh
        df['risk_distance'] = df['ATR'] * self.stop_loss_atr_mult
        
        # Giáº£ láº­p balance cá»‘ Ä‘á»‹nh hoáº·c láº¥y tá»« config
        account_balance = 10000 
        df['position_size'] = (account_balance * self.risk_per_trade) / df['risk_distance']

        # 4. GÃ¡n tÃ­n hiá»‡u
        df['signal'] = 0
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # 5. XÃ¡c Ä‘á»‹nh Stoploss Trailing (Dynamic)
        # Náº¿u Long: Stoploss = GiÃ¡ Close - (ATR * mult)
        # CÆ¡ cháº¿ Trailing sáº½ Ä‘Æ°á»£c xá»­ lÃ½ ká»¹ hÆ¡n á»Ÿ pháº§n Execution Engine
        
        return df

    def get_mlflow_params(self):
        """Tráº£ vá» cÃ¡c tham sá»‘ Ä‘á»ƒ log lÃªn MLflow"""
        return {
            "strategy_type": "AdaptiveBreakout",
            "lookback": self.lookback_period,
            "trend_ema": self.trend_filter_period,
            "sl_mult": self.stop_loss_atr_mult
        }
