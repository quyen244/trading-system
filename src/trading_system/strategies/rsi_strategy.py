# src/strategies/rsi_momentum.py

import pandas as pd
import ta
from trading_system.strategies.base_strategy import BaseStrategy

class RsiStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        super().__init__("RsiMomentum_v1", params)

        self.rsi_period = self.params.get('rsi_period', 10)
        self.rsi_lower = self.params.get('rsi_lower', 30)
        self.rsi_upper = self.params.get('rsi_upper', 70)
        self.ema_trend = self.params.get('ema_trend', 200)
        self.atr_period = self.params.get('atr_period', 14)

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.validate_data(df.copy())

        # ===== Indicators =====
        df['RSI'] = ta.momentum.RSIIndicator(
            df['close'], window=self.rsi_period
        ).rsi()

        df['EMA_Trend'] = ta.trend.EMAIndicator(
            df['close'], window=self.ema_trend
        ).ema_indicator()

        df['ATR'] = ta.volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=self.atr_period
        ).average_true_range()

        df.dropna(inplace=True)

        # ===== Signal only =====
        df['signal'] = 0

        trend_up = df['close'] > df['EMA_Trend']
        rsi_cross_up = (
            (df['RSI'].shift(1) < self.rsi_lower) &
            (df['RSI'] >= self.rsi_lower)
        )

        df.loc[trend_up & rsi_cross_up, 'signal'] = 1

        # Exit signal theo indicator
        df.loc[df['RSI'] > self.rsi_upper, 'signal'] = -1

        # ⚠️ QUAN TRỌNG: shift signal để tránh look-ahead
        df['signal'] = df['signal'].shift(1).fillna(0)

        return df
