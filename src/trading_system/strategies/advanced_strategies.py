"""
Advanced Trading Strategies

Modern, high-end, effective trading strategies:
1. Volume Profile Strategy
2. Statistical Arbitrage (Mean Reversion with Z-Score)
3. Machine Learning-based Strategy (Random Forest)

Author: Trading System
"""

import pandas as pd
import numpy as np
import ta
from typing import Optional
from trading_system.strategies.base_strategy import BaseStrategy
from trading_system.features.engineering import FeatureEngineer

class VolumeProfileStrategy(BaseStrategy):
    """
    Volume Profile Strategy - Trade based on volume concentration zones.
    
    Concept: Price tends to return to high-volume areas (value areas)
    and bounce from low-volume areas (rejection zones).
    """
    
    def __init__(self, params: dict = None):
        """
        Args:
            params:
                - lookback: Lookback period for volume profile (default: 50)
                - num_bins: Number of price bins (default: 20)
                - vwap_window: VWAP calculation window (default: 20)
        """
        super().__init__("VolumeProfile", params)
        self.lookback = self.params.get('lookback', 50)
        self.num_bins = self.params.get('num_bins', 20)
        self.vwap_window = self.params.get('vwap_window', 20)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.validate_data(data)
        df = data.copy()

        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['ATR_14'] = atr_indicator.average_true_range()
        
        # Calculate VWAP
        vwap_indicator = ta.volume.VolumeWeightedAveragePrice(
            df['high'], df['low'], df['close'], df['volume']
        )
        df['VWAP'] = vwap_indicator.volume_weighted_average_price()
        
        # Calculate volume profile
        df['POC'] = 0.0  # Point of Control (price with highest volume)
        df['VAH'] = 0.0  # Value Area High
        df['VAL'] = 0.0  # Value Area Low
        
        for i in range(self.lookback, len(df)):
            # Get lookback window
            window = df.iloc[i-self.lookback:i]
            
            # Create price bins
            price_range = window['high'].max() - window['low'].min()
            bin_size = price_range / self.num_bins
            
            # Calculate volume at each price level
            bins = np.linspace(window['low'].min(), window['high'].max(), self.num_bins)
            volume_profile = np.zeros(self.num_bins - 1)
            
            for j, row in window.iterrows():
                # Distribute volume across price range
                low_bin = int((row['low'] - window['low'].min()) / bin_size)
                high_bin = int((row['high'] - window['low'].min()) / bin_size)
                low_bin = max(0, min(low_bin, self.num_bins - 2))
                high_bin = max(0, min(high_bin, self.num_bins - 2))
                
                for k in range(low_bin, high_bin + 1):
                    volume_profile[k] += row['volume'] / (high_bin - low_bin + 1)
            
            # Find POC (Point of Control)
            poc_idx = np.argmax(volume_profile)
            poc_price = bins[poc_idx] + bin_size / 2
            df.loc[df.index[i], 'POC'] = poc_price
            
            # Find Value Area (70% of volume)
            total_volume = volume_profile.sum()
            target_volume = total_volume * 0.70
            
            # Expand from POC until we have 70% of volume
            current_volume = volume_profile[poc_idx]
            lower_idx = poc_idx
            upper_idx = poc_idx
            
            while current_volume < target_volume:
                if lower_idx > 0 and upper_idx < len(volume_profile) - 1:
                    if volume_profile[lower_idx - 1] > volume_profile[upper_idx + 1]:
                        lower_idx -= 1
                        current_volume += volume_profile[lower_idx]
                    else:
                        upper_idx += 1
                        current_volume += volume_profile[upper_idx]
                elif lower_idx > 0:
                    lower_idx -= 1
                    current_volume += volume_profile[lower_idx]
                elif upper_idx < len(volume_profile) - 1:
                    upper_idx += 1
                    current_volume += volume_profile[upper_idx]
                else:
                    break
            
            df.loc[df.index[i], 'VAH'] = bins[upper_idx + 1]
            df.loc[df.index[i], 'VAL'] = bins[lower_idx]
        df.dropna(inplace=True)
        # Generate signals
        df['signal'] = 0
        
        # Buy when price is below VAL and moving up
        buy_condition = (df['close'] < df['VAL']) & (df['close'] > df['close'].shift(1))
        
        # Sell when price is above VAH and moving down
        sell_condition = (df['close'] > df['VAH']) & (df['close'] < df['close'].shift(1))
        
        df.loc[buy_condition, 'signal'] = 1
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage Strategy using Z-Score mean reversion.
    
    Concept: Trade when price deviates significantly from its mean,
    expecting reversion to the mean.
    """
    
    def __init__(self, params: dict = None):
        """
        Args:
            params:
                - window: Lookback window for mean/std (default: 20)
                - entry_threshold: Z-score threshold for entry (default: 2.0)
                - exit_threshold: Z-score threshold for exit (default: 0.5)
        """
        super().__init__("StatArb", params)
        self.window = self.params.get('window', 20)
        self.entry_threshold = self.params.get('entry_threshold', 2.0)
        self.exit_threshold = self.params.get('exit_threshold', 0.5)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.validate_data(data)
        df = data.copy()

        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['ATR_14'] = atr_indicator.average_true_range()

        # Calculate rolling mean and std
        df['rolling_mean'] = df['close'].rolling(window=self.window).mean()
        df['rolling_std'] = df['close'].rolling(window=self.window).std()
        
        # Calculate Z-Score
        df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
        
        # Calculate RSI for confirmation
        df['RSI'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # Generate signals
        df['signal'] = 0
        df['position'] = 0  # Track current position
        
        position = 0
        for i in range(self.window, len(df)):
            z_score = df.loc[df.index[i], 'z_score']
            rsi = df.loc[df.index[i], 'RSI']
            
            # Entry signals
            if position == 0:
                # Buy when oversold (z-score < -threshold and RSI < 30)
                if z_score < -self.entry_threshold and rsi < 30:
                    df.loc[df.index[i], 'signal'] = 1
                    position = 1
                # Sell when overbought (z-score > threshold and RSI > 70)
                elif z_score > self.entry_threshold and rsi > 70:
                    df.loc[df.index[i], 'signal'] = -1
                    position = -1
            
            # Exit signals
            elif position == 1:  # Long position
                if z_score > -self.exit_threshold:
                    df.loc[df.index[i], 'signal'] = -1
                    position = 0
            
            elif position == -1:  # Short position
                if z_score < self.exit_threshold:
                    df.loc[df.index[i], 'signal'] = 1
                    position = 0
            
            df.loc[df.index[i], 'position'] = position
        
        return df


class MLMomentumStrategy(BaseStrategy):
    """
    Machine Learning-based Momentum Strategy using Random Forest.
    
    Concept: Use ML to predict price direction based on technical indicators.
    """
    
    def __init__(self, params: dict = None):
        """
        Args:
            params:
                - lookback: Feature lookback period (default: 20)
                - prediction_threshold: Probability threshold for trading (default: 0.6)
        """
        super().__init__("ML_Momentum", params)
        self.lookback = self.params.get('lookback', 20)
        self.prediction_threshold = self.params.get('prediction_threshold', 0.6)
        self.model = None
        self.feature_engineer = FeatureEngineer()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data = self.validate_data(data)
        df = data.copy()
        
        # Create features
        df = self.feature_engineer.add_all_features(df)
        
        # Create target (1 if price goes up next period, 0 otherwise)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN values
        df = df.dropna()
        
        if len(df) < 100:
            df['signal'] = 0
            return df
        
        # Train/test split (use first 70% for training)
        split_idx = int(len(df) * 0.7)
        
        feature_cols = df.columns
        
        X_train = df[feature_cols].iloc[:split_idx]
        y_train = df['target'].iloc[:split_idx]
        X_test = df[feature_cols].iloc[split_idx:]
        
        # Train Random Forest (only if not already trained)
        if self.model is None:
            try:
                from sklearn.ensemble import RandomForestClassifier
                
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42,
                    n_jobs=-1
                )
                self.model.fit(X_train, y_train)
                
            except ImportError:
                # Fallback to simple momentum if sklearn not available
                df['signal'] = 0
                df.loc[df['returns'] > 0, 'signal'] = 1
                df.loc[df['returns'] < 0, 'signal'] = -1
                return df
        
        # Make predictions
        predictions = self.model.predict_proba(df[feature_cols])[:, 1]
        
        # Generate signals based on prediction confidence
        df['signal'] = 0
        df.loc[predictions > self.prediction_threshold, 'signal'] = 1
        df.loc[predictions < (1 - self.prediction_threshold), 'signal'] = -1
        
        return df


if __name__ == "__main__":
    print("Advanced Trading Strategies")
    print("=" * 50)
    print("\nAvailable strategies:")
    print("1. VolumeProfileStrategy - Trade based on volume concentration")
    print("2. StatisticalArbitrageStrategy - Mean reversion with Z-Score")
    print("3. MLMomentumStrategy - Machine Learning-based predictions")
