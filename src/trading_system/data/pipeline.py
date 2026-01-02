import numpy as np
import pandas as pd
import ta
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from trading_system.utils.logger import setup_logger

logger = setup_logger('DataPipeline')

class CryptoDataPipeline:
    def __init__(self, window_size=10, horizon=6, barrier_width=1.5):
        self.window_size = window_size
        self.horizon = horizon
        self.barrier_width = barrier_width
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_cols = []
        self.is_fitted = False
        
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Production feature engineering without look-ahead bias."""
        df = df.copy()
        
        # 1. Cyclical Features (Time)
        if 'month' in df.columns:
             df['sin_month'] = np.sin(df['month'] * (np.pi / 6))
             df['cos_month'] = np.cos(df['month'] * (np.pi / 6))
        
        if 'hour' in df.columns:
            df['sin_hour'] = np.sin(df['hour'] * (np.pi / 12))
            df['cos_hour'] = np.cos(df['hour'] * (np.pi / 12))

        # 2. Momentum Indicators
        df['RSI_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        
        # MACD
        macd_indicator = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD'] = macd_indicator.macd()
        df['MACD_signal'] = macd_indicator.macd_signal()
        df['MACD_diff'] = macd_indicator.macd_diff() 

        # 3. Volatility Indicators
        atr_indicator = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
        df['ATR_14'] = atr_indicator.average_true_range()
        
        # Bollinger Bands
        bbands = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['BB_high'] = bbands.bollinger_hband()
        df['BB_low'] = bbands.bollinger_lband()
        df['BB_mid'] = bbands.bollinger_mavg()
        df['BB_width'] = bbands.bollinger_wband()
        df['BB_pband'] = bbands.bollinger_pband()
        
        # 4. Trend Indicators
        df['EMA_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['EMA_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['EMA_100'] = ta.trend.EMAIndicator(df['close'], window=100).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
        
        # ADX
        adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['ADX'] = adx_indicator.adx()
        df['ADX_pos'] = adx_indicator.adx_pos()
        df['ADX_neg'] = adx_indicator.adx_neg()

        # 5. Volume Indicators
        vwap_indicator = ta.volume.VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'])
        df['VWAP'] = vwap_indicator.volume_weighted_average_price()
        
        # 6. Lag Features (Returns)
        for lag in [1, 2, 3, 5]:
            df[f'return_lag_{lag}'] = df['close'].pct_change(lag)

        # 7. Target Variable (Forward Return for ML Reconstruction)
        # Shift back by 1 so that current candle predicts NEXT return
        df['target_return'] = df['close'].pct_change(1).shift(-1)
        
        # Drop NaNs created by indicators/lags
        # Note: We keep NaNs for target during live inference
        return df

    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Triple Barrier Method for labeling training data."""
        prices = df['close'].values
        vols = df['ATR_14'].values 
        
        labels = np.zeros(len(df))
        
        for t in range(len(df) - self.horizon):
            current = prices[t]
            vol = vols[t]
            upper = current + self.barrier_width * vol
            lower = current - self.barrier_width * vol
            
            future = prices[t+1 : t+1+self.horizon]
            
            hit_upper = np.where(future >= upper)[0]
            hit_lower = np.where(future <= lower)[0]
            
            first_upper = hit_upper[0] if len(hit_upper) > 0 else 999
            first_lower = hit_lower[0] if len(hit_lower) > 0 else 999
            
            if first_upper < first_lower and first_upper < self.horizon:
                labels[t] = 1 # BUY
            elif first_lower < first_upper and first_lower < self.horizon:
                labels[t] = 2 # SELL
            else:
                labels[t] = 0 # HOLD
                
        df['label'] = labels.astype(int)
        # We drop the last 'horizon' rows as they can't be labeled
        return df.iloc[:-self.horizon].copy()

    def prepare_train_test(self, df: pd.DataFrame, test_size=0.2):
        """Prepare data for training/validation."""
        df_fe = self.feature_engineering(df)
        df_labeled = self.create_labels(df_fe)
        df_labeled.dropna(inplace=True)
        
        if len(df_labeled) == 0:
             logger.error("Preparation failed: No samples left after feature engineering and labeling.")
             return None, None, None, None, None, None
        
        split_idx = int(len(df_labeled) * (1 - test_size))
        train_df = df_labeled.iloc[:split_idx]
        test_df = df_labeled.iloc[split_idx:]

        # Features to exclude from scaling
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'timeframe', 'year', 'month', 'day', 'hour', 'minute', 'target_return', 'label', 'sin_month', 'cos_month', 'sin_hour', 'cos_hour']
        self.feature_cols = [c for c in df_fe.columns if c not in exclude_cols]
        
        # Fit scaler on train only
        X_train_raw = train_df[self.feature_cols].values
        X_test_raw = test_df[self.feature_cols].values
        
        self.scaler.fit(X_train_raw)
        self.is_fitted = True
        
        X_train_scaled = self.scaler.transform(X_train_raw)
        X_test_scaled = self.scaler.transform(X_test_raw)

        # Merge back with cyclical and labels
        cyclical_cols = [c for c in ['sin_month', 'cos_month', 'sin_hour', 'cos_hour'] if c in df_fe.columns]
        
        X_train_final = np.concatenate([X_train_scaled, train_df[cyclical_cols].values], axis=1)
        X_test_final = np.concatenate([X_test_scaled, test_df[cyclical_cols].values], axis=1)
        
        # Add feature names for transparency
        self.final_feature_names = self.feature_cols + cyclical_cols
        
        X_train_win, y_ret_train, y_lab_train = self._create_window(X_train_final, train_df['target_return'].values, train_df['label'].values)
        X_test_win, y_ret_test, y_lab_test = self._create_window(X_test_final, test_df['target_return'].values, test_df['label'].values)
        
        return X_train_win, y_ret_train, y_lab_train, X_test_win, y_ret_test, y_lab_test

    def _create_window(self, data, target_returns, labels):
        X, y_ret, y_lab = [], [], []
        for i in range(self.window_size - 1, len(data)):
            X.append(data[i - self.window_size + 1 : i + 1])
            y_ret.append(target_returns[i])
            y_lab.append(labels[i])
        return np.array(X), np.array(y_ret), np.array(y_lab)
    
    def transform_live_data(self, df_window: pd.DataFrame):
        """Prepare a single window for live prediction."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before live prediction.")
            
        df_fe = self.feature_engineering(df_window)
        # Ensure we have enough data after FE
        if len(df_fe) < self.window_size:
             raise ValueError(f"Insufficient data for window size {self.window_size} after feature engineering.")
             
        data_raw = df_fe[self.feature_cols].iloc[-self.window_size:].values
        data_scaled = self.scaler.transform(data_raw)
        
        cyclical_cols = [c for c in ['sin_month', 'cos_month', 'sin_hour', 'cos_hour'] if c in df_fe.columns]
        data_cyclical = df_fe[cyclical_cols].iloc[-self.window_size:].values
        
        data_final = np.concatenate([data_scaled, data_cyclical], axis=1)
        return np.expand_dims(data_final, axis=0)

    def save(self, path='pipeline.pkl'):
        joblib.dump(self, path)
        logger.info(f"Pipeline saved to {path}")
        
    @staticmethod
    def load(path='pipeline.pkl'):
        return joblib.load(path)
