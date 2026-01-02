from abc import ABC, abstractmethod
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, name: str, params: dict = None):
        self.name = name
        self.params = params or {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze data and generate signals.
        Input: DataFrame with OHLCV and features.
        Output: DataFrame with 'signal' column (1: BUY, -1: SELL, 0: HOLD).
        """
        pass

    def validate_data(self, data: pd.DataFrame):
        """Ensure data has required columns."""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data missing required columns: {required_cols}")
        
        if data.isnull().values.any():
            logger.warning("Data contains NaNs. Filling with ffill.")
            data.fillna(method='ffill', inplace=True)
            data.dropna(inplace=True) # Drop remaining if any
            
        return data

    def get_signal(self, row: pd.Series) -> dict:
        """
        Helper to extract signal from a specific row (latest candle).
        """
        return {
            "stragegy": self.name,
            "signal": row.get('signal', 0),
            "timestamp": row.name, # Assuming datetime index
            "close": row['close']
        }