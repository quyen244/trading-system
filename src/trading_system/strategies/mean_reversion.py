import pandas as pd
from trading_system.strategies.base_strategy import BaseStrategy
import ta
class MeanReversionStrategy(BaseStrategy):
    def __init__(self, params: dict = None):
        """
        Description: Mean Reversion Strategy
        Args:
            params (dict, optional): _description_.  bb_length , bb_std , Defaults to None.
        """
        super().__init__("MeanReversion", params)
        self.bb_length = self.params.get('bb_length', 20)
        self.bb_std = self.params.get('bb_std', 2.0)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            data (pd.DataFrame): Dataframe OHLCV

        Returns:
            pd.DataFrame: 'signal', 'position_size', 'stop_loss_price'
        """
        data = self.validate_data(data)
        df = data.copy()
        
        # Bollinger Bands
        bbands = ta.volatility.BollingerBands(df['close'], window=self.bb_length, window_dev=self.bb_std)
        df['BBL'] = bbands.bollinger_lband()
        df['BBU'] = bbands.bollinger_hband()
        
        df['signal'] = 0
        
        # Buy below lower band
        df.loc[df['close'] < df['BBL'], 'signal'] = 1
        
        # Sell above upper band
        df.loc[df['close'] > df['BBU'], 'signal'] = -1
        
        return df
