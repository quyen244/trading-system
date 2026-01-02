from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from datetime import datetime
from trading_system.utils.logger import setup_logger
from trading_system.data.storage.base import StorageEngine
from trading_system.data.schemas import MarketData

logger = setup_logger("StorageEngine")

class StorageMarketData(StorageEngine):
    def __init__(self, db_url=None):
        super().__init__(db_url)
        
    def create_tables(self):
        """Create tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        
    def store_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """
        Store OHLCV data into the database.
        Assumes df has datetime index or 'timestamp' column and columns: open, high, low, close, volume.
        
        Args:
            df (pd.DataFrame): The DataFrame containing the OHLCV data.
            symbol (str): The trading pair symbol.
            timeframe (str): The timeframe of the data.
        """
        session = self.Session()
        try:
            records = []
            for index, row in df.iterrows():
                # Handle timestamp from index or column
                ts = index if isinstance(index, datetime) else row.get('timestamp')
                if not ts:
                    continue
                
                records.append({
                    'symbol': symbol,
                    'timestamp': ts,
                    'timeframe': timeframe,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                })
                
            if not records:
                return

            # Bulk upsert
            stmt = insert(MarketData).values(records)
            stmt = stmt.on_conflict_do_update(
                index_elements=['symbol', 'timestamp', 'timeframe'],
                set_={
                    'open': stmt.excluded.open,
                    'high': stmt.excluded.high,
                    'low': stmt.excluded.low,
                    'close': stmt.excluded.close,
                    'volume': stmt.excluded.volume
                }
            )
            
            session.execute(stmt)
            session.commit()
            print(f"Stored {len(records)} records for {symbol} ({timeframe})")
            return True 
            
        except Exception as e:
            session.rollback()
            print(f"Error storing data: {e}")
            raise
        finally:
            session.close()

    def get_data(self, symbol: str, timeframe: str, start_date: datetime = None, end_date: datetime = None):
        """Load data from DB into DataFrame.
        
        Args:
            symbol (str): The trading pair symbol.
            timeframe (str): The timeframe of the data.
            start_date (datetime, optional): The start date for the data. Defaults to None.
            end_date (datetime, optional): The end date for the data. Defaults to None.
        
        Returns:
            pd.DataFrame: The loaded data as a DataFrame.
        """
        query = f"SELECT * FROM market_data WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'"
        
        if start_date:
            query += f" AND timestamp >= '{start_date}'"
        if end_date:
            query += f" AND timestamp <= '{end_date}'"
            
        query += " ORDER BY timestamp ASC"
        
        try:
            data = pd.read_sql(query, self.engine, parse_dates=['timestamp']).set_index('timestamp')
            return data 
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()