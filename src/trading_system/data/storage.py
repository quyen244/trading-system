import os
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from datetime import datetime
from trading_system.utils.logger import setup_logger
from dotenv import load_dotenv
import numpy as np 

load_dotenv()

logger = setup_logger("StorageEngine")

# Define Base
Base = declarative_base()

class MarketData(Base):
    __tablename__ = 'market_data'
    
    symbol = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Float)
    timeframe = Column(String, nullable=False)
    
    __table_args__ = (
        PrimaryKeyConstraint('symbol', 'timestamp', 'timeframe'),
    )

class StorageEngine:
    def __init__(self, db_url=None):
        if db_url is None:
            # Default to local docker connection string
            # User: trader, Pass: trading_password, DB: trading_system, Port: 5433
            self.db_url = os.getenv("TRADING_DB_URL", "")
            logger.info(f"Using default DB URL: {self.db_url}")
            
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        
    def create_tables(self):
        """Create tables if they don't exist."""
        Base.metadata.create_all(self.engine)
        
    def store_market_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
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

    def load_market_data(self, symbol: str, timeframe: str, start_date: datetime = None, end_date: datetime = None):
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
        
        return pd.read_sql(query, self.engine, parse_dates=['timestamp']).set_index('timestamp')
