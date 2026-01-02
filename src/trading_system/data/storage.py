import os
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Numeric, ForeignKey, PrimaryKeyConstraint
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
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
    year = Column(Integer)
    month = Column(Integer)
    day = Column(Integer)
    hour = Column(Integer)
    minute = Column(Integer)
    
    __table_args__ = (
        PrimaryKeyConstraint('symbol', 'timestamp', 'timeframe'),
    )

class Backtest(Base):
    __tablename__ = 'backtest'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String, nullable=False) # Changed from int to String for strategy name
    symbol = Column(String, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Numeric, nullable=False)
    total_return = Column(Numeric, nullable=False)
    cagr = Column(Numeric, nullable=False)
    max_drawdown = Column(Numeric, nullable=False)
    sharpe_ratio = Column(Numeric, nullable=False)
    sortino_ratio = Column(Numeric, nullable=False)
    win_rate_daily = Column(Numeric, nullable=False)
    num_trades = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    trades = relationship("TradeRecord", back_populates="backtest")

class TradeRecord(Base):
    __tablename__ = 'trade'
    
    id = Column(Integer, primary_key=True)
    backtest_id = Column(Integer, ForeignKey('backtest.id'), nullable=False)
    symbol = Column(String, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=False)
    entry_price = Column(Numeric, nullable=False)
    exit_price = Column(Numeric, nullable=False)
    stop_loss = Column(Numeric, nullable=False)
    take_profit = Column(Numeric, nullable=False)
    position_size = Column(Numeric, nullable=False)
    side = Column(String, nullable=False)
    gross_pnl = Column(Numeric, nullable=False)
    net_pnl = Column(Numeric, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    backtest = relationship("Backtest", back_populates="trades")

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
        
        try:
            data = pd.read_sql(query, self.engine, parse_dates=['timestamp']).set_index('timestamp')
            return data 
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame() 

    def save_backtest(self, backtest_data: dict, trades: list):
        """
        Save backtest results and associated trades.
        """
        session = self.Session()
        try:
            # Create Backtest record
            bt = Backtest(**backtest_data)
            session.add(bt)
            session.flush() # Get backtest ID
            
            # Create Trade records
            trade_records = []
            for t in trades:
                t_data = t if isinstance(t, dict) else t.to_dict()
                t_data['backtest_id'] = bt.id
                # Remove exit_time if it's identical to entry_time or handle it
                trade_records.append(TradeRecord(**t_data))
            
            session.add_all(trade_records)
            session.commit()
            print(f"Saved backtest {bt.id} with {len(trade_records)} trades")
            return bt.id
        except Exception as e:
            session.rollback()
            print(f"Error saving backtest: {e}")
            raise
        finally:
            session.close()

    def get_backtests(self, strategy_id: str = None):
        """Get recent backtests."""
        query = "SELECT * FROM backtest"
        if strategy_id:
            query += f" WHERE strategy_id = '{strategy_id}'"
        query += " ORDER BY created_at DESC"
        return pd.read_sql(query, self.engine)
