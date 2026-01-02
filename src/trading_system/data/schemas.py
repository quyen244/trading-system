# Define Base
from sqlalchemy import Column, Integer, String, Float, DateTime, Numeric, ForeignKey, PrimaryKeyConstraint, JSON
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime

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
    strategy_id = Column(Integer, nullable=False) # Changed from int to String for strategy name
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
    params = Column(JSON)
    
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