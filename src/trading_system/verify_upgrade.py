import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from trading_system.backtesting.engine import BacktestEngine
from trading_system.strategies.mean_reversion import MeanReversionStrategy
from trading_system.data.storage import StorageEngine

def run_verification():
    print("Running Trading System Verification...")
    
    # 1. Mock Data
    dates = pd.date_range(start="2023-01-01", periods=100, freq="1H")
    data = pd.DataFrame({
        "open": np.random.uniform(20000, 21000, 100),
        "high": np.random.uniform(21000, 21500, 100),
        "low": np.random.uniform(19500, 20000, 100),
        "close": np.random.uniform(20000, 21000, 100),
        "volume": np.random.uniform(10, 100, 100)
    }, index=dates)
    data.index.name = "timestamp"

    # 2. Test Strategy
    strategy = MeanReversionStrategy(params={'bb_length': 20, 'bb_std': 2.0})
    
    # 3. Test Engine
    engine = BacktestEngine(initial_capital=10000)
    metrics, equity, trades = engine.run_backtest(strategy, data)
    
    print(f"Metrics: {metrics}")
    print(f"Num Trades: {len(trades)}")
    
    if len(trades) > 0:
        print(f"First Trade: {trades[0].to_dict()}")
        
    # 4. Test Persistence (if DB is available/mocked)
    try:
        bt_id = engine.save_results(strategy.name, "MOCK/BTC", metrics, trades, equity)
        if bt_id:
            print(f"Successfully saved to DB! ID: {bt_id}")
    except Exception as e:
        print(f"Persistence skipped or failed (expected if DB not running): {e}")

    print("Verification complete.")

if __name__ == "__main__":
    run_verification()
