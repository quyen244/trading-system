from trading_system.data.storage import StorageEngine
from trading_system.backtesting.engine import BacktestEngine
from trading_system.strategies.mean_reversion import MeanReversionStrategy


if __name__ == '__main__':
    storage = StorageEngine()
    df = storage.load_market_data('BTC/USDT', '1h', '2024-01-01')

    backtest_engine = BacktestEngine()
    strategy = MeanReversionStrategy()
    metrics , equity_curve , trades = backtest_engine.run_backtest(strategy , df)
    print(metrics)
    print(equity_curve)
    print(trades)

# python -m trading_system.scripts.load_market