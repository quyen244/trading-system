# Trading System Documentation

## Overview
This trading system is designed for professional backtesting and strategy optimization. It supports Long and Short positions, advanced risk management rules, and database persistence for results.

## Key Components

### 1. Backtesting Engine
- **Long/Short Support**: The engine now handles both buy and sell signals, allowing for more complex strategies.
- **Risk Management**:
    - **Stop Loss**: 1.5 × ATR (Average True Range)
    - **Take Profit**: 3 × ATR
    - **Position Sizing**: Automatically calculated based on 1% risk of capital per Stop Loss distance.
- **Fees & Slippage**:
    - **Default Fee**: 0.02% per trade volume.
    - **Slippage**: Adjustable parameter to simulate real-world execution.

### 2. Database Models
- **Backtest Result**: Stores summary metrics for each run.
- **Trade Record**: Detailed information about every individual trade executed during a backtest.

### 3. Dashboard Features
- **Backtest Page**:
    - Interactive Candlestick charts with trade markers and SL/TP lines.
    - Persistence button to save results to the PostgreSQL database.
    - Strategy comparison tool to evaluate multiple strategies side-by-side.
- **Finetune Page**:
    - Automatic parameter optimization using Grid Search or Random Search.
    - Visualization of the best-performing parameter set.

## Usage Guide
1. **Fetch Data**: Ensure market data is available in the `market_data` table.
2. **Run Backtest**: Use the ALGO TRADING section in the dashboard to select a strategy and run a simulation.
3. **Compare**: Select multiple strategies to see which performs best on the selected dataset.
4. **Optimize**: Use the Finetune page to find the best hyperparameters for your strategy.
