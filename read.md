# 📈 Quantitative Trading System with MLOps

This is a modular, high-performance Algorithmic Trading System designed for research, backtesting, and automated execution. It integrates **MLflow** for experiment tracking and **Streamlit** for real-time monitoring.

## 📄 Documentation

For a detailed technical overview, system architecture, and module descriptions, please refer to the main documentation:
👉 **[SYSTEM_ARCHITECTURE.md](./SYSTEM_ARCHITECTURE.md)**

---

## 🏗 High-Level Architecture

```mermaid
graph TD
    A[Data Ingestion] -->|Raw Data| B(Database / Data Lake)
    B --> C[Research & Dev (Notebooks)]
    B --> D[Backtesting Engine]
    C -->|New Strategy| E[MLflow Tracking]
    D -->|Log Metrics & Params| E
    E -->|Select Best Model| F[Model Registry]
    F -->|Load Model| G[Live Trading Engine]
    G -->|Execute Orders| H[Exchange API (Binance/IBKR)]
    B & E & G --> I[Streamlit Dashboard]
```

## 📂 Project Structure

```text
trading_system/
├── config/                 # Configurations (Tickers, Timeframes, API Keys)
├── data/                   # Local data storage (Parquet/CSV)
├── src/
│   ├── strategies/         # Strategy logic (Base Class & Implementation)
│   ├── backtesting/        # Simulation engine & metrics
│   ├── execution/          # Live trading & order management
│   ├── mlops/              # MLflow integration utilities
│   ├── data_loader/        # Pipeline for data ingestion
│   └── risk/               # Risk management & position sizing
├── dashboard/              # Streamlit monitoring interface
├── scripts/                # Entry points (Train, Backtest, Live)
└── docker-compose.yml      # Infrastructure (Postgres, MLflow Server)
```

## 🚀 Quick Start

### 1. Requirements
*   Docker & Docker Compose
*   Python 3.9+
*   Binance/Exchange API Keys

### 2. Environment Setup
```bash
cp .env.example .env
# Fill in your API Key and Database config in .env
```

### 3. Start Infrastructure
```bash
docker-compose up -d
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🛠 Operation Workflow

1.  **Backtest & Optimization**:
    Run backtests to find optimal parameters. Results log automatically to [localhost:5000](http://localhost:5000).
    ```bash
    python scripts/run_backtest.py --strategy StrategyRsiMomentum --symbol BTCUSDT --period 1y
    ```

2.  **Evaluate via MLflow**:
    Compare Sharpe Ratio, Drawdown, and Win Rate in the MLflow UI. Register the best model as `Production`.

3.  **Run Live**:
    Execute the bot using parameters from the Production model.
    ```bash
    python scripts/run_live.py --symbol BTCUSDT
    ```

4.  **Monitor Performance**:
    Open the Streamlit dashboard for real-time PnL and trade tracking.
    ```bash
    streamlit run src/dashboard/app.py
    ```

## 🛡 Risk Management
The system includes built-in safeguards:
*   Fixed fractional position sizing.
*   ATR-based dynamic stop-loss.
*   Daily loss limits and equity-based kill switches.

---
*Professional Quant Trading Architecture by Antigravity AI.*