import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(PROJECT_ROOT)

print(PROJECT_ROOT)
from data.ingestion import BinanceDataFetcher
from data.storage import StorageEngine
from strategies.trend import TrendFollowingStrategy, MomentumStrategy
from strategies.mean_reversion import MeanReversionStrategy
from strategies.breakout import BreakoutStrategy
from backtesting.engine import BacktestEngine
from models.model_factory import ModelFactory

def main():
    st.set_page_config(page_title="Pro Trading System", layout="wide")
    st.title("Professional Multi-Strategy Trading System")

    # Sidebar Controls
    st.sidebar.header("Settings")
    symbol = st.sidebar.text_input("Symbol", "BTC/USDT")
    timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"])
    days = st.sidebar.slider("Data Days", 10, 365, 30)
    
    # 1. Data Section
    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching data from Binance..."):
            fetcher = BinanceDataFetcher()
            fetcher.fetch_and_store(symbol, timeframe, days)
            st.success("Data fetched successfully!")

    # Load Data
    storage = StorageEngine()
    try:
        data = storage.load_market_data(symbol, timeframe)
        st.write(f"Loaded {len(data)} bars for {symbol}")
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["Backtesting", "Market Analysis", "ML Experiments"])

    with tab1:
        st.header("Backtest Runner")
        
        strat_name = st.selectbox("Select Strategy", ["Trend Following", "Momentum", "Mean Reversion", "Breakout"])
        
        if st.button("Run Backtest"):
            # Initialize Strategy
            if strat_name == "Trend Following":
                strategy = TrendFollowingStrategy({'fast_ma': 50, 'slow_ma': 200})
            elif strat_name == "Momentum":
                strategy = MomentumStrategy({'period': 14})
            elif strat_name == "Mean Reversion":
                strategy = MeanReversionStrategy({'bb_length': 20, 'bb_std': 2})
            elif strat_name == "Breakout":
                strategy = BreakoutStrategy({'lookback': 20})
            
            # Run Engine
            engine = BacktestEngine(initial_capital=10000)
            metrics, equity_curve, trades = engine.run_backtest(strategy, data)
            
            # Display Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Return", f"{metrics['Total Return']:.2%}")
            col2.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
            col3.metric("Max Drawdown", f"{metrics['Max Drawdown']:.2%}")
            col4.metric("Win Rate", f"{metrics['Win Rate (Daily)']:.2%}")
            
            # Plot Equity Curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=equity_curve.index, y=equity_curve.values, name="Equity"))
            st.plotly_chart(fig, use_container_width=True)
            
            # Trades Table
            if trades:
                st.subheader("Trade History")
                trades_df = pd.DataFrame(trades)
                st.dataframe(trades_df)

    with tab2:
        st.header("Market Analysis")
        # Simple Candlestick Chart
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['open'],
                        high=data['high'],
                        low=data['low'],
                        close=data['close'])])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("ML Experiments")
        if st.button("Train Model (Prediction)"):
            with st.spinner("Training XGBoost Model..."):
                from trading_system.features.engineering import FeatureEngineer
                fe = FeatureEngineer()
                processed_data = fe.add_all_features(data)
                
                factory = ModelFactory()
                model = factory.train_model(processed_data, model_type='xgboost')
                st.success("Model trained! Check MLflow for details.")

if __name__ == "__main__":
    main()

