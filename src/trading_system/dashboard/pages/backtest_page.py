import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from trading_system.data.storage.market import StorageMarketData 
from trading_system.visualization.charts import TradingVisualizer
from trading_system.utils.dashboard import get_strategy

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Professional Trading Terminal",
    page_icon="📈",
    layout="wide", # Tận dụng tối đa chiều ngang màn hình
)
viz = TradingVisualizer(style='dark')
# --- STYLING (Professional Dark Theme) ---
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e222d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #363a45;
    }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Market Settings")
    symbol = st.selectbox("Select Symbol", ["BTC/USDT", "ETH/USDT"], index=0)
    timeframe = st.selectbox("Timeframe", ["m15", "1h", "4h", "1d"], index=1)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime.now().date() - timedelta(days=60))
    with col2:
        end_date = st.date_input("End", datetime.now().date())
    
    st.header("⚙️ Strategy Selecting")
    strategy = st.selectbox("Select Strategy", ["Mean Reversion", "RSI", "MACD", "Break out", "MA Crossover", "MACD Divergence"], index=0)
    
    st.divider()
    st.info("System Status: Online")

# --- DATA LOADING ---
storage_engine = StorageMarketData()
market_data = storage_engine.get_data(symbol=symbol, timeframe=timeframe, start_date=str(start_date), end_date=str(end_date))

if market_data.empty:
    st.warning("No market data found for the selected parameters. Please fetch data first.")
else:
    # --- MAIN PAGE LAYOUT ---
    st.title(f"📊 {symbol} Backtesting Dashboard")
    
    # --- STRATEGY EXECUTION ---
    strat_class = get_strategy(strategy)
    
    # Simple Hyperparameter selection (can be expanded)
    st.sidebar.subheader("Strategy Parameters")
    params = {}
    if strategy == "RSI":
        params['rsi_period'] = st.sidebar.slider("RSI Period", 5, 30, 14)
    elif strategy == "MACD":
        params['fast_period'] = st.sidebar.slider("Fast Period", 5, 20, 12)
        params['slow_period'] = st.sidebar.slider("Slow Period", 21, 50, 26)
    
    # Run Backtest
    if st.sidebar.button("🚀 Run Backtest"):
        with st.spinner("Running backtest..."):
            strat_class = strat_class(params)
            metrics, equity_series, trades = strat_class.run_backtest(market_data)
            
            # Store in session state for saving later
            st.session_state['latest_backtest'] = {
                'metrics': metrics,
                'equity': equity_series,
                'trades': trades,
                'strategy_name': strategy,
                'symbol': symbol
            }

    if 'latest_backtest' in st.session_state:
        bt = st.session_state['latest_backtest']
        metrics, equity_series, trades = bt['metrics'], bt['equity'], bt['trades']
        
        # Action Buttons
        col_btn1, col_btn2 = st.columns([1, 5])
        with col_btn1:
            if st.button("💾 Save to DB"):
                strategy_inst = get_strategy(bt['strategy_name'])
                bt_id = strategy_inst.save_backtest(bt['symbol'], metrics, trades, equity_series)
                if bt_id:
                    st.success(f"Backtest saved successfully! ID: {bt_id}")
                else:
                    st.error("Failed to save backtest.")

        # --- VISUALIZATIONS ---
        # 1. Metrics Cards
        cols = st.columns(len(metrics))
        for i, (k, v) in enumerate(metrics.items()):
            cols[i].metric(label=k.replace('_', ' ').title(), value=f"{v:.2f}" if "ratio" in k else f"{v:.2%}" if "return" in k or "rate" in k or "drawdown" in k else f"{v}")

        # 2. Main Price Chart
        st.subheader("Interactive Price Chart")
        fig_price = viz.plot_candlestick_with_trades(market_data, trades, symbol=bt['symbol'])
        st.plotly_chart(fig_price, use_container_width=True)

        # 3. Performance Charts 
        
        st.pyplot(viz.plot_equity_curve(equity_series, title="Equity Curve"))
        st.pyplot(viz.plot_drawdown(equity_series, title="Drawdown Analysis"))
        
        st.pyplot(viz.plot_trade_distribution([t.to_dict() for t in trades], title="Trade Distribution"))
        
        # 4. Trades Table
        st.subheader("Trade List")
        if trades:
            trades_df = pd.DataFrame([t.to_dict() for t in trades])
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades executed during this period.")

    # --- STRATEGY COMPARISON ---
    st.divider()
    st.subheader("⚔️ Strategy Comparison")
    selected_strategies = st.multiselect("Select Strategies to Compare", ["Mean Reversion", "RSI", "MACD", "Break out", "MA Crossover"], default=["Mean Reversion", "RSI"])
    
    if st.button("📊 Run Comparison"):
        from trading_system.scripts.compare_strategies import StrategyComparator
        comparator = StrategyComparator(market_data)
        
        strat_list = []
        for s_name in selected_strategies:
            strat_list.append((s_name, (get_strategy(s_name)), {}))
        
        with st.spinner("Comparing strategies..."):
            comparison_df = comparator.compare_strategies(strat_list)
            st.dataframe(comparison_df, use_container_width=True)
            
            if hasattr(comparator, 'equity_curves'):
                st.pyplot(viz.plot_strategy_comparison(comparator.equity_curves, title="Strategy Comparison"))
