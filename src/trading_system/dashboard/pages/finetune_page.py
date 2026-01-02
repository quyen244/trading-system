import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from trading_system.data.storage import StorageEngine
from trading_system.scripts.finetune import StrategyOptimizer
from trading_system.strategies import *
from trading_system.visualization.charts import TradingVisualizer

# --- PAGE CONFIG ---
st.set_caption = "Strategy Fine-Tuning"

def get_strategy_class(name):
    strats = {
        "Mean Reversion": MeanReversionStrategy,
        "RSI": RsiStrategy,
        "MACD": MACDStrategy,
        "Break out": BreakoutStrategy,
        "MA Crossover": MovingAverageCrossoverStrategy
    }
    return strats.get(name)

st.title("🎯 Strategy Parameter Fine-Tuning")

# --- SIDEBAR: SETTINGS ---
with st.sidebar:
    st.header("⚙️ Data Settings")
    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT"])
    timeframe = st.selectbox("Timeframe", ["m15", "1h", "4h", "1d"], index=1)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", datetime.now().date() - timedelta(days=60))
    with col2:
        end_date = st.date_input("End", datetime.now().date())

    st.header("🧪 Search Strategy")
    strategy_name = st.selectbox("Select Strategy", ["Mean Reversion", "RSI", "MACD", "Break out", "MA Crossover"])
    search_type = st.radio("Search Type", ["Grid Search", "Random Search"])
    
    st.divider()

# --- DATA LOADING ---
storage = StorageEngine()
data = storage.load_market_data(symbol, timeframe, str(start_date), str(end_date))

if data.empty:
    st.warning("Please fetch market data first.")
else:
    st.info(f"Loaded {len(data)} candles for {symbol}")
    
    # --- PARAMETER CONFIGURATION ---
    st.subheader(f"Configure Parameters for {strategy_name}")
    
    param_grid = {}
    if strategy_name == "Mean Reversion":
        param_grid['bb_length'] = st.multiselect("BB Lengths", [10, 20, 30, 50], default=[20, 30])
        param_grid['bb_std'] = st.multiselect("BB Std Dev", [1.5, 2.0, 2.5], default=[2.0])
    elif strategy_name == "RSI":
        param_grid['period'] = st.multiselect("RSI Periods", [7, 14, 21, 28], default=[14, 21])
    elif strategy_name == "MACD":
        param_grid['fast_period'] = st.multiselect("Fast Periods", [5, 12, 15], default=[12])
        param_grid['slow_period'] = st.multiselect("Slow Periods", [21, 26, 30], default=[26])
    
    st.divider()
    
    if st.button("🔥 Start Optimization"):
        optimizer = StrategyOptimizer(
            strategy_class=get_strategy_class(strategy_name),
            data=data,
            optimization_metric='sharpe_ratio'
        )
        
        with st.spinner(f"Running {search_type}..."):
            if search_type == "Grid Search":
                best_params, results_df = optimizer.grid_search(param_grid)
            else:
                # Distribution for random search
                dist = {k: (min(v), max(v)) for k, v in param_grid.items()}
                best_params, results_df = optimizer.random_search(dist, n_iter=10)
            
            st.session_state['opt_results'] = {
                'best_params': best_params,
                'results_df': results_df,
                'strategy_name': strategy_name
            }

    if 'opt_results' in st.session_state:
        res = st.session_state['opt_results']
        st.success(f"Optimization Complete! Best Parameters: {res['best_params']}")
        
        # 1. Results Table
        st.subheader("Optimization Results")
        st.dataframe(res['results_df'], use_container_width=True)
        
        # 2. Visualize Best Result
        st.subheader("Best Result Analysis")
        best_strat = get_strategy_class(res['strategy_name'])(params=res['best_params'])
        metrics, equity, trades = best_strat.run_backtest(data)
        
        viz = TradingVisualizer(style='dark')
        st.plotly_chart(viz.plot_candlestick_with_trades(data, trades, symbol=symbol, title="Best performing parameters"), use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(viz.plot_equity_curve(equity, title="Best Equity Curve"))
        with col2:
            st.pyplot(viz.plot_drawdown(equity, title="Best Drawdown Analysis"))
