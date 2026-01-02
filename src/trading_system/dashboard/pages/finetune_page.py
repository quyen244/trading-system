import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from trading_system.data.storage.market import StorageMarketData
from trading_system.scripts.finetune import StrategyOptimizer
from trading_system.visualization.charts import TradingVisualizer
from trading_system.data.storage.backtest import StorageBacktest
from trading_system.utils.dashboard import backtest_data_format , get_strategy , get_id_strategy

# --- PAGE CONFIG ---
st.set_caption = "Strategy Fine-Tuning"

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
storage = StorageMarketData()
data = storage.get_data(symbol, timeframe, str(start_date), str(end_date))

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
        param_grid['rsi_period'] = st.multiselect("RSI Periods", [7, 14, 21, 28], default=[14, 21])
        param_grid['rsi_lower'] = st.multiselect("RSI Lower", [30, 40, 50], default=[30])
        param_grid['rsi_upper'] = st.multiselect("RSI Upper", [70, 80, 90], default=[70])
        param_grid['ema_trend'] = st.multiselect("EMA Trend", [200, 250, 300], default=[200])
        param_grid['atr_period'] = st.multiselect("ATR Period", [14, 21, 28], default=[14])
    elif strategy_name == "MACD":
        param_grid['fast_period'] = st.multiselect("Fast Periods", [5, 12, 15], default=[12])
        param_grid['slow_period'] = st.multiselect("Slow Periods", [21, 26, 30], default=[26])
        param_grid['signal_period'] = st.multiselect("Signal Periods", [9], default=[9])
        param_grid['histogram_threshold'] = st.multiselect("Histogram Threshold", [0], default=[0])
    elif strategy_name == "Break out":
        param_grid['lookback'] = st.multiselect("Lookback", [5, 12, 15 , 20 , 30], default=[12])
    elif strategy_name == "MA Crossover":
        param_grid['fast_period'] = st.multiselect("Fast Periods", [5, 12, 15], default=[12])
        param_grid['slow_period'] = st.multiselect("Slow Periods", [21, 26, 30], default=[26])
    
    st.divider()
    
    if st.button("🔥 Start Optimization"):
        optimizer = StrategyOptimizer(
            strategy_class=get_strategy(strategy_name),
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

            best_params = {k : int(v) if isinstance(v, float) else v for k , v in best_params.items()}

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

        # Run Backtest
        if st.button("🚀 Run Backtest"):
            with st.spinner("Running backtest..."):
                strategy_inst = get_strategy(res['strategy_name'])
                strategy_inst.params = res['best_params']
                metrics, equity_series, trades = strategy_inst.run_backtest(data)
                
                # Store in session state for saving later
                st.session_state['latest_backtest'] = {
                    'metrics': metrics,
                    'equity': equity_series,
                    'trades': trades,
                    'strategy_name': res['strategy_name'],
                    'symbol': symbol,
                    'params': res['best_params']
                }

        if 'latest_backtest' in st.session_state:
            bt = st.session_state['latest_backtest']
            metrics, equity_series, trades = bt['metrics'], bt['equity'], bt['trades']
            
            # Action Buttons
            col_btn1, col_btn2 = st.columns([1, 5])
            with col_btn1:
                if st.button("💾 Save to DB"):
                    backtest_storage = StorageBacktest()
                    strategy_inst = get_strategy(bt['strategy_name'])
                    bt_id = backtest_storage.store_data(backtest_data_format(get_id_strategy(bt['strategy_name']), bt['symbol'], metrics, equity_series, bt['params']) , trades)
                    if bt_id:
                        st.success(f"Backtest saved successfully! ID: {bt_id}")
                    else:
                        st.error("Failed to save backtest.")

            # --- VISUALIZATIONS ---
            viz = TradingVisualizer()
            # 1. Metrics Cards
            cols = st.columns(len(metrics))
            for i, (k, v) in enumerate(metrics.items()):
                cols[i].metric(label=k.replace('_', ' ').title(), value=f"{v:.2f}" if "ratio" in k else f"{v:.2%}" if "return" in k or "rate" in k or "drawdown" in k else f"{v}")

            # 2. Main Price Chart
            st.subheader("Interactive Price Chart")
            fig_price = viz.plot_candlestick_with_trades(data, trades, symbol=bt['symbol'])
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
