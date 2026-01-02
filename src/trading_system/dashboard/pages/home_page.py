import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from trading_system.data.storage.market import StorageMarketData 

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Professional Trading Terminal",
    page_icon="📈",
    layout="wide", # Tận dụng tối đa chiều ngang màn hình
)

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
        start_date = st.date_input("Start", datetime.now().date() - timedelta(days=14))
    with col2:
        end_date = st.date_input("End", datetime.now().date())
    
    st.divider()
    st.info("System Status: Online")

# --- DATA LOADING ---
storage_market = StorageMarketData()
market_data = storage_market.get_data(symbol=symbol, timeframe=timeframe, start_date=str(start_date), end_date=str(end_date))

# --- MAIN PAGE LAYOUT ---
st.title(f"📊 {symbol} Analysis Dashboard")

# Top Metrics
m1, m2, m3, m4 = st.columns(4)
current_price = market_data['close'].iloc[-1]
price_change = current_price - market_data['close'].iloc[-2]
m1.metric("Current Price", f"${current_price:,.2f}", f"{price_change:,.2f}")
m2.metric("24h High", f"${market_data['high'].max():,.2f}")
m3.metric("24h Low", f"${market_data['low'].min():,.2f}")
m4.metric("Volume", f"{market_data['volume'].sum():,.0f}")

# --- SLIDER ---
window = st.slider(
    "Số lượng nến hiển thị",
    min_value=50,
    max_value=len(market_data),
    value=200,
    step=10
)

# Cắt data theo slider
data = market_data.tail(window)

# --- CANDLESTICK CHART ---
fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['open'],
    high=data['high'],
    low=data['low'],
    close=data['close'],
    increasing_line_color='#26a69a',
    decreasing_line_color='#ef5350',
    name="Price"
)])

fig.update_layout(
    template='plotly_dark',
    plot_bgcolor='#131722',
    paper_bgcolor='#131722',
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis_rangeslider_visible=False,
    yaxis=dict(gridcolor='#363a45', zeroline=False),
    xaxis=dict(gridcolor='#363a45', zeroline=False),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# --- DATA TABLE (Optional) ---
with st.expander("See Raw Market Data"):
    st.dataframe(market_data.tail(20), use_container_width=True)


    # streamlit run src/trading_system/dashboard/pages/home_page.py