import streamlit as st

# 1. Định nghĩa các trang
home_page = st.Page("pages/home_page.py", title="Market Dashboard", icon="📊", default=True)
backtest_page = st.Page("pages/backtest_page.py", title="Strategy Backtest", icon="🧪")
settings_page = st.Page("pages/settings_page.py", title="System Settings", icon="⚙️")

# 2. Tạo Navigation với Sidebar Grouping
pg = st.navigation({
    "MARKET ANALYSIS": [home_page],
    "ALGO TRADING": [backtest_page],
    "CONFIGURATION": [settings_page]
})

# 3. Cấu hình chung cho tất cả các trang
st.set_page_config(page_title="Pro Trading System", layout="wide")

# 4. Chạy app
pg.run()
# streamlit run src/trading_system/dashboard/app.py