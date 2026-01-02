import streamlit as st

with st.sidebar:
    # Thêm Logo hoặc Tên hệ thống hoành tráng
    st.markdown("<h1 style='text-align: center; color: #26a69a;'>QUANT PRO</h1>", unsafe_allow_html=True)
    st.divider()
    
    # Thông tin tài khoản / Trạng thái
    with st.container():
        col1, col2 = st.columns([1, 3])
        col1.write("🟢") # Icon trạng thái
        col2.write("**Server: Singapore**")
    
    st.caption("Last Sync: 2023-10-27 14:30:05")
    st.divider()

    # Bạn vẫn có thể để các bộ lọc chung ở đây
    st.sidebar.selectbox("Global Currency", ["USD", "VND", "BTC"])