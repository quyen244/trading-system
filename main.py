# scripts/fetch_data.py
import sys
import os

# Thêm thư mục gốc vào path để import được src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader.download import DataLoader


def main():
    # # 1. Khởi tạo
    loader = DataLoader(exchange_id='binance')
    
    # # 2. Cấu hình tải
    symbol = 'BTC/USDT'
    timeframe = '1h'          # 1 giờ
    start_date = '2023-01-01T00:00:00Z' # Định dạng ISO 8601
    
    # # 3. Thực thi tải Historical Data (cho Backtest)
    # print(f"--- Đang tải dữ liệu {symbol} ---")
    # df = loader.fetch_historical_data(symbol, timeframe, start_date)
    
    # # In thử 5 dòng đầu
    # print("\n5 dòng dữ liệu đầu tiên:")
    # print(df.head())
    
    # # 4. Thử load lại từ ổ cứng (Test Cache)
    print("\n--- Test đọc từ Disk ---")
    df_cached = loader.load_data(symbol, timeframe)
    print(f"Đọc thành công {len(df_cached)} dòng từ file parquet.")

    return df_cached

if __name__ == "__main__":
    import os
    print(os.getcwd())