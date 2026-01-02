
import ccxt
import pandas as pd
from datetime import datetime, timezone
import time
from trading_system.utils.logger import setup_logger
from trading_system.data.storage import StorageEngine
import numpy as np 

logger = setup_logger("DataIngestion")

class BinanceDataFetcher:
    def __init__(self, db_url=None):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        self.storage = StorageEngine(db_url)
        self.storage.create_tables()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 1000, since: int = None):
        """
        Fetch OHLCV data from Binance.
        """
        try:
            print(f"Fetching {symbol} {timeframe} from Binance...")
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
            
            if not ohlcv:
                print(f"No data returned for {symbol} {timeframe}")
                return pd.DataFrame()

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Ensure proper types
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)

            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def fetch_and_store(self, symbol: str, timeframe: str,  start_date: str):
        """
        Fetch historical data for X days and store it.
        """
        since = self.exchange.parse8601(start_date)
        all_ohlcv = []
        
        logger.info(f"starting to fetch {symbol} ({timeframe}) tá»« {start_date}...")
        
        while True:
            try:
                # Táº£i tá»‘i Ä‘a giá»›i háº¡n cá»§a sÃ n (thÆ°á»ng lÃ  1000 náº¿n)
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since)
                
                if len(ohlcv) == 0:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                last_timestamp = ohlcv[-1][0]
                since = last_timestamp + 1 
                
                current_time = self.exchange.milliseconds()
                if last_timestamp >= current_time - 60000:
                    break
                
                logger.info(f"last timestamp: {datetime.fromtimestamp(last_timestamp/1000, tz=timezone.utc)}")
                
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
                time.sleep(5)
                continue

        # Chuyá»ƒn sang DataFrame
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Xá»­ lÃ½ timestamp sang dáº¡ng datetime object cho dá»… Ä‘á»c
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Loáº¡i bá» cÃ¡c dÃ²ng trÃ¹ng láº·p (náº¿u cÃ³)
        df = df[~df.index.duplicated(keep='last')]
        
        logger.info(f"fetched {len(df)} rows.")

        if self.storage.store_market_data(df, symbol, timeframe):
            logger.info(f"data stored successfully")
            return df
        else:
            logger.warning("data not stored successfully")
            return pd.DataFrame()

if __name__ == "__main__":
    # Test run
    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_and_store("BTC/USDT", "1h", "2024-01-01")
    print(df.head())

# python -m trading_system.data.ingestion