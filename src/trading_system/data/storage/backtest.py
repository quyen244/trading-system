import pandas as pd
from trading_system.utils.logger import setup_logger
from trading_system.data.storage.base import StorageEngine
from trading_system.data.schemas import *

logger = setup_logger("StorageEngine")

class StorageBacktest(StorageEngine):
    def __init__(self, db_url=None):
        super().__init__(db_url)

    def store_data(self, backtest_data: dict, trades: list = None):
        """
        Save backtest results and associated trades.
        Args:
            backtest_data (dict): Backtest results data.
            trades (list, optional): List of trades. Defaults to None.
        """
        session = self.Session()
        try:
            # Create Backtest record
            bt = Backtest(**backtest_data)
            print(bt)
            session.add(bt)
            session.flush() # Get backtest ID
            
            # # Create Trade records
            # trade_records = []
            # for t in trades:
            #     t_data = t if isinstance(t, dict) else t.to_dict()
            #     t_data['backtest_id'] = bt.id
            #     # Remove exit_time if it's identical to entry_time or handle it
            #     trade_records.append(TradeRecord(**t_data))
            session.add(bt)
            session.flush()
            session.commit()
            # session.add_all(tra
            # de_records)
            # session.commit()
            print(f"Saved backtest {bt.id}")
            return bt.id
        except Exception as e:
            session.rollback()
            print(f"Error saving backtest: {e}")
            raise
        finally:
            session.close()

    def get_data(self, strategy_id: str = None, id: int = None):
        """ Get backtest for a specific strategy or ID.
        Args:
            strategy_id (str, optional): Filter by strategy ID. Defaults to None.
            id (int, optional): Filter by backtest ID. Defaults to None.
        
        """
        query = "SELECT * FROM backtest"
        if strategy_id:
            query += f" WHERE strategy_id = '{strategy_id}' and id = '{id}'"
        query += " ORDER BY created_at DESC"
        return pd.read_sql(query, self.engine)