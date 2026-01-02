from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Trade:
    """
    Trade class representing a single completed trade.
    Matches the 'trade' table schema.
    """
    backtest_id: Optional[int] = None
    symbol: str = ""
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    side: str = "long" # "long" or "short"
    gross_pnl: float = 0.0
    net_pnl: float = 0.0

    def to_dict(self):
        return {
            "backtest_id": self.backtest_id,
            "symbol": self.symbol,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "position_size": self.position_size,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.net_pnl,
            "side": self.side
        }
