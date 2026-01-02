import unittest
import pandas as pd
import numpy as np
from src.risk.manager import RiskManager
from src.features.engineering import FeatureEngineer
from src.strategies.trend import TrendFollowingStrategy

class TestTradingSystem(unittest.TestCase):

    def test_risk_manager_sizing(self):
        rm = RiskManager(initial_capital=10000)
        # 1% Risk = 100 USD. Entry=100, SL=90 (Diff=10). Size = 100/10 = 10.
        size = rm.calculate_position_size(entry_price=100, stop_loss=90)
        self.assertAlmostEqual(size, 10.0)

    def test_risk_manager_max_drawdown(self):
        rm = RiskManager(initial_capital=10000, max_drawdown=0.10)
        # Drop to 8900 (11% DD)
        # Peak capital starts at 10000.
        allowed = rm.check_risk_allowance(portfolio_value=8900)
        self.assertFalse(allowed) # Should trigger Kill Switch
        self.assertTrue(rm.kill_switch)

    def test_feature_engineering(self):
        df = pd.DataFrame({
            'open': [10]*100,
            'high': [12]*100,
            'low': [8]*100,
            'close': np.random.normal(10, 1, 100),
            'volume': [1000]*100
        })
        fe = FeatureEngineer()
        processed = fe.add_all_features(df)
        self.assertIn("RSI_14", processed.columns)
        self.assertIn("EMA_50", processed.columns)

    def test_strategy_signal(self):
        strategy = TrendFollowingStrategy({'fast_ma': 10, 'slow_ma': 20})
        df = pd.DataFrame({
            'close': [10]*50,
            'open': [10]*50,
            'high': [10]*50,
            'low': [10]*50,
            'volume': [100]*50
        })
        # Mocking data to ensure no errors
        signals = strategy.generate_signals(df)
        self.assertIn('signal', signals.columns)

if __name__ == '__main__':
    unittest.main()
