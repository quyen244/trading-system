import sys
import os
import pandas as pd
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from trading_system.data.pipeline import CryptoDataPipeline
from trading_system.models.hybrid_model import OptimizedGRU, HybridModelTuner, MLStrategy

def generate_mock_data(n_rows=500):
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_rows, freq='h')
    
    # Create a trending/volatile price
    t = np.linspace(0, 100, n_rows)
    base_price = 100 + 10 * np.sin(t) + 0.1 * np.cumsum(np.random.normal(0, 1, n_rows))
    
    df = pd.DataFrame({
        'open': base_price + np.random.normal(0, 0.5, n_rows),
        'high': base_price + 2 + np.random.normal(0, 0.5, n_rows),
        'low': base_price - 2 + np.random.normal(0, 0.5, n_rows),
        'close': base_price + np.random.normal(0, 0.5, n_rows),
        'volume': np.random.uniform(1000, 5000, n_rows),
        'month': dates.month,
        'hour': dates.hour
    })
    return df

def test_pipeline():
    print("Testing CryptoDataPipeline...")
    df = generate_mock_data(1000)
    pipeline = CryptoDataPipeline(window_size=10, horizon=6)
    
    # Test prepare_train_test
    X_train, y_ret_train, y_lab_train, X_val, y_ret_val, y_lab_val = pipeline.prepare_train_test(df)
    
    print(f"X_train shape: {X_train.shape}")
    print(f"y_lab_train classes: {np.unique(y_lab_train)}")
    
    assert X_train.shape[1] == 10, "Window size mismatch"
    assert X_train.shape[0] == len(y_ret_train) == len(y_lab_train), "Samples count mismatch"
    
    # Test live transform
    live_df = df.iloc[-50:].copy() # Enough for indicators and window
    live_window = pipeline.transform_live_data(live_df)
    print(f"Live window shape: {live_window.shape}")
    assert live_window.shape == (1, 10, X_train.shape[2]), "Live window shape mismatch"
    print("Pipeline test PASSED\n")
    return X_train, y_ret_train, y_lab_train, X_val, y_ret_val, y_lab_val, pipeline

def test_model_training(X_train, y_ret_train, y_lab_train, X_val, y_ret_val, y_lab_val, pipeline):
    print("Testing HybridModelTuner...")
    tuner = HybridModelTuner(X_train, y_ret_train, y_lab_train, X_val, y_ret_val, y_lab_val)
    
    # Short trials for speed
    best_gru = tuner.tune_gru(n_trials=2, epochs=2)
    best_xgb = tuner.tune_xgboost(n_trials=2)
    
    assert best_gru is not None
    assert best_xgb is not None
    print("Model training test PASSED\n")
    
    print("Testing MLStrategy...")
    strategy = MLStrategy(pipeline, best_gru, best_xgb)
    
    df = generate_mock_data(100)
    signal, prob = strategy.predict(df)
    print(f"Sample prediction - Signal: {signal}, Confidence: {prob:.4f}")
    
    assert signal in [0, 1, 2]
    assert 0.0 <= prob <= 1.0
    print("MLStrategy test PASSED\n")
    
    # Test Save/Load
    print("Testing Save/Load...")
    strategy.save_model('test_models')
    loaded_strategy = MLStrategy.load_model('test_models')
    
    signal2, prob2 = loaded_strategy.predict(df)
    assert signal == signal2
    assert np.isclose(prob, prob2)
    print("Save/Load test PASSED\n")

if __name__ == "__main__":
    try:
        X_t, yr_t, yl_t, X_v, yr_v, yl_v, pipe = test_pipeline()
        test_model_training(X_t, yr_t, yl_t, X_v, yr_v, yl_v, pipe)
        print("ALL TESTS PASSED!")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
