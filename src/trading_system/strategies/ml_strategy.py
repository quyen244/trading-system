
import joblib
import os
import torch
import xgboost as xgb
import pandas as pd
import numpy as np
from trading_system.models.gru_autoencoder import OptimizedGRU
from trading_system.utils.logger import setup_logger

logger = setup_logger('MLStrategy')

class MLStrategy:
    def __init__(self, pipeline, gru_model, xgb_model, device='cpu'):
        self.pipeline = pipeline
        self.gru_model = gru_model
        self.xgb_model = xgb_model
        self.device = device
        
        self.gru_model.eval()
        self.gru_model.to(device)
        
    def predict(self, df_window: pd.DataFrame, threshold=0.6):
        """Live prediction logic."""
        try:
            # 1. Pipeline transform
            x_np = self.pipeline.transform_live_data(df_window)
            x_tensor = torch.tensor(x_np, dtype=torch.float32).to(self.device)
            
            # 2. GRU Feature Extraction
            with torch.no_grad():
                _, features = self.gru_model(x_tensor)
                features_np = features.cpu().numpy()
            
            # 3. XGBoost Classification
            probs = self.xgb_model.predict_proba(features_np)[0]
            
            # Signal: 0=Hold, 1=Buy, 2=Sell
            # Check Buy (1) and Sell (2)
            if probs[1] > threshold:
                return 1, probs[1]
            elif probs[2] > threshold:
                return 2, probs[2]
            else:
                # Use class 0 (Hold) confidence if neither buy nor sell exceeds threshold
                return 0, probs[0]
                
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return 0, 0.0

    def save_model(self, folder='models'):
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        self.pipeline.save(os.path.join(folder, 'pipeline.pkl'))
        torch.save(self.gru_model.state_dict(), os.path.join(folder, 'gru_model.pth'))
        # Save GRU config
        gru_config = {
            'input_dim': self.gru_model.gru1.input_size,
            'hidden_dim': self.gru_model.gru1.hidden_size
        }
        joblib.dump(gru_config, os.path.join(folder, 'gru_config.pkl'))
        # Use joblib for XGBoost to preserve sklearn attributes (classes_, etc.)
        joblib.dump(self.xgb_model, os.path.join(folder, 'xgb_model.pkl'))
        logger.info(f"Strategy components saved to {folder}")

    @classmethod
    def load_model(cls, folder='models', device='cpu'):
        pipeline = joblib.load(os.path.join(folder, 'pipeline.pkl'))
        
        gru_config = joblib.load(os.path.join(folder, 'gru_config.pkl'))
        gru_model = OptimizedGRU(gru_config['input_dim'], gru_config['hidden_dim'])
        gru_model.load_state_dict(torch.load(os.path.join(folder, 'gru_model.pth'), map_location=device))
        
        xgb_model = joblib.load(os.path.join(folder, 'xgb_model.pkl'))
        
        return cls(pipeline, gru_model, xgb_model, device)
