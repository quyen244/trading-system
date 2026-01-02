import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import optuna
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import f1_score
from trading_system.utils.logger import setup_logger

logger = setup_logger('HybridModel')

class OptimizedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super(OptimizedGRU, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim // 2, 1) # Reconstruction head for target_return

    def forward(self, x):
        # x shape: (Batch, Window, Features)
        out, _ = self.gru1(x)
        out = self.ln1(out)
        out = torch.relu(out)
        
        # Second GRU layer
        out, hn = self.gru2(out)
        
        # Feature Extraction: Get final hidden state
        # hn shape: (num_layers, batch, hidden_size)
        last_hidden = hn[-1] # (Batch, Hidden//2)
        features = self.dropout(self.ln2(last_hidden))
        
        # Reconstruction (prediction of target_return)
        pred = self.fc(features)
        return pred, features

class HybridModelTuner:
    def __init__(self, X_train, y_ret_train, y_lab_train, X_val, y_ret_val, y_lab_val, device='cpu'):
        # Tensors for GRU
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        self.y_ret_train_tensor = torch.tensor(y_ret_train, dtype=torch.float32).view(-1, 1).to(device)
        
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        self.y_ret_val_tensor = torch.tensor(y_ret_val, dtype=torch.float32).view(-1, 1).to(device)
        
        # Numpy for XGBoost
        self.y_lab_train = y_lab_train
        self.y_lab_val = y_lab_val
        
        self.device = device
        self.input_dim = X_train.shape[2]
        
        self.best_gru = None
        self.best_xgb = None
        
    def tune_gru(self, n_trials=10, epochs=20):
        """Tune GRU for reconstructing target_return."""
        logger.info(f"Starting GRU tuning with {n_trials} trials...")
        
        def objective(trial):
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            dropout = trial.suggest_float('dropout', 0.1, 0.5)
            
            model = OptimizedGRU(self.input_dim, hidden_dim, dropout).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            model.train()
            train_loss = 0
            for epoch in range(epochs):
                optimizer.zero_grad()
                pred, _ = model(self.X_train_tensor)
                loss = criterion(pred, self.y_ret_train_tensor)
                loss.backward()
                optimizer.step()
                train_loss = loss.item()
            
            # Use validation loss for tuning
            model.eval()
            with torch.no_grad():
                val_pred, _ = model(self.X_val_tensor)
                val_loss = criterion(val_pred, self.y_ret_val_tensor).item()
            
            return val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best GRU params: {study.best_params}")
        
        # Train final model with best params
        best_p = study.best_params
        self.best_gru = OptimizedGRU(self.input_dim, best_p['hidden_dim'], best_p['dropout']).to(self.device)
        optimizer = optim.Adam(self.best_gru.parameters(), lr=best_p['lr'])
        criterion = nn.MSELoss()
        
        self.best_gru.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred, _ = self.best_gru(self.X_train_tensor)
            loss = criterion(pred, self.y_ret_train_tensor)
            loss.backward()
            optimizer.step()
            
        return self.best_gru

    def tune_xgboost(self, n_trials=20):
        """Tune XGBoost on GRU-extracted features."""
        if not self.best_gru:
            raise ValueError("Train GRU before tuning XGBoost.")
            
        logger.info(f"Extracting features from GRU for XGBoost tuning...")
        self.best_gru.eval()
        with torch.no_grad():
            _, train_feats = self.best_gru(self.X_train_tensor)
            _, val_feats = self.best_gru(self.X_val_tensor)
            
        X_train_xgb = train_feats.cpu().numpy()
        X_val_xgb = val_feats.cpu().numpy()
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_train_xgb, self.y_lab_train, 
                    eval_set=[(X_val_xgb, self.y_lab_val)], 
                    verbose=False)
            
            preds = clf.predict(X_val_xgb)
            return f1_score(self.y_lab_val, preds, average='weighted')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best XGB params: {study.best_params}")
        self.best_xgb = xgb.XGBClassifier(**study.best_params)
        self.best_xgb.fit(X_train_xgb, self.y_lab_train)
        
        return self.best_xgb

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
