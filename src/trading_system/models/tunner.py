import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import optuna
from trading_system.utils.logger import setup_logger
from trading_system.models.gru_autoencoder import OptimizedGRU

logger = setup_logger('HybridModelTuner')

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