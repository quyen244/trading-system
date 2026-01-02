import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost

class ModelFactory:
    def __init__(self, experiment_name="Trading_Experiments"):
        mlflow.set_experiment(experiment_name)

    def prepare_data(self, df: pd.DataFrame, target_col='target_return'):
        """
        Prepare data for training.
        Target: 1 if target_return > 0 else 0
        """
        data = df.copy().dropna()
        
        # Binary target
        y = (data[target_col] > 0).astype(int)
        X = data.drop(columns=[target_col])
        
        # Drop non-numeric cols logic if needed (e.g. timestamp)
        X = X.select_dtypes(include=[np.number])
        
        return train_test_split(X, y, test_size=0.2, shuffle=False) # Time series split implicitly via shuffle=False

    def train_model(self, df: pd.DataFrame, model_type='logistic'):
        """
        Train and log model.
        """
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        with mlflow.start_run():
            if model_type == 'logistic':
                model = LogisticRegression()
                model.fit(X_train, y_train)
                mlflow.sklearn.log_model(model, "logistic_model")
                
            elif model_type == 'xgboost':
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                mlflow.xgboost.log_model(model, "xgboost_model")
                
            else:
                raise ValueError("Unknown model type")
                
            # Evaluate
            preds = model.predict(X_test)
            probs = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, probs)
            
            # Log Metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("auc", auc)
            mlflow.log_param("model_type", model_type)
            
            print(f"Trained {model_type} - Accuracy: {acc:.4f}, AUC: {auc:.4f}")
            
            return model
