import pandas as pd
import numpy as np

class FeatureSelector:
    def __init__(self, correlation_threshold=0.8):
        self.correlation_threshold = correlation_threshold

    def drop_correlated_features(self, df: pd.DataFrame, target_col='target_return') -> pd.DataFrame:
        """
        Remove features that are highly correlated with each other.
        Keeps the one with higher correlation to the target.
        """
        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Separate features and target
        if target_col in numeric_df.columns:
            features = numeric_df.drop(columns=[target_col])
        else:
            features = numeric_df
            
        # Compute correlation matrix
        corr_matrix = features.corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > self.correlation_threshold)]
        
        print(f"Dropping {len(to_drop)} features due to correlation > {self.correlation_threshold}: {to_drop}")
        
        return df.drop(columns=to_drop)

    def select_best_features(self, df: pd.DataFrame, target_col='target_return', n_features=20):
        """
        (Optional) Use ML to select best features.
        """
        # Placeholder for more advanced selection (e.g., using Random Forest importance)
        return df
