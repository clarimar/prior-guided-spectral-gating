
"""
Baseline PLS regression para comparação
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score


def test_pls(n_components=10):
    print("\n" + "="*60)
    print("📊 BASELINE: PLS REGRESSION")
    print("="*60)
    
    # Carregar dados
    with open('data/processed/tecator_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train'][:, 0]  # Fat normalizado
    y_test = data['y_test'][:, 0]
    
    print(f"\nData: Train={X_train.shape}, Test={X_test.shape}")
    
    # PLS
    print(f"\nTraining PLS with {n_components} components...")
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train, y_train)
    
    # Predict
    y_pred_train = pls.predict(X_train)
    y_pred_test = pls.predict(X_test)
    
    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\n📊 Results:")
    print(f"   Train RMSE: {train_rmse:.4f}, R²: {train_r2:.4f}")
    print(f"   Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    print("\n" + "="*60)
    print("✅ PLS BASELINE COMPLETE")
    print("="*60)
    
    return test_rmse, test_r2


if __name__ == '__main__':
    test_pls(n_components=10)

