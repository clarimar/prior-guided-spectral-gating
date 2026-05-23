
"""
PLS baseline para múltiplos datasets
Uso: python scripts/baseline_pls_multi.py --dataset gasoline
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
import argparse


def test_pls(dataset_name, n_components=10):
    print("\n" + "="*60)
    print(f"📊 PLS BASELINE: {dataset_name.upper()}")
    print("="*60)
    
    # Load data
    data_path = Path('data/processed') / f'{dataset_name}_processed.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train'][:, 0]
    y_test = data['y_test'][:, 0]
    
    print(f"\nData: Train={X_train.shape}, Test={X_test.shape}")
    
    # Adjust n_components for small datasets
    max_comp = min(n_components, X_train.shape[0]-1, X_train.shape[1])
    print(f"\nTraining PLS with {max_comp} components...")
    
    pls = PLSRegression(n_components=max_comp)
    pls.fit(X_train, y_train)
    
    # Predict
    y_pred_train = pls.predict(X_train).ravel()
    y_pred_test = pls.predict(X_test).ravel()
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                       choices=['tecator', 'gasoline'])
    args = parser.parse_args()
    
    test_pls(args.dataset)

