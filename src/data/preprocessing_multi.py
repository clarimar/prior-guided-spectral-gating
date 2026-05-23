
"""
Preprocessing para múltiplos datasets
Uso: python src/data/preprocessing_multi.py --dataset gasoline
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pickle
import argparse


def preprocess_dataset(dataset_name='tecator', test_size=0.2, n_folds=5):
    print("\n" + "=" * 60)
    print(f"PREPROCESSANDO: {dataset_name.upper()}")
    print("=" * 60)
    
    # Carregar dados
    data_dir = Path('data/raw') / dataset_name
    spectra = pd.read_csv(data_dir / 'spectra.csv').values
    targets = pd.read_csv(data_dir / 'targets.csv').values
    
    print(f"\n📂 Dados carregados:")
    print(f"   Spectra: {spectra.shape}")
    print(f"   Targets: {targets.shape}")
    
    # Para gasoline, target é 1D
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        spectra, targets, test_size=test_size, random_state=42
    )
    
    # Normalizar spectra
    scaler_X = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train)
    X_test_norm = scaler_X.transform(X_test)
    
    # Normalizar targets
    scaler_y = StandardScaler()
    y_train_norm = scaler_y.fit_transform(y_train)
    y_test_norm = scaler_y.transform(y_test)
    
    print(f"\n🔧 Normalização:")
    print(f"   Targets original: [{y_train.min():.2f}, {y_train.max():.2f}]")
    print(f"   Targets norm: [{y_train_norm.min():.2f}, {y_train_norm.max():.2f}]")
    
    # CV splits
    cv_splits = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_norm)):
        cv_splits.append({
            'fold': fold_idx,
            'train_idx': train_idx,
            'val_idx': val_idx
        })
    
    # Salvar
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    
    processed_data = {
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'y_train': y_train_norm,
        'y_test': y_test_norm,
        'y_train_raw': y_train,
        'y_test_raw': y_test,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'cv_splits': cv_splits,
        'metadata': {
            'dataset': dataset_name,
            'n_samples': len(spectra),
            'n_bands': spectra.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_folds': n_folds
        }
    }
    
    output_path = output_dir / f'{dataset_name}_processed.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\n✅ Dados processados: {output_path}")
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    print(f"   Bands: {spectra.shape[1]}, CV folds: {n_folds}")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tecator', 
                       choices=['tecator', 'gasoline'],
                       help='Dataset name')
    args = parser.parse_args()
    
    preprocess_dataset(args.dataset)
