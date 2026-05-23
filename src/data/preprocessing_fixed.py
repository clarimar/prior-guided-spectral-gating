
"""
Preprocessing com normalização de targets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pickle


def load_raw_dataset(dataset_name='tecator'):
    print(f"\n📂 Carregando {dataset_name}...")
    data_dir = Path('data/raw') / dataset_name
    spectra = pd.read_csv(data_dir / 'spectra.csv')
    targets = pd.read_csv(data_dir / 'targets.csv')
    print(f"   Spectra: {spectra.shape}")
    print(f"   Targets: {targets.shape}")
    return spectra.values, targets.values


def preprocess_dataset(dataset_name='tecator', test_size=0.2, n_folds=5):
    print("\n" + "=" * 60)
    print(f"PREPROCESSANDO: {dataset_name.upper()}")
    print("=" * 60)
    
    X, y = load_raw_dataset(dataset_name)
    
    # Split treino/teste
    print(f"\n📊 Criando split treino/teste ({int((1-test_size)*100)}%/{int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Normalizar SPECTRA
    print("🔧 Normalizando espectros (Z-score)...")
    scaler_X = StandardScaler()
    X_train_norm = scaler_X.fit_transform(X_train)
    X_test_norm = scaler_X.transform(X_test)
    
    # Normalizar TARGETS (IMPORTANTE!)
    print("🔧 Normalizando targets (Z-score)...")
    scaler_y = StandardScaler()
    y_train_norm = scaler_y.fit_transform(y_train)
    y_test_norm = scaler_y.transform(y_test)
    
    print(f"   Targets original range: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"   Targets normalized range: {y_train_norm.min():.2f} - {y_train_norm.max():.2f}")
    
    # Cross-validation splits
    print(f"🔀 Criando {n_folds}-fold cross-validation...")
    cv_splits = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_norm)):
        cv_splits.append({
            'fold': fold_idx,
            'train_idx': train_idx,
            'val_idx': val_idx
        })
    
    # Preparar output
    processed_data = {
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'y_train': y_train_norm,  # Normalizado
        'y_test': y_test_norm,    # Normalizado
        'y_train_raw': y_train,   # Original para referência
        'y_test_raw': y_test,     # Original para referência
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,     # Scaler de targets
        'cv_splits': cv_splits,
        'metadata': {
            'dataset': dataset_name,
            'n_samples': len(X),
            'n_bands': X.shape[1],
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_folds': n_folds
        }
    }
    
    # Salvar
    output_dir = Path('data/processed')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'{dataset_name}_processed.pkl'
    
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"\n✅ Dados processados salvos: {output_path}")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Bands: {X.shape[1]}")
    print(f"   CV folds: {n_folds}")
    
    return processed_data


def main():
    datasets = ['tecator']
    
    for dataset in datasets:
        try:
            preprocess_dataset(dataset)
        except Exception as e:
            print(f"❌ Erro ao processar {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✨ PREPROCESSAMENTO CONCLUÍDO!")
    print("=" * 60)


if __name__ == '__main__':
    main()
