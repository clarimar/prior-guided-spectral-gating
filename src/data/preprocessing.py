"""
Preprocessamento dos datasets
- Normalização
- Splits de treino/validação/teste
- Cross-validation folds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import pickle


def load_raw_dataset(dataset_name='tecator'):
    """Carregar dataset bruto"""
    print(f"\n📂 Carregando {dataset_name}...")
    
    data_dir = Path('data/raw') / dataset_name
    
    # Carregar espectra e targets
    spectra = pd.read_csv(data_dir / 'spectra.csv')
    targets = pd.read_csv(data_dir / 'targets.csv')
    
    print(f"   Spectra: {spectra.shape}")
    print(f"   Targets: {targets.shape}")
    
    return spectra.values, targets.values


def normalize_spectra(X_train, X_test=None):
    """Normalizar espectros (Z-score)"""
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_norm = scaler.transform(X_test)
        return X_train_norm, X_test_norm, scaler
    
    return X_train_norm, scaler


def create_cv_splits(X, y, n_folds=5, random_state=42):
    """Criar splits de cross-validation"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    cv_splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        cv_splits.append({
            'fold': fold_idx,
            'train_idx': train_idx,
            'val_idx': val_idx
        })
    
    return cv_splits


def preprocess_dataset(dataset_name='tecator', test_size=0.2, n_folds=5):
    """
    Pipeline completo de preprocessamento
    
    Returns:
        dict com dados processados e splits
    """
    print("\n" + "=" * 60)
    print(f"PREPROCESSANDO: {dataset_name.upper()}")
    print("=" * 60)
    
    # Carregar dados brutos
    X, y = load_raw_dataset(dataset_name)
    
    # Split treino/teste
    print(f"\n📊 Criando split treino/teste ({int((1-test_size)*100)}%/{int(test_size*100)}%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Normalizar
    print("🔧 Normalizando espectros (Z-score)...")
    X_train_norm, X_test_norm, scaler = normalize_spectra(X_train, X_test)
    
    # Cross-validation splits
    print(f"🔀 Criando {n_folds}-fold cross-validation...")
    cv_splits = create_cv_splits(X_train_norm, y_train, n_folds=n_folds)
    
    # Preparar output
    processed_data = {
        'X_train': X_train_norm,
        'X_test': X_test_norm,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
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
    """Preprocessar todos os datasets disponíveis"""
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
