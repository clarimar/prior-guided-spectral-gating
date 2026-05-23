"""
Computar priors chemométricos
- ANOVA F-statistic
- VIP (Variable Importance in Projection) via PLS
- Random Forest feature importance
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression


def compute_anova_scores(X, y):
    """
    Calcular ANOVA F-statistic para cada banda
    
    Args:
        X: spectra (n_samples, n_bands)
        y: target (n_samples,) ou (n_samples, 1)
    
    Returns:
        F-scores normalized to [0, 1]
    """
    print("   Computing ANOVA F-statistic...")
    
    if y.ndim > 1:
        y = y.ravel()
    
    # F-statistic
    f_scores, _ = f_regression(X, y)
    
    # Normalizar para [0, 1]
    f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min())
    
    return f_scores_norm


def compute_vip_scores(X, y, n_components=10):
    """
    Calcular VIP (Variable Importance in Projection) via PLS
    
    Args:
        X: spectra (n_samples, n_bands)
        y: target (n_samples,) ou (n_samples, 1)
        n_components: número de componentes PLS
    
    Returns:
        VIP scores normalized to [0, 1]
    """
    print(f"   Computing VIP scores (PLS with {n_components} components)...")
    
    if y.ndim > 1:
        y = y.ravel()
    
    # Ajustar número de componentes se necessário
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    
    # PLS regression
    pls = PLSRegression(n_components=n_components)
    pls.fit(X, y)
    
    # Calcular VIP
    # VIP_j = sqrt(p * sum_h(w_jh^2 * SSY_h) / sum_h(SSY_h))
    # onde p = número de variáveis, w = pesos, SSY = variância explicada
    
    t = pls.x_scores_  # scores
    w = pls.x_weights_  # weights
    q = pls.y_loadings_  # y loadings
    
    p, h = w.shape
    vip_scores = np.zeros(p)
    
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i, j] / np.linalg.norm(w[:, j]))**2 for j in range(h)])
        vip_scores[i] = np.sqrt(p * (s.T @ weight) / total_s)
    
    # Normalizar para [0, 1]
    vip_scores_norm = (vip_scores - vip_scores.min()) / (vip_scores.max() - vip_scores.min())
    
    return vip_scores_norm


def compute_rf_importance(X, y, n_estimators=100):
    """
    Calcular Random Forest feature importance
    
    Args:
        X: spectra (n_samples, n_bands)
        y: target (n_samples,) ou (n_samples, 1)
        n_estimators: número de árvores
    
    Returns:
        Feature importance normalized to [0, 1]
    """
    print(f"   Computing Random Forest importance ({n_estimators} trees)...")
    
    if y.ndim > 1:
        y = y.ravel()
    
    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    # Feature importance
    importance = rf.feature_importances_
    
    # Normalizar para [0, 1]
    importance_norm = (importance - importance.min()) / (importance.max() - importance.min())
    
    return importance_norm


def compute_all_priors(dataset='tecator', target_name='Fat'):
    """
    Computar todos os priors para um dataset e target
    
    Args:
        dataset: nome do dataset
        target_name: nome do target (Fat, Water, ou Protein)
    """
    print("\n" + "="*70)
    print(f"COMPUTING PRIORS: {dataset.upper()} - Target: {target_name}")
    print("="*70)
    
    # Carregar dados processados
    processed_path = Path('data/processed') / f'{dataset}_processed.pkl'
    
    if not processed_path.exists():
        print(f"❌ Erro: {processed_path} não existe!")
        print("   Execute primeiro: python src/data/preprocessing.py")
        return
    
    with open(processed_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    
    # Selecionar target específico (assumindo y_train tem shape (n, 3))
    if y_train.ndim > 1:
        target_idx = {'Fat': 0, 'Water': 1, 'Protein': 2}
        if target_name in target_idx:
            y = y_train[:, target_idx[target_name]]
        else:
            y = y_train[:, 0]  # default: primeiro target
    else:
        y = y_train
    
    print(f"\n📊 Data shape:")
    print(f"   X_train: {X_train.shape}")
    print(f"   y ({target_name}): {y.shape}")
    
    # Computar priors
    print(f"\n🔬 Computing priors for {target_name}:")
    
    anova_scores = compute_anova_scores(X_train, y)
    vip_scores = compute_vip_scores(X_train, y)
    rf_importance = compute_rf_importance(X_train, y)
    
    # Salvar
    priors_dir = Path('data/priors')
    priors_dir.mkdir(exist_ok=True)
    
    np.save(priors_dir / f'{dataset}_{target_name.lower()}_anova.npy', anova_scores)
    np.save(priors_dir / f'{dataset}_{target_name.lower()}_vip.npy', vip_scores)
    np.save(priors_dir / f'{dataset}_{target_name.lower()}_rf.npy', rf_importance)
    
    print(f"\n✅ Priors salvos em: {priors_dir}/")
    print(f"   - {dataset}_{target_name.lower()}_anova.npy")
    print(f"   - {dataset}_{target_name.lower()}_vip.npy")
    print(f"   - {dataset}_{target_name.lower()}_rf.npy")
    
    # Estatísticas
    print(f"\n📈 Prior statistics:")
    print(f"   ANOVA: min={anova_scores.min():.4f}, max={anova_scores.max():.4f}, mean={anova_scores.mean():.4f}")
    print(f"   VIP:   min={vip_scores.min():.4f}, max={vip_scores.max():.4f}, mean={vip_scores.mean():.4f}")
    print(f"   RF:    min={rf_importance.min():.4f}, max={rf_importance.max():.4f}, mean={rf_importance.mean():.4f}")
    
    return {
        'anova': anova_scores,
        'vip': vip_scores,
        'rf': rf_importance
    }


def main():
    """Computar priors para todos targets do Tecator"""
    
    targets = ['Fat', 'Water', 'Protein']
    
    for target in targets:
        try:
            compute_all_priors('tecator', target)
        except Exception as e:
            print(f"❌ Erro ao processar {target}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✨ PRIORS COMPUTATION COMPLETO!")
    print("="*70)


if __name__ == '__main__':
    main()
