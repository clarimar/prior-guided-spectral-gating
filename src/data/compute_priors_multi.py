
"""
Compute priors para múltiplos datasets
Uso: python src/data/compute_priors_multi.py --dataset gasoline
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_selection import f_regression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
import argparse


def compute_vip_scores(X, y, n_components=10):
    """VIP scores from PLS"""
    pls = PLSRegression(n_components=min(n_components, X.shape[0]-1, X.shape[1]))
    pls.fit(X, y)
    
    t = pls.x_scores_
    w = pls.x_weights_
    q = pls.y_loadings_
    
    p, h = w.shape
    vips = np.zeros((p,))
    
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    total_s = np.sum(s)
    
    for i in range(p):
        weight = np.array([(w[i,j] / np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vips[i] = np.sqrt(p * (s.T @ weight) / total_s)
    
    return vips


def compute_all_priors(dataset_name='tecator'):
    print("\n" + "="*60)
    print(f"COMPUTING PRIORS: {dataset_name.upper()}")
    print("="*60)
    
    # Load data
    data_path = Path('data/processed') / f'{dataset_name}_processed.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train'][:, 0]  # First target
    
    n_bands = X_train.shape[1]
    print(f"\nData: {X_train.shape[0]} samples, {n_bands} bands")
    
    # Output directory
    output_dir = Path('data/priors')
    output_dir.mkdir(exist_ok=True)
    
    # 1. ANOVA F-statistic
    print("\n1. Computing ANOVA F-statistic...")
    f_stats, _ = f_regression(X_train, y_train)
    f_stats = np.nan_to_num(f_stats, 0)
    anova_norm = (f_stats - f_stats.min()) / (f_stats.max() - f_stats.min() + 1e-10)
    
    output_path = output_dir / f'{dataset_name}_octane_anova.npy'
    np.save(output_path, anova_norm)
    print(f"   ✅ Saved: {output_path}")
    print(f"   Range: [{anova_norm.min():.4f}, {anova_norm.max():.4f}]")
    print(f"   Mean: {anova_norm.mean():.4f}")
    
    # 2. VIP scores
    print("\n2. Computing VIP scores...")
    n_comp = min(10, X_train.shape[0]-1, n_bands)
    vip_scores = compute_vip_scores(X_train, y_train, n_components=n_comp)
    vip_scores = np.nan_to_num(vip_scores, 0)
    vip_norm = (vip_scores - vip_scores.min()) / (vip_scores.max() - vip_scores.min() + 1e-10)
    
    output_path = output_dir / f'{dataset_name}_octane_vip.npy'
    np.save(output_path, vip_norm)
    print(f"   ✅ Saved: {output_path}")
    print(f"   Range: [{vip_norm.min():.4f}, {vip_norm.max():.4f}]")
    print(f"   Mean: {vip_norm.mean():.4f}")
    
    # 3. Random Forest importance
    print("\n3. Computing RF importance...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_importance = rf.feature_importances_
    rf_norm = (rf_importance - rf_importance.min()) / (rf_importance.max() - rf_importance.min() + 1e-10)
    
    output_path = output_dir / f'{dataset_name}_octane_rf.npy'
    np.save(output_path, rf_norm)
    print(f"   ✅ Saved: {output_path}")
    print(f"   Range: [{rf_norm.min():.4f}, {rf_norm.max():.4f}]")
    print(f"   Mean: {rf_norm.mean():.4f}")
    
    # Correlation between priors
    print("\n4. Prior correlations:")
    corr_anova_vip = np.corrcoef(anova_norm, vip_norm)[0, 1]
    corr_anova_rf = np.corrcoef(anova_norm, rf_norm)[0, 1]
    corr_vip_rf = np.corrcoef(vip_norm, rf_norm)[0, 1]
    
    print(f"   ANOVA-VIP: {corr_anova_vip:.4f}")
    print(f"   ANOVA-RF:  {corr_anova_rf:.4f}")
    print(f"   VIP-RF:    {corr_vip_rf:.4f}")
    
    print("\n" + "="*60)
    print("✅ ALL PRIORS COMPUTED!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='tecator',
                       choices=['tecator', 'gasoline'],
                       help='Dataset name')
    args = parser.parse_args()
    
    compute_all_priors(args.dataset)

