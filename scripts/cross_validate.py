
"""
Cross-validation com 5 folds
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.band_select_net_light import create_model_light


def cv_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
            n_epochs=200, lr=0.01, kl_weight=0.001):
    """Treinar um fold"""
    
    model = create_model_light(100, 1, 'data/priors/tecator_fat_anova.npy')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_rmse = float('inf')
    patience = 30
    no_improve = 0
    
    for epoch in range(n_epochs):
        model.train()
        y_pred, _ = model(X_train_fold)
        mse_loss = criterion(y_pred.squeeze(), y_train_fold)
        kl_loss = model.get_kl_loss()
        total_loss = mse_loss + kl_weight * kl_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_val, _ = model(X_val_fold)
            val_rmse = torch.sqrt(criterion(y_pred_val.squeeze(), y_val_fold))
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    
    # Final eval
    model.eval()
    with torch.no_grad():
        y_pred_final, _ = model(X_val_fold)
        final_rmse = torch.sqrt(criterion(y_pred_final.squeeze(), y_val_fold))
        
        ss_res = ((y_val_fold - y_pred_final.squeeze())**2).sum()
        ss_tot = ((y_val_fold - y_val_fold.mean())**2).sum()
        final_r2 = 1 - ss_res / ss_tot
    
    return final_rmse.item(), final_r2.item()


def cross_validate():
    print("\n" + "="*60)
    print("🔄 5-FOLD CROSS-VALIDATION")
    print("="*60)
    
    # Load data
    with open('data/processed/tecator_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train_all = torch.FloatTensor(data['X_train'])
    y_train_all = torch.FloatTensor(data['y_train'][:, 0])
    
    cv_splits = data['cv_splits']
    
    print(f"\nData: {X_train_all.shape[0]} samples, 5 folds\n")
    
    results = []
    
    for fold in cv_splits:
        fold_idx = fold['fold']
        train_idx = fold['train_idx']
        val_idx = fold['val_idx']
        
        X_train_fold = X_train_all[train_idx]
        y_train_fold = y_train_all[train_idx]
        X_val_fold = X_train_all[val_idx]
        y_val_fold = y_train_all[val_idx]
        
        print(f"Fold {fold_idx + 1}/5 - Training...", end=' ', flush=True)
        
        rmse, r2 = cv_fold(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        results.append({'fold': fold_idx + 1, 'rmse': rmse, 'r2': r2})
        
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    # Summary
    rmse_mean = np.mean([r['rmse'] for r in results])
    rmse_std = np.std([r['rmse'] for r in results])
    r2_mean = np.mean([r['r2'] for r in results])
    r2_std = np.std([r['r2'] for r in results])
    
    print("\n" + "="*60)
    print("📊 CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"   RMSE: {rmse_mean:.4f} ± {rmse_std:.4f}")
    print(f"   R²:   {r2_mean:.4f} ± {r2_std:.4f}")
    print(f"\n   PLS Baseline: RMSE=0.2957, R²=0.9191")
    
    if r2_mean > 0.90:
        print(f"\n   ✅ CONSISTENT HIGH PERFORMANCE!")
    elif r2_mean > 0.85:
        print(f"\n   ⚠️ Good but variable")
    else:
        print(f"\n   ❌ Inconsistent performance")
    
    print("="*60)


if __name__ == '__main__':
    cross_validate()

