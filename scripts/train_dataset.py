
"""
Treinar BandSelectNet em qualquer dataset
Uso: python scripts/train_dataset.py --dataset gasoline --prior anova
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.band_select_net_light import BandSelectNetLight


def train_model(dataset_name, prior_type, n_epochs=200, lr=0.01, kl_weight=0.001):
    print("\n" + "="*60)
    print(f"🚀 TRAINING: {dataset_name.upper()} + {prior_type.upper()} PRIOR")
    print("="*60)
    
    # Load data
    data_path = Path('data/processed') / f'{dataset_name}_processed.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train'][:, 0])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.FloatTensor(data['y_test'][:, 0])
    
    n_bands = X_train.shape[1]
    
    print(f"\n📂 Data: Train={X_train.shape}, Test={X_test.shape}")
    print(f"   Bands: {n_bands}")
    
    # Load prior
    target_name = 'fat' if dataset_name == 'tecator' else 'octane'
    prior_path = Path('data/priors') / f'{dataset_name}_{target_name}_{prior_type}.npy'
    
    # Create model
    import numpy as np
    prior_scores = np.load(prior_path)
    print(f"\n✅ Prior loaded: {prior_path}")
    
    model = BandSelectNetLight(
        n_bands=n_bands,
        n_outputs=1,
        prior_scores=prior_scores,
        hidden_dim=16,
        dropout=0.5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Parameters: {total_params:,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    print(f"\n🏋️ Hyperparameters:")
    print(f"   LR: {lr}, KL: {kl_weight}, Epochs: {n_epochs}")
    
    best_test_rmse = float('inf')
    patience = 30
    no_improve = 0
    
    for epoch in range(n_epochs):
        model.train()
        y_pred, _ = model(X_train)
        mse_loss = criterion(y_pred.squeeze(), y_train)
        kl_loss = model.get_kl_loss()
        total_loss = mse_loss + kl_weight * kl_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            y_pred_test, _ = model(X_test)
            test_rmse = torch.sqrt(criterion(y_pred_test.squeeze(), y_test))
            ss_res = ((y_test - y_pred_test.squeeze())**2).sum()
            ss_tot = ((y_test - y_test.mean())**2).sum()
            test_r2 = 1 - ss_res / ss_tot
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d} - Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
        
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS:")
    print("="*60)
    print(f"   Dataset: {dataset_name}")
    print(f"   Prior: {prior_type}")
    print(f"   Test RMSE: {test_rmse:.4f}")
    print(f"   Test R²: {test_r2:.4f}")
    print("="*60)
    
    return test_rmse.item(), test_r2.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, 
                       choices=['tecator', 'gasoline'])
    parser.add_argument('--prior', default='anova',
                       choices=['anova', 'vip', 'rf'])
    args = parser.parse_args()
    
    train_model(args.dataset, args.prior)

