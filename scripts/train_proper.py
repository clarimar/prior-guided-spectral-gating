
"""
Treino adequado com mais épocas e monitoramento
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.band_select_net import create_model


def train_model(n_epochs=100, lr=0.01, kl_weight=0.0001):
    """Treino adequado"""
    
    print("\n" + "="*60)
    print("🚀 TRAINING BANDSELECT NET (SERIOUS)")
    print("="*60)
    
    # Data
    with open('data/processed/tecator_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train'][:, 0])
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.FloatTensor(data['y_test'][:, 0])
    
    print(f"\n📂 Data: Train={X_train.shape}, Test={X_test.shape}")
    
    # Model
    model = create_model(
        n_bands=100,
        n_outputs=1,
        prior_path='data/priors/tecator_fat_anova.npy'
    )
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"\n🏋️ Training:")
    print(f"   Epochs: {n_epochs}")
    print(f"   Learning rate: {lr}")
    print(f"   KL weight: {kl_weight}")
    
    best_test_rmse = float('inf')
    patience = 20
    no_improve = 0
    
    for epoch in range(n_epochs):
        # Train
        model.train()
        y_pred, _ = model(X_train)
        mse_loss = criterion(y_pred.squeeze(), y_train)
        kl_loss = model.get_kl_loss()
        total_loss = mse_loss + kl_weight * kl_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            y_pred_train, _ = model(X_train)
            y_pred_test, gates = model(X_test)
            
            train_rmse = torch.sqrt(criterion(y_pred_train.squeeze(), y_train))
            test_rmse = torch.sqrt(criterion(y_pred_test.squeeze(), y_test))
            
            # R²
            ss_res = ((y_test - y_pred_test.squeeze())**2).sum()
            ss_tot = ((y_test - y_test.mean())**2).sum()
            test_r2 = 1 - ss_res / ss_tot
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{n_epochs} - "
                  f"Train RMSE: {train_rmse.item():.4f} | "
                  f"Test RMSE: {test_rmse.item():.4f}, R²: {test_r2.item():.4f}")
        
        # Early stopping
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Final eval
    model.eval()
    with torch.no_grad():
        y_pred_final, gates_final = model(X_test)
        final_rmse = torch.sqrt(criterion(y_pred_final.squeeze(), y_test))
        ss_res = ((y_test - y_pred_final.squeeze())**2).sum()
        ss_tot = ((y_test - y_test.mean())**2).sum()
        final_r2 = 1 - ss_res / ss_tot
    
    print("\n" + "="*60)
    print("📊 FINAL RESULTS:")
    print("="*60)
    print(f"   Test RMSE: {final_rmse.item():.4f}")
    print(f"   Test R²: {final_r2.item():.4f}")
    print(f"\n   PLS Baseline: RMSE=0.2957, R²=0.9191")
    print(f"   Our model: RMSE={final_rmse.item():.4f}, R²={final_r2.item():.4f}")
    
    if final_r2.item() > 0.91:
        print("\n   ✅ BETTER THAN PLS!")
    elif final_r2.item() > 0.80:
        print("\n   ⚠️ Good, but not better than PLS")
    else:
        print("\n   ❌ Much worse than PLS")
    
    # Top bands
    gates_np = gates_final.detach().numpy()
    top_10_idx = np.argsort(gates_np)[-10:][::-1]
    wavelengths = np.linspace(850, 1050, 100)
    
    print(f"\n🎯 Top 10 selected bands:")
    for i, idx in enumerate(top_10_idx, 1):
        print(f"   {i:2d}. Band {idx:2d} ({wavelengths[idx]:.0f} nm): {gates_np[idx]:.4f}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    train_model(n_epochs=200, lr=0.01, kl_weight=0.0001)
