#cat > scripts/quick_train.py << 'EOF'
"""
Quick training test - treinar por algumas épocas para validar pipeline
Uso: python scripts/quick_train.py
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


def load_data():
    """Carregar dados processados"""
    with open('data/processed/tecator_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def quick_train(n_epochs=5, kl_weight=0.001):
    """Treino rápido para testar pipeline"""
    
    print("\n" + "="*60)
    print("🚀 QUICK TRAINING TEST")
    print("="*60)
    
    # Carregar dados
    print("\n📂 Loading data...")
    data = load_data()
    
    X_train = torch.FloatTensor(data['X_train'])
    y_train = torch.FloatTensor(data['y_train'][:, 0])  # Apenas Fat
    X_test = torch.FloatTensor(data['X_test'])
    y_test = torch.FloatTensor(data['y_test'][:, 0])
    
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Criar modelo
    print("\n🔧 Creating model...")
    model = create_model(
        n_bands=100,
        n_outputs=1,
        prior_path='data/priors/tecator_fat_anova.npy'
    )
    
    # Loss e optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print(f"\n🏋️ Training for {n_epochs} epochs...")
    print(f"   KL weight: {kl_weight}")
    print()
    
    for epoch in range(n_epochs):
        model.train()
        
        # Forward
        y_pred, gates = model(X_train)
        
        # Loss
        mse_loss = criterion(y_pred.squeeze(), y_train)
        kl_loss = model.get_kl_loss()
        total_loss = mse_loss + kl_weight * kl_loss
        
        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            y_pred_test, _ = model(X_test)
            test_mse = criterion(y_pred_test.squeeze(), y_test)
            test_rmse = torch.sqrt(test_mse)
        
        print(f"Epoch {epoch+1}/{n_epochs} - "
              f"Train Loss: {total_loss.item():.4f} "
              f"(MSE: {mse_loss.item():.4f}, KL: {kl_loss.item():.4f}) | "
              f"Test RMSE: {test_rmse.item():.4f}")
    
    print("\n✅ Training completed!")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_final, gates_final = model(X_test)
        final_rmse = torch.sqrt(criterion(y_pred_final.squeeze(), y_test))
        
        # R²
        ss_res = ((y_test - y_pred_final.squeeze())**2).sum()
        ss_tot = ((y_test - y_test.mean())**2).sum()
        r2 = 1 - ss_res / ss_tot
    
    print("\n📊 Final Test Performance:")
    print(f"   RMSE: {final_rmse.item():.4f}")
    print(f"   R²: {r2.item():.4f}")
    
    # Top bands
    gates_np = gates_final.detach().numpy()
    top_5_idx = np.argsort(gates_np)[-5:][::-1]
    wavelengths = np.linspace(850, 1050, 100)
    
    print("\n🎯 Top 5 selected bands:")
    for i, idx in enumerate(top_5_idx, 1):
        print(f"   {i}. Band {idx} ({wavelengths[idx]:.0f} nm): weight={gates_np[idx]:.4f}")
    
    print("\n" + "="*60)
    print("✅ PIPELINE VALIDATION SUCCESSFUL!")
    print("="*60)


if __name__ == '__main__':
    quick_train(n_epochs=10, kl_weight=0.001)
