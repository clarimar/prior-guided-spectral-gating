import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.band_select_net_light import create_model_light

print("\n" + "="*60)
print("🧬 TRAINING WITH VIP PRIOR")
print("="*60)

with open('data/processed/tecator_processed.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = torch.FloatTensor(data['X_train'])
y_train = torch.FloatTensor(data['y_train'][:, 0])
X_test = torch.FloatTensor(data['X_test'])
y_test = torch.FloatTensor(data['y_test'][:, 0])

model = create_model_light(100, 1, 'data/priors/tecator_fat_vip.npy')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

best_test_rmse = float('inf')
patience = 30
no_improve = 0

for epoch in range(200):
    model.train()
    y_pred, _ = model(X_train)
    mse_loss = criterion(y_pred.squeeze(), y_train)
    kl_loss = model.get_kl_loss()
    total_loss = mse_loss + 0.001 * kl_loss
    
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
        print(f"Epoch {epoch+1:3d} - RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")
    
    if test_rmse < best_test_rmse:
        best_test_rmse = test_rmse
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

print("\n" + "="*60)
print("FINAL (VIP):")
print(f"   Test RMSE: {test_rmse:.4f}")
print(f"   Test R²: {test_r2:.4f}")
print("="*60)
