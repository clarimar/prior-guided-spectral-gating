import torch, torch.nn as nn, torch.optim as optim, pickle, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.band_select_net_light import create_model_light

with open('data/processed/tecator_processed.pkl', 'rb') as f:
    data = pickle.load(f)

X_train = torch.FloatTensor(data['X_train'])
y_train = torch.FloatTensor(data['y_train'][:, 0])
X_test = torch.FloatTensor(data['X_test'])
y_test = torch.FloatTensor(data['y_test'][:, 0])

model = create_model_light(100, 1, None)  # SEM prior!
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

print("\n" + "="*60)
print("🎲 TRAINING WITHOUT PRIOR (Random Init)")
print("="*60)

for epoch in range(200):
    model.train()
    y_pred, _ = model(X_train)
    mse_loss = criterion(y_pred.squeeze(), y_train)
    kl_loss = model.get_kl_loss()
    total_loss = mse_loss + 0.001 * kl_loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 50 == 0:
        model.eval()
        with torch.no_grad():
            y_pred_test, _ = model(X_test)
            test_rmse = torch.sqrt(criterion(y_pred_test.squeeze(), y_test))
            ss_res = ((y_test - y_pred_test.squeeze())**2).sum()
            ss_tot = ((y_test - y_test.mean())**2).sum()
            test_r2 = 1 - ss_res / ss_tot
        print(f"Epoch {epoch+1} - Test RMSE: {test_rmse:.4f}, R²: {test_r2:.4f}")

model.eval()
with torch.no_grad():
    y_pred_final, _ = model(X_test)
    final_rmse = torch.sqrt(criterion(y_pred_final.squeeze(), y_test))
    ss_res = ((y_test - y_pred_final.squeeze())**2).sum()
    ss_tot = ((y_test - y_test.mean())**2).sum()
    final_r2 = 1 - ss_res / ss_tot

print("\n" + "="*60)
print("📊 FINAL (RANDOM):")
print(f"   Test RMSE: {final_rmse:.4f}, R²: {final_r2:.4f}")
print("="*60)
