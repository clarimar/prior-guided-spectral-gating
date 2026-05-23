
"""
Analisar gates para qualquer dataset
Uso: python scripts/analyze_gates_multi.py --dataset gasoline
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.band_select_net_light import BandSelectNetLight


def analyze_gates(dataset_name, prior_type='anova'):
    print("\n" + "="*60)
    print(f"🔬 ANALYZING GATES: {dataset_name.upper()}")
    print("="*60)
    
    # Load data
    data_path = Path('data/processed') / f'{dataset_name}_processed.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_test = torch.FloatTensor(data['X_test'])
    n_bands = X_test.shape[1]
    
    # Load prior
    target_name = 'fat' if dataset_name == 'tecator' else 'octane'
    prior_path = Path('data/priors') / f'{dataset_name}_{target_name}_{prior_type}.npy'
    prior_scores = np.load(prior_path)
    
    # Create and eval model
    model = BandSelectNetLight(n_bands=n_bands, n_outputs=1, prior_scores=prior_scores)
    model.eval()
    
    with torch.no_grad():
        _, gates = model(X_test)
    
    gates_np = gates.detach().numpy()
    
    # Top 20 bands
    top_20_idx = np.argsort(gates_np)[-20:][::-1]
    
    print(f"\n🎯 TOP 20 SELECTED BANDS:")
    print(f"{'Rank':<5} {'Band':<5} {'Weight':<10} {'ANOVA':<10}")
    print("-" * 40)
    
    for i, idx in enumerate(top_20_idx, 1):
        print(f"{i:<5} {idx:<5} {gates_np[idx]:.6f}   {prior_scores[idx]:.6f}")
    
    # Correlation
    corr = np.corrcoef(gates_np, prior_scores)[0, 1]
    print(f"\n📊 Correlation with {prior_type.upper()} prior: {corr:.4f}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                       choices=['tecator', 'gasoline'])
    args = parser.parse_args()
    
    analyze_gates(args.dataset)

