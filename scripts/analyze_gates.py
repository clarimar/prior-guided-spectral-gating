
"""
Analisar gates aprendidas
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from models.band_select_net_light import create_model_light


def analyze_gates():
    print("\n" + "="*60)
    print("🔬 ANALYZING LEARNED GATES")
    print("="*60)
    
    # Load model
    model = create_model_light(100, 1, 'data/priors/tecator_fat_anova.npy')
    
    # Load data to get sample
    with open('data/processed/tecator_processed.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_test = torch.FloatTensor(data['X_test'])
    
    # Get gates
    model.eval()
    with torch.no_grad():
        _, gates = model(X_test)
    
    gates_np = gates.detach().numpy()
    wavelengths = np.linspace(850, 1050, 100)
    
    # Load ANOVA prior
    anova_prior = np.load('data/priors/tecator_fat_anova.npy')
    
    # Top 20 bands
    top_20_idx = np.argsort(gates_np)[-20:][::-1]
    
    print(f"\n🎯 TOP 20 SELECTED BANDS:")
    print(f"{'Rank':<5} {'Band':<5} {'Wavelength':<12} {'Weight':<10} {'ANOVA':<10}")
    print("-" * 50)
    
    for i, idx in enumerate(top_20_idx, 1):
        print(f"{i:<5} {idx:<5} {wavelengths[idx]:.0f} nm{'':<6} "
              f"{gates_np[idx]:.6f}   {anova_prior[idx]:.6f}")
    
    # Correlation with prior
    corr = np.corrcoef(gates_np, anova_prior)[0, 1]
    print(f"\n📊 Correlation with ANOVA prior: {corr:.4f}")
    
    # Known fat peaks
    print(f"\n🧪 Known fat absorption peaks:")
    print(f"   C-H bonds: ~920-930 nm")
    print(f"   O-H bonds: ~970 nm")
    
    # Check if top bands are near these peaks
    ch_region = (wavelengths >= 920) & (wavelengths <= 930)
    oh_region = (wavelengths >= 965) & (wavelengths <= 975)
    
    ch_weight = gates_np[ch_region].sum()
    oh_weight = gates_np[oh_region].sum()
    
    print(f"\n   Total weight in C-H region (920-930nm): {ch_weight:.4f}")
    print(f"   Total weight in O-H region (965-975nm): {oh_weight:.4f}")
    
    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax[0].plot(wavelengths, anova_prior, 'b-', label='ANOVA Prior', alpha=0.7)
    ax[0].set_ylabel('ANOVA Score')
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)
    
    ax[1].plot(wavelengths, gates_np, 'r-', label='Learned Gates', linewidth=2)
    ax[1].axvspan(920, 930, alpha=0.2, color='green', label='C-H region')
    ax[1].axvspan(965, 975, alpha=0.2, color='orange', label='O-H region')
    ax[1].set_ylabel('Gate Weight')
    ax[1].set_xlabel('Wavelength (nm)')
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path('results/tecator/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'learned_gates_vs_prior.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved: {output_path}")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    analyze_gates()

