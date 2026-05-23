"""
Testar o modelo BandSelectNet
Uso: python scripts/test_model.py
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.band_select_net import create_model


def test_basic():
    """Teste básico sem prior"""
    print("\n" + "="*60)
    print("🧪 TESTE 1: Modelo sem prior")
    print("="*60)
    
    model = create_model(n_bands=100, n_outputs=1, prior_path=None)
    
    x = torch.randn(32, 100)
    y_pred, gates = model(x)
    kl_loss = model.get_kl_loss()
    
    print(f"✅ Input: {x.shape}")
    print(f"✅ Output: {y_pred.shape}")
    print(f"✅ Gates: {gates.shape}")
    print(f"✅ KL divergence: {kl_loss.item():.4f}")


def test_with_prior():
    """Teste com prior real"""
    print("\n" + "="*60)
    print("🧪 TESTE 2: Modelo com prior ANOVA (Fat)")
    print("="*60)
    
    model = create_model(
        n_bands=100,
        n_outputs=1,
        prior_path='data/priors/tecator_fat_anova.npy'
    )
    
    print(f"\n📊 Arquitetura:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Forward pass
    batch_size = 16
    x = torch.randn(batch_size, 100)
    y_pred, gates = model(x)
    kl_loss = model.get_kl_loss()
    
    print(f"\n🚀 Forward pass:")
    print(f"   Input: {x.shape}")
    print(f"   Output: {y_pred.shape}")
    print(f"   KL loss: {kl_loss.item():.4f}")
    
    # Gate statistics
    gates_np = gates.detach().numpy()
    print(f"\n🎯 Gates statistics:")
    print(f"   Min: {gates_np.min():.4f}")
    print(f"   Max: {gates_np.max():.4f}")
    print(f"   Mean: {gates_np.mean():.4f}")
    print(f"   Sum: {gates_np.sum():.4f}")
    
    top_5_idx = np.argsort(gates_np)[-5:][::-1]
    wavelengths = np.linspace(850, 1050, 100)
    print(f"\n📍 Top 5 bands:")
    for i, idx in enumerate(top_5_idx, 1):
        print(f"   {i}. Band {idx} ({wavelengths[idx]:.0f} nm): {gates_np[idx]:.4f}")


def test_multi_output():
    """Teste com múltiplos outputs"""
    print("\n" + "="*60)
    print("🧪 TESTE 3: Multi-output (Fat, Water, Protein)")
    print("="*60)
    
    model = create_model(
        n_bands=100,
        n_outputs=3,  # Fat, Water, Protein
        prior_path='data/priors/tecator_fat_anova.npy'
    )
    
    x = torch.randn(16, 100)
    y_pred, gates = model(x)
    
    print(f"✅ Input: {x.shape}")
    print(f"✅ Output: {y_pred.shape} (batch_size, 3)")
    print(f"✅ Gates: {gates.shape}")


def main():
    print("\n" + "="*60)
    print("🚀 TESTANDO BANDSELECT NET")
    print("="*60)
    
    try:
        test_basic()
        test_with_prior()
        test_multi_output()
        
        print("\n" + "="*60)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
 
