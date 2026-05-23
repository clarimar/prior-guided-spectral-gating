"""
Lightweight BandSelectNet - adequado para datasets pequenos
~2,000 parâmetros (vs 109,000 da versão original)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralGatingLayer(nn.Module):
    """Learnable spectral gating layer"""
    
    def __init__(self, n_bands, prior_scores=None, temperature=5.0):
        super().__init__()
        self.n_bands = n_bands
        self.temperature = temperature
        
        if prior_scores is not None:
            prior_scores = np.clip(prior_scores, 1e-7, 1 - 1e-7)
            logits = np.log(prior_scores / (1 - prior_scores))
            init_gates = torch.tensor(logits, dtype=torch.float32)
        else:
            init_gates = torch.randn(n_bands) * 0.01
        
        self.gates = nn.Parameter(init_gates)
        
        if prior_scores is not None:
            self.register_buffer('prior_probs', torch.tensor(prior_scores, dtype=torch.float32))
        else:
            self.register_buffer('prior_probs', torch.ones(n_bands) / n_bands)
    
    def forward(self, x):
        gate_probs = F.softmax(self.gates / self.temperature, dim=0)
        weighted = x * gate_probs.unsqueeze(0)
        return weighted, gate_probs
    
    def get_kl_divergence(self):
        gate_probs = F.softmax(self.gates / self.temperature, dim=0)
        kl = (gate_probs * (torch.log(gate_probs + 1e-10) - torch.log(self.prior_probs + 1e-10))).sum()
        return kl


class BandSelectNetLight(nn.Module):
    """
    Lightweight version - apenas FC layers após gating
    ~2,000 parâmetros total
    """
    
    def __init__(self, n_bands, n_outputs=1, prior_scores=None, 
                 hidden_dim=16, dropout=0.5):
        super().__init__()
        
        self.n_bands = n_bands
        self.n_outputs = n_outputs
        
        self.gating = SpectralGatingLayer(n_bands, prior_scores, temperature=5.0)
        
        self.fc1 = nn.Linear(n_bands, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_outputs)
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x):
        x_gated, gate_probs = self.gating(x)
        x = F.relu(self.bn(self.fc1(x_gated)))
        x = self.dropout(x)
        out = self.fc2(x)
        return out, gate_probs
    
    def get_kl_loss(self):
        return self.gating.get_kl_divergence()


def create_model_light(n_bands, n_outputs=1, prior_path=None):
    """Create lightweight model"""
    
    prior_scores = None
    if prior_path is not None:
        prior_scores = np.load(prior_path)
        print(f"✅ Prior carregado: {prior_path}")
    
    model = BandSelectNetLight(
        n_bands=n_bands,
        n_outputs=n_outputs,
        prior_scores=prior_scores,
        hidden_dim=16,
        dropout=0.5
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Total parameters: {total_params:,}")
    
    return model


if __name__ == '__main__':
    model = create_model_light(100, 1, 'data/priors/tecator_fat_anova.npy')
    x = torch.randn(32, 100)
    out, gates = model(x)
    print(f"✅ Input: {x.shape}, Output: {out.shape}, Gates: {gates.shape}")
