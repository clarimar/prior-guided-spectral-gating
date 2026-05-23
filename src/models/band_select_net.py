#cat > src/models/band_select_net.py << 'EOF'
"""
Spectral Band Selection Network com Chemometric Priors
- Learnable spectral gating layer
- Prior initialization (ANOVA, VIP, RF)
- KL divergence regularization for interpretability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralGatingLayer(nn.Module):
    """
    Learnable spectral gating layer com prior initialization
    """
    def __init__(self, n_bands, prior_scores=None, temperature=5.0):
        """
        Args:
            n_bands: número de bandas espectrais
            prior_scores: scores de prior (ANOVA/VIP/RF) - numpy array [0, 1]
            temperature: temperatura para softmax (default=1.0)
        """
        super().__init__()
        
        self.n_bands = n_bands
        self.temperature = temperature
        
        # Inicializar gates
        if prior_scores is not None:
            # Inicializar com prior (logit transform para não saturar)
            # prior [0, 1] -> logit -> learnable parameter
            prior_scores = np.clip(prior_scores, 1e-7, 1 - 1e-7)
            logits = np.log(prior_scores / (1 - prior_scores))
            init_gates = torch.tensor(logits, dtype=torch.float32)
        else:
            # Random initialization
            init_gates = torch.randn(n_bands) * 0.01
        
        # Learnable gates
        self.gates = nn.Parameter(init_gates)
        
        # Salvar prior para regularização
        if prior_scores is not None:
            self.register_buffer('prior_probs', torch.tensor(prior_scores, dtype=torch.float32))
        else:
            self.register_buffer('prior_probs', torch.ones(n_bands) / n_bands)
    
    def forward(self, x):
        """
        Args:
            x: input spectra (batch_size, n_bands)
        Returns:
            weighted spectra (batch_size, n_bands)
            gate probabilities (n_bands,)
        """
        # Softmax para obter probabilidades
        gate_probs = F.softmax(self.gates / self.temperature, dim=0)
        
        # Aplicar gates
        weighted = x * gate_probs.unsqueeze(0)
        
        return weighted, gate_probs
    
    def get_kl_divergence(self):
        """
        KL divergence entre learned gates e prior
        KL(learned || prior)
        """
        gate_probs = F.softmax(self.gates / self.temperature, dim=0)
        
        # KL divergence
        kl = (gate_probs * (torch.log(gate_probs + 1e-10) - torch.log(self.prior_probs + 1e-10))).sum()
        
        return kl


class BandSelectNet(nn.Module):
    """
    CNN com spectral gating layer para band selection
    """
    def __init__(self, n_bands, n_outputs=1, prior_scores=None, 
                 hidden_dim=64, dropout=0.3):
        """
        Args:
            n_bands: número de bandas espectrais
            n_outputs: número de outputs (1 para regressão simples, 3 para multi-output)
            prior_scores: chemometric prior scores
            hidden_dim: dimensão das camadas ocultas
            dropout: dropout rate
        """
        super().__init__()
        
        self.n_bands = n_bands
        self.n_outputs = n_outputs
        
        # Spectral gating layer
        self.gating = SpectralGatingLayer(n_bands, prior_scores, temperature=5.0)
        
        # CNN layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calcular dimensão após pooling
        pooled_dim = n_bands // 4
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * pooled_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, n_outputs)
    
    def forward(self, x):
        """
        Args:
            x: input spectra (batch_size, n_bands)
        Returns:
            predictions (batch_size, n_outputs)
            gate_probs (n_bands,)
        """
        # Spectral gating
        x_gated, gate_probs = self.gating(x)
        
        # Reshape para Conv1d: (batch, channels, length)
        x_gated = x_gated.unsqueeze(1)
        
        # CNN forward
        x = F.relu(self.bn1(self.conv1(x_gated)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out, gate_probs
    
    def get_kl_loss(self):
        """Retornar KL divergence loss"""
        return self.gating.get_kl_divergence()


def create_model(n_bands, n_outputs=1, prior_type='anova', prior_path=None):
    """
    Factory function para criar modelo
    
    Args:
        n_bands: número de bandas
        n_outputs: número de outputs
        prior_type: 'anova', 'vip', 'rf', ou None (random)
        prior_path: caminho para arquivo .npy de prior
    
    Returns:
        BandSelectNet model
    """
    prior_scores = None
    
    if prior_path is not None:
        prior_scores = np.load(prior_path)
        print(f"✅ Prior carregado: {prior_path}")
        print(f"   Shape: {prior_scores.shape}, Range: [{prior_scores.min():.3f}, {prior_scores.max():.3f}]")
    
    model = BandSelectNet(
        n_bands=n_bands,
        n_outputs=n_outputs,
        prior_scores=prior_scores
    )
    
    return model


if __name__ == '__main__':
    # Teste
    print("Testing BandSelectNet...")
    
    # Criar modelo com prior dummy
    prior = np.random.rand(100)
    model = create_model(n_bands=100, n_outputs=1, prior_path=None)
    
    # Input dummy
    x = torch.randn(32, 100)  # batch=32, bands=100
    
    # Forward pass
    out, gates = model(x)
    kl = model.get_kl_loss()
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Gates shape: {gates.shape}")
    print(f"KL divergence: {kl.item():.4f}")
    print("\n✅ Model test passed!")
#EOF
