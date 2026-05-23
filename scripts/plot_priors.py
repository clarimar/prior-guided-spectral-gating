#cat > scripts/plot_priors.py << 'EOF'
"""
Plot chemometric priors para qualquer target
Uso: python scripts/plot_priors.py --target fat
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def plot_priors(target='fat', dataset='tecator', output_dir='results/tecator/figures'):
    """Plot priors para um target específico"""
    
    # Criar diretório
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carregar priors
    priors_dir = Path('data/priors')
    anova = np.load(priors_dir / f'{dataset}_{target}_anova.npy')
    vip = np.load(priors_dir / f'{dataset}_{target}_vip.npy')
    rf = np.load(priors_dir / f'{dataset}_{target}_rf.npy')
    
    # Wavelengths (Tecator: 850-1050 nm)
    wavelengths = np.linspace(850, 1050, len(anova))
    
    # Plot stacked
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    
    axes[0].plot(wavelengths, anova, 'b-', linewidth=1.5)
    axes[0].set_ylabel('ANOVA', fontsize=11, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_ylim([-0.05, 1.05])
    
    axes[1].plot(wavelengths, vip, 'g-', linewidth=1.5)
    axes[1].set_ylabel('VIP', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_ylim([-0.05, 1.05])
    
    axes[2].plot(wavelengths, rf, 'r-', linewidth=1.5)
    axes[2].set_ylabel('RF Importance', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Wavelength (nm)', fontsize=11)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    output1 = output_dir / f'priors_{target}.png'
    plt.savefig(output1, dpi=300, bbox_inches='tight')
    print(f"✅ {output1}")
    plt.close()
    
    # Plot overlay
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(wavelengths, anova, 'b-', linewidth=2, label='ANOVA', alpha=0.8)
    ax.plot(wavelengths, vip, 'g-', linewidth=2, label='VIP', alpha=0.8)
    ax.plot(wavelengths, rf, 'r-', linewidth=2, label='RF', alpha=0.8)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Normalized Importance', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-0.05, 1.05])
    plt.tight_layout()
    
    output2 = output_dir / f'priors_{target}_overlay.png'
    plt.savefig(output2, dpi=300, bbox_inches='tight')
    print(f"✅ {output2}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot chemometric priors')
    parser.add_argument('--target', default='fat', choices=['fat', 'water', 'protein'])
    parser.add_argument('--all', action='store_true', help='Plot all targets')
    
    args = parser.parse_args()
    
    if args.all:
        for target in ['fat', 'water', 'protein']:
            plot_priors(target)
    else:
        plot_priors(args.target)
    
    print("\n✨ Done!")


if __name__ == '__main__':
    main()
 

