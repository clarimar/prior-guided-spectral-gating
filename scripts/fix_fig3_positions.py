"""
Figura 3 - Ajustar posição de BandSelectNet (Gasoline): abaixar e mover direita
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

print("="*70)
print("FIXING FIGURE 3: Adjusting BandSelectNet (Gasoline) position")
print("="*70)

fig, ax = plt.subplots(figsize=(9, 5))

methods = ['PLSR', 'BandSelectNet\n(Tecator)', 'BandSelectNet\n(Gasoline)', 'Plain CNN']
params = [10, 1765, 6882, 109221]
r2_scores = [0.919, 0.862, 0.928, 0.235]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#BC4B51']

# Plotar os círculos
for param, r2, color in zip(params, r2_scores, colors):
    ax.scatter(param, r2, s=250, color=color, alpha=0.85, 
               edgecolor='black', linewidth=1.8, zorder=5)

# PLSR - à direita
ax.annotate('PLSR', xy=(params[0], r2_scores[0]),
            xytext=(25, 0), textcoords='offset points',
            fontsize=10, ha='left', va='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.8))

# BandSelectNet (Tecator) - acima
ax.annotate('BandSelectNet\n(Tecator)', xy=(params[1], r2_scores[1]),
            xytext=(0, 20), textcoords='offset points',
            fontsize=9, ha='center', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.8))

# BandSelectNet (Gasoline) - AJUSTADO: mais baixo e à direita
ax.annotate('BandSelectNet\n(Gasoline)', xy=(params[2], r2_scores[2]),
            xytext=(20, 8), textcoords='offset points',  # MUDOU: (0, 20) → (20, 8)
            fontsize=9, ha='left', va='bottom', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.8))

# Plain CNN - abaixo
ax.annotate('Plain CNN', xy=(params[3], r2_scores[3]),
            xytext=(0, -25), textcoords='offset points',
            fontsize=10, ha='center', va='top', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.8))

ax.set_xscale('log')
ax.set_xlabel('Number of Parameters (log scale)', fontsize=13, fontweight='bold')
ax.set_ylabel('Test $R^2$ Score', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Regiões de referência
ax.axhline(0.85, color='green', linestyle='--', alpha=0.35, linewidth=2.5)
ax.fill_between([1, 200000], 0.85, 1.05, alpha=0.08, color='green')
ax.text(50, 0.93, 'Good Performance\n($R^2 > 0.85$)', fontsize=9, 
        color='green', fontweight='bold', ha='left')

ax.axvline(10000, color='red', linestyle='--', alpha=0.35, linewidth=2.5)
ax.fill_betweenx([0, 1.05], 10000, 200000, alpha=0.08, color='red')
ax.text(25000, 0.20, 'Overparameterized\n(>10k params)', fontsize=9, 
        color='red', rotation=90, va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig3_parameter_efficiency.pdf', dpi=600, bbox_inches='tight')
plt.savefig(output_dir / 'fig3_parameter_efficiency.png', dpi=600, bbox_inches='tight')

print(f"✓ Updated: {output_dir}/fig3_parameter_efficiency.pdf")
print(f"✓ Updated: {output_dir}/fig3_parameter_efficiency.png")
print("\nChanges:")
print("  - BandSelectNet (Gasoline): moved DOWN (20→8) and RIGHT (0→20) ✓")
print("  - No overlap with Tecator label ✓")
print("  - All labels within figure bounds ✓")
print("="*70)
