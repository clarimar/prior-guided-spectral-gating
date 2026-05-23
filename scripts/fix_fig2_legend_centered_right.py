"""
Figura 2 - Legenda no centro à direita COM ESPAÇAMENTO (não sobrepõe)
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
print("FIXING FIGURE 2: Legend centered right WITH SPACING")
print("="*70)

fig, ax = plt.subplots(figsize=(8.5, 5))  # Ligeiramente mais larga

methods = ['PLS', 'BandSelectNet\n+ANOVA (CV)', 'Plain CNN']
tecator_r2 = [0.919, 0.862, 0.235]
gasoline_r2 = [0.941, 0.928, np.nan]
tecator_err = [0, 0.036, 0]

x = np.arange(len(methods))
width = 0.35

bars1 = ax.bar(x - width/2, tecator_r2, width, label='Tecator (food)', 
               color='#2E86AB', edgecolor='black', linewidth=1.2, alpha=0.85)
bars2 = ax.bar(x + width/2, gasoline_r2, width, label='Gasoline (petrochem.)',
               color='#F18F01', edgecolor='black', linewidth=1.2, alpha=0.85)

# Error bars apenas para BandSelectNet CV
ax.errorbar([1 - width/2], [tecator_r2[1]], yerr=[tecator_err[1]], 
            fmt='none', ecolor='black', capsize=5, capthick=1.8, zorder=10)

# Linha de referência
ax.axhline(0.85, color='gray', linestyle='--', alpha=0.4, linewidth=1.8)

# Texto "Good Performance" - POSIÇÃO ORIGINAL (NÃO MODIFICADA)
ax.text(2.35, 0.87, 'Good Performance\n($R^2 > 0.85$)', 
        fontsize=9, color='gray', ha='left')

ax.set_ylabel('Test $R^2$ Score', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)

# LEGENDA: Center right COM ESPAÇAMENTO
# bbox_to_anchor=(1.0, 0.5) = x=1.0 (borda direita), y=0.5 (centro vertical)
ax.legend(loc='center left', bbox_to_anchor=(0.85, 0.5), 
          frameon=False, fontsize=11)

ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Valores nos bars
for bar, val in zip(bars1, tecator_r2):
    if val > 0.3:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, val in zip(bars2, gasoline_r2):
    if not np.isnan(val) and val > 0.3:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig2_performance_comparison.pdf', dpi=600, bbox_inches='tight')
plt.savefig(output_dir / 'fig2_performance_comparison.png', dpi=600, bbox_inches='tight')

print(f"✓ Updated: {output_dir}/fig2_performance_comparison.pdf")
print(f"✓ Updated: {output_dir}/fig2_performance_comparison.png")
print("\nLegend: CENTER RIGHT with spacing (bbox_to_anchor=(0.85, 0.5))")
print("No overlap with bars!")
print("="*70)
