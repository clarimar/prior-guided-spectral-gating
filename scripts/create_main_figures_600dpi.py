"""
Criar figuras principais para o paper em 600 DPI (publication quality)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuração para publicação em alta resolução
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 600,  # Alta resolução
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

print("="*70)
print("GENERATING HIGH-RESOLUTION FIGURES (600 DPI)")
print("="*70)

# ============================================================
# FIGURE 2: Multi-Dataset Performance Comparison
# ============================================================
print("\n[1/2] Creating Figure 2: Performance Comparison...")

fig, ax = plt.subplots(figsize=(8, 5))

methods = ['PLS', 'BandSelectNet\n+ANOVA (CV)', 'Plain CNN']
tecator_r2 = [0.919, 0.862, 0.235]
gasoline_r2 = [0.975, 0.928, np.nan]  # R1: PLS was 0.941 (H=10); R2: PLS is 0.975 (H*=5 via one-SE rule)
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
ax.text(2.35, 0.87, 'Good Performance\n($R^2 > 0.85$)', fontsize=9, color='gray', ha='left')

ax.set_ylabel('Test $R^2$ Score', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Valores nos bars
for i, (bar, val) in enumerate(zip(bars1, tecator_r2)):
    if val > 0.3:
        offset = 0.06 if i == 1 else 0.03   # barra 1 (CV Tecator) tem error bar, precisa mais espaço
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + offset,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

for bar, val in zip(bars2, gasoline_r2):
    if not np.isnan(val) and val > 0.3:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.03,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig2_performance_comparison.pdf', dpi=600, bbox_inches='tight')
plt.savefig(output_dir / 'fig2_performance_comparison.png', dpi=600, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/fig2_performance_comparison.pdf (600 DPI)")
print(f"  ✓ Saved: {output_dir}/fig2_performance_comparison.png (600 DPI)")
plt.close()

# ============================================================
# FIGURE 3: Parameter Efficiency
# ============================================================
print("\n[2/2] Creating Figure 3: Parameter Efficiency...")

fig, ax = plt.subplots(figsize=(8, 5))

methods = ['PLS', 'Ours\n(Tecator)', 'Ours\n(Gasoline)', 'Plain\nCNN']
params = [10, 1765, 6882, 109221]
r2_scores = [0.919, 0.862, 0.928, 0.235]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#BC4B51']

# Posições dos labels em relação a cada bolinha:
# 0 PLS -> esquerda; 1 Ours(Tecator) -> esquerda;
# 2 Ours(Gasoline) -> direita; 3 Plain CNN -> esquerda.
# label_side: 'left' desloca para -18 px com ha='right';
# 'right' desloca para +18 px com ha='left'.
label_sides = ['left', 'left', 'right', 'left']
for i, (method, param, r2, color) in enumerate(zip(methods, params, r2_scores, colors)):
    ax.scatter(param, r2, s=250, color=color, alpha=0.85,
               edgecolor='black', linewidth=1.8, zorder=5)
    side = label_sides[i]
    dx = -18 if side == 'left' else 18
    ha = 'right' if side == 'left' else 'left'
    # troca as quebras de linha do label por espaco (single-line, mais discreto)
    label_txt = method.replace('\n', ' ')
    ax.annotate(label_txt, xy=(param, r2),
                xytext=(dx, 0), textcoords='offset points',
                fontsize=10, ha=ha, va='center', fontweight='bold')

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
ax.text(2000, 1.02, 'Good Performance ($R^2 > 0.85$)', fontsize=9, color='green', fontweight='bold', ha='center')

ax.axvline(10000, color='red', linestyle='--', alpha=0.35, linewidth=2.5)
ax.fill_betweenx([0, 1.05], 10000, 200000, alpha=0.08, color='red')
ax.text(45000, 0.05, 'Overparameterized (>10k params)', fontsize=9, color='red', fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig(output_dir / 'fig3_parameter_efficiency.pdf', dpi=600, bbox_inches='tight')
plt.savefig(output_dir / 'fig3_parameter_efficiency.png', dpi=600, bbox_inches='tight')
print(f"  ✓ Saved: {output_dir}/fig3_parameter_efficiency.pdf (600 DPI)")
print(f"  ✓ Saved: {output_dir}/fig3_parameter_efficiency.png (600 DPI)")
plt.close()

print("\n" + "="*70)
print("✅ ALL MAIN FIGURES CREATED AT 600 DPI")
print("="*70)
print(f"\nFiles created in: {output_dir}/")
print("  - fig2_performance_comparison.pdf (600 DPI)")
print("  - fig2_performance_comparison.png (600 DPI)")
print("  - fig3_parameter_efficiency.pdf (600 DPI)")
print("  - fig3_parameter_efficiency.png (600 DPI)")
