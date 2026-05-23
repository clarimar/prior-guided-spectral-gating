"""
Criar figuras principais para o paper (Performance e Parameter Efficiency)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'legend.fontsize': 10,
})

output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

# ============================================================
# FIGURE 2: Multi-Dataset Performance Comparison
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

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
ax.text(2.35, 0.87, 'Good Performance\n($R^2 > 0.85$)', fontsize=9, color='gray', ha='left')

ax.set_ylabel('Test $R^2$ Score', fontsize=13, fontweight='bold')
ax.set_ylim([0, 1.05])
ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=11)
ax.legend(loc='upper left', frameon=False, fontsize=11)
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
plt.savefig(output_dir / 'fig2_performance_comparison.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig2_performance_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir}/fig2_performance_comparison.pdf")
print(f"✅ Saved: {output_dir}/fig2_performance_comparison.png")
plt.close()

# ============================================================
# FIGURE 3: Parameter Efficiency
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

methods = ['PLS', 'Ours\n(Tecator)', 'Ours\n(Gasoline)', 'Plain\nCNN']
params = [10, 1765, 6882, 109221]
r2_scores = [0.919, 0.862, 0.928, 0.235]
colors = ['#2E86AB', '#A23B72', '#F18F01', '#BC4B51']

for i, (method, param, r2, color) in enumerate(zip(methods, params, r2_scores, colors)):
    ax.scatter(param, r2, s=250, color=color, alpha=0.85, 
               edgecolor='black', linewidth=1.8, zorder=5)
    
    offset_y = 0.05 if i % 2 == 0 else -0.07
    ax.annotate(method, xy=(param, r2),
                xytext=(0, offset_y), textcoords='offset points',
                fontsize=10, ha='center', va='center', fontweight='bold')

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
ax.text(20, 0.93, 'Good Performance\n($R^2 > 0.85$)', fontsize=9, color='green', fontweight='bold')

ax.axvline(10000, color='red', linestyle='--', alpha=0.35, linewidth=2.5)
ax.fill_betweenx([0, 1.05], 10000, 200000, alpha=0.08, color='red')
ax.text(25000, 0.20, 'Overparameterized\n(>10k params)', fontsize=9, color='red', 
        rotation=90, va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'fig3_parameter_efficiency.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'fig3_parameter_efficiency.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir}/fig3_parameter_efficiency.pdf")
print(f"✅ Saved: {output_dir}/fig3_parameter_efficiency.png")
plt.close()

print("\n" + "="*60)
print("✅ MAIN FIGURES CREATED SUCCESSFULLY")
print("="*60)
