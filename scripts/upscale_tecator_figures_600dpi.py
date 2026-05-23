"""
Converter figuras do Tecator existentes para 600 DPI
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

source_dir = Path('results/tecator/figures')
output_dir = Path('figures')
output_dir.mkdir(exist_ok=True)

figures_to_convert = {
    'learned_gates_vs_prior.png': 'fig1_tecator_gates.png',
    'priors_fat.png': 'fig_s1_priors_comparison.png',
    'priors_fat_overlay.png': 'fig_s2_priors_overlay.png'
}

print("="*70)
print("UPSCALING TECATOR FIGURES TO 600 DPI")
print("="*70)

for source_name, target_name in figures_to_convert.items():
    source_path = source_dir / source_name
    target_path = output_dir / target_name
    
    if not source_path.exists():
        print(f"⚠️  Skipping {source_name} - file not found")
        continue
    
    print(f"\nProcessing: {source_name}")
    
    # Carregar imagem
    img = mpimg.imread(source_path)
    
    # Criar figura em alta resolução
    dpi = 600
    height_inches = img.shape[0] / dpi
    width_inches = img.shape[1] / dpi
    
    fig, ax = plt.subplots(figsize=(width_inches, height_inches), dpi=dpi)
    ax.imshow(img)
    ax.axis('off')
    
    # Salvar em 600 DPI
    plt.savefig(target_path, dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"  ✓ Saved: {target_path} (600 DPI)")

print("\n" + "="*70)
print("✅ TECATOR FIGURES UPSCALED TO 600 DPI")
print("="*70)
