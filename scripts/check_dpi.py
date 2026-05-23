from PIL import Image
from pathlib import Path

print("="*80)
print("VERIFICAÇÃO DE DPI DAS FIGURAS")
print("="*80)
print(f"{'Arquivo':<45} {'Dimensões':<15} {'DPI':>10}")
print("-"*80)

for img_path in sorted(Path('figures').glob('*.png')):
    img = Image.open(img_path)
    dpi = img.info.get('dpi', (0, 0))
    width, height = img.size
    print(f"{img_path.name:<45} {width:5}x{height:<5} @ {dpi[0]:>6.0f} DPI")

print("="*80)
