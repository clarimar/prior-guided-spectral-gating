#!/bin/bash

echo "======================================================================"
echo "GENERATING ALL PAPER FIGURES IN 600 DPI (PUBLICATION QUALITY)"
echo "======================================================================"
echo ""

# Criar diretório de saída
mkdir -p figures

# 1. Converter figuras existentes do Tecator
echo "[Step 1/2] Upscaling existing Tecator figures to 600 DPI..."
python scripts/upscale_tecator_figures_600dpi.py

echo ""

# 2. Gerar figuras principais (Performance e Parameter Efficiency)
echo "[Step 2/2] Creating main figures at 600 DPI..."
python scripts/create_main_figures_600dpi.py

echo ""
echo "======================================================================"
echo "✅ ALL FIGURES GENERATED AT 600 DPI"
echo "======================================================================"
echo ""
echo "Summary:"
ls -lh figures/ | grep -E "\.(pdf|png)$"

echo ""
echo "Total size:"
du -sh figures/

echo ""
echo "======================================================================"
echo "Ready for publication!"
echo "======================================================================"
