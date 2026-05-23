"""
Download automático dos 3 datasets
Rodar de: ~/Dropbox/chemometric-band-selection/
Uso: python src/data/download.py --all
"""

import os
import argparse
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Barra de progresso para downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download com barra de progresso"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_tecator():
    """
    Download Tecator Meat dataset
    Fonte: R package 'pls'
    """
    print("\n📦 TECATOR MEAT DATASET")
    print("=" * 50)
    
    output_dir = Path('data/raw/tecator')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Tentar via rpy2 (se instalado)
        print("Tentando download via R...")
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        pandas2ri.activate()
        
        # Instalar e carregar pacote pls
        ro.r('if (!require("pls")) install.packages("pls", repos="http://cran.us.r-project.org")')
        ro.r('library(pls)')
        ro.r('data(meatspec)')
        
        # Extrair dados
        X = np.array(ro.r('meatspec$X'))
        Y = np.array(ro.r('meatspec$Y'))
        
        # Salvar
        pd.DataFrame(X).to_csv(output_dir / 'spectra.csv', index=False)
        pd.DataFrame(Y, columns=['Fat', 'Water', 'Protein']).to_csv(
            output_dir / 'targets.csv', index=False
        )
        
        print(f"✅ Tecator baixado via R!")
        print(f"   Spectra: {X.shape}")
        print(f"   Targets: {Y.shape}")
        
    except ImportError:
        print("⚠️  rpy2 não instalado. Usando método alternativo...")
        
        # Alternativa: baixar de repositório público (se disponível)
        # Ou instruções para download manual
        print("\n📋 DOWNLOAD MANUAL:")
        print("1. Abrir R ou RStudio")
        print("2. Executar:")
        print("   install.packages('pls')")
        print("   library(pls)")
        print("   data(meatspec)")
        print("   write.csv(meatspec$X, 'tecator_spectra.csv', row.names=FALSE)")
        print("   write.csv(meatspec$Y, 'tecator_targets.csv', row.names=FALSE)")
        print("3. Mover arquivos para: data/raw/tecator/")
        
        # Tentar download direto de URL alternativa (se existir)
        try_alternative_tecator()


def try_alternative_tecator():
    """Tentar baixar Tecator de fonte alternativa"""
    # URLs públicas conhecidas para Tecator
    urls = [
        "http://lib.stat.cmu.edu/datasets/tecator",
        # Adicionar outras URLs públicas se conhecidas
    ]
    
    print("\n🔍 Procurando fontes alternativas...")
    print("(Se não funcionar, use download manual via R)")


def download_shootout():
    """
    Download NIR Shootout 2002
    Fonte: IDRC / Eigenvector Research
    """
    print("\n📦 NIR SHOOTOUT 2002 DATASET")
    print("=" * 50)
    
    output_dir = Path('data/raw/shootout')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # URLs conhecidas
    base_url = "http://www.eigenvector.com/data/"
    
    print("📋 INSTRUÇÕES DE DOWNLOAD:")
    print("\n1. Acessar: https://eigenvector.com/resources/data-sets/")
    print("   OU: http://www.idrc-chambersburg.org/shootout_2002.htm")
    print("\n2. Baixar: 'nir_shootout_2002.mat' ou arquivos CSV")
    print("\n3. Salvar em: data/raw/shootout/")
    print("\nFormato esperado:")
    print("  - calibration.csv (ou .txt)")
    print("  - validation.csv")
    print("  - test.csv")
    
    print("\n⏳ Aguardando download manual...")
    print("(Arquivo .mat pode ser convertido depois)")


def download_corn():
    """
    Download Corn dataset
    Fonte: Eigenvector Research ou JchemoData
    """
    print("\n📦 CORN DATASET")
    print("=" * 50)
    
    output_dir = Path('data/raw/corn')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📋 INSTRUÇÕES DE DOWNLOAD:")
    print("\n1. Acessar: https://eigenvector.com/resources/data-sets/")
    print("   Procurar por: 'Corn' ou 'Multi-spectrometer'")
    print("\n2. OU clonar: https://github.com/mlesnoff/JchemoData.jl")
    print("   Dataset: data/corn.jld2")
    print("\n3. Salvar em: data/raw/corn/")
    print("\nFormato esperado:")
    print("  - m5_spectra.csv")
    print("  - mp5_spectra.csv")
    print("  - mp6_spectra.csv")
    print("  - targets.csv (moisture, oil, protein, starch)")
    
    print("\n⏳ Aguardando download manual...")


def verify_downloads():
    """Verificar quais datasets foram baixados"""
    print("\n" + "=" * 50)
    print("📊 VERIFICAÇÃO DE DATASETS")
    print("=" * 50)
    
    datasets = {
        'Tecator': Path('data/raw/tecator'),
        'Shootout': Path('data/raw/shootout'),
        'Corn': Path('data/raw/corn'),
    }
    
    for name, path in datasets.items():
        if path.exists() and any(path.iterdir()):
            files = list(path.glob('*'))
            print(f"✅ {name}: {len(files)} arquivo(s)")
            for f in files:
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"   - {f.name} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {name}: Não encontrado")
    
    print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--all', action='store_true', help='Baixar todos os datasets')
    parser.add_argument('--tecator', action='store_true', help='Baixar apenas Tecator')
    parser.add_argument('--shootout', action='store_true', help='Baixar apenas Shootout')
    parser.add_argument('--corn', action='store_true', help='Baixar apenas Corn')
    parser.add_argument('--verify', action='store_true', help='Verificar downloads')
    
    args = parser.parse_args()
    
    print("🚀 DOWNLOAD DE DATASETS - CHEMOMETRIC BAND SELECTION")
    print("=" * 60)
    
    if args.verify:
        verify_downloads()
        return
    
    if args.all or args.tecator:
        download_tecator()
    
    if args.all or args.shootout:
        download_shootout()
    
    if args.all or args.corn:
        download_corn()
    
    if not any([args.all, args.tecator, args.shootout, args.corn]):
        print("\n⚠️  Nenhum dataset especificado!")
        print("\nUso:")
        print("  python src/data/download.py --all")
        print("  python src/data/download.py --tecator")
        print("  python src/data/download.py --verify")
    
    verify_downloads()


if __name__ == '__main__':
    main()
def download_tecator():
    """
    Download Tecator Meat dataset
    Fonte: R package 'pls'
    """
    print("\n📦 TECATOR MEAT DATASET")
    print("=" * 50)
    
    output_dir = Path('data/raw/tecator')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Método 1: Via rpy2 (API moderna)
    try:
        print("Tentando download via R (rpy2)...")
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        
        # Ativar conversão numpy (API moderna)
        numpy2ri.activate()
        
        # Importar pacotes R
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)  # Selecionar mirror automático
        
        # Instalar pacote pls se necessário
        print("Verificando pacote 'pls'...")
        try:
            pls = importr('pls')
        except:
            print("Instalando pacote 'pls'...")
            utils.install_packages('pls')
            pls = importr('pls')
        
        # Carregar dados
        print("Carregando dataset meatspec...")
        ro.r('data(meatspec, package="pls")')
        
        # Extrair dados
        X = np.array(ro.r('meatspec$X'))
        Y = np.array(ro.r('meatspec$Y'))
        
        # Salvar
        spectra_path = output_dir / 'spectra.csv'
        targets_path = output_dir / 'targets.csv'
        
        pd.DataFrame(X).to_csv(spectra_path, index=False)
        pd.DataFrame(Y, columns=['Fat', 'Water', 'Protein']).to_csv(
            targets_path, index=False
        )
        
        print(f"✅ Tecator baixado via R!")
        print(f"   Spectra: {X.shape}")
        print(f"   Targets: {Y.shape}")
        print(f"   Salvo em: {output_dir}")
        
        numpy2ri.deactivate()
        return True
        
    except Exception as e:
        print(f"⚠️  Erro no download via rpy2: {e}")
        print("\n" + "=" * 50)
        print("📋 MÉTODO ALTERNATIVO: Download manual via R")
        print("=" * 50)
        
        # Criar script R para download manual
        r_script = """
# Script para baixar Tecator
install.packages('pls')
library(pls)
data(meatspec)

# Definir caminho (ajustar se necessário)
output_dir <- "~/Dropbox/chemometric-band-selection/data/raw/tecator"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Salvar dados
write.csv(meatspec$X, 
          file.path(output_dir, "spectra.csv"), 
          row.names = FALSE)
write.csv(meatspec$Y, 
          file.path(output_dir, "targets.csv"), 
          row.names = FALSE)

# Verificar
cat("✅ Tecator dataset salvo em:", output_dir, "\\n")
cat("Arquivos:\\n")
list.files(output_dir)
"""
        
        # Salvar script R
        script_path = Path('scripts/download_tecator.R')
        script_path.parent.mkdir(exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(r_script)
        
        print(f"\n📝 Script R criado: {script_path}")
        print("\n🔧 Para executar:")
        print(f"   Rscript {script_path}")
        print("\n   OU abrir R/RStudio e executar:")
        print(f"   source('{script_path}')")
        
        return False
