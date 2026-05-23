"""
Download automático dos 3 datasets
Rodar de: ~/Dropbox/chemometric-band-selection/
Uso: python src/data/download.py --tecator
"""

import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def download_tecator():
    """Download Tecator Meat dataset"""
    print("\n📦 TECATOR MEAT DATASET")
    print("=" * 50)
    
    output_dir = Path('data/raw/tecator')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("Tentando download via R (rpy2)...")
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        from rpy2.robjects.packages import importr
        
        # Ativar conversão numpy
        numpy2ri.activate()
        
        # Importar utils para instalação
        utils = importr('utils')
        utils.chooseCRANmirror(ind=1)
        
        # Tentar importar pls, instalar se necessário
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
        
        # Extrair
        X = np.array(ro.r('meatspec$X'))
        Y = np.array(ro.r('meatspec$Y'))
        
        # Salvar
        pd.DataFrame(X).to_csv(output_dir / 'spectra.csv', index=False)
        pd.DataFrame(Y, columns=['Fat', 'Water', 'Protein']).to_csv(
            output_dir / 'targets.csv', index=False
        )
        
        print(f"✅ Tecator baixado!")
        print(f"   Spectra: {X.shape}")
        print(f"   Targets: {Y.shape}")
        
        numpy2ri.deactivate()
        return True
        
    except Exception as e:
        print(f"⚠️  Erro: {e}")
        print("\n📋 DOWNLOAD MANUAL:")
        print("\nAbra R ou RStudio e execute:")
        print("-" * 50)
        print("install.packages('pls')")
        print("library(pls)")
        print("data(meatspec)")
        print("write.csv(meatspec$X, '~/Dropbox/chemometric-band-selection/data/raw/tecator/spectra.csv', row.names=FALSE)")
        print("write.csv(meatspec$Y, '~/Dropbox/chemometric-band-selection/data/raw/tecator/targets.csv', row.names=FALSE)")
        print("-" * 50)
        
        # Criar script R
        create_r_script()
        return False


def create_r_script():
    """Criar script R para download manual"""
    r_script = """# Download Tecator dataset
install.packages('pls', repos='http://cran.us.r-project.org')
library(pls)
data(meatspec)

output_dir <- '~/Dropbox/chemometric-band-selection/data/raw/tecator'
dir.create(output_dir, recursive=TRUE, showWarnings=FALSE)

write.csv(meatspec$X, file.path(output_dir, 'spectra.csv'), row.names=FALSE)
write.csv(meatspec$Y, file.path(output_dir, 'targets.csv'), row.names=FALSE)

cat('✅ Arquivos salvos em:', output_dir, '\\n')
"""
    
    script_dir = Path('scripts')
    script_dir.mkdir(exist_ok=True)
    script_path = script_dir / 'download_tecator.R'
    
    with open(script_path, 'w') as f:
        f.write(r_script)
    
    print(f"\n📝 Script R criado: {script_path}")
    print(f"   Execute: Rscript {script_path}")


def download_shootout():
    """Download NIR Shootout 2002"""
    print("\n📦 NIR SHOOTOUT 2002 DATASET")
    print("=" * 50)
    print("\n📋 DOWNLOAD MANUAL:")
    print("1. Visite: https://eigenvector.com/resources/data-sets/")
    print("2. Procure: 'NIR Shootout 2002' ou 'nir_shootout_2002.mat'")
    print("3. Baixe e coloque em: data/raw/shootout/")


def download_corn():
    """Download Corn dataset"""
    print("\n📦 CORN DATASET")
    print("=" * 50)
    print("\n📋 DOWNLOAD MANUAL:")
    print("1. Visite: https://eigenvector.com/resources/data-sets/")
    print("2. Procure: 'Corn' ou 'Multi-spectrometer'")
    print("3. Baixe e coloque em: data/raw/corn/")


def verify_downloads():
    """Verificar datasets baixados"""
    print("\n" + "=" * 50)
    print("📊 VERIFICAÇÃO DE DATASETS")
    print("=" * 50)
    
    datasets = {
        'Tecator': Path('data/raw/tecator'),
        'Shootout': Path('data/raw/shootout'),
        'Corn': Path('data/raw/corn'),
    }
    
    for name, path in datasets.items():
        if path.exists():
            files = list(path.glob('*'))
            if files:
                print(f"✅ {name}: {len(files)} arquivo(s)")
                for f in files:
                    size_mb = f.stat().st_size / 1024 / 1024
                    print(f"   - {f.name} ({size_mb:.2f} MB)")
            else:
                print(f"❌ {name}: Pasta vazia")
        else:
            print(f"❌ {name}: Pasta não existe")


def main():
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--all', action='store_true', help='Baixar todos')
    parser.add_argument('--tecator', action='store_true', help='Baixar Tecator')
    parser.add_argument('--shootout', action='store_true', help='Baixar Shootout')
    parser.add_argument('--corn', action='store_true', help='Baixar Corn')
    parser.add_argument('--verify', action='store_true', help='Verificar downloads')
    
    args = parser.parse_args()
    
    print("🚀 DOWNLOAD DE DATASETS")
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
        print("\nUso:")
        print("  python src/data/download.py --tecator")
        print("  python src/data/download.py --all")
        print("  python src/data/download.py --verify")
    
    print("\n")
    verify_downloads()


if __name__ == '__main__':
    main()
