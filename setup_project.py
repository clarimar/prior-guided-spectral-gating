"""
Criar estrutura de pastas do projeto
Rodar de: ~/Dropbox/chemometric-band-selection/
"""
import os

def create_directory_structure():
    """Criar toda a estrutura de pastas"""
    
    dirs = [
        # Data
        'data/raw/tecator',
        'data/raw/shootout',
        'data/raw/corn',
        'data/processed',
        'data/priors',
        
        # Source code
        'src/data',
        'src/models',
        'src/training',
        'src/utils',
        
        # Configs
        'configs/datasets',
        'configs/experiments',
        
        # Experiments
        'experiments',
        
        # Notebooks
        'notebooks',
        
        # Results
        'results/tecator/metrics',
        'results/tecator/predictions',
        'results/tecator/figures',
        'results/shootout/metrics',
        'results/shootout/predictions',
        'results/shootout/figures',
        'results/corn/metrics',
        'results/corn/predictions',
        'results/corn/figures',
        
        # Scripts
        'scripts',
        
        # Tests
        'tests',
    ]
    
    print("🔨 Criando estrutura de pastas...")
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        
        # Criar __init__.py em pastas Python
        if dir_path.startswith('src/'):
            init_file = os.path.join(dir_path, '__init__.py')
            if not os.path.exists(init_file):
                with open(init_file, 'w') as f:
                    f.write('"""Auto-generated package file"""\n')
    
    print(f"✅ Estrutura criada! Total: {len(dirs)} diretórios")
    print("\n📂 Estrutura:")
    print("chemometric-band-selection/")
    print("├── data/         (datasets)")
    print("├── src/          (código)")
    print("├── configs/      (configurações)")
    print("├── experiments/  (scripts de treino)")
    print("├── notebooks/    (análises)")
    print("├── results/      (resultados)")
    print("├── scripts/      (utilitários)")
    print("└── tests/        (testes)")

if __name__ == '__main__':
    create_directory_structure()
