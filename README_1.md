chemometric-band-selection/
├── data/              # Datasets (Tecator, Shootout, Corn)
├── src/               # Source code
├── experiments/       # Training scripts
├── results/           # Experimental results
└── notebooks/         # Analysis notebooks

## Setup

```bash
cd ~/Dropbox/chemometric-band-selection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Download datasets
python src/data/download.py --all

# Train model
python experiments/run_single.py --dataset tecator
```

## Documentation

See `PROJECT_OVERVIEW.md` for complete project description.

## Citation

```bibtex
@article{yourname2026,
  title={Chemometric Prior-Guided Deep Learning for Interpretable NIR Spectroscopy},
  author={Your Name},
  journal={TBD},
  year={2026}
}
```
