# KAN_Based

This is a minimal GitHub release version of the KAN-based pathology feature fusion project. It keeps only the core training code for our method and removes comparison methods, extra experiment scripts, generated figures, logs, and local outputs.

## Files

```text
.
├── Train4CRC100K.py      # Main training entry point
├── module/
│   └── KCNet.py          # GraphKANFusion model
├── datasets/
│   └── CRC100K.py        # CRC100K feature dataset loader
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch version for your machine first, then install the remaining packages.

## Data Preparation

This code expects pre-extracted features from three pathology foundation models:

- Virchow
- UNI
- Hibou

Before training, edit the paths in `datasets/CRC100K.py`:

```python
FEATURE_ROOT_DIR = '/path/to/features'
TRAIN_CSV = '/path/to/train.csv'
VAL_CSV = '/path/to/val.csv'
```

The feature directory is expected to contain:

```text
features/
├── virchow/
├── uni/
└── hibou/
```

Each CSV should contain at least:

- `path`: image or tile path used to locate the corresponding `.pt` feature file
- `label`: class label

## Training

Run from the repository root:

```bash
python Train4CRC100K.py
```

Training logs and checkpoints are written by PyTorch Lightning according to the logger and checkpoint settings in `Train4CRC100K.py`.

## Notes

- This folder is intended as a clean release package, not the full local experiment workspace.
- Datasets, checkpoints, generated figures, and experiment logs are not included.
- Update local paths, GPU settings, batch size, class count, and training hyperparameters in `Train4CRC100K.py` before running on a new machine.
