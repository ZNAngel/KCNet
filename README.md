# KCNet

This release implements the method in **Calibrated Residual Council Fusion with KAN-NBC for Heterogeneous Pathology Foundation Models**, following `manuscript_0811_polished.pdf` (Methods, Sections 3.1-3.5, equations 1-21; supplied 2026-09-16). Frozen PFM token features are calibrated, refined within each expert, and fused through a reference classifier plus a council correction.

This update covers the method and its training objective. Dataset preparation, experimental sampling, and the full evaluation protocol are not reproduced by this minimal release.

## Files

```text
.
├── Train4CRC100K.py      # Main training entry point
├── module/
│   ├── KCNet.py          # CFC, GFR, RCF; GraphKANFusion / KCNet
│   └── losses.py         # Manuscript training objective
├── datasets/
│   └── CRC100K.py        # CRC100K feature dataset loader
├── requirements.txt
├── tests/test_kcnet.py   # Formula, masking, gradient and integration checks
├── .gitignore
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch version for your machine first, then install the remaining packages.

Python 3.10 or later is required.

## Method correspondence

| Manuscript | Implementation |
| --- | --- |
| KAN-NBC, Eq. (1) | `KANNBC`: learned linear maps and scalar scales over `SiLU(x)` and `x * SiLU(x)`; no splines or knot grids |
| CFC, Eqs. (3)-(6) | Independent shared/private projections; sparse directed co-attention of shared tokens; one residual and LayerNorm after summing messages; KAN-NBC gating |
| GFR, Eqs. (7)-(9) | Fully connected directed token graph with self-loops, feature-dependent edge weights, original tokens as values, masked pooling, and expert identities |
| RCF, Eqs. (10)-(15) | Identity-free concatenation reference; identity-aware stack followed by a linear projection and four-head council; mean consensus and additive correction |
| Auxiliary supervision, Eq. (17) | An MLP classifier on each GFR pooled embedding |
| Orthogonality, Eq. (18) | Batch mean of the squared Frobenius norm of `S.T @ P / valid_length`, summed over experts; uses pre-interaction shared/private branches |
| Consistency, Eq. (19) | Sum of MSE on masked attention-pooled, pre-interaction shared embeddings for selected expert pairs |
| Sparsity, Eq. (20) | L1 of KAN-NBC nonlinear-branch scales in CFC/GFR only; no penalty on correction logits |
| Total objective, Eq. (21) | Main CE + weighted sums of auxiliary CE, orthogonality, pairwise MSE + scale sparsity |

Only the correction head's final linear weight and bias are zero-initialized. Thus initial predictions equal the current reference branch exactly; all branches remain trainable. There is no classifier after the logit sum.

The graph uses `d_g = d / 4`; the default width is 1536. The implementation uses one graph layer, dropout 0.1, two-layer KAN-NBC updates with hidden width `2d`, and shares the gate and attention pooler across experts. These internal choices retain the release's conventions where the manuscript does not specify additional details. Each graph layer applies `LayerNorm(F + Phi_upd(A @ F))`, with no extra residual inside `Phi_upd`.

Padding masks have shape `[B, L_m]` with **True meaning padding**. Padding is excluded from co-attention, graph sources, pooling, and orthogonality. Valid lengths normalize the orthogonality term per sample. A sample must contain at least one valid token for every configured expert. Different experts may have different token lengths; the model does not require token correspondence.

## Expert configurations and model API

VUH is the default, using feature keys `virchow`, `uni`, `hibou` for Virchow2, UNI2-h and Hibou-L. The corresponding dimensions are 1280, 1536, 1024, and masks are `mask_v`, `mask_u`, `mask_h`.

```python
from module.KCNet import GraphKANFusion, VUP_EXPERT_DIMS
from module.losses import KCNetObjective

model = GraphKANFusion(num_classes=9)  # VUH
outputs = model(features)
losses = KCNetObjective()(model, outputs, labels, model.get_masks(features))
losses["total"].backward()

# VUP: replace Hibou-L with Prov-GigaPath (1536 channels).
vup_model = GraphKANFusion(num_classes=2, expert_dims=VUP_EXPERT_DIMS)
```

VUP expects `virchow`, `uni`, `gigapath` and masks `mask_v`, `mask_u`, `mask_p`. The bundled CRC100K loader still supplies VUH; using VUP requires supplying the corresponding feature dictionary.

Both configurations use target-source edges `(v, u), (u, h/p)`, meaning `h/p -> u -> v`. Messages are computed simultaneously from the original shared features, as in Eq. (4). Consistency uses only pairs `(v, u), (u, h/p)`; it does not add the third pair or average over pairs. Custom `expert_dims`, `interaction_edges`, and `consistency_pairs` are supported. Dictionary insertion order determines expert concatenation and output order. Custom expert masks use `mask_<name>`.

The six-element return tuple remains:

```python
final_logits, expert_logits, correction, shared_tokens, private_tokens, shared_pools = outputs
```

`shared_tokens` now correctly contains the original shared branches before co-attention. `KCNet` aliases `GraphKANFusion`; `KANLinear` is a legacy import alias for `KANNBC`. Module structure and parameter names changed, so **old checkpoints are not directly compatible and require retraining**. Loading them with `strict=False` does not reproduce the updated method.

## Objective weights

The manuscript specifies validation selection but does not report the selected values of `alpha_aux`, `alpha_orth`, `alpha_con`, or `lambda_sp`. Defaults are 0.3, 0.3, 0.3, and 1e-4, retaining the earlier release's coefficient values as configurable starting points, not verified paper settings. In particular, Eq. (18) has a different scale from the old cosine penalty and requires validation tuning.

Eq. (18) does not average over feature channels. At width 1536 it can greatly exceed cross-entropy: a synthetic two-sample check with 1-4 valid tokens per expert produced an unweighted orthogonality sum of about 3.58 million versus main CE 2.15. This is a scale check, not a dataset result; tune `alpha_orth` before a real training run.

```python
from Train4CRC100K import GraphFusionModule

training_module = GraphFusionModule(
    model, alpha_aux=0.3, alpha_orth=0.3, alpha_con=0.3, lambda_sp=1e-4,
)
```

The training module logs the main, auxiliary, orthogonality, consistency and sparsity terms separately.

## Data Preparation

This code expects pre-extracted features from three pathology foundation models:

- Virchow2 (`virchow`)
- UNI2-h (`uni`)
- Hibou-L (`hibou`)

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

The CRC100K class count is derived from its nine-label mapping. Batch size 8, learning rate 1e-4, 200 epochs, GPU selection, and the existing data loader remain example-run settings. The manuscript instead specifies a maximum of 64 tokens, seeded train/evaluation sampling, batch size 128, learning rate 1e-3 and 5 epochs. Implement that experiment protocol separately before attempting to reproduce reported metrics.

## Verification

```bash
pip install pytest
python -m pytest tests -q
```

Checks cover numerical values and gradients of Eq. (18), directed CFC updates, graph masking and original values, batched versus unpadded predictions, zero padding gradients, RCF initialization and subsequent council learning, selected MSE pairs, scale-only sparsity, VUH/VUP/custom expert sets, and training-module integration. They use synthetic features and do not establish dataset accuracy.

## Notes

- This folder is intended as a clean release package, not the full local experiment workspace.
- Datasets, checkpoints, generated figures, and experiment logs are not included.
- Update local paths, GPU settings, batch size, class count, and training hyperparameters in `Train4CRC100K.py` before running on a new machine.
