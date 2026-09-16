"""KCNet: CFC, GFR and RCF from manuscript_0811_polished, Eqs. (1)-(17).

Inputs are frozen PFM features [batch, tokens, channels]. Boolean masks use
True for padding. KAN-NBC combines two fixed SiLU-based responses; it is not
a spline KAN. The historical GraphKANFusion entry point is retained.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F


PROJECTION_DIM = 1536
DIM_VIRCHOW = 1280
DIM_UNI = 1536
DIM_HIBOU = 1024
DIM_GIGAPATH = 1536
VUH_EXPERT_DIMS = {"virchow": DIM_VIRCHOW, "uni": DIM_UNI, "hibou": DIM_HIBOU}
VUP_EXPERT_DIMS = {"virchow": DIM_VIRCHOW, "uni": DIM_UNI, "gigapath": DIM_GIGAPATH}
MASK_KEYS = {"virchow": "mask_v", "uni": "mask_u", "hibou": "mask_h", "gigapath": "mask_p"}


def _zero_padding(x, mask):
    return x if mask is None else x.masked_fill(mask.unsqueeze(-1), 0.0)


def _masked_softmax(scores, mask, dim=-1):
    """Exclude padding, returning zero weights for a fully masked row."""
    if mask is None:
        return torch.softmax(scores, dim=dim)
    scores = scores.masked_fill(mask, float("-inf"))
    scores = scores.masked_fill(mask.all(dim=dim, keepdim=True), 0.0)
    return torch.softmax(scores, dim=dim).masked_fill(mask, 0.0)


class KANNBC(nn.Module):
    """Eq. (1): gamma_b W_b(SiLU(x)) + gamma_s W_s(x * SiLU(x))."""

    def __init__(self, in_features, out_features, scale_base=1.0, scale_nonlinear=1.0):
        super().__init__()
        self.base_linear = nn.Linear(in_features, out_features)
        self.nonlinear_linear = nn.Linear(in_features, out_features)
        self.scale_base = nn.Parameter(torch.tensor([float(scale_base)]))
        self.scale_nonlinear = nn.Parameter(torch.tensor([float(scale_nonlinear)]))
        nn.init.xavier_uniform_(self.base_linear.weight)
        nn.init.xavier_uniform_(self.nonlinear_linear.weight)

    def forward(self, x):
        activated = F.silu(x)
        return (self.scale_base * self.base_linear(activated)
                + self.scale_nonlinear * self.nonlinear_linear(x * activated))


# Keep the old import name; checkpoint parameter names have changed.
KANLinear = KANNBC


class KANBlock(nn.Module):
    """GFR update mapping Phi_upd; the graph layer owns the outer residual."""

    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            KANNBC(dim, dim * 2), nn.LayerNorm(dim * 2), nn.SiLU(),
            nn.Dropout(drop), KANNBC(dim * 2, dim), nn.Dropout(drop),
        )

    def forward(self, x):
        return self.net(x)


class DynamicGraphLayer(nn.Module):
    """Eq. (7): dense directed token graph, including self-loops.

    Edge weights depend on features. Values are the original calibrated
    tokens; no value projection or spatial adjacency is used.
    """

    def __init__(self, dim, dropout=0.1):
        super().__init__()
        if dim < 4 or dim % 4:
            raise ValueError("GFR width must be a positive multiple of four.")
        self.proj_q = KANNBC(dim, dim // 4)
        self.proj_k = KANNBC(dim, dim // 4)
        self.update_kan = KANBlock(dim, dropout)
        self.norm = nn.LayerNorm(dim)
        self.scale = (dim // 4) ** -0.5

    def forward(self, x, mask=None):
        x = _zero_padding(x, mask)
        scores = torch.matmul(self.proj_q(x), self.proj_k(x).transpose(-2, -1)) * self.scale
        # [B, 1, L] masks sources in [B, L, L] without adding a batch axis.
        weights = _masked_softmax(scores, None if mask is None else mask[:, None, :])
        aggregated = torch.matmul(weights, x)
        return _zero_padding(self.norm(x + self.update_kan(aggregated)), mask)


class DynamicGraphKANEncoder(nn.Module):
    def __init__(self, dim, num_layers=1, dropout=0.1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GFR requires at least one graph layer.")
        self.layers = nn.ModuleList(
            DynamicGraphLayer(dim, dropout) for _ in range(num_layers)
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class KANCoAttention(nn.Module):
    """Eq. (5): linear Q/K/V attention returning only exchanged evidence.

    CFC applies the residual and normalization after summing incoming
    messages, Eq. (4). Despite the legacy name, Q/K/V are linear.
    """

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        if num_heads < 1 or dim % num_heads:
            raise ValueError("Attention width must be divisible by num_heads.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_q, x_kv, mask_kv=None):
        batch_size, query_length, dim = x_q.shape
        source_length = x_kv.shape[1]
        x_kv = _zero_padding(x_kv, mask_kv)
        q = self.q_proj(x_q).reshape(batch_size, query_length, self.num_heads, -1).transpose(1, 2)
        k = self.k_proj(x_kv).reshape(batch_size, source_length, self.num_heads, -1).transpose(1, 2)
        v = self.v_proj(x_kv).reshape(batch_size, source_length, self.num_heads, -1).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = None if mask_kv is None else mask_kv[:, None, None, :]
        attention = _masked_softmax(scores, mask)
        output = torch.matmul(attention, v).transpose(1, 2).reshape(batch_size, query_length, dim)
        output = self.dropout(self.out_proj(output))
        if mask_kv is not None:
            output = output.masked_fill(mask_kv.all(dim=1)[:, None, None], 0.0)
        return output


class ModalityDecomposition(nn.Module):
    """Eq. (3): independent shared and private KAN-NBC projections."""

    def __init__(self, in_dim, proj_dim):
        super().__init__()
        self.shared_proj = KANNBC(in_dim, proj_dim)
        self.private_proj = KANNBC(in_dim, proj_dim)
        self.shared_norm = nn.LayerNorm(proj_dim)
        self.private_norm = nn.LayerNorm(proj_dim)

    def forward(self, x):
        return self.shared_norm(self.shared_proj(x)), self.private_norm(self.private_proj(x))


class KANGatedFusion(nn.Module):
    """Eq. (6): token- and channel-wise shared/private gate."""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            KANNBC(dim * 2, dim), nn.SiLU(), KANNBC(dim, dim), nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, shared, private):
        alpha = self.gate(torch.cat([shared, private], dim=-1))
        return self.norm(shared * alpha + private * (1 - alpha))


class MaskedAttentionPooling(nn.Module):
    """Eq. (8), also used on pre-interaction shared tokens in Eq. (19)."""

    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x, mask=None):
        x = _zero_padding(x, mask)
        scores = self.attn(x).squeeze(-1)
        weights = _masked_softmax(scores, mask)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class ExpertConsultationLayer(nn.Module):
    """Eq. (13): four-head council attention followed by a residual FFN."""

    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, expert_feats):
        attention, _ = self.attn(expert_feats, expert_feats, expert_feats, need_weights=False)
        x = self.norm(expert_feats + attention)
        return self.norm2(x + self.ffn(x))


class GraphKANFusion(nn.Module):
    """Configurable KCNet; defaults to the manuscript's VUH configuration.

    expert_dims insertion order defines concatenation and output-list order.
    interaction_edges use (target, source), with simultaneous updates from
    the original shared tokens. Defaults form an adjacent directed chain;
    consistency pairs select the same adjacent expert pairs. Custom experts
    use mask_<name>; VUH/VUP retain mask_v, mask_u, mask_h / mask_p.

    Forward preserves the six-output API: final logits, auxiliary logits,
    correction, original shared tokens, private tokens, pooled shared tokens.
    The two losses on shared tokens use S, not the interacted S_tilde.
    """

    def __init__(
        self, num_classes=15, *, expert_dims: Mapping[str, int] | None = None,
        proj_dim=PROJECTION_DIM, interaction_edges: Sequence[tuple[str, str]] | None = None,
        consistency_pairs: Sequence[tuple[str, str]] | None = None,
        num_heads=4, graph_layers=1, dropout=0.1,
    ):
        super().__init__()
        self.expert_dims = dict(VUH_EXPERT_DIMS if expert_dims is None else expert_dims)
        if not self.expert_dims or any(dim < 1 for dim in self.expert_dims.values()):
            raise ValueError("Supply at least one expert with a positive input dimension.")
        if any(not name or "." in name for name in self.expert_dims):
            raise ValueError("Expert names must be nonempty and cannot contain '.'.")
        if proj_dim < 4 or proj_dim % 4 or num_heads < 1 or proj_dim % num_heads:
            raise ValueError("proj_dim must be divisible by four and by num_heads.")
        if num_classes < 1:
            raise ValueError("num_classes must be positive.")
        self.expert_names = tuple(self.expert_dims)
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        default_pairs = tuple(zip(self.expert_names[:-1], self.expert_names[1:]))
        self.interaction_edges = self._validate_pairs(
            default_pairs if interaction_edges is None else interaction_edges, "interaction_edges"
        )
        self.consistency_pairs = self._validate_pairs(
            default_pairs if consistency_pairs is None else consistency_pairs, "consistency_pairs",
            undirected=True,
        )
        self.mask_keys = {name: MASK_KEYS.get(name, f"mask_{name}") for name in self.expert_names}

        # CFC, Eqs. (3)-(6). The gate and attention pooler are shared across experts.
        self.decompositions = nn.ModuleDict({
            name: ModalityDecomposition(dim, proj_dim) for name, dim in self.expert_dims.items()
        })
        self.co_attentions = nn.ModuleList(
            KANCoAttention(proj_dim, num_heads, dropout) for _ in self.interaction_edges
        )
        self.interaction_norms = nn.ModuleDict({name: nn.LayerNorm(proj_dim) for name in self.expert_names})
        self.fusion_gate = KANGatedFusion(proj_dim)

        # GFR, Eqs. (7)-(9).
        self.spatial_encoders = nn.ModuleDict({
            name: DynamicGraphKANEncoder(proj_dim, graph_layers, dropout) for name in self.expert_names
        })
        self.pooler = MaskedAttentionPooling(proj_dim)
        self.expert_ids = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, proj_dim) * 0.02) for name in self.expert_names
        })

        # Per-expert MLP auxiliary heads, Eq. (17).
        self.expert_heads = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2), nn.LayerNorm(proj_dim // 2),
            nn.SiLU(), nn.Dropout(dropout), nn.Linear(proj_dim // 2, num_classes),
        ) for name in self.expert_names})

        # RCF, Eqs. (10)-(15).
        self.baseline_classifier = nn.Sequential(
            nn.Linear(proj_dim * len(self.expert_names), proj_dim), nn.LayerNorm(proj_dim),
            nn.SiLU(), nn.Linear(proj_dim, num_classes),
        )
        self.council_projection = nn.Linear(proj_dim, proj_dim)
        self.council_layer = ExpertConsultationLayer(proj_dim, num_heads, dropout)
        self.council_readout = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2), nn.SiLU(), nn.Linear(proj_dim // 2, num_classes),
        )
        nn.init.zeros_(self.council_readout[-1].weight)
        nn.init.zeros_(self.council_readout[-1].bias)

    def _validate_pairs(self, pairs, label, undirected=False):
        result = tuple(tuple(pair) for pair in pairs)
        seen = set()
        for pair in result:
            if len(pair) != 2 or any(name not in self.expert_dims for name in pair) or pair[0] == pair[1]:
                raise ValueError(f"{label} must connect two distinct configured experts: {pair}")
            key = frozenset(pair) if undirected else pair
            if key in seen:
                raise ValueError(f"Duplicate pair in {label}: {pair}")
            seen.add(key)
        return result

    def get_masks(self, batch_data):
        """Return masks in expert order, including None for unpadded streams."""
        return [batch_data.get(self.mask_keys[name]) for name in self.expert_names]

    def nbc_sparsity_loss(self):
        """Unweighted Eq. (20): only nonlinear scales in CFC and GFR."""
        modules = (self.decompositions, self.fusion_gate, self.spatial_encoders)
        scales = [layer.scale_nonlinear.abs().sum()
                  for module in modules for layer in module.modules() if isinstance(layer, KANNBC)]
        return torch.stack(scales).sum()

    def forward(self, batch_data):
        masks = dict(zip(self.expert_names, self.get_masks(batch_data)))
        shared, private = {}, {}
        batch_size = None
        for name, dim in self.expert_dims.items():
            x, mask = batch_data[name], masks[name]
            if x.ndim != 3 or x.shape[-1] != dim or x.shape[0] < 1 or x.shape[1] < 1:
                raise ValueError(f"{name} must have nonempty shape [B, L, {dim}], got {tuple(x.shape)}")
            if batch_size is not None and x.shape[0] != batch_size:
                raise ValueError("All expert streams must have the same batch size.")
            batch_size = x.shape[0]
            if mask is not None:
                if mask.dtype != torch.bool or mask.shape != x.shape[:2] or mask.device != x.device:
                    raise ValueError(f"{self.mask_keys[name]} must be a boolean [B, L] mask on the input device.")
                if mask.all(dim=1).any():
                    raise ValueError(f"Every sample needs at least one valid token for expert {name}.")
            s, p = self.decompositions[name](_zero_padding(x, mask))
            shared[name], private[name] = _zero_padding(s, mask), _zero_padding(p, mask)

        shared_pools = [self.pooler(shared[name], masks[name]) for name in self.expert_names]
        messages = {name: torch.zeros_like(shared[name]) for name in self.expert_names}
        for (target, source), attention in zip(self.interaction_edges, self.co_attentions):
            messages[target] = messages[target] + attention(shared[target], shared[source], masks[source])

        embeddings = []
        for name in self.expert_names:
            updated = self.interaction_norms[name](shared[name] + messages[name])
            calibrated = _zero_padding(self.fusion_gate(updated, private[name]), masks[name])
            refined = self.spatial_encoders[name](calibrated, masks[name])
            embeddings.append(self.pooler(refined, masks[name]))

        expert_logits = [self.expert_heads[name](z) for name, z in zip(self.expert_names, embeddings)]
        baseline = self.baseline_classifier(torch.cat(embeddings, dim=-1))
        identity_aware = torch.stack([
            z + self.expert_ids[name] for name, z in zip(self.expert_names, embeddings)
        ], dim=1)
        council = self.council_layer(self.council_projection(identity_aware))
        correction = self.council_readout(council.mean(dim=1))
        return (baseline + correction, expert_logits, correction,
                [shared[name] for name in self.expert_names],
                [private[name] for name in self.expert_names], shared_pools)


KCNet = GraphKANFusion
