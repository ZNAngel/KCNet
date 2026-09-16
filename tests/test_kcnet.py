"""Numerical and integration checks for the manuscript method, on CPU."""

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from module.KCNet import (
    DynamicGraphLayer, GraphKANFusion, KANCoAttention, KANNBC,
    MaskedAttentionPooling, VUH_EXPERT_DIMS, VUP_EXPERT_DIMS,
)
from module.losses import KCNetObjective, orthogonality_loss


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    torch.set_num_threads(1)


def small_model(**kwargs):
    return GraphKANFusion(
        num_classes=3, expert_dims={"virchow": 6, "uni": 8, "hibou": 4},
        proj_dim=8, dropout=0.0, **kwargs,
    )


def make_batch(model, batch_size=3):
    batch = {}
    for index, (name, dim) in enumerate(model.expert_dims.items()):
        length = index + 3
        batch[name] = torch.randn(batch_size, length, dim)
        valid = torch.tensor([max(1, length - i) for i in range(batch_size)])
        batch[model.mask_keys[name]] = torch.arange(length)[None, :] >= valid[:, None]
    return batch


def reference_orthogonality(shared, private, mask):
    values = []
    for s, p, padded in zip(shared, private, mask):
        s, p = s[~padded], p[~padded]
        values.append(((s.T @ p) / len(s)).square().sum())
    return torch.stack(values).mean()


@pytest.mark.parametrize("length,width", [(3, 8), (7, 4)])
def test_orthogonality_matches_equation_values_and_gradients(length, width):
    shared = torch.randn(2, length, width, dtype=torch.float64, requires_grad=True)
    private = torch.randn_like(shared, requires_grad=True)
    mask = torch.zeros(2, length, dtype=torch.bool)
    mask[1, -2:] = True
    actual = orthogonality_loss(shared, private, mask)
    expected = reference_orthogonality(shared, private, mask)
    torch.testing.assert_close(actual, expected)
    grads = torch.autograd.grad(actual, (shared, private))
    reference_grads = torch.autograd.grad(expected, (shared, private))
    for grad, reference in zip(grads, reference_grads):
        torch.testing.assert_close(grad, reference)
        assert torch.count_nonzero(grad[mask]) == 0


def test_orthogonality_is_cross_channel_penalty_not_token_cosine():
    shared = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    private = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    assert F.cosine_similarity(shared, private, dim=-1).sum() == 0
    assert orthogonality_loss(shared, private).item() == 1.0


def test_nbc_fixed_basis_and_independent_scales():
    layer = KANNBC(2, 2, scale_base=2, scale_nonlinear=-3)
    with torch.no_grad():
        for linear in (layer.base_linear, layer.nonlinear_linear):
            linear.weight.copy_(torch.eye(2))
            linear.bias.zero_()
    x = torch.tensor([[-2.0, 3.0]])
    # SiLU(x) = x * sigmoid(x); no learned spline basis is involved.
    expected = 2 * x * x.sigmoid() - 3 * x.square() * x.sigmoid()
    torch.testing.assert_close(layer(x), expected)


def test_coattention_returns_message_without_query_residual_or_normalization():
    attention = KANCoAttention(4, num_heads=1, dropout=0.0)
    with torch.no_grad():
        for linear in (attention.q_proj, attention.k_proj, attention.v_proj, attention.out_proj):
            linear.weight.copy_(torch.eye(4))
            linear.bias.zero_()
    query = torch.randn(2, 3, 4)
    sources = torch.randn(2, 5, 4)
    mask = torch.tensor([[False, False, True, True, True], [False, False, False, False, True]])
    expected = torch.stack([
        (q @ kv[~padded].T / 2).softmax(-1) @ kv[~padded]
        for q, kv, padded in zip(query, sources, mask)
    ])
    torch.testing.assert_close(attention(query, sources, mask), expected)


def test_graph_uses_original_values_and_one_outer_residual():
    graph = DynamicGraphLayer(8, dropout=0.0)
    # Zero Q/K gives uniform weights including self-loops. Identity Phi_upd
    # lets the test distinguish original-token values from projected values.
    with torch.no_grad():
        for projection in (graph.proj_q, graph.proj_k):
            for parameter in projection.parameters():
                parameter.zero_()
    graph.update_kan = nn.Identity()
    x = torch.randn(3, 5, 8)
    mask = torch.tensor([[False] * 5, [False, False, True, True, True], [False] * 4 + [True]])
    expected = torch.zeros_like(x)
    for index in range(3):
        valid = x[index, ~mask[index]]
        expected[index, ~mask[index]] = F.layer_norm(valid + valid.mean(0), (8,))
    actual = graph(x, mask)
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)


def test_cfc_sums_incoming_messages_from_original_shared_tokens():
    edges = [("virchow", "uni"), ("uni", "hibou"), ("virchow", "hibou")]
    model = small_model(interaction_edges=edges).eval()

    class SourceMean(nn.Module):
        def forward(self, query, source, mask):
            valid = (~mask).unsqueeze(-1)
            mean = (source * valid).sum(1, keepdim=True) / valid.sum(1, keepdim=True)
            return mean.expand_as(query)

    model.co_attentions = nn.ModuleList(SourceMean() for _ in edges)
    norm_inputs, decomposed = {}, {}
    handles = []
    for name in model.expert_names:
        handles.append(model.interaction_norms[name].register_forward_pre_hook(
            lambda module, inputs, name=name: norm_inputs.update({name: inputs[0].detach()})
        ))
        handles.append(model.decompositions[name].register_forward_hook(
            lambda module, inputs, output, name=name: decomposed.update({name: output})
        ))
    batch = make_batch(model)
    outputs = model(batch)
    shared = dict(zip(model.expert_names, outputs[3]))
    for name in model.expert_names:
        mask = batch[model.mask_keys[name]]
        expected = shared[name].clone()
        for target, source in edges:
            if target == name:
                source_mask = batch[model.mask_keys[source]]
                expected = expected + shared[source].sum(1, keepdim=True) / (~source_mask).sum(1)[:, None, None]
        torch.testing.assert_close(norm_inputs[name], expected)
        torch.testing.assert_close(shared[name][~mask], decomposed[name][0][~mask])
        index = model.expert_names.index(name)
        torch.testing.assert_close(outputs[4][index][~mask], decomposed[name][1][~mask])
    for handle in handles:
        handle.remove()


def test_batched_predictions_equal_individual_unpadded_predictions():
    model = small_model().eval()
    # Exercise both RCF paths, not just the initially zero correction.
    nn.init.normal_(model.council_readout[-1].weight, std=0.1)
    batch = make_batch(model)
    for name in model.expert_names:
        batch[name][batch[model.mask_keys[name]]] = float("nan")
    batched = model(batch)
    batched_losses = KCNetObjective()(model, batched, torch.tensor([0, 1, 2]), model.get_masks(batch))
    individual_losses = []
    for sample in range(3):
        single = {name: batch[name][sample, ~batch[model.mask_keys[name]][sample]].unsqueeze(0)
                  for name in model.expert_names}
        result = model(single)
        for index in (0, 2):
            torch.testing.assert_close(batched[index][sample:sample + 1], result[index], atol=2e-6, rtol=1e-5)
        for index in (1, 5):
            for batched_expert, single_expert in zip(batched[index], result[index]):
                torch.testing.assert_close(batched_expert[sample:sample + 1], single_expert, atol=2e-6, rtol=1e-5)
        individual_losses.append(KCNetObjective()(model, result, torch.tensor([sample])))
    for key, value in batched_losses.items():
        torch.testing.assert_close(value, torch.stack([loss[key] for loss in individual_losses]).mean())


def test_padding_has_zero_gradient_through_model_and_objective():
    model = small_model()
    batch = make_batch(model)
    for name in model.expert_names:
        batch[name].requires_grad_()
    losses = KCNetObjective()(model, model(batch), torch.tensor([0, 1, 2]), model.get_masks(batch))
    losses["total"].backward()
    for name in model.expert_names:
        gradient = batch[name].grad
        mask = batch[model.mask_keys[name]]
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient[mask]) == 0
        assert torch.count_nonzero(gradient[~mask]) > 0


def test_zero_initialization_identity_routing_and_council_learning():
    model = small_model()
    captured = {}
    handles = [
        model.baseline_classifier.register_forward_pre_hook(
            lambda module, args: captured.update({"baseline_input": args[0].detach()})
        ),
        model.baseline_classifier.register_forward_hook(
            lambda module, args, output: captured.update({"baseline": output.detach()})
        ),
        model.council_projection.register_forward_pre_hook(
            lambda module, args: captured.update({"council_input": args[0].detach()})
        ),
    ]
    batch = make_batch(model)
    labels = torch.tensor([0, 1, 2])
    output = model(batch)
    assert torch.count_nonzero(output[2]) == 0
    torch.testing.assert_close(output[0], captured["baseline"], atol=0, rtol=0)
    identities = torch.stack([model.expert_ids[name] for name in model.expert_names], dim=1)
    expected = captured["baseline_input"].reshape(3, 3, 8) + identities
    torch.testing.assert_close(captured["council_input"], expected)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    F.cross_entropy(output[0], labels).backward()
    assert model.council_readout[-1].weight.grad.abs().sum() > 0
    assert model.council_projection.weight.grad.abs().sum() == 0
    optimizer.step()
    optimizer.zero_grad()
    output = model(batch)
    assert output[2].abs().sum() > 0
    F.cross_entropy(output[0], labels).backward()
    assert model.council_projection.weight.grad.abs().sum() > 0
    assert model.expert_ids["virchow"].grad.abs().sum() > 0
    for handle in handles:
        handle.remove()


def test_objective_uses_selected_mse_pairs_and_sums_experts():
    model = small_model()
    outputs = list(model(make_batch(model, batch_size=1)))
    outputs[5] = [torch.full((1, 8), float(value)) for value in (0, 2, 5)]
    labels = torch.tensor([1])
    objective = KCNetObjective(alpha_aux=0.2, alpha_orth=0.4, alpha_con=0.6, lambda_sp=0.01)
    losses = objective(model, outputs, labels)
    # Adjacent-pair MSE is 4 + 9. No v-h term and no pair averaging.
    assert losses["con"].item() == 13
    expected_orth = sum(reference_orthogonality(s, p, torch.zeros(s.shape[:2], dtype=torch.bool))
                        for s, p in zip(outputs[3], outputs[4]))
    torch.testing.assert_close(losses["orth"], expected_orth)
    expected_aux = sum(F.cross_entropy(logits, labels) for logits in outputs[1])
    torch.testing.assert_close(losses["aux"], expected_aux)
    # 6 projections + 2 gated fusion mappings + 12 graph mappings.
    torch.testing.assert_close(losses["sparsity"], torch.tensor(0.2))
    expected = F.cross_entropy(outputs[0], labels) + 0.2 * expected_aux + 0.4 * expected_orth + 0.6 * 13 + 0.2
    torch.testing.assert_close(losses["total"], expected)


def test_sparsity_only_penalizes_cfc_gfr_nonlinear_scales():
    model = small_model()
    model.nbc_sparsity_loss().backward()
    for name, parameter in model.named_parameters():
        if name.endswith("scale_nonlinear"):
            torch.testing.assert_close(parameter.grad, torch.ones_like(parameter))
        else:
            assert parameter.grad is None, name


@pytest.mark.parametrize("dims", [VUH_EXPERT_DIMS, VUP_EXPERT_DIMS, {"a": 4}, {"a": 4, "b": 6, "c": 8, "d": 5}])
def test_configurable_experts_forward_and_backward(dims):
    model = GraphKANFusion(num_classes=3, expert_dims=dims, proj_dim=8, dropout=0.0)
    batch = make_batch(model, batch_size=2)
    outputs = model(batch)
    assert outputs[0].shape == (2, 3)
    assert len(outputs[1]) == len(dims)
    assert model.baseline_classifier[0].in_features == len(dims) * 8
    losses = KCNetObjective()(model, outputs, torch.tensor([0, 1]), model.get_masks(batch))
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_all_masked_primitives_are_finite_but_empty_experts_are_rejected():
    x = torch.randn(2, 3, 8, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool)
    pool = MaskedAttentionPooling(8)(x, mask)
    attention = KANCoAttention(8)(x, x, mask)
    graph = DynamicGraphLayer(8)(x, mask)
    for result in (pool, attention, graph):
        assert torch.count_nonzero(result) == 0
    (pool.sum() + attention.sum() + graph.sum()).backward()
    assert torch.isfinite(x.grad).all()
    model = small_model()
    batch = make_batch(model)
    batch["mask_v"][0] = True
    with pytest.raises(ValueError, match="at least one valid token"):
        model(batch)
    with pytest.raises(ValueError, match="at least one valid token"):
        orthogonality_loss(x, x, mask)


@pytest.mark.parametrize("kwargs", [
    {"interaction_edges": [("virchow", "missing")]},
    {"interaction_edges": [("uni", "uni")]},
    {"consistency_pairs": [("uni", "virchow"), ("virchow", "uni")]},
])
def test_invalid_expert_pairs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        small_model(**kwargs)


def test_lightning_training_and_validation_integration():
    from Train4CRC100K import GraphFusionModule, NUM_CLASSES
    from datasets.CRC100K import collate_fn_masked

    assert NUM_CLASSES == 9
    model = small_model()
    samples = [{"v": torch.randn(index + 2, 6), "u": torch.randn(index + 3, 8),
                "h": torch.randn(index + 1, 4), "label": index} for index in range(3)]
    features, labels = collate_fn_masked(samples)
    module = GraphFusionModule(model, alpha_aux=0.1, alpha_orth=0.01, alpha_con=0.2, lambda_sp=0.001)
    logged = {}
    module.log = lambda name, value, **kwargs: logged.update({name: torch.as_tensor(value).detach()})
    loss = module.training_step((features, labels), 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(logged) >= {"train_loss", "train_main", "train_aux", "train_orth", "train_con", "train_sparsity"}
    validation = module.validation_step((features, labels), 0)
    assert validation["preds"].shape == labels.shape
    module.on_validation_epoch_end()
    assert "val_f1" in logged
    assert module.training_step((None, None), 0) is None
    assert module.validation_step((None, None), 0) is None
