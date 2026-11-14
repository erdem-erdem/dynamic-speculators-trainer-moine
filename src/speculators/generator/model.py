"""Set-conditioned full-factor hypernetwork and differentiable LoRA injection."""

import math

import torch
from torch import nn
from torch.nn import functional

from speculators.generator.config import Architecture

SUMMARY_NDIM = 2


class ResidualMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, width),
        )

    def forward(self, x):
        return x + self.net(x)


class LoRAGenerator(nn.Module):
    """Permutation-invariant context encoder, one A/B pair per projection."""

    def __init__(
        self,
        input_dim,
        shapes,
        rank,
        architecture=None,
        max_examples=8,
        *,
        feedback_dim=0,
        layout=None,
        head_sharing="per_module",
    ):
        """`layout[key] = (module_type, layer)`; None means `{key: (key, 0)}`.

        `head_sharing="per_module"` gives one A/B head per key (original
        behaviour); `"per_type"` shares one head per module type across layers,
        which are told apart by `layer_embedding`. `"per_type_bias"` (hybrid,
        ABL24K 2026-11-14) shares the head *weights* per module type but gives
        every key its own A/B bias, i.e. a per-layer static LoRA plus a shared
        prompt-dependent part; at init it equals `"per_type"` per key.
        """
        super().__init__()
        cfg = architecture or Architecture()
        self.shapes, self.rank, self.max_examples = shapes, rank, max_examples
        self.layout = (
            dict(layout) if layout is not None else {k: (k, 0) for k in shapes}
        )
        if list(self.layout) != list(shapes):
            raise ValueError("Generator layout keys must equal shapes keys, in order")
        if head_sharing not in {"per_module", "per_type", "per_type_bias"}:
            raise ValueError(f"Unknown head_sharing {head_sharing!r}")
        self.head_sharing = head_sharing
        self.module_types = list(dict.fromkeys(t for t, _ in self.layout.values()))
        layers = [layer for _, layer in self.layout.values()]
        if any(not isinstance(layer, int) or layer < 0 for layer in layers):
            raise ValueError("Generator layout layers must be non-negative ints")
        head_shapes = {}
        for key, (module_type, _) in self.layout.items():
            head = key if head_sharing == "per_module" else module_type
            if head_shapes.setdefault(head, shapes[key]) != shapes[key]:
                raise ValueError(
                    f"per_type head {head!r} has differing shapes across layers"
                )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, cfg.encoder_hidden),
            nn.GELU(),
            nn.Linear(cfg.encoder_hidden, cfg.example_dim),
            nn.LayerNorm(cfg.example_dim),
        )
        self.condition = nn.Sequential(
            nn.Linear(2 * cfg.example_dim + 1, cfg.example_dim),
            nn.GELU(),
            nn.Linear(cfg.example_dim, cfg.condition_dim),
        )
        self.module_embedding = nn.Embedding(len(self.module_types), cfg.descriptor_dim)
        self.layer_embedding = nn.Embedding(max(layers) + 1, cfg.descriptor_dim)
        self.feedback_dim = feedback_dim
        if feedback_dim:
            self.feedback_encoder = nn.Linear(
                feedback_dim, cfg.condition_dim, bias=False
            )
            nn.init.zeros_(self.feedback_encoder.weight)
        self.decoder = nn.Sequential(
            nn.Linear(cfg.condition_dim + 2 * cfg.descriptor_dim, cfg.decoder_width),
            nn.GELU(),
            *(ResidualMLP(cfg.decoder_width) for _ in range(cfg.residual_blocks)),
        )
        self.heads = nn.ModuleDict()
        own_bias = head_sharing == "per_type_bias"
        for name, (in_features, out_features) in head_shapes.items():
            self.heads[name] = nn.ModuleDict(
                {
                    "A": nn.Linear(cfg.decoder_width, rank * in_features, bias=not own_bias),
                    "B": nn.Linear(cfg.decoder_width, out_features * rank, bias=not own_bias),
                }
            )
            for head in self.heads[name].values():
                nn.init.zeros_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)
            if not own_bias:
                nn.init.kaiming_uniform_(
                    self.heads[name]["A"].bias.view(rank, in_features), a=math.sqrt(5)
                )
        # Per-key static factors (hybrid): same init as a head bias.
        self.biases = nn.ParameterDict()
        if own_bias:
            for key, (in_features, out_features) in shapes.items():
                a_bias = nn.Parameter(torch.empty(rank * in_features))
                nn.init.kaiming_uniform_(a_bias.view(rank, in_features), a=math.sqrt(5))
                self.biases[f"{key}__A"] = a_bias
                self.biases[f"{key}__B"] = nn.Parameter(torch.zeros(out_features * rank))

    def forward(self, summaries, lengths, feedback=None):
        encoded = self.encode_examples(summaries, lengths)
        condition = self.condition(self.pool_examples(encoded))
        return self.decode_factors(condition, feedback)

    def encode_examples(self, summaries, lengths):
        count = summaries.shape[0]
        if summaries.ndim != SUMMARY_NDIM or not 1 <= count <= self.max_examples:
            raise ValueError(f"Expected 1–{self.max_examples} conditioning examples")
        if lengths.shape != (count,) or torch.any(lengths <= 0):
            raise ValueError("Each conditioning example needs a positive prompt length")
        dtype = self.encoder[0].weight.dtype
        return self.encoder(
            torch.cat(
                [summaries.to(dtype), lengths.float().log1p().to(dtype).unsqueeze(-1)],
                dim=-1,
            )
        )

    @staticmethod
    def pool_examples(encoded):
        return torch.cat(
            [
                encoded.mean(0),
                encoded.std(0, correction=0),
                encoded.new_tensor([math.log1p(len(encoded))]),
            ]
        )

    def forward_cohorts(self, contexts, feedback=None):
        """Batch independent variable-sized sets; never pool across cohorts."""
        if not contexts:
            raise ValueError("At least one conditioning cohort is required")
        # Encode each set with its own size validation, then batch the large
        # shared condition/decoder/factor-head operations across all cohorts.
        pooled = torch.stack(
            [
                self.pool_examples(self.encode_examples(summaries, lengths))
                for summaries, lengths in contexts
            ]
        )
        return self.decode_factors(self.condition(pooled), feedback)

    def decode_factors(self, condition, feedback):
        leading = condition.shape[:-1]
        if self.feedback_dim:
            if feedback is None or feedback.shape != (*leading, self.feedback_dim):
                raise ValueError("Feedback generator requires a causal feedback vector")
            condition = condition + self.feedback_encoder(feedback.to(condition))
        elif feedback is not None:
            raise ValueError("This generator was trained without feedback inputs")
        result = {}
        for name, (in_features, out_features) in self.shapes.items():
            module_type, layer = self.layout[name]
            descriptor = torch.cat(
                [
                    condition,
                    self.module_embedding.weight[
                        self.module_types.index(module_type)
                    ].expand(*leading, -1),
                    self.layer_embedding.weight[layer].expand(*leading, -1),
                ],
                dim=-1,
            )
            decoded = self.decoder(descriptor)
            head = self.heads[
                name if self.head_sharing == "per_module" else module_type
            ]
            a, b = head["A"](decoded), head["B"](decoded)
            if self.head_sharing == "per_type_bias":
                a = a + self.biases[f"{name}__A"]
                b = b + self.biases[f"{name}__B"]
            result[name] = (
                a.view(*leading, self.rank, in_features),
                b.view(*leading, out_features, self.rank),
            )
        return result


class AdapterLinear(nn.Module):
    """Ordinary LoRA parameters can be substituted by torch.func.functional_call."""

    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        # Both generated and directly optimized factors use float32 optimizer
        # state, while autocast handles the frozen drafter's compute precision.
        self.a = nn.Parameter(
            torch.empty(
                rank, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.b = nn.Parameter(
            torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * functional.linear(
            functional.linear(self.dropout(x), self.a), self.b
        )


def leaf_targets(model, target_modules=None):
    """`{leaf name: path}` for single-layer drafters (one module per leaf name)."""
    selected = set(
        target_modules if target_modules is not None else ("o_proj", "v_proj")
    )
    matches = {name: [] for name in selected}
    for path, layer in model.named_modules():
        name = path.rsplit(".", 1)[-1]
        if name in matches:
            matches[name].append((path, layer))
    if not selected or any(
        len(items) != 1 or not isinstance(items[0][1], nn.Linear)
        for items in matches.values()
    ):
        raise ValueError("Expected exactly one linear layer per selected projection")
    return {name: items[0][0] for name, items in sorted(matches.items())}


def install_adapters(model, rank, alpha, dropout, *, target_modules=None, targets=None):
    """Replace each target linear by `AdapterLinear`; return `{key: path}`.

    `targets` (`{key: path}`) comes from the drafter family; without it the
    single-layer leaf-name selection of `target_modules` is used.
    """
    if targets is None:
        targets = leaf_targets(model, target_modules)
    elif target_modules is not None:
        raise ValueError("Pass either targets or target_modules, not both")
    if not targets:
        raise ValueError("No adapter targets selected")
    # Validate all targets before mutating the model.
    for key, path in targets.items():
        if "." in key:
            raise ValueError(f"Adapter key {key!r} must not contain dots")
        if not isinstance(model.get_submodule(path), nn.Linear):
            raise ValueError(f"Adapter target {path!r} is not nn.Linear")
    if len(set(targets.values())) != len(targets):
        raise ValueError("Adapter targets repeat a module path")
    model.requires_grad_(False)
    # Module-traversal order keeps the adapters' random initialisation
    # identical to the original single-layer implementation.
    wanted = set(targets.values())
    ordered = [path for path, _ in model.named_modules() if path in wanted]
    for path in ordered:
        parent, _, child = path.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        setattr(
            owner, child, AdapterLinear(model.get_submodule(path), rank, alpha, dropout)
        )
    return dict(sorted(targets.items()))


def factor_parameters(paths, factors):
    return {
        f"{paths[name]}.{suffix}": tensor
        for name, pair in factors.items()
        for suffix, tensor in zip(("a", "b"), pair, strict=True)
    }


def detached_factors(factors):
    """Leaf adapters collect per-microbatch gradients without retaining graphs."""
    return {
        name: tuple(value.detach().requires_grad_(True) for value in pair)
        for name, pair in factors.items()
    }


def backward_factors(generated, leaves):
    """Apply accumulated adapter derivatives to the generator exactly once."""
    values, gradients = [], []
    for name, pair in generated.items():
        for value, leaf in zip(pair, leaves[name], strict=True):
            if leaf.grad is not None:
                values.append(value)
                gradients.append(leaf.grad)
    if not values:
        raise ValueError("Drafter produced no gradients for generated LoRA factors")
    torch.autograd.backward(values, gradients)


def factor_loss(generated, target):
    """Equal weight for each saved A/B factor; no gauge alignment or conversion."""
    return torch.stack(
        [
            functional.l1_loss(value.float(), reference.to(value).float())
            for name, pair in generated.items()
            for value, reference in zip(pair, target[name], strict=True)
        ]
    ).mean()
