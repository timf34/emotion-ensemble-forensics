"""Residual-stream capture in the EMOTION-VECTOR layer convention.

    block b, 0 <= b < n_blocks : residual stream at the OUTPUT of decoder block b (0-based)
                                 == HF output_hidden_states[b + 1]

This is the convention of timf34/emotion-vectors-collection ("layer = b is the residual
stream after decoder block b"), so emotion vector [b] is read from exactly this tensor.
It is NOT the convention of grader-shifts-persona-space / Betley et al. (layer L = INPUT of
block L = hidden_states[L]): emotion block 40 is their layer 41. Captured with a forward
hook on block b.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def find_text_model(model: torch.nn.Module, n_layers: int | None = None) -> torch.nn.Module:
    """The text decoder (the unique module with a ModuleList `layers` and `embed_tokens`)."""
    matches = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and hasattr(module, "embed_tokens"):
            if n_layers is None or len(layers) == n_layers:
                matches.append((name or "<root>", module))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one decoder stack (n_layers={n_layers}), found "
                           f"{[(n, len(m.layers)) for n, m in matches]}")
    return matches[0][1]


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    raise RuntimeError(f"unsupported decoder block output type: {type(x)}")


@contextmanager
def capture_block_outputs(model: torch.nn.Module, blocks: list[int],
                          n_layers: int | None = None) -> Iterator[dict[int, torch.Tensor]]:
    """Context manager filling {block: hidden [batch, seq, hidden]} (block OUTPUTS) on each forward."""
    layers = find_text_model(model, n_layers).layers
    n = len(layers)
    store: dict[int, torch.Tensor] = {}
    handles = []
    for b in blocks:
        if not 0 <= b < n:
            raise ValueError(f"block {b} out of range 0..{n - 1}")

        def hook(module, args, output, b=b):
            store[b] = _first_tensor(output).detach()

        handles.append(layers[b].register_forward_hook(hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()
