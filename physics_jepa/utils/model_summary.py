import torch
import torch.nn as nn
from typing import Iterable, Tuple, Union, Dict, Any, List

# Conv types we care about
CONV_TYPES = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
)

def _as_tuple(x) -> Tuple[int, ...]:
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return tuple(int(v) for v in x)
    return (int(x),)

def summarize_convs(
    model: nn.Module,
    example_input: Union[torch.Tensor, Tuple[torch.Tensor, ...], None] = None
) -> str:
    """
    Pretty-print a summary of all convolution layers in `model`.
    If `example_input` is given, also shows each layer's output shape.
    """
    # Optionally collect output shapes via forward hooks
    out_shapes: Dict[str, Any] = {}
    handles = []

    if example_input is not None:
        model_was_training = model.training
        model.eval()

        def make_hook(name):
            def hook(_m, _inp, out):
                def shape(o):
                    if isinstance(o, torch.Tensor):
                        return tuple(o.shape)
                    if isinstance(o, (list, tuple)):
                        return [shape(t) for t in o]
                    return type(o).__name__
                out_shapes[name] = shape(out)
            return hook

        for name, m in model.named_modules():
            if isinstance(m, CONV_TYPES):
                handles.append(m.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            if isinstance(example_input, tuple):
                model(*example_input)
            else:
                model(example_input)

        # Clean up hooks
        for h in handles:
            h.remove()
        model.train(model_was_training)

    # Build rows
    rows: List[Dict[str, Any]] = []
    for name, m in model.named_modules():
        if isinstance(m, CONV_TYPES):
            kind = m.__class__.__name__  # e.g., "Conv2d" or "ConvTranspose3d"
            k = _as_tuple(m.kernel_size)
            s = _as_tuple(m.stride)
            p = m.padding
            d = _as_tuple(m.dilation)
            in_ch = getattr(m, "in_channels", None)
            out_ch = getattr(m, "out_channels", None)
            groups = getattr(m, "groups", 1)
            bias = getattr(m, "bias", None) is not None
            params = sum(p.numel() for p in m.parameters())

            # Depthwise-ish tag for standard convs (not transposed)
            depthwise = (groups == in_ch and isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)))

            rows.append({
                "name": name or "(root)",
                "type": kind,
                "in→out": f"{in_ch}→{out_ch}",
                "kernel": k,
                "stride": s,
                "pad": p,
                "dil": d,
                "groups": groups,
                "bias": "✓" if bias else "✗",
                "depthwise": "✓" if depthwise else "",
                "params": params,
                "out_shape": out_shapes.get(name, "") if example_input is not None else "",
            })

    # Column order & widths
    cols = ["name", "type", "in→out", "kernel", "stride", "pad", "dil", "groups", "bias", "depthwise", "params", "out_shape"]
    # Simple fixed-width table
    col_widths = {
        "name": 36, "type": 16, "in→out": 10, "kernel": 12, "stride": 12, "pad": 12,
        "dil": 10, "groups": 8, "bias": 6, "depthwise": 9, "params": 12, "out_shape": 20
    }

    def fmt(cell, w):
        s = str(cell)
        return s if len(s) <= w else s[: w - 1] + "…"

    header = "  ".join(f"{c:{col_widths[c]}}" for c in cols)
    line = "-" * len(header)
    out = [header, line]
    for r in rows:
        out.append("  ".join(fmt(r[c], col_widths[c]).ljust(col_widths[c]) for c in cols))
    return "\n".join(out)

# --------------------------
# Usage
# --------------------------
# print(summarize_convs(model))                         # just layer hyperparams
# print(summarize_convs(model, example_input=x))        # also prints each layer's output shape

