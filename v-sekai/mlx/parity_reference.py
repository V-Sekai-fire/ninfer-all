"""Witness: every text tensor the .ninfer path loads equals an independent MLX conversion.

Reference: 3hs4n/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-mlx-2bit, made from the same PQ2_0
GGUF by a different converter (its own GDN value-head regrouping, norm offset and sign vectors).
Ternary matrices are compared after dequantisation, exactly; sign vectors exactly; dense tensors
exactly after widening both to fp32, except A_log: both converters recover it as log(-ssm_a) from
the GGUF's fp32 -exp(A_log), so it may differ by one fp32 ulp, which is reported. The negative control reorders one matrix's rows and swaps
two sign vectors, and both must be caught.
Run: pixi run -e mlx python v-sekai/mlx/parity_reference.py models/bonsai2-27b-heretic.ninfer sources/heretic-v2-mlx-2bit
"""

import sys
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ninfer_mlx  # noqa: E402


def dequant(w, prefix):
    return mx.dequantize(w[prefix + ".weight"], w[prefix + ".scales"], w[prefix + ".biases"],
                         group_size=128, bits=2).astype(mx.float32)


def main(artifact, reference):
    text = ninfer_mlx.NinferText(Path(artifact))
    config = text.config()
    ours, signs = ninfer_mlx.text_weights(text, config)
    theirs = mx.load(str(Path(reference) / "model.safetensors"))
    theirs = {k: v for k, v in theirs.items() if k.startswith("language_model.")}

    results, failures = Counter(), []
    quantized = set(signs)
    for prefix in sorted(quantized):
        same = mx.array_equal(dequant(ours, prefix), dequant(theirs, prefix)).item()
        s_ours, s_theirs = signs[prefix], theirs.get(prefix + ".signs")
        same_signs = (s_ours is None and s_theirs is None) or (
            s_ours is not None and s_theirs is not None and mx.array_equal(s_ours, s_theirs).item())
        results["ternary exact" if same else "ternary DIFFER"] += 1
        results["signs exact" if same_signs else "signs DIFFER"] += 1
        if not (same and same_signs):
            failures.append(prefix)
    dense = [k for k in ours if not any(k.startswith(p + ".") for p in quantized)]
    for key in dense:
        a, b = ours[key].astype(mx.float32), theirs[key].astype(mx.float32)
        if a.shape != b.shape:
            failures.append(f"{key} shape {a.shape} vs {b.shape}")
            continue
        same = mx.array_equal(a, b).item()
        if not same and key.endswith("A_log"):
            ulps = np.abs(np.asarray(a).view(np.int32).astype(np.int64)
                          - np.asarray(b).view(np.int32).astype(np.int64)).max()
            if ulps <= 1:
                results["A_log within 1 ulp"] += 1
                print(f"note {key}: {int(ulps)} fp32 ulp apart (log(-ssm_a) rounding)")
                continue
        results["dense exact" if same else "dense DIFFER"] += 1
        if not same:
            gap = mx.max(mx.abs(a - b)).item()
            failures.append(f"{key} max |diff| {gap:.3g}")
    extra = sorted(set(theirs) - set(ours) - {p + ".signs" for p in quantized})
    print(dict(results))
    print(f"reference tensors not loaded by the .ninfer path: {len(extra)} {extra[:5]}")
    for f in failures[:20]:
        print("DIFFER", f)

    # Negative controls.
    probe = "language_model.model.layers.3.self_attn.q_proj"
    shuffled = {k: v[::-1] if k.startswith(probe + ".") and k != probe + ".signs" else v
                for k, v in ours.items() if k.startswith(probe)}
    caught_rows = not mx.array_equal(dequant(shuffled, probe), dequant(theirs, probe)).item()
    a, b = "language_model.model.layers.0.mlp.gate_proj", "language_model.model.layers.0.mlp.down_proj"
    caught_signs = signs[a].shape != theirs[b + ".signs"].shape or \
        not mx.array_equal(signs[a], theirs[b + ".signs"]).item()
    print(("ok  " if caught_rows else "FAIL"), "control: reversed q_proj rows are rejected")
    print(("ok  " if caught_signs else "FAIL"), "control: a 5120 sign vector does not pass for 17408")
    bad = len(failures) + (not caught_rows) + (not caught_signs)
    print(f"{bad} failure(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
