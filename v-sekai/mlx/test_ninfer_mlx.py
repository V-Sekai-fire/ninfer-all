"""Witness: the MLX repack of t2_g128_fp16 decodes to the artifact's values exactly.

The oracle decodes the row-split planes straight from docs/maintainer/storage-layouts.md
(two's-complement lanes, fp16 scale per 128) without touching the loader's LUT.
Run: pixi run -e mlx python v-sekai/mlx/test_ninfer_mlx.py models/bonsai2-27b-heretic.ninfer
"""

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import ninfer_mlx  # noqa: E402


def oracle(text, object_id, rows):
    record = text.artifact.object(object_id)
    n, k = record.shape
    data = text.artifact.read_object(object_id)
    if record.format in ("q4_g64_fp16", "q5_g64_fp16", "q6_g64_fp16"):
        b, high = {"q4_g64_fp16": (4, 0), "q5_g64_fp16": (5, 1), "q6_g64_fp16": (6, 2)}[record.format]
        kp = (k + 127) // 128 * 128
        base_total = n * kp // 2
        base = np.frombuffer(data, np.uint8, count=base_total).reshape(n, kp // 2)
        u = np.stack([base & 15, base >> 4], -1).reshape(n, kp).astype(np.int64)
        at = (base_total + 255) // 256 * 256
        if high:
            hb = np.frombuffer(data, np.uint8, count=n * kp * high // 8, offset=at).reshape(n, -1)
            bitsarr = np.unpackbits(hb, axis=1, bitorder="little").reshape(n, kp, high)
            for j in range(high):
                u |= bitsarr[..., j].astype(np.int64) << (4 + j)
            at = (at + n * kp * high // 8 + 255) // 256 * 256
        q = np.where(u >= 1 << (b - 1), u - (1 << b), u).astype(np.float64)
        scales = np.frombuffer(data, np.float16, count=n * kp // 64, offset=at).reshape(n, kp // 64)
        full = q * np.repeat(scales.astype(np.float64), 64, axis=1)
        return full[rows[0]:rows[1], :k]
    if record.format == "q8_g32_fp16":
        codes = np.frombuffer(data, np.int8, count=n * k).reshape(n, k)[rows[0]:rows[1]].astype(np.float64)
        offset = (n * k + 255) // 256 * 256
        scales = np.frombuffer(data, np.float16, count=n * k // 32, offset=offset).reshape(n, k // 32)
        return codes * np.repeat(scales[rows[0]:rows[1]].astype(np.float64), 32, axis=1)
    code_bytes = n * k // 4
    packed = np.frombuffer(data, np.uint8, count=code_bytes).reshape(n, k // 4)[rows[0]:rows[1]]
    lanes = np.stack([(packed >> (2 * i)) & 3 for i in range(4)], -1).reshape(rows[1] - rows[0], k)
    signed = np.where(lanes >= 2, lanes.astype(np.int16) - 4, lanes).astype(np.float64)
    assert not np.any(lanes == 2), "word 0b10 is outside the t2 language"
    offset = (code_bytes + 255) // 256 * 256
    scales = np.frombuffer(data, np.float16, count=n * k // 128, offset=offset).reshape(n, k // 128)
    return signed * np.repeat(scales[rows[0]:rows[1]].astype(np.float64), 128, axis=1)


def mlx_decode(words, scales, biases=None, bits=2, group=128):
    """Dequantise in fp32 (fp16 scales would round q8's s * u)."""
    s = mx.array(scales).astype(mx.float32)
    b = -s if biases is None else mx.array(biases).astype(mx.float32)
    return np.asarray(mx.dequantize(mx.array(words), s, b, group_size=group, bits=bits))


def check(text, name, rows_limit=512):
    binding = text.bindings[name]
    parts = binding.get("parts") or [{"object": binding["object"], "range": None}]
    words, scales, biases, bits, group = text.rowsplit(name)
    expected, start = [], 0
    for part in parts:
        record = text.artifact.object(part["object"])
        k = record.shape[1]
        begin, end = part["range"] or (0, record.shape[0] * k)
        expected.append(oracle(text, part["object"], (begin // k, end // k)))
    expected = np.concatenate(expected)
    take = slice(0, min(rows_limit, expected.shape[0]))
    got = mlx_decode(words[take], scales[take], biases[take], bits, group)[:, :expected.shape[1]]
    return np.array_equal(got.astype(np.float64), expected[take]), expected[take]


def main(path):
    text = ninfer_mlx.NinferText(Path(path))
    names = [
        "text/token_embedding",
        "text/layers/0/gdn/key",        # second half of a shared object
        "text/layers/0/gdn/z",
        "text/layers/3/attention/gate",
        "text/layers/63/mlp/down",
        "text/output_head",
        "mtp/input_projection",          # q8_g32
        "mtp/layers/0/attention/gate",  # q8, third part of a shared object
        "mtp/layers/0/mlp/down",
        "proposal/head",
        "vision/patch_embedding",              # q6_g64
        "vision/layers/0/attention/key",       # q4, part of a fused parent
        "vision/layers/0/attention/output",    # q5
        "vision/layers/26/mlp/fc2",            # q5, K=4304 padded to 4480
        "vision/merger/fc1",                   # q8
    ]
    failures = 0
    for name in names:
        ok, _ = check(text, name)
        print(("ok  " if ok else "FAIL"), name)
        failures += not ok

    # Negative controls: a wrong lane map and a wrong row slice must both be caught.
    saved = ninfer_mlx._LUT.copy()
    ninfer_mlx._LUT[:] = ninfer_mlx._LUT[::-1]
    caught_lut = not check(text, "text/layers/0/gdn/z")[0]
    ninfer_mlx._LUT[:] = saved
    words, scales = text.t2("text/layers/0/gdn/key")
    ref = check(text, "text/layers/0/gdn/query")[1]
    caught_rows = not np.array_equal(mlx_decode(words[:512], scales[:512]).astype(np.float64), ref)
    print(("ok  " if caught_lut else "FAIL"), "control: reversed lane map is rejected")
    print(("ok  " if caught_rows else "FAIL"), "control: key rows do not pass for query rows")
    failures += (not caught_lut) + (not caught_rows)
    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(sys.argv[1])
