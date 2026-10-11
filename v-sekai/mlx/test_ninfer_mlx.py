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
    code_bytes = n * k // 4
    packed = np.frombuffer(data, np.uint8, count=code_bytes).reshape(n, k // 4)[rows[0]:rows[1]]
    lanes = np.stack([(packed >> (2 * i)) & 3 for i in range(4)], -1).reshape(rows[1] - rows[0], k)
    signed = np.where(lanes >= 2, lanes.astype(np.int16) - 4, lanes).astype(np.float64)
    assert not np.any(lanes == 2), "word 0b10 is outside the t2 language"
    offset = (code_bytes + 255) // 256 * 256
    scales = np.frombuffer(data, np.float16, count=n * k // 128, offset=offset).reshape(n, k // 128)
    return signed * np.repeat(scales[rows[0]:rows[1]].astype(np.float64), 128, axis=1)


def mlx_decode(words, scales):
    s = mx.array(scales)
    return np.asarray(mx.dequantize(mx.array(words), s, -s, group_size=128, bits=2).astype(mx.float32))


def check(text, name, rows_limit=512):
    binding = text.bindings[name]
    parts = binding.get("parts") or [{"object": binding["object"], "range": None}]
    words, scales = text.t2(name)
    expected, start = [], 0
    for part in parts:
        record = text.artifact.object(part["object"])
        k = record.shape[1]
        begin, end = part["range"] or (0, record.shape[0] * k)
        expected.append(oracle(text, part["object"], (begin // k, end // k)))
    expected = np.concatenate(expected)
    take = slice(0, min(rows_limit, expected.shape[0]))
    got = mlx_decode(words[take], scales[take])
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
