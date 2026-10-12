"""Serve a ninfer-all v3 .ninfer text model on Apple silicon through MLX.

The artifact's t2_g128_fp16 codes are repacked into MLX 2-bit affine words with
scale s and bias -s, which decode to the same values without rounding. Uses that
carry hadamard_signs read hadamard_transform(x, s); see docs/maintainer/qwen3_5-model.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.artifact.reader import Artifact  # noqa: E402

GROUP = 128
HADAMARD_BLOCK = 1024
FULL_ATTENTION_INTERVAL = 4

# t2 stores a two's-complement word per lane (0 -> 0, 1 -> +1, 3 -> -1); MLX wants t + 1.
_T2_TO_MLX = np.array([1, 2, 0, 0], np.uint8)
_LUT = np.zeros(256, np.uint8)
for _byte in range(256):
    _LUT[_byte] = sum(int(_T2_TO_MLX[(_byte >> (2 * lane)) & 3]) << (2 * lane) for lane in range(4))


# format -> (bits, group, high-plane bits per lane)
_ROW_SPLIT = {
    "t2_g128_fp16": (2, 128, 0),
    "q4_g64_fp16": (4, 64, 0),
    "q5_g64_fp16": (5, 64, 1),
    "q6_g64_fp16": (6, 64, 2),
    "q8_g32_fp16": (8, 32, 0),
}


def _pack_bitstream(u: mx.array, bits: int) -> mx.array:
    """Pack [N, K] b-bit words LSB-first into MLX's uint32 layout (8 lanes per b bytes)."""
    n, k = u.shape
    lanes = u.astype(mx.uint64).reshape(n, k // 8, 8)
    packed = mx.zeros((n, k // 8), mx.uint64)
    for i in range(8):
        packed = packed | (lanes[..., i] << (bits * i))
    as_bytes = mx.stack([(packed >> (8 * j)) & 0xFF for j in range(bits)], axis=-1).astype(mx.uint8)
    return as_bytes.reshape(n, k * bits // 8).view(mx.uint32)


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class NinferText:
    """Reads the text component of a .ninfer artifact into MLX arrays."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.artifact = Artifact.open(self.path)
        self.bindings = self.artifact.directory.bindings
        self.text = self.artifact.directory.components["text"]

    def config(self) -> dict:
        raw = dict(self.text["config"])
        rope = dict(raw.pop("rope_parameters"))
        rope.setdefault("rope_type", "default")
        rope.setdefault("mrope_interleaved", True)
        raw["rope_parameters"] = rope
        raw.pop("layer_types")
        raw["model_type"] = "qwen3_5"
        raw["full_attention_interval"] = FULL_ATTENTION_INTERVAL
        return raw

    def check_layer_types(self) -> None:
        for index, kind in enumerate(self.text["config"]["layer_types"]):
            full = (index + 1) % FULL_ATTENTION_INTERVAL == 0
            if kind != ("full_attention" if full else "linear_attention"):
                raise ValueError(f"layer {index} is {kind}; MLX Qwen3.5 expects every fourth layer full")

    def signs(self, parameter: str) -> mx.array | None:
        """The hadamard_signs auxiliary every Use of `parameter` shares, or None if unrotated."""
        found = set()
        for use in self.artifact.directory.uses:
            if use["parameter"] == parameter:
                aux = use.get("auxiliaries", {}).get("hadamard_signs")
                found.add(aux["object"] if aux else None)
        if len(found) != 1:
            raise ValueError(f"{parameter}: Uses disagree on hadamard_signs: {found}")
        object_id = found.pop()
        if object_id is None:
            return None
        raw = np.frombuffer(self._raw(object_id), np.uint16)
        return mx.array(raw).view(mx.bfloat16).astype(mx.float32)

    def resource(self, name: str) -> bytes:
        return self.artifact.read_object(self.text["resources"][name])

    @lru_cache(maxsize=4)
    def _raw(self, object_id: str) -> bytes:
        return self.artifact.read_object(object_id)

    def _record(self, object_id: str):
        return self.artifact.object(object_id)

    def _dense_object(self, object_id: str) -> mx.array:
        record = self._record(object_id)
        data = self._raw(object_id)
        if record.format == "bf16":
            return mx.array(np.frombuffer(data, np.uint16).reshape(record.shape)).view(mx.bfloat16)
        if record.format == "fp32":
            return mx.array(np.frombuffer(data, np.float32).reshape(record.shape))
        raise ValueError(f"{object_id}: {record.format} is not a dense format")

    def dense(self, name: str) -> mx.array:
        """A bf16/fp32 binding; a parts binding joins its element ranges into one flat array."""
        binding = self.bindings[name]
        if "object" in binding:
            return self._dense_object(binding["object"])
        pieces = [self._dense_object(p["object"]).reshape(-1)[p["range"][0]:p["range"][1]]
                  for p in binding["parts"]]
        return mx.concatenate(pieces)

    def _rows(self, object_id: str, row_begin: int, row_end: int):
        """Rows [row_begin, row_end) of a row-split object as MLX affine (words, scales, biases, bits, group).

        Codes are signed two's complement; MLX stores u = q + 2^(b-1) (q XOR 2^(b-1) in b bits) with
        bias -2^(b-1) s, except t2, whose {-1,0,1} becomes u = q + 1 with bias -s. Each decodes to
        q * s exactly. K is the layout's padded width (a multiple of 128); padding decodes to zero.
        The bit work runs on the GPU.
        """
        record = self._record(object_id)
        if record.layout != "row_split_k128_v1":
            raise ValueError(f"{object_id}: layout {record.layout} is not row-split")
        fmt = record.format
        if fmt not in _ROW_SPLIT:
            raise ValueError(f"{object_id}: {fmt} has no MLX repack")
        bits, group, high_bits = _ROW_SPLIT[fmt]
        rows, columns = record.shape
        kp = _align(columns, 128)
        base_bits = {2: 2, 8: 8}.get(bits, 4)
        base_total = rows * kp * base_bits // 8
        high_total = rows * kp * high_bits // 8
        high_at = _align(base_total, 256)
        scales_at = _align(high_at + high_total, 256) if high_bits else high_at
        data = self._raw(object_id)
        groups = kp // group
        scales = mx.array(np.frombuffer(data, np.float16, count=rows * groups,
                                        offset=scales_at).reshape(rows, groups)[row_begin:row_end])
        base = mx.array(np.frombuffer(data, np.uint8, count=base_total).reshape(rows, -1)[row_begin:row_end])
        if bits == 2:
            words = mx.take(mx.array(_LUT), base).view(mx.uint32)
            return words, scales, -scales, bits, group
        if bits == 8:
            words = (base ^ 0x80).view(mx.uint32)
            return words, scales, scales * -128.0, bits, group
        n = row_end - row_begin
        lanes = mx.stack([base & 0x0F, base >> 4], axis=-1).reshape(n, kp).astype(mx.uint32)
        if high_bits:
            high = mx.array(np.frombuffer(data, np.uint8, count=high_total, offset=high_at)
                            .reshape(rows, -1)[row_begin:row_end])
            stream = mx.stack([(high >> i) & 1 for i in range(8)], axis=-1).reshape(n, kp, high_bits)
            for j in range(high_bits):
                lanes = lanes | (stream[..., j].astype(mx.uint32) << (4 + j))
        u = lanes ^ (1 << (bits - 1))
        return _pack_bitstream(u, bits), scales, scales * -float(1 << (bits - 1)), bits, group

    def rowsplit(self, name: str):
        """One logical binding as (words, scales, biases, bits, group), joining its parts by rows."""
        binding = self.bindings[name]
        parts = binding.get("parts") or [{"object": binding["object"], "range": None}]
        pieces = []
        for part in parts:
            record = self._record(part["object"])
            columns = record.shape[1]
            begin, end = part["range"] or (0, record.shape[0] * columns)
            if begin % columns or end % columns:
                raise ValueError(f"{name}: range {begin}..{end} does not cover whole rows")
            pieces.append(self._rows(part["object"], begin // columns, end // columns))
        if len({(p[3], p[4]) for p in pieces}) != 1:
            raise ValueError(f"{name}: parts mix formats")
        return (mx.concatenate([p[0] for p in pieces]), mx.concatenate([p[1] for p in pieces]),
                mx.concatenate([p[2] for p in pieces]), pieces[0][3], pieces[0][4])

    def t2(self, name: str):
        """(words, scales) of a t2_g128_fp16 binding."""
        words, scales, _, bits, _ = self.rowsplit(name)
        if bits != 2:
            raise ValueError(f"{name} is not t2_g128_fp16")
        return words, scales

    def int32(self, name: str) -> mx.array:
        return mx.array(np.frombuffer(self._raw(self.bindings[name]["object"]), np.int32))


def hadamard_transform(x: mx.array, signs: mx.array) -> mx.array:
    """Per 1024 block: multiply by the signs, then the normalised Sylvester WHT, in fp32; bf16 out."""
    shape = x.shape
    v = (x.astype(mx.float32) * signs).reshape(*shape[:-1], -1, HADAMARD_BLOCK)
    return mx.hadamard_transform(v, scale=HADAMARD_BLOCK ** -0.5).reshape(shape).astype(x.dtype)


class TernaryLinear(nn.QuantizedLinear):
    """Rotates the input when the Use is Hadamard-rotated; writes bf16 like ninfer's linears."""

    def __call__(self, x):
        signs = self.__dict__.get("_signs")
        rotated = x if signs is None else hadamard_transform(x, signs)
        return super().__call__(rotated).astype(x.dtype)


class TernaryEmbedding(nn.QuantizedEmbedding):
    """Rows stored rotated go back to the primal basis: signs * WHT(code * scale), bf16 out."""

    def __call__(self, x):
        rows = super().__call__(x).astype(mx.float32)
        signs = self.__dict__.get("_signs")
        if signs is not None:
            shape = rows.shape
            blocks = rows.reshape(*shape[:-1], -1, HADAMARD_BLOCK)
            rows = mx.hadamard_transform(blocks, scale=HADAMARD_BLOCK ** -0.5).reshape(shape) * signs
        return rows.astype(mx.bfloat16)


class OffsetRMSNorm(nn.Module):
    """Qwen3.5 RMSNorm: normalise in fp32, multiply by (1 + w) in fp32, return the input dtype."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dims,), mx.float32)
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)


def _quantized(words: mx.array, scales: mx.array, biases: mx.array, prefix: str) -> dict:
    return {prefix + ".weight": words, prefix + ".scales": scales, prefix + ".biases": biases}


def _offset_norm(text: NinferText, name: str) -> mx.array:
    return text.dense(name).astype(mx.float32) + 1.0


class _Weights:
    """Collects MLX parameters; quantized[path] = (hadamard signs or None, bits, group)."""

    def __init__(self, text: NinferText) -> None:
        self.text, self.weights, self.quantized = text, {}, {}

    def shared_signs(self, names: list[str]):
        found = {n: self.text.signs(n) for n in names}
        first = found[names[0]]
        for n, v in found.items():
            if (v is None) != (first is None) or (v is not None and not mx.array_equal(v, first)):
                raise ValueError(f"{names}: fused parts read differently rotated inputs")
        return first

    def linear(self, path: str, binding: str | list[str], signs_from: list[str] | None = None) -> None:
        names = [binding] if isinstance(binding, str) else binding
        pieces = [self.text.rowsplit(n) for n in names]
        if len({(p[3], p[4]) for p in pieces}) != 1:
            raise ValueError(f"{names}: fused parts mix formats")
        cat = [mx.concatenate([p[i] for p in pieces]) for i in range(3)]
        self.weights.update(_quantized(*cat, path))
        self.quantized[path] = (self.shared_signs(signs_from or names), pieces[0][3], pieces[0][4])

    def query_gate(self, path: str, query: str, gate: str, heads: int, head_dim: int) -> None:
        # mlx-lm splits each head's q_proj rows as [query rows, gate rows].
        q, g = self.text.rowsplit(query), self.text.rowsplit(gate)
        stacked = [mx.stack([q[i].reshape(heads, head_dim, -1), g[i].reshape(heads, head_dim, -1)], 1)
                   .reshape(2 * heads * head_dim, -1) for i in range(3)]
        self.weights.update(_quantized(*stacked, path))
        self.quantized[path] = (self.shared_signs([query, gate]), q[3], q[4])

    def offset_norm(self, path: str, binding: str) -> None:
        self.weights[path] = _offset_norm(self.text, binding)

    def attention_block(self, src: str, dst: str, config: dict) -> None:
        g, a = src + "attention/", dst + "self_attn."
        self.query_gate(a + "q_proj", g + "query", g + "gate", config["num_attention_heads"], config["head_dim"])
        self.linear(a + "k_proj", g + "key")
        self.linear(a + "v_proj", g + "value")
        self.linear(a + "o_proj", g + "output")
        self.offset_norm(a + "q_norm.weight", g + "query_norm")
        self.offset_norm(a + "k_norm.weight", g + "key_norm")

    def ffn_block(self, src: str, dst: str) -> None:
        m, p = src + "mlp/", dst + "mlp."
        self.linear(p + "gate_proj", m + "gate")
        self.linear(p + "up_proj", m + "up")
        self.linear(p + "down_proj", m + "down")


def text_weights(text: NinferText, config: dict) -> tuple[dict, dict]:
    """Map ninfer text bindings onto mlx-lm Qwen3.5 parameter paths."""
    w = _Weights(text)
    base = "language_model.model."
    # The token table has no Use; its rows restore with the output head's hidden-width signs.
    w.linear(base + "embed_tokens", "text/token_embedding", signs_from=["text/output_head"])
    w.linear("language_model.lm_head", "text/output_head")
    w.offset_norm(base + "norm.weight", "text/final_norm")
    for i in range(config["num_hidden_layers"]):
        src, dst = f"text/layers/{i}/", f"{base}layers.{i}."
        w.offset_norm(dst + "input_layernorm.weight", src + "input_norm")
        w.offset_norm(dst + "post_attention_layernorm.weight", src + "post_attention_norm")
        if (i + 1) % FULL_ATTENTION_INTERVAL:
            g, a = src + "gdn/", dst + "linear_attn."
            w.linear(a + "in_proj_qkv", [g + "query", g + "key", g + "value"])
            w.linear(a + "in_proj_z", g + "z")
            w.linear(a + "out_proj", g + "output")
            w.weights[a + "in_proj_a.weight"] = text.dense(g + "a_projection")
            w.weights[a + "in_proj_b.weight"] = text.dense(g + "b_projection")
            w.weights[a + "conv1d.weight"] = text.dense(g + "convolution").T[:, :, None]
            w.weights[a + "A_log"] = text.dense(g + "a_log")
            w.weights[a + "dt_bias"] = text.dense(g + "dt_bias")
            w.weights[a + "norm.weight"] = text.dense(g + "norm")
        else:
            w.attention_block(src, dst, config)
        w.ffn_block(src, dst)
    return w.weights, w.quantized


def _materialise(module: nn.Module, weights: dict, quantized: dict) -> None:
    def predicate(path, _module):
        if path not in quantized:
            return False
        _, bits, group = quantized[path]
        return {"bits": bits, "group_size": group}

    nn.quantize(module, class_predicate=predicate)
    module.load_weights(list(weights.items()), strict=True)
    for path, sub in module.named_modules():
        if path in quantized:
            sub.__class__ = TernaryEmbedding if isinstance(sub, nn.QuantizedEmbedding) else TernaryLinear
            # A plain attribute keeps the signs out of the parameter tree.
            sub.__dict__["_signs"] = quantized[path][0]
    mx.eval(module.parameters())
    module.eval()


def _install_norms(model: nn.Module, eps: float) -> None:
    targets = ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm")
    for path, module in list(model.named_modules()):
        if isinstance(module, nn.RMSNorm) and (path.endswith(targets) or path == "language_model.model.norm"
                                                or path.endswith("_norm")):
            parent_path, _, leaf = path.rpartition(".")
            parent = model
            for part in parent_path.split("."):
                parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
            setattr(parent, leaf, OffsetRMSNorm(module.weight.shape[0], eps))


def _resource_dir(text: NinferText) -> Path:
    artifact_id = text.artifact.artifact_id
    cache = Path(tempfile.gettempdir()) / f"ninfer-mlx-{artifact_id.hex() if isinstance(artifact_id, bytes) else artifact_id}"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        (cache / name).write_bytes(text.resource(name))
    return cache


def load(path: str | Path, tokenizer_config: dict | None = None, **_ignored):
    """Load a .ninfer text model as an (mlx-lm Model, TokenizerWrapper) pair."""
    from mlx_lm.models import qwen3_5
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("ninfer_mlx runs inference on the Metal GPU; none is available")
    text = NinferText(Path(path))
    text.check_layer_types()
    config = text.config()
    model = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(config))
    _install_norms(model, config["rms_norm_eps"])

    _materialise(model, *text_weights(text, config))

    resources = _resource_dir(text)
    generation = json.loads(text.resource("generation_config.json"))
    tokenizer = load_tokenizer(resources, tokenizer_config, eos_token_ids=generation.get("eos_token_id"))
    return model, tokenizer


def _patched_load(original):
    def wrapper(path_or_repo, *args, **kwargs):
        if str(path_or_repo).endswith(".ninfer"):
            return load(path_or_repo, kwargs.get("tokenizer_config"))
        return original(path_or_repo, *args, **kwargs)
    return wrapper


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate", help="greedy text generation from one prompt")
    gen.add_argument("--model", required=True)
    gen.add_argument("--prompt", required=True)
    gen.add_argument("--max-tokens", type=int, default=128)
    gen.add_argument("--no-chat", action="store_true", help="feed the prompt without the chat template")
    sub.add_parser("serve", help="OpenAI-compatible server; the remaining flags go to mlx_lm.server")
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["serve"]:
        import mlx_lm.server as server

        server.load = _patched_load(server.load)
        sys.argv = ["mlx_lm.server", *argv[1:]]
        server.main()
        return
    args = parser.parse_args(argv)

    if args.command == "generate":
        from mlx_lm import generate

        model, tokenizer = load(args.model)
        prompt = args.prompt
        if not args.no_chat:
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        print(generate(model, tokenizer, prompt, max_tokens=args.max_tokens, verbose=True))


if __name__ == "__main__":
    main()
