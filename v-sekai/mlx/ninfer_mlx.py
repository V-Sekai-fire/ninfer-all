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

    def dense(self, name: str) -> mx.array:
        binding = self.bindings[name]
        record = self._record(binding["object"])
        data = self._raw(binding["object"])
        if record.format == "bf16":
            return mx.array(np.frombuffer(data, np.uint16).reshape(record.shape)).view(mx.bfloat16)
        if record.format == "fp32":
            return mx.array(np.frombuffer(data, np.float32).reshape(record.shape))
        raise ValueError(f"{name}: {record.format} is not a dense format")

    def _t2_rows(self, object_id: str, row_begin: int, row_end: int):
        record = self._record(object_id)
        if record.format != "t2_g128_fp16" or record.layout != "row_split_k128_v1":
            raise ValueError(f"{object_id}: {record.format}/{record.layout} is not t2 row-split")
        rows, columns = record.shape
        if columns % GROUP:
            raise ValueError(f"{object_id}: K={columns} is not a multiple of {GROUP}")
        groups = columns // GROUP
        code_bytes = rows * columns // 4
        data = self._raw(object_id)
        codes = np.frombuffer(data, np.uint8, count=code_bytes).reshape(rows, columns // 4)
        scales = np.frombuffer(data, np.float16, count=rows * groups,
                               offset=_align(code_bytes, 256)).reshape(rows, groups)
        # The byte remap runs on the GPU; mx.take gathers each code byte's MLX word.
        packed = mx.take(mx.array(_LUT), mx.array(codes[row_begin:row_end]))
        return packed.view(mx.uint32), mx.array(scales[row_begin:row_end])

    def t2(self, name: str):
        """Return (packed uint32 [N, K/16], fp16 scales [N, K/128]) for one logical binding."""
        binding = self.bindings[name]
        parts = binding.get("parts") or [{"object": binding["object"], "range": None}]
        words, scales = [], []
        for part in parts:
            record = self._record(part["object"])
            columns = record.shape[1]
            begin, end = part["range"] or (0, record.shape[0] * columns)
            if begin % columns or end % columns:
                raise ValueError(f"{name}: range {begin}..{end} does not cover whole rows")
            w, s = self._t2_rows(part["object"], begin // columns, end // columns)
            words.append(w)
            scales.append(s)
        return mx.concatenate(words), mx.concatenate(scales)


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


def _quantized(words: mx.array, scales: mx.array, prefix: str) -> dict:
    return {prefix + ".weight": words, prefix + ".scales": scales, prefix + ".biases": -scales}


def _offset_norm(text: NinferText, name: str) -> mx.array:
    return text.dense(name).astype(mx.float32) + 1.0


def text_weights(text: NinferText, config: dict) -> tuple[dict, dict]:
    """Map ninfer bindings onto mlx-lm Qwen3.5 parameter paths.

    Returns the weights and, for every quantized path, its input's Hadamard signs (or None).
    """
    weights, quantized = {}, {}
    base = "language_model.model."

    def shared_signs(names: list[str]):
        ids = {n: text.signs(n) for n in names}
        first = ids[names[0]]
        for n, v in ids.items():
            if (v is None) != (first is None) or (v is not None and not mx.array_equal(v, first)):
                raise ValueError(f"{names}: fused parts read differently rotated inputs")
        return first

    def linear(path: str, binding: str | list[str], signs_from: list[str] | None = None) -> None:
        names = [binding] if isinstance(binding, str) else binding
        pieces = [text.t2(n) for n in names]
        words = mx.concatenate([p[0] for p in pieces])
        scales = mx.concatenate([p[1] for p in pieces])
        weights.update(_quantized(words, scales, path))
        quantized[path] = shared_signs(signs_from or names)

    # The token table has no Use; its rows restore with the output head's hidden-width signs.
    linear(base + "embed_tokens", "text/token_embedding", signs_from=["text/output_head"])
    linear("language_model.lm_head", "text/output_head")
    weights[base + "norm.weight"] = _offset_norm(text, "text/final_norm")

    heads, head_dim = config["num_attention_heads"], config["head_dim"]
    for i in range(config["num_hidden_layers"]):
        src, dst = f"text/layers/{i}/", f"{base}layers.{i}."
        weights[dst + "input_layernorm.weight"] = _offset_norm(text, src + "input_norm")
        weights[dst + "post_attention_layernorm.weight"] = _offset_norm(text, src + "post_attention_norm")
        if (i + 1) % FULL_ATTENTION_INTERVAL:
            g, a = src + "gdn/", dst + "linear_attn."
            linear(a + "in_proj_qkv", [g + "query", g + "key", g + "value"])
            linear(a + "in_proj_z", g + "z")
            linear(a + "out_proj", g + "output")
            weights[a + "in_proj_a.weight"] = text.dense(g + "a_projection")
            weights[a + "in_proj_b.weight"] = text.dense(g + "b_projection")
            weights[a + "conv1d.weight"] = text.dense(g + "convolution").T[:, :, None]
            weights[a + "A_log"] = text.dense(g + "a_log")
            weights[a + "dt_bias"] = text.dense(g + "dt_bias")
            weights[a + "norm.weight"] = text.dense(g + "norm")
        else:
            g, a = src + "attention/", dst + "self_attn."
            qw, qs = text.t2(g + "query")
            gw, gs = text.t2(g + "gate")
            # mlx-lm splits each head's q_proj rows as [query rows, gate rows].
            words = mx.stack([qw.reshape(heads, head_dim, -1), gw.reshape(heads, head_dim, -1)], 1)
            scales = mx.stack([qs.reshape(heads, head_dim, -1), gs.reshape(heads, head_dim, -1)], 1)
            weights.update(_quantized(words.reshape(2 * heads * head_dim, -1),
                                      scales.reshape(2 * heads * head_dim, -1), a + "q_proj"))
            quantized[a + "q_proj"] = shared_signs([g + "query", g + "gate"])
            linear(a + "k_proj", g + "key")
            linear(a + "v_proj", g + "value")
            linear(a + "o_proj", g + "output")
            weights[a + "q_norm.weight"] = _offset_norm(text, g + "query_norm")
            weights[a + "k_norm.weight"] = _offset_norm(text, g + "key_norm")
        m, p = src + "mlp/", dst + "mlp."
        linear(p + "gate_proj", m + "gate")
        linear(p + "up_proj", m + "up")
        linear(p + "down_proj", m + "down")
    return weights, quantized


def _install_norms(model: nn.Module, eps: float) -> None:
    targets = ("input_layernorm", "post_attention_layernorm", "q_norm", "k_norm")
    for path, module in list(model.named_modules()):
        if isinstance(module, nn.RMSNorm) and (path.endswith(targets) or path == "language_model.model.norm"):
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

    weights, quantized = text_weights(text, config)
    nn.quantize(model, group_size=GROUP, bits=2,
                class_predicate=lambda p, m: p in quantized)
    model.load_weights(list(weights.items()), strict=True)
    for path_, module in model.named_modules():
        if path_ in quantized:
            module.__class__ = TernaryEmbedding if isinstance(module, nn.QuantizedEmbedding) else TernaryLinear
            # A plain attribute keeps the signs out of the parameter tree.
            module.__dict__["_signs"] = quantized[path_]
    mx.eval(model.parameters())
    model.eval()

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
