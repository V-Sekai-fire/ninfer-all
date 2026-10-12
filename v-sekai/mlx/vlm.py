"""Load a ninfer-all v3 .ninfer artifact into mlx-vlm's Qwen3.5 vision-language model on Metal.

Text and the shared heads use ninfer_mlx's exact repacks and Hadamard handling. Vision adds the
q4/q5/q6/q8 row-split tower; MTP and DFlash2 become mlx-vlm drafters (Qwen3_5MTPDraftModel,
DFlash2DraftModel) built from the artifact's own components, so mlx-vlm's speculative decoding
and OpenAI server run the artifact without other files. Token ids and frontend resources come from
the artifact's tokenizer and resources.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
import ninfer_mlx  # noqa: E402
from ninfer_mlx import NinferText, TernaryLinear, _Weights, _materialise  # noqa: E402

_VISION_TOKENS = {
    "image_token_id": "<|image_pad|>",
    "video_token_id": "<|video_pad|>",
    "vision_start_token_id": "<|vision_start|>",
    "vision_end_token_id": "<|vision_end|>",
}


class Fp32RMSNorm(nn.Module):
    """RMSNorm computed in fp32 with the stored weight (already 1 + w for zero-centred norms)."""

    def __init__(self, dims: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dims,), mx.float32)
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)


class PatchProjection(nn.Module):
    """The q6 patch embedding as a linear over (C, T, H, W)-flattened patches."""

    def __init__(self, hidden: int, patch_values: int) -> None:
        super().__init__()
        self.linear = nn.Linear(patch_values, hidden, bias=True)
        # mlx-vlm casts pixel values to patch_embed.proj.weight.dtype; keep that bf16.
        self.__dict__["weight"] = mx.zeros((1,), mx.bfloat16)


class PatchEmbedding(nn.Module):
    def __init__(self, hidden: int, patch_values: int) -> None:
        super().__init__()
        self.hidden_size = hidden
        self.proj = PatchProjection(hidden, patch_values)

    def __call__(self, x):
        flat = x.reshape(-1, self.proj.linear.weight.shape[-1] * 32 // self.proj.linear.bits)
        return self.proj.linear(flat).reshape(-1, self.hidden_size)


class PaddedLinear(nn.Module):
    """A linear whose stored K was padded by the layout; the input is zero-padded to match."""

    def __init__(self, k: int, k_pad: int, n: int) -> None:
        super().__init__()
        self.k, self.k_pad = k, k_pad
        self.inner = nn.Linear(k_pad, n, bias=True)

    def __call__(self, x):
        pad = [(0, 0)] * (x.ndim - 1) + [(0, self.k_pad - self.k)]
        return self.inner(mx.pad(x, pad))


def _swap_rmsnorms(module: nn.Module, eps: float) -> None:
    for path, sub in list(module.named_modules()):
        if type(sub) is nn.RMSNorm:
            parent_path, _, leaf = path.rpartition(".")
            parent = module
            for part in filter(None, parent_path.split(".")):
                parent = parent[int(part)] if isinstance(parent, list) else getattr(parent, part)
            setattr(parent, leaf, Fp32RMSNorm(sub.weight.shape[0], eps))


def _token_ids(tokenizer_json: bytes) -> dict:
    vocab = {}
    data = json.loads(tokenizer_json)
    for item in data.get("added_tokens", []):
        vocab[item["content"]] = item["id"]
    vocab.update({k: v for k, v in data["model"]["vocab"].items() if k in _VISION_TOKENS.values()})
    return {key: vocab[token] for key, token in _VISION_TOKENS.items()}


def hf_config(text: NinferText) -> dict:
    """A Qwen3_5ForConditionalGeneration config assembled from the artifact."""
    components = text.artifact.directory.components
    t = dict(components["text"]["config"])
    rope = dict(t.pop("rope_parameters"))
    rope.setdefault("rope_type", "default")
    rope.setdefault("mrope_interleaved", True)
    t.update(model_type="qwen3_5_text", rope_parameters=rope,
             full_attention_interval=ninfer_mlx.FULL_ATTENTION_INTERVAL, attn_output_gate=True)
    v = dict(components["vision"]["config"])
    v.update(out_hidden_size=t["hidden_size"], in_channels=3, hidden_act="gelu_pytorch_tanh",
             deepstack_visual_indexes=[], model_type="qwen3_5")
    config = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"],
              "text_config": t, "vision_config": v, "tie_word_embeddings": t["tie_word_embeddings"]}
    config.update(_token_ids(text.resource("tokenizer.json")))
    return config


def vision_weights(w: _Weights, config: dict) -> None:
    v = config["vision_config"]
    vt = "vision_tower."
    w.linear(vt + "patch_embed.proj.linear", "vision/patch_embedding")
    w.weights[vt + "patch_embed.proj.linear.bias"] = w.text.dense("vision/patch_embedding_bias")
    w.weights[vt + "pos_embed.weight"] = w.text.dense("vision/position_embedding")
    for i in range(v["depth"]):
        src, dst = f"vision/layers/{i}/", f"{vt}blocks.{i}."
        for norm in ("norm1", "norm2"):
            w.weights[f"{dst}{norm}.weight"] = w.text.dense(f"{src}{norm}_weight")
            w.weights[f"{dst}{norm}.bias"] = w.text.dense(f"{src}{norm}_bias")
        a = src + "attention/"
        w.linear(dst + "attn.qkv", [a + "query", a + "key", a + "value"])
        w.weights[dst + "attn.qkv.bias"] = mx.concatenate(
            [w.text.dense(a + r + "_bias") for r in ("query", "key", "value")])
        w.linear(dst + "attn.proj", a + "output")
        w.weights[dst + "attn.proj.bias"] = w.text.dense(a + "output_bias")
        w.linear(dst + "mlp.linear_fc1", src + "mlp/fc1")
        w.weights[dst + "mlp.linear_fc1.bias"] = w.text.dense(src + "mlp/fc1_bias")
        w.linear(dst + "mlp.linear_fc2.inner", src + "mlp/fc2")
        w.weights[dst + "mlp.linear_fc2.inner.bias"] = w.text.dense(src + "mlp/fc2_bias")
    w.weights[vt + "merger.norm.weight"] = w.text.dense("vision/merger/norm_weight")
    w.weights[vt + "merger.norm.bias"] = w.text.dense("vision/merger/norm_bias")
    for fc in ("fc1", "fc2"):
        w.linear(f"{vt}merger.linear_{fc}", f"vision/merger/{fc}")
        w.weights[f"{vt}merger.linear_{fc}.bias"] = w.text.dense(f"vision/merger/{fc}_bias")


def _prepare_vision_modules(model, config: dict) -> None:
    v = config["vision_config"]
    tower = model.vision_tower
    tower.patch_embed = PatchEmbedding(v["hidden_size"], 3 * v["temporal_patch_size"] * v["patch_size"] ** 2)
    k = v["intermediate_size"]
    for block in tower.blocks:
        block.mlp.linear_fc2 = PaddedLinear(k, ninfer_mlx._align(k, 128), v["hidden_size"])


def _guard_fused_decode() -> None:
    """mlx-vlm fuses four QuantizedLinears for decode; that would skip a Hadamard-rotated input."""
    from mlx_vlm.models.qwen3_5 import language

    original = language._decode_quantized_linears_fused
    if getattr(original, "_ninfer_guard", False):
        return

    def guarded(linears, x):
        if any(isinstance(l, TernaryLinear) and l.__dict__.get("_signs") is not None for l in linears):
            return None
        return original(linears, x)

    guarded._ninfer_guard = True
    language._decode_quantized_linears_fused = guarded


def _resource_dir(text: NinferText, config: dict) -> Path:
    artifact_id = text.artifact.artifact_id
    tag = artifact_id.hex() if isinstance(artifact_id, bytes) else str(artifact_id)
    out = Path(tempfile.gettempdir()) / f"ninfer-vlm-{tag}"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        (out / name).write_bytes(text.resource(name))
    for name, rid in text.artifact.directory.components["vision"]["resources"].items():
        (out / name).write_bytes(text.artifact.read_object(rid))
    (out / "config.json").write_text(json.dumps(config))
    return out


def load(path: str | Path, **_ignored):
    """(mlx-vlm Qwen3.5 Model, processor) for a .ninfer artifact, text and vision, on the GPU."""
    from mlx_vlm.utils import get_model_and_args, load_processor

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise RuntimeError("inference runs on the Metal GPU; none is available")
    text = NinferText(Path(path))
    text.check_layer_types()
    config = hf_config(text)
    module, _ = get_model_and_args(config)
    model_config = module.ModelConfig.from_dict(config)
    model_config.text_config = module.TextConfig.from_dict(config["text_config"])
    model_config.vision_config = module.VisionConfig.from_dict(config["vision_config"])
    model = module.Model(model_config)
    eps = config["text_config"]["rms_norm_eps"]
    _swap_rmsnorms(model, eps)
    _prepare_vision_modules(model, config)
    weights, quantized = ninfer_mlx.text_weights(text, text.config())
    w = _Weights(text)
    w.weights, w.quantized = weights, quantized
    vision_weights(w, config)
    _materialise(model, w.weights, w.quantized)
    _guard_fused_decode()
    resources = _resource_dir(text, config)
    generation = json.loads(text.resource("generation_config.json"))
    processor = load_processor(resources, eos_token_ids=generation.get("eos_token_id"))
    model.__dict__["_ninfer_text"] = text
    model.__dict__["_ninfer_config"] = config
    return model, processor


def load_mtp_drafter(model):
    """mlx-vlm's Qwen3.5 MTP drafter carrying the artifact's MTP head (zero-centred norms)."""
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.config import Qwen3_5MTPConfig
    from mlx_vlm.speculative.drafters.qwen3_5_mtp.qwen3_5_mtp import Qwen3_5MTPDraftModel

    text, config = model._ninfer_text, model._ninfer_config
    drafter = Qwen3_5MTPDraftModel(Qwen3_5MTPConfig.from_dict({"text_config": config["text_config"]}))
    _swap_rmsnorms(drafter, config["text_config"]["rms_norm_eps"])
    w = _Weights(text)
    w.linear("fc", "mtp/input_projection")
    w.offset_norm("pre_fc_norm_embedding.weight", "mtp/embedding_norm")
    w.offset_norm("pre_fc_norm_hidden.weight", "mtp/hidden_norm")
    w.offset_norm("norm.weight", "mtp/final_norm")
    w.offset_norm("layers.0.input_layernorm.weight", "mtp/layers/0/input_norm")
    w.offset_norm("layers.0.post_attention_layernorm.weight", "mtp/layers/0/post_attention_norm")
    w.attention_block("mtp/layers/0/", "layers.0.", text.config())
    w.ffn_block("mtp/layers/0/", "layers.0.")
    _materialise(drafter, w.weights, w.quantized)
    return drafter


def load_dflash2_drafter(model, block_size: int = 6):
    """mlx-vlm's DFlash2 drafter carrying the artifact's DFlash2 head (plain norms)."""
    from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

    text, config = model._ninfer_text, model._ninfer_config
    raw = dict(text.artifact.directory.components["dflash2"]["config"])
    tc = config["text_config"]
    raw["dflash_config"] = dict(raw["dflash_config"], block_size=block_size)
    raw.update(hidden_size=tc["hidden_size"], vocab_size=tc["vocab_size"],
               num_target_layers=tc["num_hidden_layers"], use_sliding_window=True,
               tie_word_embeddings=False)
    drafter = DFlash2DraftModel(DFlash2Config.from_dict(raw))
    _swap_rmsnorms(drafter, raw["rms_norm_eps"])
    w = _Weights(text)
    dense = text.dense
    w.linear("fc", "dflash2/feature_projection")
    w.weights["hidden_norm.weight"] = dense("dflash2/context_norm").astype(mx.float32)
    w.weights["norm.weight"] = dense("dflash2/final_norm").astype(mx.float32)
    for i in range(raw["num_hidden_layers"]):
        src, dst = f"dflash2/layers/{i}/", f"layers.{i}."
        a = src + "attention/"
        w.linear(dst + "self_attn.q_proj", a + "query")
        w.linear(dst + "self_attn.k_proj", a + "key")
        w.linear(dst + "self_attn.v_proj", a + "value")
        w.linear(dst + "self_attn.o_proj", a + "output")
        w.weights[dst + "self_attn.q_norm.weight"] = dense(a + "query_norm").astype(mx.float32)
        w.weights[dst + "self_attn.k_norm.weight"] = dense(a + "key_norm").astype(mx.float32)
        w.weights[dst + "input_layernorm.weight"] = dense(src + "input_norm").astype(mx.float32)
        w.weights[dst + "post_attention_layernorm.weight"] = dense(src + "post_attention_norm").astype(mx.float32)
        w.ffn_block(src, dst)
        for conv in ("attention_conv", "mlp_conv"):
            w.weights[f"{dst}{conv}.base_kernel"] = dense(f"{src}{conv}/base_kernel")
            w.weights[f"{dst}{conv}.kernel_projection.weight"] = dense(f"{src}{conv}/kernel_projection")
    cs = "candidate_selector."
    w.weights[cs + "hidden_projection.weight"] = dense("dflash2/candidate_selector/hidden_projection")
    w.weights[cs + "predecessor_codebook.weight"] = dense("dflash2/candidate_selector/predecessor_codebook")
    w.weights[cs + "successor_codebook.weight"] = dense("dflash2/candidate_selector/successor_codebook")
    _materialise(drafter, w.weights, w.quantized)
    return drafter


_TARGET = {}


def _patch_server() -> None:
    """Route mlx-vlm's server loads for .ninfer paths and ninfer:<drafter> names to this module."""
    import mlx_vlm.server.generation as generation
    import mlx_vlm.speculative.drafters as drafters

    original_load, original_drafter = generation.load, drafters.load_drafter

    def load_any(path, *args, **kwargs):
        if str(path).endswith(".ninfer"):
            model, processor = load(path)
            _TARGET["model"] = model
            return model, processor
        return original_load(path, *args, **kwargs)

    def load_drafter_any(path, kind=None, **kwargs):
        if str(path) == "ninfer:mtp":
            return load_mtp_drafter(_TARGET["model"]), "mtp"
        if str(path) == "ninfer:dflash2":
            block = int(os.environ.get("MLX_VLM_DRAFT_BLOCK_SIZE", "6"))
            return load_dflash2_drafter(_TARGET["model"], block_size=block), "dflash"
        return original_drafter(path, kind=kind, **kwargs)

    generation.load = load_any
    drafters.load_drafter = load_drafter_any


def main(argv=None) -> None:
    """vlm.py serve <mlx_vlm.server flags>: the OpenAI server on a .ninfer artifact (text and images).

    --draft-model ninfer:mtp or ninfer:dflash2 drafts with the artifact's own heads.
    """
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["serve"]:
        raise SystemExit("usage: vlm.py serve --model <artifact.ninfer> [--draft-model ninfer:mtp|ninfer:dflash2] ...")
    from mlx_vlm.server import cli

    _patch_server()
    sys.argv = ["mlx_vlm.server", *argv[1:]]
    cli.main()


if __name__ == "__main__":
    main()
