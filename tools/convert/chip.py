"""The chip tower component (docs/maintainer/chip-tower.md)."""

from __future__ import annotations

from .config_fields import _f32, _fixed, _positive
from .methods import exact_direct

ARCHITECTURE = "ChipTower"
LAYOUT = "rv1-tower"
GDN_FIELDS = (
    "linear_num_key_heads",
    "linear_key_head_dim",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
)


def _required(config: dict, key: str, value, label: str) -> None:
    if key not in config:
        raise ValueError(f"{label}.{key}: expected {value!r}, got nothing")
    _fixed(config, key, value, label)


def _token_ids(values, limit: int, label: str, count: int | None = None) -> list[int]:
    if (
        not isinstance(values, list)
        or (count is not None and len(values) != count)
        or any(type(v) is not int or not 0 <= v < limit for v in values)
    ):
        raise ValueError(f"chip.{label} must list token ids below {limit}")
    return list(values)


def chip_config(raw: dict, target: dict) -> dict:

    _required(raw, "architectures", [ARCHITECTURE], "chip")
    _required(raw, "model_type", "chip_tower", "chip")
    _fixed(raw, "hidden_act", "silu", "chip")
    if "linear_attention" not in target["layer_types"]:
        raise ValueError("chip: the text model has no Gated DeltaNet layers")
    result = {"architectures": [ARCHITECTURE], "model_type": "chip_tower"}
    for key in ("hidden_size", "vocab_size", "intermediate_size"):
        result[key] = _positive(raw.get(key), "chip." + key)
    for key in GDN_FIELDS:
        _required(raw, key, target[key], "chip")
        result[key] = target[key]
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), "chip.rms_norm_eps")
    if result["rms_norm_eps"] != target["rms_norm_eps"]:
        raise ValueError("chip.rms_norm_eps differs from the text model's")
    layers = raw.get("layer_types")
    if (
        not isinstance(layers, list)
        or len(layers) < 2
        or layers[-1] != "full_attention"
        or any(kind != "linear_attention" for kind in layers[:-1])
    ):
        raise ValueError(
            "chip.layer_types must be linear_attention layers and one closing full_attention"
        )
    result["num_hidden_layers"] = len(layers) - 1
    result["layer_types"] = layers[:-1]
    chip = raw.get("chip_config")
    if not isinstance(chip, dict):
        raise ValueError("chip requires chip_config")
    for key, value in (
        ("layout", LAYOUT),
        ("chip_dims_offset", 0),
        ("chip_layers", result["num_hidden_layers"]),
        ("hidden", result["hidden_size"]),
        ("chip_token_offset", target["vocab_size"]),
    ):
        _required(chip, key, value, "chip.chip_config")
    offset = target["vocab_size"]
    chip_vocab = _positive(chip.get("chip_vocab"), "chip.chip_vocab")
    if result["vocab_size"] < offset + chip_vocab:
        raise ValueError("chip.vocab_size does not hold the text and chip tokens")
    nfam = _positive(chip.get("nfam"), "chip.nfam")
    npay = _positive(chip.get("npay"), "chip.npay")
    code = chip.get("code_dims")
    if (
        not isinstance(code, list)
        or len(code) != nfam + npay
        or len(set(code)) != len(code)
        or any(type(d) is not int or not 0 <= d < result["hidden_size"] for d in code)
    ):
        raise ValueError("chip.code_dims must name nfam + npay distinct residual dims")
    firmware = chip.get("firmware")
    if (
        not isinstance(firmware, list)
        or not firmware
        or len(set(firmware)) != len(firmware)
        or any(
            not isinstance(name, str) or not name or "/" in name or "." in name
            for name in firmware
        )
    ):
        raise ValueError("chip.firmware must list at least one firmware name")
    result["chip_config"] = {
        "layout": LAYOUT,
        "chip_token_offset": offset,
        "chip_vocab": chip_vocab,
        "ballast_dims": _positive(chip.get("ballast_dims"), "chip.ballast_dims"),
        "nfam": nfam,
        "npay": npay,
        "code_dims": list(code),
        "tool_call": _token_ids(chip.get("tool_call"), offset, "tool_call", 2),
        "printable": _token_ids(chip.get("printable"), offset, "printable"),
        "eos": _token_ids(chip.get("eos"), offset, "eos"),
        "firmware": list(firmware),
    }
    return result


def add_chip(builder, store, config: dict) -> None:

    h = config["hidden_size"]
    builder.add(
        "chip/token_embedding", store, "model.embed_tokens.weight", (config["vocab_size"], h)
    )
    for layer in range(config["num_hidden_layers"]):
        builder.block(
            f"chip/layers/{layer}/",
            f"model.layers.{layer}.",
            store,
            config,
            "linear_attention",
        )
    state = (
        config["num_hidden_layers"],
        config["linear_num_value_heads"],
        config["linear_key_head_dim"],
        config["linear_value_head_dim"],
    )
    for name in config["chip_config"]["firmware"]:
        builder.add(
            f"chip/firmware/{name}/state",
            store,
            f"firmware.{name}.state",
            state,
            direct="fp32",
        )


def assign_chip(model, recipe) -> None:

    names = [name for name in model.parameters if name.startswith("chip/")]
    for name in names:
        recipe.assign(
            name, format=model.parameters[name].direct_format, method=exact_direct
        )
    recipe.separate(names)


__all__ = ["ARCHITECTURE", "LAYOUT", "add_chip", "assign_chip", "chip_config"]
