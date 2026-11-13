from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from tools.artifact.reader import Artifact
from tools.artifact.schema import binding_parts
from tools.convert.chip import assign_chip, chip_config
from tools.convert.model import Model, Parameter
from tools.convert.qwen3_5 import build_model, text_config
from tools.convert.recipe import Recipe
from tools.convert.sources.logical import array_source

from .test_qwen3_5 import _checkpoint, _config, _values
from .test_recipe import _write


def _tower():
    text = _config()["text_config"]
    return {
        "architectures": ["ChipTower"],
        "model_type": "chip_tower",
        "hidden_act": "silu",
        "hidden_size": 8,
        "vocab_size": 16,
        "intermediate_size": 12,
        "num_hidden_layers": 3,
        "layer_types": ["linear_attention", "linear_attention", "full_attention"],
        "rms_norm_eps": 1e-6,
        **{key: value for key, value in text.items() if key.startswith("linear_")},
        "chip_config": {
            "layout": "rv1-tower",
            "chip_token_offset": 8,
            "chip_vocab": 6,
            "chip_layers": 2,
            "hidden": 8,
            "chip_dims_offset": 0,
            "ballast_dims": 3,
            "nfam": 1,
            "npay": 2,
            "code_dims": [4, 5, 6],
            "tool_call": [1, 2],
            "printable": [3, 4],
            "eos": [7],
            "firmware": ["calc"],
        },
    }


def _tensors():
    qkv = torch.arange(16 * 8).to(torch.bfloat16).reshape(16, 8)
    embedding = torch.arange(16 * 8).to(torch.bfloat16).reshape(16, 8)
    state = torch.arange(2 * 2 * 4 * 4).float().reshape(2, 2, 4, 4) * 2.0**20
    return {
        "model.embed_tokens.weight": embedding,
        "model.layers.1.linear_attn.in_proj_qkv.weight": qkv,
        "firmware.calc.state": state,
    }


def test_chip_tower_is_a_component_beside_an_unchanged_text_model(tmp_path):
    tensors = _tensors()
    with (
        _checkpoint(tmp_path / "text", _config(), {"unused": torch.ones(1)}) as base,
        _checkpoint(tmp_path / "tower", _tower(), tensors) as tower,
    ):
        plain = build_model(base)
        model = build_model(
            base, components=("text", "chip"), companions={"chip": tower}
        )
        chip = {name for name in model.parameters if name.startswith("chip/")}
        assert set(model.parameters) - chip == set(plain.parameters)
        assert model.components["chip"]["target"] == "text"
        config = model.components["chip"]["config"]
        assert config["num_hidden_layers"] == 2
        assert config["layer_types"] == ["linear_attention"] * 2
        assert config["chip_config"]["code_dims"] == [4, 5, 6]
        assert torch.equal(
            _values(model, "chip/token_embedding"), tensors["model.embed_tokens.weight"]
        )
        qkv = tensors["model.layers.1.linear_attn.in_proj_qkv.weight"]
        for role, rows in (
            ("query", slice(0, 4)),
            ("key", slice(4, 8)),
            ("value", slice(8, 16)),
        ):
            assert torch.equal(_values(model, f"chip/layers/1/gdn/{role}"), qkv[rows])
        state = model.parameters["chip/firmware/calc/state"]
        assert state.shape == (2, 2, 4, 4) and state.direct_format == "fp32"
        assert torch.equal(_values(model, state.name), tensors["firmware.calc.state"])
        assert model.parameters["chip/layers/0/mlp/gate"].shape == (12, 8)
        assert all(model.parameters[name].residency == "chip" for name in chip)
        assert not any(name.startswith("chip/layers/2/") for name in chip)


def test_chip_tower_is_refused_when_it_does_not_fit_the_text_model(tmp_path):
    target = text_config(_config(), mtp=False)
    assert chip_config(_tower(), target)["hidden_size"] == 8
    for change, message in (
        (lambda c: c.update(architectures=["Qwen3_5ForCausalLM"]), "architectures"),
        (lambda c: c.update(linear_num_value_heads=1), "linear_num_value_heads"),
        (lambda c: c.pop("linear_key_head_dim"), "linear_key_head_dim"),
        (lambda c: c["chip_config"].update(chip_token_offset=9), "chip_token_offset"),
        (lambda c: c["chip_config"].update(layout="rv1"), "layout"),
        (lambda c: c["chip_config"].update(chip_vocab=9), "vocab_size"),
        (lambda c: c["chip_config"].update(code_dims=[4, 5, 8]), "code_dims"),
        (lambda c: c["chip_config"].update(tool_call=[1, 8]), "tool_call"),
        (lambda c: c["chip_config"].update(firmware=[]), "firmware"),
        (
            lambda c: c.update(layer_types=["linear_attention", "linear_attention"]),
            "layer_types",
        ),
    ):
        raw = deepcopy(_tower())
        change(raw)
        with pytest.raises(ValueError, match=message):
            chip_config(raw, target)
    with _checkpoint(tmp_path / "text", _config(), {"unused": torch.ones(1)}) as base:
        with pytest.raises(ValueError, match="requires its source"):
            build_model(base, components=("text", "chip"))
    with (
        _checkpoint(
            tmp_path / "moe", _config(moe=True), {"unused": torch.ones(1)}
        ) as base,
        _checkpoint(tmp_path / "tower", _tower(), _tensors()) as tower,
    ):
        with pytest.raises(ValueError, match="Dense"):
            build_model(base, components=("text", "chip"), companions={"chip": tower})


def _chip_model(weight, state):
    model = Model(
        {"text": {"config": {}}, "chip": {"config": {}, "target": "text"}}
    )
    model.add(
        Parameter(
            "chip/layers/0/mlp/gate",
            tuple(weight.shape),
            array_source(weight, "gate"),
            inputs=("chip/layers/0/ffn_input",),
            residency="chip",
        )
    )
    model.add(
        Parameter(
            "chip/layers/0/mlp/up",
            tuple(weight.shape),
            array_source(weight + 1, "up"),
            inputs=("chip/layers/0/ffn_input",),
            residency="chip",
        )
    )
    model.add(
        Parameter(
            "chip/firmware/calc/state",
            tuple(state.shape),
            array_source(state, "state"),
            direct_format="fp32",
            residency="chip",
        )
    )
    model.packing_groups = [("chip/layers/0/mlp/gate", "chip/layers/0/mlp/up")]
    return model


def test_chip_parameters_are_stored_word_for_word_in_their_own_objects(tmp_path):
    # Circuit magnitudes: far beyond a quantized format's range, and beyond FP16's.
    weight = torch.tensor(
        [[2.0**40, -128.0, 0.0, 2.0**-12], [0.0, 3.0 * 2.0**30, -1.0, 0.0]],
        dtype=torch.bfloat16,
    )
    state = torch.tensor([[2.0**24, 128.0], [0.0, 255.0]], dtype=torch.float32)
    model = _chip_model(weight, state)
    recipe = Recipe(model)
    assign_chip(model, recipe)
    path = tmp_path / "chip.ninfer"
    _write(path, model, recipe.prepare(device="cpu", rows_per_chunk=1))
    with Artifact(path) as artifact:
        parts = {
            name: binding_parts(binding, artifact.by_id)[0]
            for name, binding in artifact.directory.bindings.items()
        }
        assert len({part[0] for part in parts.values()}) == 3
        for name, expected in (
            ("chip/layers/0/mlp/gate", weight),
            ("chip/layers/0/mlp/up", weight + 1),
            ("chip/firmware/calc/state", state),
        ):
            object_id, begin, end = parts[name]
            obj = artifact.object(object_id)
            assert obj.format == ("fp32" if expected.dtype == torch.float32 else "bf16")
            assert (begin, end) == (0, expected.numel())
            raw = expected.contiguous().numpy().tobytes() if expected.dtype == torch.float32 \
                else expected.contiguous().view(torch.int16).numpy().tobytes()
            assert bytes(artifact.read_object(object_id))[: len(raw)] == raw


def test_a_value_the_format_would_round_is_refused(tmp_path):
    # 1 + 2^-10 needs eleven significand bits; BF16 keeps eight.
    weight = torch.tensor([[1.0 + 2.0**-10, 2.0]], dtype=torch.float32)
    state = torch.zeros(1, 2)
    model = _chip_model(weight, state)
    recipe = Recipe(model)
    assign_chip(model, recipe)
    with pytest.raises(ValueError, match="not exact in bf16"):
        _write(
            tmp_path / "inexact.ninfer",
            model,
            recipe.prepare(device="cpu", rows_per_chunk=1),
        )
