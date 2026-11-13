"""Compare artifacts converted with and without a chip tower (docs/maintainer/chip-tower.md)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.artifact.reader import Artifact  # noqa: E402
from tools.artifact.schema import binding_parts  # noqa: E402
from tools.convert.qwen3_5 import build_model  # noqa: E402
from tools.convert.sources.safetensors import SafetensorsSource  # noqa: E402


class Hashes:
    def __init__(self, artifact: Artifact, plant: str | None = None) -> None:
        self.artifact, self.plant, self.seen = artifact, plant, {}

    def __call__(self, object_id: str) -> tuple:
        if object_id not in self.seen:
            obj = self.artifact.object(object_id)
            digest = hashlib.sha256()
            first = object_id == self.plant
            for chunk in self.artifact.iter_object(object_id):
                if first and len(chunk):
                    chunk = bytes([chunk[0] ^ 1]) + bytes(chunk[1:])
                    first = False
                digest.update(chunk)
            self.seen[object_id] = (
                getattr(obj, "format", "resource"),
                tuple(getattr(obj, "shape", ())),
                digest.hexdigest(),
            )
        return self.seen[object_id]


def bound(artifact: Artifact, hashes: Hashes) -> dict[str, list]:
    out = {}
    for name, binding in artifact.directory.bindings.items():
        out[name] = [
            (hashes(object_id), begin, end)
            for object_id, begin, end in binding_parts(binding, artifact.by_id)
        ]
    return out


def chip_words(artifact: Artifact, model, name: str) -> bool:
    parameter = model.parameters[name]
    ((object_id, begin, end),) = binding_parts(
        artifact.directory.bindings[name], artifact.by_id
    )
    fp32 = parameter.direct_format == "fp32"
    width = 4 if fp32 else 2
    raw = bytes(artifact.read_object(object_id))[begin * width : end * width]
    values = parameter.source.values()
    if fp32:
        expected = values.to(torch.float32).contiguous().numpy().tobytes()
    else:
        expected = values.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes()
    return raw == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("with_chip", type=Path)
    parser.add_argument("without_chip", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--chip", type=Path, required=True)
    parser.add_argument("--plant", action="store_true")
    args = parser.parse_args()
    with Artifact(args.with_chip) as a, Artifact(args.without_chip) as b:
        plant = None
        if args.plant:
            binding = a.directory.bindings["text/layers/0/mlp/down"]
            plant = binding_parts(binding, a.by_id)[0][0]
        ha, hb = Hashes(a, plant), Hashes(b)
        pa, pb = bound(a, ha), bound(b, hb)
        chip = sorted(name for name in pa if name.startswith("chip/"))
        rest = sorted(name for name in pa if not name.startswith("chip/"))
        differing = [name for name in rest if pa[name] != pb.get(name)]
        missing = sorted(set(pb) - set(rest))
        all_a = {ha(obj.id) for obj in a.objects}
        all_b = {hb(obj.id) for obj in b.objects}
        chip_objects = {part[0] for name in chip for part in pa[name]}
        with SafetensorsSource(args.model) as base, SafetensorsSource(args.chip) as tower:
            model = build_model(
                base, components=("text", "chip"), companions={"chip": tower}
            )
            wrong = [name for name in chip if not chip_words(a, model, name)]
            unbound = sorted(set(n for n in model.parameters if n.startswith("chip/")) - set(chip))
        report = {
            "parameters_compared": len(rest),
            "parameters_differing": len(differing),
            "parameters_missing": len(missing),
            "objects_with": len(all_a),
            "objects_without": len(all_b),
            "objects_only_without": len(all_b - all_a),
            "objects_added_not_chip": len(all_a - all_b - chip_objects),
            "chip_parameters": len(chip),
            "chip_parameters_wrong": len(wrong),
            "chip_parameters_unbound": len(unbound),
            "first_differing": differing[:3],
            "planted": bool(args.plant),
        }
        report["ok"] = not (
            differing
            or missing
            or all_b - all_a
            or all_a - all_b - chip_objects
            or wrong
            or unbound
            or not chip
        )
    print(json.dumps(report))
    sys.exit(int(report["ok"]) if args.plant else int(not report["ok"]))


if __name__ == "__main__":
    main()
