"""Witness for the artifact's drafters under mlx-vlm's speculative decoding (greedy).

For each drafter: plain greedy text vs drafted greedy text on the same prompts, the acceptance
mlx-vlm records, and decode rate. Equal text, or a first difference only after a token the plain
run chose by a bf16 tie, passes (batched verification rounds differently from one-token decode).
Control: DFlash2's feature projection with its five per-layer column blocks rotated by one (each
block then reads a tap 14 layers away) must lose most of its acceptance. Two weaker controls were
tried first and rejected: shifting every tap one layer deeper (neighbouring residual states are
nearly equal; 0.50 vs 0.56) and listing the taps in reverse (mlx-vlm captures layers in index
order, so nothing changed; 0.56 vs 0.56).
Run: pixi run -e mlx python v-sekai/mlx/test_drafters.py models/bonsai2-27b-heretic.ninfer [tokens]
"""

import sys
from pathlib import Path

import mlx.core as mx
from mlx_vlm import generate
from mlx_vlm.speculative.common import speculative_stats_since, speculative_stats_snapshot

sys.path.insert(0, str(Path(__file__).parent))
import vlm  # noqa: E402

PROMPTS = [
    "Explain how a lighthouse lens focuses light.",
    "Write a Python function that merges two sorted lists, with a docstring.",
    "List the planets of the solar system with one fact about each.",
]


def run(model, processor, prompt, n, **draft):
    text = processor.apply_chat_template([{"role": "user", "content": prompt}],
                                         add_generation_prompt=True, tokenize=False)
    result = generate(model, processor, text, max_tokens=n, temperature=0.0, verbose=False, **draft)
    return result.text, result.generation_tps


def measure(model, processor, drafter, kind, n, block=None):
    rows = []
    extra = {"draft_block_size": block} if block else {}
    for prompt in PROMPTS:
        base, base_tps = run(model, processor, prompt, n)
        snap = speculative_stats_snapshot(drafter)
        got, tps = run(model, processor, prompt, n, draft_model=drafter, draft_kind=kind, **extra)
        rounds, accepted, drafted = speculative_stats_since(drafter, snap)
        first = next((i for i, (a, b) in enumerate(zip(got, base)) if a != b), None)
        rows.append((prompt, got == base, first, accepted / max(drafted or 1, 1), (accepted + rounds) / max(rounds, 1),
                     base_tps, tps, base[max(0, (first or 0) - 30):(first or 0) + 10] if first is not None else ""))
    return rows


def main(path, n=192, only=None):
    model, processor = vlm.load(path)
    failures = 0
    for kind, factory, block in (("mtp", vlm.load_mtp_drafter, None),
                                 ("dflash", lambda m: vlm.load_dflash2_drafter(m, block_size=6), 6)):
        if only and kind != only:
            continue
        drafter = factory(model)
        print(f"== {kind}")
        for prompt, equal, first, rate, per_round, base_tps, tps, context in measure(model, processor, drafter, kind, n, block):
            print(f"  {prompt[:40]!r:44} equal={equal} first_char_diff={first} accept {rate:.2f} "
                  f"({per_round:.2f} tok/round)  plain {base_tps:5.2f} tok/s  drafted {tps:5.2f} tok/s")
            if not equal:
                print(f"     plain text around the difference: {context!r}")
        if kind == "dflash":
            healthy = sum(r[3] for r in measure(model, processor, drafter, kind, 64, block)) / len(PROMPTS)
            fc = drafter.fc
            saved = {k: getattr(fc, k) for k in ("weight", "scales", "biases")}
            taps = len(drafter.config.target_layer_ids)
            for k, v in saved.items():
                blocks = mx.split(v, taps, axis=1)
                setattr(fc, k, mx.concatenate(blocks[1:] + blocks[:1], axis=1))
            blind = sum(r[3] for r in measure(model, processor, drafter, kind, 64, block)) / len(PROMPTS)
            for k, v in saved.items():
                setattr(fc, k, v)
            ok = blind < healthy / 2
            failures += not ok
            print(f"{'ok  ' if ok else 'FAIL'} control: rotated feature blocks accept {blind:.2f} vs {healthy:.2f}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], int(args[1]) if len(args) > 1 else 192, args[2] if len(args) > 2 else None)
