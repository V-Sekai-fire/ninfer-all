"""Witness: greedy MTP decoding emits the same tokens as plain greedy decoding of the target.

Verification runs several tokens through one forward pass, and MLX's batched matmuls round
differently from its single-token ones, so a token whose top two logits are within one bf16 step
(TIE) may flip. A run passes when the streams are equal or first differ at such a tie.

Also reports acceptance and decode rate. Two negative controls: skipping the GDN rollback must
break the equality, and an MTP fed zero target hidden states must lose most of its acceptance.
Run: pixi run -e mlx python v-sekai/mlx/test_mtp.py models/bonsai2-27b-heretic.ninfer [tokens] [drafts]
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).parent))
import ninfer_mlx  # noqa: E402
import speculative  # noqa: E402

PROMPTS = [
    "Explain how a lighthouse lens focuses light.",
    "Write a Python function that merges two sorted lists, with a docstring.",
    "List the planets of the solar system with one fact about each.",
    "Translate into French: The weather is lovely today and we are going to the market.",
]


TIE = 0.25  # one bf16 step for logits in [32, 64); decode logits here stay below that


def plain(model, ids, n):
    out, margins, t1 = [], [], time.perf_counter()
    for i, (tok, logprobs) in enumerate(generate_step(mx.array(ids), model, max_tokens=n, sampler=make_sampler(temp=0.0))):
        if i == 1:
            t1 = time.perf_counter()
        top = mx.sort(logprobs)[-2:]
        margins.append((top[1] - top[0]).item())
        out.append(int(tok))
    plain.margins = margins
    return out, (n - 1) / (time.perf_counter() - t1)


def main(path, n=256, drafts=3):
    model, tokenizer = ninfer_mlx.load(path, draft=True)
    decoder = speculative.MTPDecoder(model, drafts=drafts)
    failures = 0
    for prompt in PROMPTS:
        ids = tokenizer.encode(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False))
        reference, plain_rate = plain(model, ids, n)
        got = list(decoder.generate(ids, n))
        s = decoder.stats
        same = got == reference
        first = next((i for i, (a, b) in enumerate(zip(got, reference)) if a != b), None)
        tie = first is not None and plain.margins[first] <= TIE
        ok = same or tie
        failures += not ok
        where = "" if first is None else f" first_diff={first} (plain top-2 margin {plain.margins[first]:.3f})"
        print(f"{'ok  ' if ok else 'FAIL'} {prompt[:40]!r:44} equal={same}{where} "
              f"accept {s.acceptance:.2f} ({s.tokens_per_round:.2f} tok/round)  "
              f"plain {plain_rate:5.2f} tok/s  mtp {(s.tokens - 1) / s.decode_seconds:5.2f} tok/s", flush=True)

    ids = tokenizer.encode(tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPTS[0]}], add_generation_prompt=True, tokenize=False))
    reference, _ = plain(model, ids, n)
    saved = speculative._rollback_gdn
    speculative._rollback_gdn = lambda trace, keep: None
    broken = list(decoder.generate(ids, n))
    speculative._rollback_gdn = saved
    caught = broken != reference
    print(("ok  " if caught else "FAIL"), "control: no GDN rollback changes the output")
    list(decoder.generate(ids, n))
    healthy = decoder.stats.acceptance
    target = model._drafter.mtp.hidden_norm
    real = ninfer_mlx.OffsetRMSNorm.__call__
    ninfer_mlx.OffsetRMSNorm.__call__ = lambda self, x: real(self, mx.zeros_like(x) if self is target else x)
    list(decoder.generate(ids, n))
    blind = decoder.stats.acceptance
    ninfer_mlx.OffsetRMSNorm.__call__ = real
    dropped = blind < healthy / 2
    print(("ok  " if dropped else "FAIL"), f"control: MTP without target hidden accepts {blind:.2f} vs {healthy:.2f}")
    failures += (not caught) + (not dropped)
    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[0], *(int(a) for a in args[1:]))
