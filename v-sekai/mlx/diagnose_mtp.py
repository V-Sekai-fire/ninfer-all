"""Diagnose MTP/plain divergences: is the flipped token a near-tie that batched verification rounds
differently, or a rollback error? Prints the plain run's top-2 logit margin at the first divergence
and whether a teacher-forced batched pass (no speculation, no rollback) also disagrees there.
Also times the pieces of one speculative round.
"""

import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import KVCache, make_prompt_cache
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).parent))
import ninfer_mlx  # noqa: E402
import speculative  # noqa: E402

PROMPTS = ["Explain how a lighthouse lens focuses light.",
           "List the planets of the solar system with one fact about each."]


def timeit(fn, n=5):
    fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1000


def main(path, n=128):
    model, tokenizer = ninfer_mlx.load(path, draft=True)
    decoder = speculative.MTPDecoder(model, drafts=3)
    lm = model.language_model
    for prompt in PROMPTS:
        ids = tokenizer.encode(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False))
        ref, margins = [], []
        for tok, logprobs in generate_step(mx.array(ids), model, max_tokens=n, sampler=make_sampler(temp=0.0)):
            top = mx.sort(logprobs)[-2:]
            margins.append((top[1] - top[0]).item())
            ref.append(int(tok))
        got = list(decoder.generate(ids, n))
        i = next((j for j, (a, b) in enumerate(zip(got, ref)) if a != b), None)
        cache = make_prompt_cache(model)
        forced = mx.argmax(lm(mx.array(ids + ref[:-1])[None], cache=cache)[0, len(ids) - 1:], -1).tolist()
        forced_diff = [j for j, (a, b) in enumerate(zip(forced, ref)) if a != b]
        print(f"{prompt[:30]!r}: first MTP divergence {i}; plain margin there "
              f"{margins[i] if i is not None else float('nan'):.4f} (median {sorted(margins)[len(margins)//2]:.2f}); "
              f"teacher-forced batch disagrees with plain at {forced_diff[:5]}")

    ids = mx.array(tokenizer.encode("Hello there, how are you today?"))[None]
    cache = make_prompt_cache(model)
    hidden = lm.model(ids, cache=cache)
    mx.eval(hidden)
    one, four = ids[:, :1], ids[:, :4]

    def target(x):
        h = lm.model(x, cache=cache)
        mx.eval(lm.lm_head(h))
        for c in cache:
            if isinstance(c, KVCache):
                c.trim(x.shape[1])

    decoder.mtp_cache = KVCache()
    mx.eval(decoder._mtp(ids, hidden))

    def mtp_step():
        m = decoder._mtp(one, hidden[:, -1:, :])
        mx.eval(decoder._propose(m))
        decoder.mtp_cache.trim(1)

    print(f"target forward: 1 token {timeit(lambda: target(one)):.1f} ms, 4 tokens {timeit(lambda: target(four)):.1f} ms")
    print(f"MTP step + proposal head: {timeit(mtp_step):.1f} ms")


if __name__ == "__main__":
    main(sys.argv[1])
