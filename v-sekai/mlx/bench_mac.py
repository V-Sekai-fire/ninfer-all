"""Decode and time-to-first-token for the .ninfer artifact on the Mac's GPU, shaped like
docs/performance/reference-2026-09.md: short chat = five 512-token answers; at depth, a
512-token answer after a document of that length. Greedy, one request, no speculation.
Run: pixi run -e mlx python v-sekai/mlx/bench_mac.py models/bonsai2-27b-heretic.ninfer [depths...]
"""

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

sys.path.insert(0, str(Path(__file__).parent))
import ninfer_mlx  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ANSWER = 512
CHAT = [
    "Explain how a lighthouse lens focuses light.",
    "Write a short story about a robot learning to paint.",
    "Describe the water cycle for a ten-year-old.",
    "Compare TCP and UDP and when to use each.",
    "Give a recipe for vegetable soup and explain each step.",
]


def run(model, ids):
    """Prefill ids, then decode ANSWER tokens without stopping at EOS. Returns (ttft s, decode tok/s)."""
    sampler = make_sampler(temp=0.0)
    start = time.perf_counter()
    steps = generate_step(mx.array(ids), model, max_tokens=ANSWER, sampler=sampler)
    token, _ = next(steps)
    mx.eval(token)
    first = time.perf_counter()
    count = 1
    for token, _ in steps:
        count += 1
    mx.eval(token)
    end = time.perf_counter()
    return first - start, (count - 1) / (end - first)


def document(tokenizer, length):
    text = "\n\n".join(p.read_text(errors="ignore") for p in sorted((ROOT / "docs").rglob("*.md")))
    ids = tokenizer.encode(text)
    while len(ids) < length:
        ids = ids + ids
    body = tokenizer.decode(ids[: length - 64])
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": body + "\n\nSummarise the document above in about 400 words."}],
        add_generation_prompt=True, tokenize=False)
    return tokenizer.encode(prompt)


def main(path, depths):
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    memory = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30
    print(f"host {chip}, {memory:.0f} GiB, macOS {platform.mac_ver()[0]}, mlx {mx.__version__}, device {mx.default_device()}")
    model, tokenizer = ninfer_mlx.load(path)
    warm = tokenizer.encode("Hello")
    list(generate_step(mx.array(warm), model, max_tokens=8))
    results = {}

    rates = []
    for question in CHAT:
        ids = tokenizer.encode(tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True, tokenize=False))
        ttft, rate = run(model, ids)
        rates.append(rate)
        print(f"short chat  prompt {len(ids):5d}  ttft {ttft:6.2f} s  decode {rate:6.2f} tok/s", flush=True)
    results["short chat"] = {"decode": sum(rates) / len(rates), "min": min(rates), "max": max(rates)}

    for depth in depths:
        ids = document(tokenizer, depth)
        ttft, rate = run(model, ids)
        results[str(depth)] = {"prompt": len(ids), "ttft": ttft, "decode": rate}
        print(f"depth {depth:6d}  prompt {len(ids):6d}  ttft {ttft:7.2f} s  decode {rate:6.2f} tok/s", flush=True)
    results["peak_memory_gib"] = mx.get_peak_memory() / 2**30
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], [int(d) for d in sys.argv[2:]] or [1024, 8192])
