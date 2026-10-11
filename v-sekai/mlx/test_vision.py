"""Witnesses for the vision path.

1. Weights: every quantized vision matrix the loader builds, dequantised, against the bf16 original
   in Qwen3.8-27B shard 1 (the artifact's vision source). A wrong lane order, plane offset or row
   range drops cosine to about 0; honest rounding does not. The bar is 0.95: q4 rounding alone
   reaches 0.987 on block 0's qkv, so 0.99 (the first bar tried) rejected correct q4 weights.
   Control: the qkv rows rolled by one head must fail.
2. Zero-shot image: a red circle on the left, a blue square on the right and the text "42", asked
   at temperature 0. Pass = the reply names both shapes, both colours, their sides and "42".
Run: pixi run -e mlx python v-sekai/mlx/test_vision.py models/bonsai2-27b-heretic.ninfer sources/Qwen3.8-27B
"""

import re
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import vlm  # noqa: E402

COS_BAR = 0.95


def dequant(module):
    m = module.inner if hasattr(module, "inner") else module
    m = m.linear if hasattr(m, "linear") else m
    s = m.scales.astype(mx.float32)
    w = mx.dequantize(m.weight, s, m.biases.astype(mx.float32), group_size=m.group_size, bits=m.bits)
    return np.asarray(w, dtype=np.float64)


def cosine(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def weights_check(model, source):
    shard = mx.load(str(Path(source) / "model-00001-of-00018.safetensors"))
    tower = model.vision_tower
    pairs = [("patch_embed", tower.patch_embed.proj, "model.visual.patch_embed.proj.weight")]
    for i, block in enumerate(tower.blocks):
        p = f"model.visual.blocks.{i}."
        pairs += [(f"blocks.{i}.attn.qkv", block.attn.qkv, p + "attn.qkv.weight"),
                  (f"blocks.{i}.attn.proj", block.attn.proj, p + "attn.proj.weight"),
                  (f"blocks.{i}.mlp.fc1", block.mlp.linear_fc1, p + "mlp.linear_fc1.weight"),
                  (f"blocks.{i}.mlp.fc2", block.mlp.linear_fc2, p + "mlp.linear_fc2.weight")]
    pairs += [("merger.fc1", tower.merger.linear_fc1, "model.visual.merger.linear_fc1.weight"),
              ("merger.fc2", tower.merger.linear_fc2, "model.visual.merger.linear_fc2.weight")]
    worst, failures, by_bits = (1.0, ""), 0, {}
    for name, module, key in pairs:
        ref = np.asarray(shard[key].astype(mx.float32), dtype=np.float64)
        ref = ref.reshape(ref.shape[0], -1)
        got = dequant(module)[:, :ref.shape[1]]
        c = cosine(got, ref)
        worst = min(worst, (c, name))
        inner = module.inner if hasattr(module, "inner") else module
        bits = (inner.linear if hasattr(inner, "linear") else inner).bits
        by_bits[bits] = min(by_bits.get(bits, 1.0), c)
        failures += c <= COS_BAR
    print("worst cosine by bits:", {b: round(c, 5) for b, c in sorted(by_bits.items())})
    print(f"{'ok  ' if not failures else 'FAIL'} vision weights: {len(pairs)} matrices, worst cosine "
          f"{worst[0]:.5f} ({worst[1]}), {failures} below {COS_BAR}")
    qkv = dequant(tower.blocks[0].attn.qkv)
    ref = np.asarray(shard["model.visual.blocks.0.attn.qkv.weight"].astype(mx.float32), dtype=np.float64)
    rolled = cosine(np.roll(qkv, 72, axis=0), ref)
    caught = rolled <= COS_BAR
    print(f"{'ok  ' if caught else 'FAIL'} control: qkv rows rolled by one head give cosine {rolled:.4f}")
    return failures + (not caught)


def probe_image() -> Path:
    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((40, 150, 180, 290), fill=(220, 20, 20))
    draw.rectangle((270, 150, 410, 290), fill=(20, 40, 220))
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 72)
    except OSError:
        font = ImageFont.load_default()
    draw.text((180, 330), "42", fill=(0, 0, 0), font=font)
    path = Path(tempfile.gettempdir()) / "ninfer-vision-probe.png"
    image.save(path)
    return path


def zero_shot(model, processor):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    image = probe_image()
    question = ("Describe this image: which shapes are there, their colours, which side each is on, "
                "and any text. Answer in one short paragraph.")
    prompt = apply_chat_template(processor, model._ninfer_config, question, num_images=1)
    reply = generate(model, processor, prompt, image=[str(image)], max_tokens=700, temperature=0.0, verbose=False)
    text = getattr(reply, "text", reply)
    answer = text.split("</think>")[-1].lower()
    print("reply:", answer.strip()[:400])
    checks = {
        "circle": "circle" in answer, "square": "square" in answer,
        "red": "red" in answer, "blue": "blue" in answer, "42": "42" in answer,
        "red circle left": bool(re.search(r"(red[^.]*circle[^.]*left|left[^.]*red[^.]*circle)", answer)),
        "blue square right": bool(re.search(r"(blue[^.]*square[^.]*right|right[^.]*blue[^.]*square)", answer)),
    }
    missing = [k for k, v in checks.items() if not v]
    print(("ok  " if not missing else "FAIL"), "zero-shot image", "" if not missing else f"missing {missing}")
    return int(bool(missing))


def main(artifact, source):
    model, processor = vlm.load(artifact)
    failures = weights_check(model, source) + zero_shot(model, processor)
    print(f"{failures} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
