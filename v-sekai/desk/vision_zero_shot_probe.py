#!/usr/bin/env python3
"""Zero-shot vision probe for an OpenAI-compatible ninfer-serve / llama-server.

Usage: vision_zero_shot_probe.py [base_url] [model_id]
Defaults: http://100.86.241.78:8080  (model auto-picked from /v1/models)
Pass = reply names red circle upper-left, "42" upper-right, blue square lower-right.
Set BONSAI_API_KEY to send a bearer token (servers started with --api-key).
"""
import base64, io, json, os, sys, urllib.request
from PIL import Image, ImageDraw, ImageFont

base = sys.argv[1] if len(sys.argv) > 1 else "http://100.86.241.78:8080"

AUTH = {"Authorization": "Bearer " + os.environ["BONSAI_API_KEY"]} if os.environ.get("BONSAI_API_KEY") else {}

def get(path):
    return json.load(urllib.request.urlopen(urllib.request.Request(base + path, headers=AUTH), timeout=30))

models = get("/v1/models")["data"]
model = sys.argv[2] if len(sys.argv) > 2 else models[0]["id"]
print("model:", model, "modalities:", next((m.get("modalities") for m in models if m["id"] == model), None))

im = Image.new("RGB", (512, 512), "white")
d = ImageDraw.Draw(im)
d.ellipse((40, 40, 220, 220), fill="red")
d.rectangle((280, 280, 470, 470), fill="blue")
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 90)
except Exception:
    font = ImageFont.load_default()
d.text((300, 60), "42", fill="black", font=font)
buf = io.BytesIO(); im.save(buf, "PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

body = {"model": model, "max_tokens": 200, "temperature": 0, "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
    {"type": "text", "text": "Describe the shapes, their colors and positions, and any text in this image. Be brief."}]}]}
req = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json", **AUTH})
j = json.load(urllib.request.urlopen(req, timeout=240))
print(j["choices"][0]["message"].get("content"))
print("usage:", j.get("usage"))
