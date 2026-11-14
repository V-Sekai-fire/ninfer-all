# Chip tower: standard operating procedures

How to serve, use, check and rebuild a Bonsai 2 artifact that carries a chip tower. The design is
in [`docs/maintainer/chip-tower.md`](../docs/maintainer/chip-tower.md). Every command runs from the
repository root unless a step says otherwise.

## What it is

The artifact holds one more component beside the text model: an RV32IM processor in eleven Gated
DeltaNet layers. With `--chip`, the server runs it. When the model writes a `calc` tool call, the
processor runs the calc firmware and writes the tool response into the model's own output. No
program outside the artifact computes the answer.

Without `--chip` the same artifact serves as the artifact without the tower does.

## Before you start

- One NVIDIA card with 24 GB, Fedora 44 (native or WSL2), and the engine built from this
  repository at a commit that has `--chip` (`ninfer-serve --help` lists it).
- About 13 GB of disk for the artifact.
- A `--chip` server has one request lane and one device. With the DFlash2 drafter it decodes text
  at about 95 tokens per second on an RTX 4090; the DFlash2 server of RFD66606-2311 without the
  tower gives about 180, and a `--chip` server without a drafter about 72. Keep the server
  without the tower for work that does not need it.

## Procedure 1: get the artifact

1. Download it:

       hf download chibifire/ninfer-bonsai2-heretic \
           models/chip/bonsai2-27b-heretic-chip.ninfer models/chip/SHA256SUMS \
           --repo-type dataset --local-dir .

2. Check it. The command must print `OK`:

       cd models/chip && sha256sum -c SHA256SUMS

3. Do not serve a file that fails the check.

## Procedure 2: serve it

1. Check that the card is free: `nvidia-smi` shows at least 17 GB of free memory (the test
   server used 16 GB with this serving line).
2. Start the server. This is the serving line of RFD66606-2311 with `--chip` added:

       ninfer-serve models/chip/bonsai2-27b-heretic-chip.ninfer \
           --device 0 --host 127.0.0.1 --port <port> --api-key ${NINFER_API_KEY} \
           --model-id bonsai-27b --max-context 262144 --kv-capacity 262144 \
           --kv-dtype rk8v4 --vision --max-concurrency 1 --no-thinking \
           --spec dflash2 --draft-tokens 5 --chip calc-bridge

3. Wait for the line `listening on` in its log.
4. On a host that keeps the server, run it as a systemd service unit of its own, on its own port,
   as RFD66606-2311 describes for the DFlash2 server. Do not start it with `nohup` or `&`.

The server stops at startup, with a message that names the option, when `--chip` is given with
`--spec mtp` or `--spec dflash`, with `--draft-tokens` above 11, with n-gram drafting (its width
is 0 under `--chip` unless you set it), with `--max-concurrency` above 1, with more than one device, or
with a firmware name the artifact does not hold. `--chip-trace` needs `--chip`.

## Procedure 3: ask it

Send an ordinary chat completion. Do not declare a `calc` tool; the server adds it. Do not run a
tool on the client.

    curl -s http://127.0.0.1:<port>/v1/chat/completions \
        -H "Authorization: Bearer ${NINFER_API_KEY}" -H "Content-Type: application/json" \
        -d '{"model": "bonsai-27b", "temperature": 0, "max_tokens": 400,
             "messages": [{"role": "user", "content": "Use calc: what is 123456789 * 987654321?"}]}'

The answer is one message. Its content holds, in order:

1. the call the model wrote, from `<tool_call>` to `</tool_call>`;
2. the response the processor wrote, from `<tool_response>` to `</tool_response>`, then an empty
   `<think>` block;
3. the model's own sentence, written after it read the response.

`finish_reason` is `stop`. The message has no `tool_calls` entry.

A client that shows the answer to a person removes parts 1 and 2 and shows part 3.

### What the calc tool takes

- Integers, decimals and fractions of up to 62 digits.
- `+ - * /` and parentheses. `/` is exact: the result is an integer or a reduced fraction.
- On integers also `//` (floor division), `%`, `**` and `gcd(a,b)`.
- On decimals `+ - *`.
- A result that needs more than 62 digits returns `error`. So does an expression the firmware
  cannot read.

The model decides when to call the tool. A prompt that names it ("Use calc: ...") makes the call
reliable. How often the model calls it without being asked is not measured.

## Procedure 4: check that the tower works

Run this after each install, after each change of artifact, and after each engine upgrade.

1. Send the request of procedure 3.
2. The content must hold `<tool_response>`, then `121932631112635269`, then `</tool_response>`.
3. The server log, at the info level, must hold one line for the request that starts with
   `chip burst | request`.
4. Send `"Use calc: what is 84 / 18?"`. The response must be `14/3`.

If step 2 shows a `tool_calls` entry and empty content, the server runs without `--chip`.

## Procedure 5: rebuild the tower and the artifact

Do this when the firmware changes, when the text model's geometry changes, or to reproduce the
published files.

1. In a checkout of V-Sekai-fire/qwenputer, directory `experimental/riscv`, with the host model's
   `config.json` and `tokenizer.json` in `HOST`:

       uv sync --extra hf --extra golden
       uv run python -m chipformer.rv.export_tower HOST --out TOWER \
           --firmware tests/data/rv/calc-bridge.elf

2. Check the tower against the reference machine, and write the golden bundle. The command must
   print `"ok": true` with `"inexact_positions": 0`:

       uv run python -m chipformer.rv.towercheck TOWER HOST --mode decode --served \
           --device cuda --golden TOWER/golden-calc-bridge.parquet \
           --exprs "123456789*987654321" "2*(3+4)-50" "(1.2 - 3) * 4.5 - 0.75" "-84/18"

3. Run the three controls. Each must print `"ok": false`:

       for c in rom weight fp16; do
           uv run python -m chipformer.rv.towercheck TOWER HOST --exprs "12*34" \
               --mode decode --served --device cuda --control $c
       done

4. In this repository, put the tower at `sources/tower-27b` (a copy or a link), fetch the other
   sources, and convert with and without the tower. Use the relative path: an absolute path is
   written into the artifact's report.

       pixi run --locked nu v-sekai/fetch-sources.nu
       pixi run --locked nu v-sekai/convert.nu --chip sources/tower-27b \
           --out models/chip/bonsai2-27b-heretic-chip.ninfer
       pixi run --locked nu v-sekai/convert.nu --out models/bonsai2-27b-heretic.ninfer

5. Run the object-hash witness, then its control. The first must exit 0 with `"ok": true`. The
   second must report `"ok": false` and also exit 0:

       pixi run --locked python v-sekai/chip_witness.py \
           models/chip/bonsai2-27b-heretic-chip.ninfer models/bonsai2-27b-heretic.ninfer \
           --model sources/Qwen3.8-27B --chip sources/tower-27b
       pixi run --locked python v-sekai/chip_witness.py \
           models/chip/bonsai2-27b-heretic-chip.ninfer models/bonsai2-27b-heretic.ninfer \
           --model sources/Qwen3.8-27B --chip sources/tower-27b --plant

6. Run the two engine tests against the golden bundle. Each prints one line per run; the plain run
   must show `"inexact_positions": 0`, and the `fp16` and `weight` runs must not:

       build/tests/ninfer_tests ninfer_chip_tower_test sources/tower-27b \
           sources/tower-27b/golden-calc-bridge.raw
       build/tests/ninfer_chip_runtime_test models/chip/bonsai2-27b-heretic-chip.ninfer \
           sources/tower-27b/golden-calc-bridge.raw

7. Do procedure 4 on a test port before the artifact replaces one in service.
8. Before the files are published, check that no report names a host directory:
   `grep -l /home/ models/chip/*.conversion.json` must print nothing.

Each long step is a named systemd oneshot unit on the host that runs it, with a job count and a
memory limit, and each unit that needs another names it with `Requires=` and `After=`.

## Procedure 6: read a burst

Start the server with `--chip-trace FILE` to keep what the processor did. Each burst appends one
JSON line to `FILE`: the request id, the token ids of the call, and the token ids of the burst.
Burst ids at or above 248320 are processor tokens. Ids below it are the bytes the firmware
printed, as text tokens.

Do not leave the trace on for a server in service: the file grows with every call.

## Faults

| What you see | Cause | What to do |
| --- | --- | --- |
| The server stops at startup and names `--chip` | `--chip` with a drafter other than DFlash2, more than 11 draft tokens, n-gram drafting, more than one lane, more than one device, or an unknown firmware name | Remove the other option, or correct the name |
| `tool_calls` entry, empty content | The server runs without `--chip`, or `max_tokens` ended the call before `</tool_call>` | Start it with `--chip`; raise `max_tokens` |
| `<tool_response>` holds `error` | The result needs more than 62 digits, or the firmware cannot read the expression | Ask for a smaller result, or an operator from the list above |
| `finish_reason` is `length` and the response is cut | The output limit fell inside the response | Raise `max_tokens`; each digit of a response is one token |
| The first token of each answer is slow | Each prompt token is one tower step, about 2 ms | Expected. Keep prompts short on a `--chip` server |
| The log holds `the chip tower names no token` | The firmware state or a tower weight is not what the compiler wrote | Do procedure 1, step 2. If the file is good, do procedure 5, steps 6 and 7 |
| The model keeps calling the tool after `error` | The model does not know why the firmware refused | Tell it the 62-digit limit in the prompt |
| A cancelled request ends late | A burst runs to its end before the server reads the cancel | Expected. A burst takes 4 to 16 s for long results |

## What is not supported

- A drafter other than DFlash2 beside the tower, n-gram drafting, and more than 11 draft tokens.
- More than one request lane, or more than one device.
- Requests in thinking mode, streamed requests, the Responses and Anthropic endpoints, structured
  output, and histories that hold earlier tower calls. None of these has run with `--chip`.
- Firmware other than `calc-bridge`.
