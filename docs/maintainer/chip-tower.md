# The chip tower component

A chip tower is an RV32IM processor compiled into Gated DeltaNet layers. It runs beside a
Qwen3.5-family text model. A compiler (qwenputer) computes every weight; none is trained. The
processor's registers, ROM and RAM are the recurrent state rows of its heads.

The tower has its own residual width, its own MLP width, its own token table and its own
states. Its Gated DeltaNet head geometry is the text model's, so the text model's recurrent
kernels can run it. The text model's parameters and objects do not change when the tower is
selected.

## Source directory

`--source chip=DIR` names a directory that qwenputer's `chipformer.rv.export_tower` writes:

- `config.json`: architecture `ChipTower`, model type `chip_tower`, the tower's text config, and
  `chip_config` with the layout.
- `model.safetensors`: the layers under their Hugging Face names, the token table
  `model.embed_tokens.weight`, and each firmware's parked state `firmware.NAME.state` with shape
  (layers, value heads, key head dim, value head dim), fp32.

The source's `layer_types` ends with one `full_attention` layer. It is all zero: the reference
decoder uses it to count positions. The converter drops it.

## Converter

    python -m tools.convert --model MODEL --recipe bonsai2_27b_ternary \
        --source chip=DIR --components text,vision,mtp,dflash2,chip ...

`v-sekai/convert.nu --chip DIR` does this for the Bonsai 2 artifact.

`tools/convert/chip.py` holds three things.

`chip_config(raw, target)` checks the tower against the text model and returns the component
config. It refuses a tower when:

- the architecture, model type or layout is not the registered one;
- a Gated DeltaNet geometry field (key heads, key head dim, value heads, value head dim, conv
  kernel) is missing or differs from the text model's;
- `rms_norm_eps` differs;
- `layer_types` is not linear-attention layers followed by one full-attention layer;
- `chip_token_offset` is not the text model's vocabulary size, or the tower's vocabulary does
  not hold the text and chip tokens;
- `code_dims` does not name `nfam + npay` distinct residual dims;
- `tool_call`, `printable` or `eos` lists a token id outside the text vocabulary;
- no firmware is named.

`add_chip(builder, store, config)` adds the logical parameters: `chip/token_embedding`, one
linear-attention block per layer under `chip/layers/N/` (the same roles as a text block), and
`chip/firmware/NAME/state` for each firmware.

`assign_chip(model, recipe)` stores every tower parameter in its direct format (bf16, and fp32
for `a_log`, `dt_bias` and the states) with method `exact_direct`, and keeps each parameter in
its own object.

The text model must be Dense. A tower beside MoE text is refused.

## Why `exact_direct`

The circuit is exact only if each stored word is the compiler's value. One rounded word breaks
the processor. `cast_direct` rounds silently, and a quantizing method cannot hold the tower's
magnitudes, which reach 2^40 in the token table and beyond in the projections.

`exact_direct` casts to the target BF16 or FP32 format, casts back, and raises if any value
changed.

## Witness: `v-sekai/chip_witness.py`

    python v-sekai/chip_witness.py WITH.ninfer WITHOUT.ninfer \
        --model sources/Qwen3.8-27B --chip TOWER [--plant]

It compares an artifact converted with the tower against one converted without it. It passes
when:

- every parameter outside `chip/` is bound to objects of the same format, shape, range and
  SHA-256 in both artifacts;
- no object of the artifact without the tower is missing from the one with it;
- the only objects the tower adds are the ones `chip/` binds;
- every `chip/` object holds its source's words, and every tower parameter is bound.

`--plant` is the control. It flips one bit of one text object while that object is hashed. The
exit status is then 0 only if the check fails.

## Engine test

`tests/models/qwen3_5/test_chip_tower.cpp` (`ninfer_chip_tower_test`) runs the exported tower
through the public Ops and compares the result with the reference bit for bit.

    build/tests/ninfer_tests ninfer_chip_tower_test TOWER GOLDEN.raw \
        [--control all|none|fp16|weight] [--limit N] [--dump POSITION DIR]

`TOWER` is the source directory. `GOLDEN.raw` is the bundle that qwenputer's
`chipformer.rv.towercheck --golden` writes next to its parquet table: `ids.i32`, `kept.f32` and
`golden.json` (positions, the kept residual dims in column order, and `inject_at`). Without
arguments the test reads `NINFER_CHIP_TOWER` and `NINFER_CHIP_TOWER_GOLDEN`, and is skipped when
they are not set.

The test reads `model.safetensors` directly, so it does not depend on the converter. It uploads
each matrix as a dense BF16 `Weight`, as the direct BF16 Op tests do. For each token it calls,
per layer: `embedding`, `rmsnorm` (unit offset), `linear` four times (q/k/v, z, a, b),
`causal_conv1d_silu`, `gdn_gating`, `gated_delta_net` (one token, normalized q/k),
`gated_rmsnorm`, `linear`, `residual_add`, `rmsnorm`, `linear` twice, `silu_mul`, `linear` and
`residual_add`. The tower's shapes are not registered shapes, so every `linear` call takes the
runtime-shape BF16 route (one warp per output row at one token, FP32 accumulator). The fused
GDN input projection and the fused SwiGLU Ops are not used: they accept registered shapes only.

The test keeps one FP32 recurrent state and one BF16 convolution window per layer. Both start at
zero. Before position `inject_at` it overwrites every recurrent state with the firmware state
and keeps the convolution windows. The Op stores a state as (head, value dim, key dim), and the
tower stores it as (head, key dim, value dim), so the test transposes each head.

The test prints one JSON line per run: the number of positions, the number of inexact positions,
and the position, the residual dim and the two values of the first mismatch.

The two controls must fail the comparison. `fp16` keeps the recurrent states in FP16, which
`gated_delta_net` accepts as a state dtype. `weight` flips the sign bit of the first nonzero
`down_proj` weight of the middle layer. With `--control all` (the default) the exit status is 0
only if the plain run is exact and both controls fail.

`--dump POSITION DIR` writes every intermediate tensor of one position as raw files
(`L<layer>.<name>.bin`), to compare with the reference when a run is not exact.

## Loading

`--chip NAME` (`EngineOptions::chip_firmware`, `LoadOptions::chip_firmware`) selects the tower and
one firmware. Without it the `chip` component is not read, and the engine does what it did before.

`parse_config` reads the component config into `ChipConfig` (`src/models/qwen3_5/config.h`). It
refuses a tower whose Gated DeltaNet geometry or `rms_norm_eps` differs from the text model's, whose
chip token ids do not follow the text vocabulary, whose `code_dims` do not name `nfam + npay`
residual dims, or which does not hold firmware `NAME`.

`bind_chip` (`src/models/qwen3_5/load/chip.cpp`) binds `chip/token_embedding`, every block under
`chip/layers/N/` with the text block's bindings, and `chip/firmware/NAME/state`. The handles are
`ModelWeights::chip`. All of it is device memory on the first device.

## Runtime

`chip::Tower` (`src/models/qwen3_5/chip/tower.h`) holds one conversation: an FP32 recurrent state
and a BF16 convolution window per layer. `step` runs one token with the Op sequence of the Op
test. The artifact stores query, key and value as three matrices, so `step` calls `linear` once
for each into one buffer. One row of a BF16 projection is one FP32 dot product, so the words are
those of a single call.

The constructor reads the firmware state once, turns each head from (key dim, value dim) to the
Op's (value dim, key dim), and keeps the result on the device. `load_firmware` copies it over the
recurrent states.

`owner` decodes the output code (`chip/code.h`). A code bit is set where its residual dim is
positive. The tower owns the next position when any code dim is nonzero. The family id selects a
chip token family, a control token, or the echo family, whose payload `p` names text token
`printable[p]`. The family table is the `rv1-tower` layout's and is fixed in `code.h`.

`begin` runs a prompt from zero states and then loads the firmware. `advance` runs text tokens the
text model wrote. `burst` starts at an owned position: it runs each chip token through the tower,
collects each text token the tower names (the firmware's echo), runs that token too, and stops
when the tower no longer owns the next position.

### Runtime test

    build/tests/ninfer_chip_runtime_test ARTIFACT GOLDEN.raw \
        [--firmware NAME] [--control all|none|fp16|weight] [--device-profile NAME] [--limit N]

`tests/models/qwen3_5/test_chip_runtime.cpp` loads the converted artifact with the engine's
loader, installs the device route profile as the engine does, and replays the bundle's ids through
`chip::Tower`. It compares the kept dims bit for bit. After the prompt it also compares the token
`owner` names with the stream's next token wherever the tower owns the position. The controls are
those of the Op test, and each must fail the kept-dim comparison. It is its own program, not part
of the `ninfer_tests` bundle.

## Decoding

The Engine owns one `chip::Tower` (`ModelInstance::chip`), so `--chip` requires one lane on one
device. `--chip` runs without a drafter or with `--spec dflash2` and `--draft-tokens` of at most 11.
It is refused with `--spec mtp`, `--spec dflash` and n-gram drafting; with `--chip` the n-gram
width defaults to 0. The section "Drafting" gives the reasons.

With `--chip`, the server adds the `calc` tool definition (the one in qwenputer's `rv/chat.py`) to
every request that does not declare a tool named `calc` (`chip_calc_tool` in
`src/serve/translate.cpp`). The definition goes through the renderer that a client's tools go
through, so the prompt equals the prompt of a client that declares the same tool.

`EngineCore::follow_chip` runs after every commit of model tokens:

- At the request's first commit it runs the prompt's token ids through the tower (`begin`).
- It runs the committed text tokens (`advance`).
- If the tower owns the position after the last one, it runs the burst and gives the echo tokens
  to the request's `OutputSession` (`set_chip_echo`). The request becomes control-ready, and the
  next control unit forces the echo through the text model with `Program::append_forced_tokens`,
  the path the thinking-budget close uses. The text model's KV cache and recurrent state then
  hold the echo, and decoding goes on from its last token.

Chip tokens run through the tower only. They are not in `generated_token_ids`, in the text
model's positions, or in the output budget. Echo tokens are generated tokens and count against
the budget.

While a tool call is open (`<tool_call>` committed, `</tool_call>` not yet),
`OutputSession::model_token_budget_remaining` gives a round one model token. The close tag is then
the last token of its round, so a speculative round cannot write past it before the burst.

If the echo does not leave the text model one token of budget, the echo is cut to the remaining
budget less one. The text model's next round then ends the request.

The answered call is returned inline. When the echo commits,
`ToolCallOutputDecoder::resolve_in_model` releases the held tool-call region as content, and the
echo's bytes follow it, special tokens included (`<tool_response>`, `</tool_response>`, `<think>`,
`</think>`). The request's result has no structured call for it, and the text the model writes
after the echo is content. A grammar of the request does not read the echo.

A tower error (an output code that names no token, a burst of more than 60000 chip tokens) is
logged, ends the tower's part in that request, and leaves the text model decoding.

Each burst writes one Info diagnostic: `chip burst | request N | C chip tokens | E echo tokens
(K kept) | T ms`.

`--chip-trace FILE` (`EngineOptions::chip_trace_path`) appends one JSON line per burst to `FILE`:
the request id, the model's call tokens from `<tool_call>` on, and the burst's stream (chip tokens
at `chip_token_offset + t` and echo tokens, in order). The served witness reads it.

A request whose echo was cut ends with its limit reason, whatever its last round names.

### Limits

- The tower runs one token per step, about 2 ms on an RTX 4090. A prompt costs that for every
  token before the first output token, on every request, and prefix-cache reuse does not shorten
  it. The chunked Ops are not used: their exactness on the tower is not measured.
- A burst runs inside one Engine unit. A cancellation is seen after it.
- A malformed tool call shorter than a drafted round can be closed and passed in one round (see
  "Drafting"). The burst is then skipped with a Warning diagnostic and a `skipped` trace event.

## Drafting

A DFlash2 round commits the drafts the target accepts and one more token. Its draft extent is
`min(proposed, budget - 1, ...)` (`ProgramImpl::decode_dflash_batch`), so a round budget of one
gives a round of exactly one token. While a tool call is open the budget is one, so every token
after `<tool_call>` ends its own round, and the burst starts at the end of the round that commits
`</tool_call>`.

The Program does not let a round that goes on accept fewer tokens than it produced
(`ProgramImpl::commit`). The round that commits `<tool_call>` is an ordinary round of up to
`draft_tokens + 1` tokens. If it also commits `</tool_call>` and one more token, that token is
committed before the echo. This needs a call of at most `draft_tokens` tokens. The shortest
well-formed call is 12 tokens (25 for `calc` with a one-character expression), so
`--draft-tokens` is limited to 11 (`kChipMaximumDraftTokens`). A malformed call can still be
that short. The Engine steps the tower one token at a time inside a round, so it sees such a
position: it writes a Warning diagnostic and counts it; it does not start a burst there.

N-gram drafting is refused because it copies runs of earlier text, an earlier tool call and its
response among them. MTP and DFlash are refused because they were not measured.

The echo is forced with `Program::append_forced_tokens`. Under a drafter that call first appends
the drafter's context up to the frontier, then runs the forced tokens through the prefill path
with the drafter attached, and moves the drafter's context frontier with them. The drafter is
then where a prompt that held the echo would leave it.

The `--chip-trace` file holds three events: `burst`; `open_round`, a round that committed more
than one token while a call was open; and `skipped`, an owned position where no burst could
start. Each `burst` line carries the two running counts. Both counts must be zero.

`--chip-test-no-round-cap` switches the one-token budget off. It exists for the witness that
shows the budget is what keeps the counts at zero. Do not use it to serve.
