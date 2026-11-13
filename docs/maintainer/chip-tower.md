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
