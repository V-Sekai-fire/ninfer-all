# V-Sekai release line

This directory adds a Windows and Linux release of ninfer-all for one RTX 4090 (sm_89).
The release serves one model: Ternary Bonsai 2 27B Uncensored Heretic v2. Everything else in the
repository is upstream [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all).

The release format copies the desync pattern of
[V-Sekai-fire/service-language-model](https://github.com/V-Sekai-fire/service-language-model).
`payload-pack.exe` packs the zstd chunks of one desync chunk store, unchanged, into
`payload-data-NNN.bin` volumes below the GitHub asset limit. Both platforms share the store,
so the model's chunks are stored once. `payload.caidx` indexes the Windows payload, and
`setup.exe` installs it online from its pinned tag or offline from the assets beside it.
`payload-linux.caidx` indexes the Linux payload, and `ninfer-install` installs it the same way.

## Steps

Every script is nushell and runs inside the pixi environment, from the repository root.
Pixi supplies CUDA 13, CMake, Ninja, FFmpeg (LGPL build), libcurl, Go, Python and nushell,
and on Linux GCC 14 with a glibc 2.28 sysroot. The Windows host needs Visual Studio 2022
with the C++ tools.

Steps 1 and 2 use the default environment, which runs on Windows, Linux and macOS (Apple
silicon). The converter is pure Python on the CPU, so any of them can make the artifact.
Steps 3 and 4 use the `build` environment, which exists only on Windows and Linux, because the
engine needs CUDA.

1. `pixi run --locked nu v-sekai/fetch-sources.nu` downloads the conversion sources into `sources/`.
2. `pixi run --locked nu v-sekai/convert.nu` writes `models/bonsai2-27b-heretic.ninfer`.
3. `pixi run --locked -e build nu v-sekai/build.nu` builds `ninfer-serve` and `ninfer-calibrate` and
   stages them with their library closure in `dist/<os>/`. Run it once on Windows and once
   on Linux. WSL2 on the RTX 4090 host is enough: it passes the Windows NVIDIA driver through,
   so the same host builds and GPU-tests both platforms. Copy the Linux `dist/linux` and
   `dist/tools-linux` into the Windows checkout's `dist/`.
4. `pixi run --locked -e build nu v-sekai/release.nu --tag v0.12.0-dev.N`, on Windows with both
   `dist` trees present, stages both payloads, restores each one, compares every file by
   SHA-256, and publishes a prerelease.
5. Run the `V-Sekai Linux install check` workflow with the tag. It installs the release
   online with `ninfer-install` on a clean runner and checks every file against
   `payload-linux.sha256`. It needs no GPU.

## Artifact sources

- Body: [OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-GGUF](https://huggingface.co/OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-GGUF), `PQ2_0`, imported as `t2_g128_fp16` without rounding.
- Vision tower, tokenizer and frontend resources: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
- MTP head: [ProCreations/Ternary-Bonsai-2-27B-MTP](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-MTP).
- DFlash2 adapter: [ProCreations/Ternary-Bonsai-2-27B-DFlash2](https://huggingface.co/ProCreations/Ternary-Bonsai-2-27B-DFlash2).
- Recipe: `bonsai2_27b_ternary`, as [docs/weight-conversion.md](../docs/weight-conversion.md#ternary-bonsai-2-27b) describes it.

## Serving

After installation, run `nu serve.nu` in the install directory
(`C:\ProgramData\V-Sekai\NInfer` by default on Windows, the `--target` folder on Linux).
`nu serve.nu --help` lists the options. A Linux host needs an NVIDIA driver of the CUDA 13
branch and glibc 2.28 or newer.

## macOS (Apple silicon)

The CUDA engine does not build on macOS. `v-sekai/mlx/ninfer_mlx.py` serves the same `.ninfer`
artifact through MLX instead. It reads the text component and repacks each `t2_g128_fp16`
matrix into MLX 2-bit affine words (code `t + 1`, scale `s`, bias `-s`), which decode to the
stored values without rounding. It then runs them in mlx-lm's Qwen3.5 model. The loader applies
each Use's `hadamard_signs` rotation before the product and restores the token table to the
primal basis, as [docs/maintainer/qwen3_5-model.md](../docs/maintainer/qwen3_5-model.md#hadamard-rotated-checkpoints)
describes. It also keeps the bf16 activation boundaries of the CUDA engine's embedding gather
and linears.

`v-sekai/mlx/vlm.py` loads every component into mlx-vlm. The vision tower's q4/q5/q6/q8
row-split matrices are repacked to MLX affine words of the same width, also without rounding.
The MTP and DFlash2 heads become mlx-vlm's `qwen3_5_mtp` and `dflash2` drafters. Drafting is
greedy and opt-in. On an M2 Pro both drafters give the same text as plain decoding, but they
decode slower than it, because MLX's batched 2-bit matmul costs more per verify pass than the
accepted tokens save. `ninfer_mlx.py generate --mtp N` runs the earlier standalone MTP decoder
in `speculative.py`.

1. `pixi run --locked nu v-sekai/macos-convert-job.nu install` fetches the sources and converts
   them as a launchd oneshot. The log is `~/Library/Logs/ninfer-convert.log`.
2. `pixi run --locked -e mlx python v-sekai/mlx/ninfer_mlx.py generate --model models/bonsai2-27b-heretic.ninfer --prompt "..."`
   runs one prompt.
3. `pixi run --locked nu v-sekai/macos-serve-job.nu install` runs mlx-vlm's OpenAI server on
   the artifact as a launchd service at `http://127.0.0.1:8090/v1`, one sequence at a time. It
   accepts text and image messages. `--draft ninfer:mtp` or `--draft ninfer:dflash2` turns on
   a drafter. `uninstall` removes the service.

These checks cover the loader:

- `v-sekai/mlx/test_ninfer_mlx.py <artifact>` decodes 15 tensors (t2, q4, q5, q6 and q8, from
  text, MTP, proposal and vision) straight from the layout specification. The MLX repack must
  dequantise each one to the same values. A reversed lane map and a wrong row range must both
  fail.
- `v-sekai/mlx/test_vision.py <artifact> <Qwen3.8-27B dir>` compares all 111 vision matrices
  with the bf16 source by cosine, with a bar of 0.95. q4 rounding alone reaches 0.987. Rows
  rolled by one head must fail. It also asks about a generated image with a red circle, a blue
  square and the text "42", and requires the answer to name each shape, colour, side and the
  text.
- `v-sekai/mlx/test_drafters.py <artifact> [tokens] [mtp|dflash]` compares drafted greedy text
  with plain greedy text and reports acceptance. A DFlash2 feature projection with its per-layer
  blocks rotated must lose most of its acceptance.
- `v-sekai/mlx/test_mtp.py <artifact>` does the same for `speculative.py`. It accepts a first
  difference only where the plain run's top two logits were within one bf16 step.
- `v-sekai/mlx/parity_reference.py <artifact> <dir>` compares every text tensor the loader
  builds with an independent MLX conversion of the same PQ2_0 GGUF
  ([3hs4n/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-mlx-2bit](https://huggingface.co/3hs4n/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-mlx-2bit)).
  Ternary matrices are compared after dequantisation and sign vectors bit for bit. Dense
  tensors must match exactly, except that `A_log` may differ by one fp32 ulp, because both
  converters recover it as `log(-ssm_a)`.

Neither check compares logits with the CUDA engine. That comparison needs an NVIDIA host.
