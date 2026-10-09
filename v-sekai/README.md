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
