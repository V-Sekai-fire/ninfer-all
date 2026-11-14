---
license: other
pretty_name: NInfer Bonsai 2 27B Heretic artifact and sources
---

# NInfer Bonsai 2 27B Heretic: artifact and sources

The converted `.ninfer` artifact for Ternary Bonsai 2 27B Heretic v2, with every source file the
conversion read, so the artifact can be rebuilt or served without the upstream repositories.

## What it is for

- `models/` holds the artifact that the V-Sekai-fire/interactor-language-ninfer CUDA engine and its Apple silicon
  MLX loader (`v-sekai/mlx/`) serve, with the converter's `conversion.json` report.
- `sources/` holds the converter inputs. Their upstream repositories are fetched without a pinned
  revision, so this copy is the fixed input set.

## Provenance

One folder per upstream repository under `sources/`, named as `v-sekai/fetch-sources.nu` names it:

- `Heretic-v2-GGUF`: OS-Software/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-GGUF (PQ2_0 GGUF).
- `Qwen3.8-27B`: Qwen/Qwen3.8-27B configs, tokenizer and shard 1 (the vision tower).
- `Ternary-Bonsai-2-27B-MTP`, `Ternary-Bonsai-2-27B-DFlash2`: ProCreations drafter heads.

The independent MLX conversion that `v-sekai/mlx/parity_reference.py` compares against
(3hs4n/Ternary-Bonsai-2-27B-Uncensored-Heretic-v2-mlx-2bit) is not archived here; download it
from that public repository when needed.

`models/` is the output of `v-sekai/convert.nu` (recipe `bonsai2_27b_ternary`) on these sources.

## The artifact with a chip tower

`models/chip/` holds the same artifact with one more component, `chip`: an RV32IM processor
compiled into eleven Gated DeltaNet layers, which `ninfer-serve --chip calc-bridge` runs beside
the text model to answer a `calc` tool call from inside the artifact. Every other parameter is
bound to objects with the same bytes as in the artifact converted without the tower by the same
converter commit; `SHA256SUMS` beside it names both files.

- `sources/tower-27b/` is the tower's source: `config.json` and `model.safetensors` (the layers,
  the token table and the parked `calc-bridge` firmware state), written by
  `chipformer.rv.export_tower` in V-Sekai-fire/qwenputer for Qwen3.8-27B's head geometry.
- `sources/tower-27b/golden-calc-bridge.parquet` and `golden-calc-bridge.raw/` are the reference
  stream (17,792 positions) that the engine's tower tests are held to.
- `models/chip/` is the output of `v-sekai/convert.nu --chip sources/tower-27b`.

The design and its witnesses are in `docs/maintainer/chip-tower.md` of V-Sekai-fire/interactor-language-ninfer.

## Use

From a ninfer-all checkout:

    hf download chibifire/ninfer-bonsai2-heretic --repo-type dataset --local-dir .

Add a folder with `hf upload chibifire/ninfer-bonsai2-heretic <dir> <dir> --repo-type dataset` from
the checkout root, keeping the same layout.

## Licence

Each `sources/` folder keeps its upstream licence files and terms. The artifact is derived from
them. The dataset is private because it redistributes third-party weights.
