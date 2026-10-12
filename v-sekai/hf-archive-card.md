---
license: other
pretty_name: NInfer Bonsai 2 27B Heretic artifact and sources
---

# NInfer Bonsai 2 27B Heretic: artifact and sources

The converted `.ninfer` artifact for Ternary Bonsai 2 27B Heretic v2, with every source file the
conversion read, so the artifact can be rebuilt or served without the upstream repositories.

## What it is for

- `models/` holds the artifact that the V-Sekai-fire/ninfer-all CUDA engine and its Apple silicon
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

## Use

From a ninfer-all checkout:

    hf download chibifire/ninfer-bonsai2-heretic --repo-type dataset --local-dir .

Add a folder with `hf upload chibifire/ninfer-bonsai2-heretic <dir> <dir> --repo-type dataset` from
the checkout root, keeping the same layout.

## Licence

Each `sources/` folder keeps its upstream licence files and terms. The artifact is derived from
them. The dataset is private because it redistributes third-party weights.
