<!--
Maintainer template, not a published file.

Backbone-only DINOv2 checkpoints (facebook/dinov2-{small,base,large,giant}
without the imagenet1k-1-layer classifier head) are being converted and
published separately; they are not in these repos yet. See
docs/cli.md "Backbone-only checkpoints". Do not mention them in the card
until the GGUFs actually exist on the hub.

How to use: fill in {{REPO}}, {{SOURCE}}, {{SIZE_BYTES}}, {{SIZE_HUMAN}},
{{REGISTERS}}, {{HIDDEN}} from the lookup table at the bottom, then upload
the result as README.md to the target repo. The YAML frontmatter must be
the first bytes of the uploaded file (this comment block stays out).
Upload command:

  hf upload {{REPO}} README.md README.md --repo-type model

or via the hub web UI: Add file -> paste body.
-->
---
license: apache-2.0
tags:
  - gguf
  - dinov2
  - embeddings
  - image-feature-extraction
---

# {{REPO}}

f16 GGUF conversion of [`{{SOURCE}}`](https://huggingface.co/{{SOURCE}}) for
[dinov2.cpp](https://github.com/espetro/dinov2.cpp), a C++ port of Meta's
DINOv2 vision encoder that runs on ggml. No Python or PyTorch needed at
runtime.

- Format: GGUF, f16 weights
- File: `model.gguf` ({{SIZE_HUMAN}}, {{SIZE_BYTES}} bytes)
- Produced by `scripts/dinov2-to-gguf.py` and published by the
  `convert-and-publish-gguf` CI workflow (monthly + on demand)

## Install and run

```bash
# 1. Download this weight
hf download {{REPO}} --local-dir models
# -> models/model.gguf

# 2. Get a dinov2-cli binary
#    prebuilt: https://github.com/espetro/dinov2.cpp/releases
#    or build: cmake --preset release && cmake --build --preset release

# 3. Run
dinov2-cli -m models/model.gguf -i image.jpg -c                    # top-5 ImageNet labels
dinov2-cli -m models/model.gguf -i image.jpg --print-embeddings    # cls/pooled vectors as JSON
dinov2-cli -m models/model.gguf -i image.jpg -o pca.png            # PCA map of patch features
```

## What this variant contains

- Backbone: ViT-{{VARIANT_LETTER}}/14 distilled DINOv2, hidden size {{HIDDEN}}
- Register tokens: {{REGISTERS}}
- ImageNet1k linear classifier head: included (`-c` works out of the box)
- Outputs: `cls` embedding, `pooled` `[cls || mean(patches)]`, per-patch
  tokens with `--print-patch-tokens`

Register-token checkpoints reduce high-norm patch-token artifacts and give
smoother dense feature maps in the settings studied in
[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588).
Prefer a `with-registers` repo for patch/dense feature work; prefer the
plain repo for exact baseline reproduction or task-specific comparisons.

## Numerical parity

Embeddings from this GGUF are checked against the HF reference
implementation by a nightly CI job and committed run reports. On the small
checkpoints the recorded CLS cosine similarity is 0.9999+ and pooled cosine
is 0.9999+ (gates: cls/pooled >= 0.999, patch mean >= 0.99). Full evidence,
per-checkpoint provenance, and reproduction commands:
[docs/parity/](https://github.com/espetro/dinov2.cpp/tree/main/docs/parity).

## Links

- Engine and CLI: https://github.com/espetro/dinov2.cpp
- CLI reference: https://github.com/espetro/dinov2.cpp/blob/main/docs/cli.md
- Publishing pipeline: https://github.com/espetro/dinov2.cpp/blob/main/docs/hf-publishing.md
- Upstream model: https://huggingface.co/{{SOURCE}}

<!--
Lookup table (measured 2026-09-23 via curl -sIL on /resolve/main/model.gguf,
x-linked-size header). SIZE_HUMAN uses decimal GB/MB.

| REPO                                                | SOURCE                                                     | SIZE_BYTES  | SIZE_HUMAN | REGISTERS | HIDDEN | VARIANT_LETTER |
|-----------------------------------------------------|------------------------------------------------------------|-------------|------------|-----------|--------|----------------|
| dinov2-cpp-core/dinov2-small-gguf                   | facebook/dinov2-small-imagenet1k-1-layer                   | 46,901,760  | ~46.9 MB   | 0         | 384    | S              |
| dinov2-cpp-core/dinov2-base-gguf                    | facebook/dinov2-base-imagenet1k-1-layer                    | 178,678,272 | ~178.7 MB  | 0         | 768    | B              |
| dinov2-cpp-core/dinov2-large-gguf                   | facebook/dinov2-large-imagenet1k-1-layer                   | 616,453,984 | ~616.5 MB  | 0         | 1024   | L              |
| dinov2-cpp-core/dinov2-giant-gguf                   | facebook/dinov2-giant-imagenet1k-1-layer                   | 2,285,418,624 | ~2.29 GB | 0         | 1536   | g              |
| dinov2-cpp-core/dinov2-with-registers-small-gguf    | facebook/dinov2-with-registers-small-imagenet1k-1-layer    | 46,907,968  | ~46.9 MB   | 4         | 384    | S              |
| dinov2-cpp-core/dinov2-with-registers-base-gguf     | facebook/dinov2-with-registers-base-imagenet1k-1-layer     | 178,690,624 | ~178.7 MB  | 4         | 768    | B              |
| dinov2-cpp-core/dinov2-with-registers-large-gguf    | facebook/dinov2-with-registers-large-imagenet1k-1-layer    | 616,470,432 | ~616.5 MB  | 4         | 1024   | L              |
| dinov2-cpp-core/dinov2-with-registers-giant-gguf    | facebook/dinov2-with-registers-giant-imagenet1k-1-layer    | 2,285,443,296 | ~2.29 GB | 4         | 1536   | g              |

Register count per the DINOv2 release is 4 for all with-registers sizes;
confirm against the checkpoint config if a card claims it.
-->
