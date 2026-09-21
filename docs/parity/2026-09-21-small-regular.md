# Regular small classifier parity

## Scope and provenance

* Date: 2026-09-21
* Repository SHA at run: `dd33647e3fd42686d48dcbbaae298fdb4ca6b3fb`
* CLI version: `dinov2-cli 0.3.0`
* HF reference: `facebook/dinov2-small-imagenet1k-1-layer`
* GGUF path used: `models/dinov2-small/model.gguf`
* GGUF SHA256: `b7ca009aa416f6be85ea95363f4ef13d3c139999f96ca3fd47226849afb712e`
* Image: `assets/tench.jpg`
* Environment: Darwin 25.6.0, Apple arm64, MacBookPro17,1, Apple M1, Metal backend, 8 CPUs

The checksum identifies the exact GGUF file used for this run. The model weights and build artifacts remain local and are not part of this evidence change.

## Command and gates

```sh
.venv/bin/python scripts/parity_check.py --cli build-debug/bin/dinov2-cli --gguf models/dinov2-small/model.gguf --hf-model facebook/dinov2-small-imagenet1k-1-layer --image assets/tench.jpg --cls-threshold 0.999 --pooled-threshold 0.999 --patches-threshold 0.99 --patches-token-min-threshold 0 --prob-tolerance 0.05
```

Gates were CLS cosine >= 0.999, pooled cosine >= 0.999, flat patch cosine and patch-token mean cosine >= 0.99, matching classification top-1 index and label, and top-1 probability absolute difference < 0.05. Patch-token minimum was informational only.

## Complete human-readable result

```text
== assets/tench.jpg
check                 ours                        ref                         metric              threshold     result
n_patches             1320                        1320                        match               equal         PASS
hidden                384                         384                         match               equal         PASS
cls                   norm 49.6426                norm 49.6581                cos 0.999911        >= 0.999      PASS
cls                                                                           max|d| 0.110233     info          -
pooled                norm 58.7241                norm 58.7457                cos 0.999928        >= 0.999      PASS
pooled                                                                        max|d| 0.110233     info          -
patches flat          1320x384                    1320x384                    cos 0.999570        >= 0.99       PASS
patches token                                                                 cos mean 0.999589   >= 0.99       PASS
patches token                                                                 below 0.99: 7/1320  info          -
patches token                                                                 cos p1 0.994550     info          -
patches token                                                                 cos p5 0.998686     info          -
patches token                                                                 cos min 0.979825    info          -
classify top-1        0 "tench, Tinca tinca"      0 "tench, Tinca tinca"      idx/label           equal         PASS
classify prob         0.982613                    0.982727                    |diff| 0.000114     < 0.05        PASS
verdict: PASS (8/8 checks passed; gates: cls>=0.999 pooled>=0.999 patches flat+mean>=0.99 classify (patch token min: informational))
```

## Verdict

**PASS**. This is evidence for this regular small checkpoint, image, CLI build, and environment only. It does not imply parity for other checkpoints or platforms.
