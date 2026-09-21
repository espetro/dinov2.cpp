# Register small classifier parity

## Scope and provenance

* Date: 2026-09-21
* Recorded CLI/model/parity run code state: `482ea51ad79412c6f7df6d6c538a983345302004`
* CLI version: `dinov2-cli 0.4.0`
* HF reference: local `/tmp/dinov2-final-work/hf/dinov2-with-registers-small-imagenet1k-1-layer` downloaded from `facebook/dinov2-with-registers-small-imagenet1k-1-layer`
* GGUF path used: `/tmp/dinov2-final-work/out/dinov2-with-registers-small-imagenet1k-1-layer/ggml-model.gguf` converted at this SHA
* GGUF SHA256: `8a78711f45b218dd213ca1cc76d1c6089ba21bbfe055be397d00eaf30db9ad82`
* Image: `assets/tench.jpg`
* Environment: Darwin 25.6.0, Apple arm64, MacBookPro17,1, Apple M1, Metal backend, 8 CPUs

The checksum identifies the exact GGUF file used for this rerun. All CLI output and parity checks were produced at the final SHA above. The model weights and build artifacts remain local and are not part of this evidence commit.

## Command and gates

```sh
/tmp/dinov2-final-venv/bin/python scripts/parity_check.py --cli build-debug/bin/dinov2-cli --gguf /tmp/dinov2-final-work/out/dinov2-with-registers-small-imagenet1k-1-layer/ggml-model.gguf --hf-model /tmp/dinov2-final-work/hf/dinov2-with-registers-small-imagenet1k-1-layer --image assets/tench.jpg --cls-threshold 0.999 --pooled-threshold 0.999 --patches-threshold 0.99 --patches-token-min-threshold 0 --prob-tolerance 0.05
```

Gates were CLS cosine >= 0.999, pooled cosine >= 0.999, flat patch cosine and patch-token mean cosine >= 0.99, matching classification top-1 index and label, and top-1 probability absolute difference < 0.05. Patch-token minimum was informational only.

## Complete human-readable result

```text
== assets/tench.jpg
check                 ours                        ref                         metric              threshold     result
n_patches             1320                        1320                        match               equal         PASS
hidden                384                         384                         match               equal         PASS
cls                   norm 26.2428                norm 26.2490                cos 0.999984        >= 0.999      PASS
cls                                                                           max|d| 0.050792     info          -
pooled                norm 31.7880                norm 31.8052                cos 0.999984        >= 0.999      PASS
pooled                                                                        max|d| 0.050792     info          -
patches flat          1320x384                    1320x384                    cos 0.999847        >= 0.99       PASS
patches token                                                                 cos mean 0.999836   >= 0.99       PASS
patches token                                                                 below 0.99: 1/1320  info          -
patches token                                                                 cos p1 0.997877     info          -
patches token                                                                 cos p5 0.999353     info          -
patches token                                                                 cos min 0.979199    info          -
classify top-1        0 "tench, Tinca tinca"      0 "tench, Tinca tinca"      idx/label           equal         PASS
classify prob         0.984967                    0.978853                    |diff| 0.006114     < 0.05        PASS
verdict: PASS (8/8 checks passed; gates: cls>=0.999 pooled>=0.999 patches flat+mean>=0.99 classify (patch token min: informational))
```

## Verdict

**PASS**. This is evidence for this register-token small checkpoint, image, CLI build, and environment only. It does not imply parity for other checkpoints or platforms.
