# Audit-hardening rerun: both small checkpoints

## Scope and provenance

* Date: 2026-09-22
* Code state: `fix/audit-hardening` @ `d2b6847` (five audit commits on top of `ba2e3d5`)
* Purpose: confirm the F3 flash-attention K/V padding fix and the F4
  position-embedding rewrite did not regress parity. The recorded runs below
  use the default (non-flash) attention path; F3 only changes `-fa` output.
* GGUF paths: `models/dinov2-small/model.gguf`,
  `models/dinov2-with-registers-small/model.gguf`
* Image: `assets/tench.jpg`
* Environment: Darwin arm64, Metal backend

## Commands

The README reproduce commands, verbatim, once per checkpoint.

## Results vs the 2026-09-21 records

Regular small (`facebook/dinov2-small-imagenet1k-1-layer`):

| metric | 2026-09-21 | 2026-09-22 | delta |
| --- | --- | --- | --- |
| cls cosine | 0.999911 | 0.999911 | 0 |
| cls max abs diff | 0.110232 | 0.110233 | +1e-6 |
| pooled cosine | 0.999928 | 0.999928 | 0 |
| patches flat cosine | 0.999571 | 0.999570 | -1e-6 |
| patch-token mean cosine | 0.999589 | 0.999589 | 0 |
| top-1 prob (ours) | 0.982613 | 0.982613 | 0 |
| top-1 prob (ref) | 0.982652 | 0.982727 | +7.5e-5 |

Verdict: PASS (8/8).

Register small (`facebook/dinov2-with-registers-small-imagenet1k-1-layer`):

| metric | 2026-09-21 | 2026-09-22 | delta |
| --- | --- | --- | --- |
| cls cosine | 0.999984 | 0.999984 | 0 |
| cls max abs diff | 0.050792 | 0.050797 | +5e-6 |
| pooled cosine | 0.999984 | 0.999983 | -1e-6 |
| patches flat cosine | 0.999847 | 0.999847 | 0 |
| patch-token mean cosine | 0.999836 | 0.999836 | 0 |
| top-1 prob (ours) | 0.984967 | 0.984967 | 0 |
| top-1 prob (ref) | 0.978853 | 0.978378 | -4.75e-4 |

Verdict: PASS (8/8).

## Notes

The ours-side deltas are at most a few ulps, consistent with Metal kernel
nondeterminism; F4 is mathematically identical for this input (the 44x30
grid still interpolates from the 37x37 table through the same Catmull-Rom
kernel, now via one interleaved `resize_planes` call instead of per-channel
gather/resize/scatter). The `ref` column moved slightly because the HF
reference is recomputed live and the reference stack (transformers / image
processor) is environment-dependent; the CLI output itself did not move.

No numerical changes were expected from F3 in these runs because parity
uses the default attention path. The `-fa` path now omits the unmasked
zero-padding of the K/V sequence dim, which corrects a real bias; there is
no recorded `-fa` baseline to diff against.
