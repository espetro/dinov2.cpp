# container: vision embeddings in a tiny scratch image

A `FROM scratch` image that ships a statically linked `dinov2-cli` plus the
pinned `dinov2-small` GGUF weight. No shell, no libc, no package manager:
the whole runtime is one binary and one model file, roughly 50 MB total.

Tier-2 example: not part of the shipped binary or CI gates. Build context
is this directory only; the Dockerfile clones the source at `DINOV2_REF`
(default `v0.4.0`) and downloads `model.gguf` from Hugging Face with a
pinned sha256, so the build fails loudly if either upstream changes.

## Build

```bash
docker build -t dinov2 examples/container/
```

Multi-stage layout:

1. `ubuntu:24.04` stage fetches `model.gguf` (sha256 verified), clones the
   repo with the `ggml` submodule, and builds `dinov2-cli` with
   `-DBUILD_SHARED_LIBS=OFF` and
   `-DCMAKE_EXE_LINKER_FLAGS="-static -static-libgcc -static-libstdc++"`.
   A `readelf` guard fails the build if the binary still has dynamic
   dependencies.
2. `FROM scratch` copies only `/dinov2-cli` and `/model.gguf`.

`GGML_NATIVE=OFF` keeps the binary portable, so the same Dockerfile
produces correct images on x86-64 and arm64 hosts (including emulated
`--platform` builds).

## Run

```bash
# version / help (no image needed)
docker run --rm dinov2 --version
docker run --rm dinov2            # prints usage via CMD ["--help"]

# embed an image; mount a folder of inputs at /imgs
docker run --rm -v "$PWD/imgs:/imgs" dinov2 \
  -m /model.gguf -i /imgs/a.jpg --print-embeddings | jq .

# every flag of dinov2-cli works the same way
docker run --rm -v "$PWD/imgs:/imgs" dinov2 \
  -m /model.gguf -i /imgs/a.jpg -i /imgs/b.jpg --batch 2 --print-embeddings
```

The image's `ENTRYPOINT` is `dinov2-cli`, so arguments after the image name
go straight to the CLI. `ENTRYPOINT ["/dinov2-cli"]` means there is no
shell inside; pass flags directly, and use `--entrypoint` only if you
bind-mount your own binary for debugging.

## Size

Measured on an arm64 host (x86-64 is within a few MB):

| image | size |
| --- | --- |
| `dinov2` (this demo, scratch, arm64 build) | 45 MB |
| `gcr.io/distroless/cc-debian12` runtime alone | ~30 MB |
| `pytorch/pytorch` (CUDA + PyTorch + Python) | ~7 GB |

Same embedding quality as the desktop CLI; the 100x+ difference vs the
PyTorch image is entirely the runtime: no Python, no CUDA stack, no OS
userland. If a fully static build ever regresses (for example a future
ggml backend that requires dlopen), the documented fallback is
`gcr.io/distroless/cc-debian12` plus the dynamically linked release
binary, which lands around 80 MB with the same model.

## Build args

| arg | default | purpose |
| --- | --- | --- |
| `DINOV2_REF` | `v0.4.0` | git tag/branch/commit cloned in the build stage |
| `MODEL_URL` | HF `dinov2-small-gguf` resolve URL | where `model.gguf` is fetched |
| `MODEL_SHA256` | pinned digest | build fails if the download does not match |

```bash
docker build -t dinov2 --build-arg DINOV2_REF=main examples/container/
```

## Notes

- The model is baked in for zero config startup. For a slimmer image,
  drop the `COPY` of `model.gguf` and mount it instead:
  `docker run -v "$PWD/models:/models" dinov2 -m /models/model.gguf ...`.
- Static linking covers OpenMP (`libgomp`), pthread, libc and libstdc++.
  `ggml` links `dl` on Linux, but with `GGML_BACKEND_DL=OFF` no `dlopen`
  call survives into the binary, so a glibc static link is safe here.
- Networking is unnecessary at runtime; run with `--network none` for a
  fully offline encode.
