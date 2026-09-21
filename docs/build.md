# Building from source

If the prebuilt binaries in [releases](https://github.com/espetro/dinov2.cpp/releases) don't cover your platform, or you want per-device optimizations, build from source.

## Simple build

All image decoding dependencies (stb) are vendored, so no external image libraries are required.

```bash
# macOS / Linux
cmake --preset release && cmake --build --preset release
./build-release/bin/dinov2-cli -m models/model.gguf -i assets/tench.jpg -c
```

```bash
# Windows (Ninja)
cmake --preset release
cmake --build --preset release
.\build-release\bin\dinov2-cli.exe -m models\model.gguf -i assets\tench.jpg -c
```

Use `-c` for classification output. Backbone PCA features are opt-in via `-o out.png` (PNG output). For the full flag reference see [cli.md](cli.md).

## CLI options

```text
usage: ./bin/dinov2-cli [options]

Model:
  -m FNAME, --model     model path (default: ../model.gguf)
  -fa, --flash_attn     enable flash attention, less accurate (default: off)
  -t N, --threads       number of threads to use during computation (default: 4)

Input:
  -i FNAME, --inp       input image file (default: ../assets/tench.jpg)
  -s N, --seed          RNG seed (default: 42)

Output modes:
  -c, --classify        classify the image and print top-k labels (default: off)
  -k N, --topk          top k classes to print (default: 5)
  --print-embeddings    emit one JSON object on stdout with cls/pooled embeddings
  --print-patch-tokens  include per-patch token vectors in the JSON output
  --l2-normalize        L2-normalize emitted embedding vectors
  -o FNAME, --out       write PCA visualization of patch features to FNAME (default: off)

Benchmark:
  --bench               enable bench loop (default repeats=5, warmup=1); skips PCA image output
  --bench-runs N        number of timed runs (overrides default 5 when --bench is set)
  --bench-warmup N      number of warmup runs discarded before timing (default: 1)
  --bench-json          emit one JSON object per line to stdout instead of markdown row

Misc:
  -h, --help            show this help message and exit
  --version             print version and exit
```

The optimal thread count is usually the number of physical cores; more is not always better.
The `--bench` path skips PCA visualization -- the `-o` flag is ignored when `--bench` is set.

## Per-device optimizations

Generate per-device instructions with `-march=native` in the compiler flags (multi-threading, vectorization, loop unrolling).

### For AMD host processors

Use AMD's specialized compiler to make full use of your processor's architecture: [AMD Optimizing C/C++ and Fortran Compilers (AOCC)](https://www.amd.com/en/developer/aocc.html). Modern CPUs see the greatest benefit; older CPUs may see little to no improvement.

### Using OpenMP

Compile with `-fopenmp` (add it to the compiler flags in CMakeLists.txt) to enable multithreaded runs:

```bash
OMP_NUM_THREADS=4 ./bin/dinov2-cli -t 4 -m models/model.gguf -i assets/tench.jpg -c
```

## Quantization

Quantized GGUFs (q4_0, q4_1, q5_0, q5_1, q8_0) are not produced by this
repo. They are downloaded from [`dinov2-cpp-core/<variant>-gguf`](https://huggingface.co/dinov2-cpp-core)
on Hugging Face, which publishes them via the
`convert-and-publish-gguf` workflow. The pre-built `dinov2-cli` binary
loads any ggml-supported quant type transparently — pass any of those
GGUFs as `-m`.

## Benchmarks

To measure inference speed on your machine, build first, then:

```bash
cmake --preset release && cmake --build --preset release

# Run the dinov2-cli binary directly with --bench (single-shot, JSON output):
./build/bin/dinov2-cli -m models/dinov2-vit-small-patch14/model.gguf \
    -i assets/tench.jpg -t 4 -c \
    --bench --bench-runs 5 --bench-json

# Or sweep the model size matrix via the bench script:
scripts/bench.sh                                   # 4 models x f16, default 5 repeats
scripts/bench.sh --models small,base --repeats 10
scripts/bench.sh --out ./my-results.md             # write to a custom path
```

Flags: `--models <csv>` (default `small,base,large,giant`), `--repeats <N>`
(default 5), `--threads <N>` (default 12), `--warmup <N>` (default 1),
`--out <PATH>` (default `./benchmark_results.md`), `--aggregate <inputs> --out <PATH>`
(concatenate per-platform tables).
Pre-stage GGUFs under `models/dinov2-vit-{size}-patch14/model.gguf` first;
the script will exit with a clear error if a model is missing. Only f16
GGUFs are supported as input — pull pre-quantized variants from
[`dinov2-cpp-core/<variant>-gguf`](https://huggingface.co/dinov2-cpp-core)
directly.

Both scripts use 4 threads by default; `threadpoolctl` limits PyTorch's
thread count for a fair comparison.
