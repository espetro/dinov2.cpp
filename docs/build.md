# Building from source

If the prebuilt binaries in [releases](https://github.com/espetro/dinov2.cpp/releases) don't cover your platform, or you want per-device optimizations, build from source.

## Simple build

All image decoding dependencies (stb) are vendored, so no external image libraries are required.

```bash
# macOS / Linux
cmake --preset release && cmake --build --preset release
./build-release/bin/inference -m ggml-model.gguf -i assets/tench.jpg -c
```

```bash
# Windows (Ninja)
cmake --preset release
cmake --build --preset release
.\build-release\bin\inference.exe -m ggml-model.gguf -i assets\tench.jpg -c
```

Use `-c` for classification output. Omitting the flag returns backbone PCA features (written to `pca_visual.png` by default).

## CLI options

```text
usage: ./bin/inference [options]

options:
  -h, --help              show this help message and exit
  -m FNAME, --model       model path (default: ../ggml-model.gguf)
  -i FNAME, --inp         input file (default: ../assets/tench.jpg)
  -o FNAME, --out         output file for backbone PCA features (default: pca_visual.png)
  -k N, --topk            top k classes to print (default: 5)
  -t N, --threads         number of threads to use during computation (default: 4)
  -c, --classify          whether to classify the image or get backbone PCA features (default: 0)
  -fa, --flash_attn       whether to enable flash_attn, less accurate (default: 0)
```

The optimal thread count is usually the number of physical cores; more is not always better.

## Per-device optimizations

Generate per-device instructions with `-march=native` in the compiler flags (multi-threading, vectorization, loop unrolling).

### For AMD host processors

Use AMD's specialized compiler to make full use of your processor's architecture: [AMD Optimizing C/C++ and Fortran Compilers (AOCC)](https://www.amd.com/en/developer/aocc.html). Modern CPUs see the greatest benefit; older CPUs may see little to no improvement.

### Using OpenMP

Compile with `-fopenmp` (add it to the compiler flags in CMakeLists.txt) to enable multithreaded runs:

```bash
OMP_NUM_THREADS=4 ./bin/inference -t 4 -m ggml-model.gguf -i assets/tench.jpg
```

## Quantization

ggml quantization types q4_0, q4_1, q5_0, q5_1 and q8_0 are supported. Quantize an f16 GGUF model with the `quantize` binary:

```bash
./bin/quantize ggml-model.gguf ggml-model-quant.gguf 7   # q5_1
```

Type codes: 2=q4_0, 3=q4_1, 6=q5_0, 7=q5_1, 8=q8_0. Then use `ggml-model-quant.gguf` like any f16 model. Quantized-model benchmark numbers are in [benchmarks.md](benchmarks.md).

## Benchmarks

To measure inference speed on your machine:

```bash
chmod +x scripts/benchmark.*
pip install memory_profiler threadpoolctl

python scripts/benchmark.py        # PyTorch baseline
./scripts/benchmark.sh             # dinov2.cpp, non-quantized
./scripts/benchmark.sh 4 1         # quantized models, 4 threads
```

Both scripts use 4 threads by default; `threadpoolctl` limits PyTorch's thread count for a fair comparison.
