# Architecture

## Goals

`dinov2.cpp` is a from-scratch C++ port of Meta's DINOv2 vision encoder that runs on
the [ggml](https://github.com/ggml-org/ggml) tensor library. It is **one CLI binary**
(`dinov2-cli`) that loads any ggml-supported GGUF weight, decodes an image with
vendored `stb_image`, runs the encoder graph, and prints classification
predictions, embeddings JSON, or PCA-visualised patch features. It is **not** a Python wrapper, a
training framework, a model converter, or a multi-backend serving system: those
concerns live in adjacent repos (`dinov2-cpp-core` for GGUF conversion and HF
distribution, `ggml` for backends).

## Layer model

The dependency rule: source code dependencies point strictly **downward**. The
CLI shell only includes the public header; the implementation includes ggml;
ggml is a vendored submodule that includes nothing of ours.

```mermaid
graph TD
    A["dinov2-cli (dinov2-cli.cpp)<br/>CLI shell: argv parsing, image load,<br/>graph build, output formatting"]
    B["dinov2.h<br/>public API: structs, signatures,<br/>IMAGENET defaults"]
    C["dinov2.cpp<br/>encoder graph:<br/>attn, mlp, swiglu_ffn, build_graph,<br/>dino_predict"]
    D["src/image.h + src/image.cpp<br/>stb wrappers: load_image,<br/>dino_preprocess, dino_classify_preprocess,<br/>preprocess_resize_crop (bounded/hf/crop518)"]
    E["ggml/ submodule<br/>tensor library + CPU backend"]

    A --> B
    B --> C
    B --> D
    C --> E
    D --> E

    classDef cli    fill:#dde7f3,stroke:#557,color:#000;
    classDef api    fill:#f3e7dd,stroke:#755,color:#000;
    classDef impl   fill:#e7f3dd,stroke:#575,color:#000;
    classDef vend   fill:#f0f0f0,stroke:#888,color:#000;
    class A cli
    class B api
    class C,D impl
    class E vend
```

Arrows never point upward. `dinov2-cli.cpp` does not include `ggml.h` directly;
it talks to ggml only through the opaque struct types in `dinov2.h`.

## File-by-file index (role groups)

```mermaid
graph LR
    classDef core  fill:#e7f3dd,stroke:#575,color:#000;
    classDef lib   fill:#dde7f3,stroke:#557,color:#000;
    classDef cli   fill:#f3e7dd,stroke:#755,color:#000;
    classDef test  fill:#fde7f3,stroke:#a55,color:#000;
    classDef tool  fill:#f3f3dd,stroke:#770,color:#000;
    classDef build fill:#eee,stroke:#777,color:#000;
    classDef ci    fill:#eef,stroke:#77a,color:#000;
    classDef doc   fill:#fee,stroke:#a77,color:#000;
    classDef asset fill:#efe,stroke:#7a7,color:#000;
    classDef sub   fill:#f0f0f0,stroke:#888,color:#000;

    Core["dinov2.h<br/>dinov2.cpp"]
    Lib["src/image.h<br/>src/image.cpp"]
    CLI["dinov2-cli.cpp"]
    Test["tests/test_image.cpp<br/>tests/test_dinov2.cpp"]
    Tool["scripts/bench.sh<br/>scripts/dinov2-to-gguf.py<br/>scripts/publish-gguf.sh"]
    Build["CMakeLists.txt<br/>CMakePresets.json<br/>src/stb_image.h<br/>src/stb_image_write.h<br/>src/doctest.h"]
    CI[".github/workflows/release.yml<br/>.github/workflows/build.yml<br/>.github/workflows/convert-and-publish-gguf.yml"]
    Doc["README.md<br/>CONTRIBUTING.md<br/>docs/cli.md<br/>docs/build.md<br/>docs/benchmarks.md<br/>docs/hf-publishing.md<br/>ARCHITECTURE.md<br/>RELEASE_NOTES_v0.2.0.md<br/>RELEASE_NOTES_v0.3.0.md"]
    Asset["assets/logo/<br/>assets/tench.jpg<br/>assets/logo.png"]
    Sub["ggml/ (submodule)"]

    class Core core
    class Lib lib
    class CLI cli
    class Test test
    class Tool tool
    class Build build
    class CI ci
    class Doc doc
    class Asset asset
    class Sub sub
```

Abridged table:

| Path | Role | One-line purpose |
|:-----|:-----|:-----------------|
| `dinov2.h`, `dinov2.cpp` | Core | Public API + encoder graph. |
| `src/image.{h,cpp}` | Library | stb-backed image load + `dino_preprocess` / feature-mode `--preprocess` recipes (bounded 518 bound, hf, crop518). |
| `dinov2-cli.cpp` | CLI | `main`: arg parsing + bench loop. |
| `tests/test_{image,dinov2}.cpp` | Test | doctest pure-function coverage. |
| `scripts/{bench.sh,dinov2-to-gguf.py,publish-gguf.sh}` | Tool | Bench sweep + PyTorch→GGUF + HF upload. |
| `CMakeLists.txt`, `CMakePresets.json`, vendored stb/doctest | Build | CMake build + vendored single-file deps. |
| `.github/workflows/*` | CI | 3 workflows: release, build, convert-and-publish. |
| `README.md`, `CONTRIBUTING.md`, `docs/*`, `ARCHITECTURE.md`, `RELEASE_NOTES_*.md` | Doc | User + contributor-facing docs. |
| `assets/*` | Asset | Logo + default input image. |
| `ggml/` | Submodule | Vendored ggml at pinned SHA. |

## Build targets

```mermaid
graph LR
    CLI["dinov2-cli.cpp"]
    LIB["dinov2.cpp"]
    IMG["src/image.cpp"]
    TST_IMG["tests/test_image.cpp"]
    TST_LIB["tests/test_dinov2.cpp"]
    GGML["ggml/ (add_subdirectory)"]

    CLI_EXE["dinov2-cli (add_executable)"]
    TEST_IMG["test_image (add_executable)"]
    TEST_LIB["test_dinov2 (add_executable)"]
    CTEST["ctest"]

    CLI --> CLI_EXE
    LIB --> CLI_EXE
    IMG --> CLI_EXE
    LIB --> TEST_LIB
    IMG --> TEST_LIB
    TST_LIB --> TEST_LIB
    IMG --> TEST_IMG
    TST_IMG --> TEST_IMG
    GGML --> CLI_EXE
    GGML --> TEST_LIB
    TEST_IMG --> CTEST
    TEST_LIB --> CTEST
```

`test_image` and `test_dinov2` register themselves with `add_test(NAME …)`;
`ctest --test-dir build` runs both. `dinov2-cli` is not a test (it needs a
real GGUF + image that the test environment does not guarantee).

## CLI surface

`dinov2-cli --help`:

```text
usage: ./bin/dinov2-cli [options]

Model:
  -m FNAME, --model     model path (default: ../model.gguf)
  -fa, --flash_attn     enable flash attention, less accurate (default: off)
  -t N, --threads       number of threads to use during computation (default: 4)

Input:
  -i FNAME, --inp       input image file; repeat or comma-separate for several
                        (default: ../assets/tench.jpg)
  -s N, --seed          RNG seed (default: 42)
  --batch N             max images per forward pass; inputs run in chunks of N
                        (default: 1, max: 64)

Output modes:
  -c, --classify        classify each input image and print top-k labels (default: off)
  -k N, --topk          top k classes to print (default: 5)
  --print-embeddings    emit embeddings JSON on stdout, one object per input image (JSONL)
  --print-patch-tokens  include per-patch token vectors in the JSON output
  --l2-normalize        L2-normalize emitted embedding vectors
  -o FNAME, --out       write PCA visualization of patch features to FNAME; with multiple
                        inputs FNAME is a directory for <input-stem>.pca.png files

Benchmark:
  --bench               enable bench loop (default repeats=5, warmup=1); skips PCA image output
  --bench-runs N        number of timed runs (overrides default 5 when --bench is set)
  --bench-warmup N      number of warmup runs discarded before timing (default: 1)
  --bench-json          emit one JSON object per line to stdout instead of markdown row

Misc:
  -h, --help            show this help message and exit
  --version             print version and exit

Workflows:
  dinov2-cli -m model.gguf -i img.jpg -c                        # classify: top-k labels
  dinov2-cli -m model.gguf -i img.jpg --print-embeddings        # embeddings JSON on stdout
  dinov2-cli -m model.gguf -i img.jpg --print-embeddings --print-patch-tokens
                                                                # + per-patch tokens
  dinov2-cli -m model.gguf -i a.jpg -i b.jpg --batch 2 --print-embeddings
                                                                # batch: one JSON line per image
  dinov2-cli -m model.gguf -i img.jpg -o pca.png                # PCA viz of patch features
  dinov2-cli -m model.gguf -i img.jpg --bench --bench-json      # benchmark, JSON lines

docs: https://raw.githubusercontent.com/espetro/dinov2.cpp/main/docs/cli.md
```

Flow:

```mermaid
flowchart TD
    start([argv]) --> parse["dino_params_parse"]
    parse --> load_img["load_image (stb)"]
    load_img --> load_model["dino_model_load<br/>(gguf -> ggml tensors)"]
    load_model --> alloc["ggml_gallocr_new"]
    alloc --> branch{--bench?}
    branch -- no --> single["chunked single-shot predict (n_batch per pass)<br/>+ JSONL / top-k / PCA outputs"]
    branch -- yes --> loop["warmup x N + bench x M<br/>each run covers all inputs<br/>emit JSON or stderr row"]
    single --> cleanup["free ctx + buffer + backend"]
    loop --> cleanup
    cleanup --> exit([return 0])
```

Every flag maps to one `dino_params` field (`dinov2.h:67-88`).
`--print-embeddings` and `--bench-json` are the stable machine-readable
outputs; `scripts/bench.sh` parses `--bench-json` lines one by one.

## Public API surface

Verbatim from `dinov2.h` (the entire 120-line public surface):

```c++
struct ImgSize { int width = 0; int height = 0; };

constexpr float IMAGENET_DEFAULT_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_DEFAULT_STD[3]  = {0.229f, 0.224f, 0.225f};

uint32_t get_val_u32(const struct gguf_context *ctx, const char *key);
const char *get_val_str(const struct gguf_context *ctx, const char *key);

void l2_normalize(std::vector<float> &v);

struct dino_hparams {
    uint32_t hidden_size, num_hidden_layers, num_attention_heads;
    uint32_t num_classes, num_register_tokens, patch_size, img_size;
    uint32_t ftype; float eps; std::string interpolation;
    std::map<int, std::string> id2label;
    uint32_t n_enc_head_dim() const, n_img_size() const,
              n_patch_size() const, n_img_embd() const;
};

struct dino_model {
    dino_hparams hparams;
    struct ggml_context *ctx;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer;
    std::map<std::string, struct ggml_tensor *> tensors;
};

struct dino_params {
    uint32_t seed = 42, topk = 5;
    uint32_t n_batch = 1;           // max images per forward pass (--batch)
    bool enable_flash_attn = false;
    uint32_t n_threads = std::min(4u, std::thread::hardware_concurrency());
    bool classify = false;
    bool print_embeddings = false;
    bool print_patch_tokens = false;
    bool l2_normalize = false;
    std::string model = "../model.gguf";
    std::vector<std::string> fnames_inp = {"../assets/tench.jpg"};
    std::string image_out = "";     // PCA visualization is opt-in via -o
    float eps = 1e-6f;
    uint32_t bench_repeats = 0;     // --bench default: 5
    uint32_t bench_warmup = 1;
    bool bench_json = false;
};

// encoder graph
struct ggml_tensor *attn(...), *mlp(...), *swiglu_ffn(...);
void forward_features(...), forward_head(...);

struct dino_output {
    std::optional<std::vector<uint32_t>> preds;        // top-k indices (classify)
    std::optional<std::vector<float>>    pred_scores;  // top-k probabilities (classify)
    std::optional<std::vector<float>>    cls_token;    // hidden_size floats (both modes)
    std::optional<std::vector<float>>    pooled;       // [cls || mean(patches)] (feature)
    std::optional<std::vector<float>>    patch_tokens; // n_patches x hidden (feature)
};

ImageF dino_classify_preprocess(const Image &img, const dino_hparams &params);
ImageF dino_preprocess(const Image &img, const dino_hparams &params);

bool dino_model_load(ImgSize img_size, const std::string &fname,
                     dino_model &model, const dino_params &params);

std::vector<float> interpolate_pos_embed(ImgSize img_size,
                                         const float *pos_embed_data,
                                         const dino_hparams &hparams);

struct ggml_cgraph *build_graph(ImgSize img_size, struct ggml_context *ctx_cgraph,
                                const dino_model &model, const dino_params &params);

// batch form: 1..n_batch same-dims images -> one output per image
std::vector<dino_output> dino_predict(const dino_model &model,
                                      const std::vector<ImageF> &imgs,
                                      const dino_params &params, ggml_gallocr_t allocr);
// single-image convenience wrapper around the batch form
std::unique_ptr<dino_output> dino_predict(const dino_model &model, const ImageF &img,
                                          const dino_params &params, ggml_gallocr_t allocr);

void print_usage(FILE *out, int argc, char **argv, const dino_params &params);
bool dino_params_parse(int argc, char **argv, dino_params &params);
```

`dinov2.h` includes `ggml.h` and `src/image.h` so callers transitively pick up
ggml's types and `Image` / `ImageF`. There are no `extern "C"` exports — this
is C++, consumed by the CLI shell in the same translation-unit set.

## What ships per release

Each `v*` tag triggers `.github/workflows/release.yml`, which builds **one
binary per platform** (the `quantize` binary that existed through v0.2.0 is
gone as of v0.3.0):

| Platform    | Asset                                     |
|:------------|:------------------------------------------|
| macOS arm64 | `dinov2-v0.3.0-bin-macos-arm64.tar.gz`    |
| Linux x64   | `dinov2-v0.3.0-bin-ubuntu-x64.tar.gz`     |
| Linux arm64 | `dinov2-v0.3.0-bin-ubuntu-arm64.tar.gz`   |
| Windows x64 | `dinov2-bin-win-cpu-x64.zip` (contains `.exe`) |

The `release` aggregation job downloads all four artifacts, runs `sha256sum`
on each, and attaches `SHA256SUMS.txt` plus the per-platform archives to the
GitHub Release.

## What doesn't ship

Deliberately untracked by `.gitignore` and not in release archives:

- `build/`, `build-*/`, `cmake-build-debug/` — local CMake output.
- `.venv/`, `.venv-publish/` — per-run converter venv (created by `publish-gguf.sh`).
- `*.gguf`, `/models`, `/data` — pre-converted weights live on HF; users pull
  them with `hf download`.
- `.agents/` — agent working notes (memory + plans); force-added per global
  AGENTS.md policy but not in release tarballs.
- `.publish-logs/` — per-run audit trail from `publish-gguf.sh`.
- `.DS_Store`, `.env` — OS + local secrets.

## Where to read next

- [CONTRIBUTING.md](CONTRIBUTING.md): dev harness, CMake presets, sanitizer
  builds, clang-format gate, PR guidelines.
- [docs/cli.md](docs/cli.md): full CLI reference: flags, output modes,
  embeddings JSON schema, workflows, exit codes.
- [docs/build.md](docs/build.md): per-device optimisations, OpenMP, sanitizer
  presets.
- [docs/benchmarks.md](docs/benchmarks.md): how to read the tables,
  methodology, reproducing locally.
- [docs/hf-publishing.md](docs/hf-publishing.md): `HF_TOKEN` setup, the
  `convert-and-publish-gguf` workflow, the 8 published HF repos.
- [README.md](README.md): quickstart, downloads, why this fork exists.
- [RELEASE_NOTES_v0.3.0.md](RELEASE_NOTES_v0.3.0.md): breaking changes
  (`inference` → `dinov2-cli`, `quantize` removed), upgrade notes.
