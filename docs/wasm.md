# WebAssembly target

`dinov2.cpp` can be compiled to WebAssembly with Emscripten and run fully in
the browser (or a web worker): the encoder, preprocessing and GGUF loader all
live in one `dinov2-wasm.wasm` (~1 MB) plus a small ES6 JS loader. The wasm
build uses ggml's SIMD128 kernels (`ggml/src/ggml-cpu/arch/wasm/`).

This is a tier-2 target: `DINOV2_BUILD_WASM` is OFF by default, the CI job in
`.github/workflows/extras.yml` is `continue-on-error`, and nothing here can
break the native engine build.

## Prerequisites: emsdk

The `wasm` preset locates the Emscripten toolchain through the `EMSDK`
environment variable, so install emsdk the upstream way:

```bash
git clone https://github.com/emscripten-core/emsdk.git ~/emsdk
~/emsdk/emsdk install latest
~/emsdk/emsdk activate latest
source ~/emsdk/emsdk_env.sh   # exports EMSDK, puts emcc on PATH
```

The Homebrew `emscripten` formula also works if you skip the preset and point
CMake at its toolchain file directly:

```bash
cmake -B build-wasm -G "Unix Makefiles" \
  -DCMAKE_TOOLCHAIN_FILE="$(brew --prefix emscripten)/libexec/cmake/Modules/Platform/Emscripten.cmake" \
  -DDINOV2_BUILD_WASM=ON -DDINOV2_BUILD_CLI=OFF -DDINOV2_BUILD_TESTS=OFF \
  -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm32 -DGGML_OPENMP=OFF
```

Note `EMSCRIPTEN_SYSTEM_PROCESSOR=wasm32`: emsdk defaults
`CMAKE_SYSTEM_PROCESSOR` to `x86`, which would make ggml fall back to its
generic (scalar) quant kernels instead of the SIMD128 wasm ones. The preset
sets this for you.

## Building

```bash
cmake --preset wasm
cmake --build --preset wasm
# -> build-wasm/bin/dinov2-wasm.js + dinov2-wasm.wasm
```

Emscripten settings baked in (`wasm/CMakeLists.txt`): `-sALLOW_MEMORY_GROWTH`,
`-sMAXIMUM_MEMORY=4GB`, `-sMODULARIZE -sEXPORT_ES6` (so the loader is an ES6
module exporting a `Dinov2Module()` factory), `-sENVIRONMENT=web,worker`, and
embind. For a node smoke test reconfigure with
`-DDINOV2_WASM_ENVIRONMENT=web,worker,node`.

## Running the demo

The demo (`wasm/index.html` + `wasm/main.js`) expects the build artifacts
next to it:

```bash
cp build-wasm/bin/dinov2-wasm.js build-wasm/bin/dinov2-wasm.wasm wasm/
cd wasm && python3 -m http.server 8000
# open http://localhost:8000
```

Model source, in the URL field:

- `model.gguf` (default): fetched relative to the page, so drop a GGUF next to
  `index.html`. Get one with `hf download dinov2-cpp-core/dinov2-small-gguf
  model.gguf --local-dir wasm` (~50 MB f16 ViT-S).
- A remote URL also works when the host sends CORS headers. huggingface.co
  `/resolve/` URLs do (verified), e.g.
  `https://huggingface.co/dinov2-cpp-core/dinov2-small-gguf/resolve/main/model.gguf`.

Then pick two images: the page decodes them via `createImageBitmap` + canvas,
encodes each (RGB8/RGBA8 in, preprocessing happens inside the library), and
shows the cosine similarity of the CLS embeddings plus a per-patch heatmap of
patch-to-CLS cosine.

## JS API

```js
import Dinov2Module from './dinov2-wasm.js';
const Module = await Dinov2Module();

const handle = Module.loadModel(new Uint8Array(ggufBytes)); // 0 on failure
const info   = Module.modelInfo(handle);   // {hiddenSize, patchSize, nRegisterTokens, hasClassifier, nClasses}
const enc    = Module.encode(handle, pixels, width, height, channels); // channels: 3 or 4
// enc.ok: {cls, pooled, patches, gridW, gridH, nPatches, hiddenSize}, all Float32Array copies
// !enc.ok: {status} a dino_status code
Module.freeModel(handle);
```

`encode` runs the bounded feature recipe with L2-normalized outputs, so
`cosine(a, b)` is just a dot product. Similarity helpers live in JS
(`wasm/main.js`), not in the binding.

## Performance and limitations

Honest estimate: single-threaded wasm is roughly 1.5-3x slower than native
multi-threaded CPU for the same image. Expect on the order of **4-10 s per
image for ViT-S at the 518 px bounded recipe** on a recent laptop, faster for
smaller inputs (a 64x48 test image encodes in well under 1 s). This is an
estimate, not a measurement; SIMD128 helps the quant kernels but there is no
GPU or threaded path yet.

- **Single thread only.** ggml's wasm pthread pool is possible but requires
  cross-origin isolation (COOP/COEP headers, `SharedArrayBuffer`); see the
  commented block in `wasm/CMakeLists.txt`. Not shipped in v1.
- **Whole model in memory.** `loadModel` copies the fetched bytes once into a
  std::vector, then ggml copies weights into its backend buffer. Peak usage is
  roughly 2x the GGUF size plus activations; `MAXIMUM_MEMORY=4GB` caps the
  address space. ViT-S/B are fine; think twice about the 2.2 GB giant.
- **encode() blocks.** Call it from a Worker (`ENVIRONMENT=web,worker`
  already permits that) if the UI must stay responsive.
- Feature mode only from the shim (`classify` is not exposed); add it if a
  classifier GGUF ever ships.
