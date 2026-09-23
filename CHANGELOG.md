# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This file is **auto-generated** by [git-cliff](https://git-cliff.org) from the
conventional-commit history at every `v*` tag push. See
[.github/workflows/changelog.yml](.github/workflows/changelog.yml) for the
generator and [cliff.toml](cliff.toml) for the section/commit-type mapping.
Hand-edit only the lines **above** the `<!-- git-cliff: end of header -->`
marker; everything below is regenerated.

<!-- git-cliff: end of header -->
## [0.5.0] - 2026-09-22

### Added
- Load models from buffer and callback readers
- Public C API in include/dinov2.h + src/dinov2-c.cpp
- Add tier-2 Emscripten wasm target with embind shim over the C API
- Add minimal browser demo for the wasm build
- Add in-memory image decode helper
- Add tier-2 dinov2-server HTTP embeddings microservice
- Backbone-only GGUF publishing matrix (#23)
- GitHub Pages wasm demo + visual-regression CI proof (#26)
- *(examples)* Scratch container demo (45 MB image) (#27)
- *(wasm)* Repo links, encode feedback, unload model button

### Fixed
- *(ci)* Fetch full history in changelog tag workflow
- *(loader)* Free tmp_ctx and validate GGUF before use
- *(inference)* Read tensors through the backend, not ->data
- *(attention)* Stop zero-padding the K/V seq dim for flash attention
- *(embeddings)* Correct pos-embed early return and collapse interp loop
- *(core)* Audit cleanup items in predict, graph sizing, and CLI
- *(loader)* Guard ggml_backend_cpu_init failure
- *(loader)* Read num_classes metadata unconditionally
- *(inference)* Size the cgraph node capacity to the layer count
- Check gguf kv types before reading values
- Stop C++ exceptions at the C boundary
- Drop stale outputs on any failed dino_encode
- Validate topk only when classify is requested
- Claim hygiene (webp bug, limitations, ARM bench) (#21)

### Changed
- *(changelog)* Regenerate for v0.4.0
- *(changelog)* Correct v0.4.0 regen, add link rows, bump footer
- Untrack .superpowers and .agents/plans
- Point at latest release instead of hardcoded versions
- *(parity)* Record audit-hardening rerun for both small checkpoints
- *(cli)* Cover the -c/-o parse-time rejection
- *(cli)* Sync with current -o and -s behavior
- *(build)* Note -s is a no-op and drop em dashes
- *(architecture)* Refresh help output and public API surface
- *(spec)* Design library extraction and C API
- Split CLI helpers and params out of the engine
- Introduce dino_ctx owning allocator and run state
- Run inference through ggml_backend_sched and registry backend init
- C API coverage on synthetic GGUF fixtures
- Library layering, README C API section, stability stub
- Clang-format-18 over refactored sources
- Move internal header to src/dinov2-impl.h
- Move per-chunk outputs into ctx last_outputs
- Warm only the backend registry in dino_backend_init
- Fix device names and fill contract gaps in the C header
- Cover the C API and remaining test sources in the format gate
- Wire orphaned quality checks into a contract job
- Add nightly parity workflow
- Add tier-2 extras workflow with a wasm build job
- Document the wasm build, demo and JS API
- *(examples)* Add ci-visual-regression GitHub Action
- *(examples)* Add minimal folder dedup example
- Expand stability contract and add tiers page
- Point architecture and readme at tier-2 surfaces
- Add server and examples jobs to extras workflow
- Marketing drafts and trust page (#22)
- PCA demo GIF hero and server asciinema demo (#24)
- *(examples)* Add measured dedup run report on a synthetic corpus (#25)
- *(readme)* Consolidate entry points, benchmarks, and doc links (#28)
- *(hf)* Mark backbone GGUF repos as pending publish
- *(gguf)* Let single-variant dispatch reach the publish job
- *(license)* Add Joaquin Terrasa copyright line
- Bump version to 0.5.0
- Relax flash-attn parity tolerance for sanitizer fp noise
- *(release)* Add v0.5.0 changelog links

### Miscellaneous
- Merge pull request #15 from espetro/changelog/v0.4.0
- Merge pull request #16 from espetro/fix/audit-hardening
- Extract a dinov2 library target
- Merge pull request #17 from espetro/refactor/library-and-c-api
- Merge pull request #18 from espetro/ci/wire-checks
- Merge pull request #19 from espetro/feat/wasm
- Merge pull request #20 from espetro/feat/tier2-surfaces
## [0.4.0] - 2026-09-21

### Added
- Return cls, pooled, and score embeddings from dino_predict
- Add output-mode, version, and help flags to CLI parsing
- Emit embeddings JSON and restore top-k output in CLI
- *(scripts)* Add parity check vs PyTorch reference
- *(params)* Add n_batch and --batch flag with bounds checking
- *(graph)* Batch-aware encoder graph in forward_features and attn
- *(predict)* Multi-image dino_predict API
- *(cli)* Accept multiple -i inputs and comma-separated image lists
- *(cli)* Run multi-image inference in n_batch chunks
- *(cli)* Report per-image throughput in bench output
- *(cli)* Add preview binary embeddings output
- Harden backbone-only DINOv2 loading
- *(preprocess)* Bound feature mode with --preprocess, --no-resize, --max-tokens
- *(cli)* Add index and grid to records, D2EMB v2 header

### Fixed
- Preserve aspect ratio in classify preprocess resize
- Pool over actual patch-token count in classify head
- Route dino_model_load logs to stderr
- *(scripts)* Gate parity on patch flat+mean cosine, keep token min informational
- *(dinov2)* Exclude register tokens from classify pooling
- *(dinov2)* Bounds-check value flags in dino_params_parse
- *(scripts)* Fail parity check on empty image list and CLI timeout
- Add missing scale param to attn declaration
- *(cli)* Address batched inference review findings
- *(cli)* Enforce strict numeric parsing
- *(cli)* Address Task 1 review nits
- *(cli)* Address binary embeddings review findings
- *(ci)* Use supported Hugging Face CLI
- *(bench)* Parse benchmark JSON portably
- Address final parity review findings
- Support backbone-only DINOv2 conversion
- *(converter)* Harden DINOv2 backbone resolution
- Check ggml graph and model buffer allocation failures

### Changed
- *(readme)* Drop Topics section and topic-count badge
- *(plans)* Point CUDA smoke-test step at the live HF slug
- Apply clang-format-18 to bench path + bench flag declarations
- Drop unused print_t_f32 debug helper
- Cover classify preprocess aspect and l2_normalize
- Ignore CMake preset build-* directories
- Add cli.md reference and sync CLI docs with embeddings output
- Complete --help blocks and fix model path in cli.md example
- *(cli)* Fix build preset, threads default, and flag-table lead-in
- Update agent memory for embeddings-dropin learnings
- Batch inference coverage with synthetic tiny model
- Cover flash-attention batch path in dino_predict
- Document batch inference across CLI reference and guides
- *(spec)* Define register-token recommendations
- Recommend register-token models for feature workflows
- Specify parity benchmark and binary preview
- Add reproducible parity evidence
- *(parity)* Clarify evidence provenance and downloads
- *(bench)* Report blocked Ubuntu evidence run
- Record Ubuntu benchmark rerun blocker
- *(bench)* Record Ubuntu benchmark artifact
- *(bench)* Preserve model provenance
- *(bench)* Record current Ubuntu benchmark evidence
- Harden backbone loader review coverage
- *(bench)* Publish Ubuntu-only benchmark evidence
- Refresh final parity evidence
- Refresh final parity evidence
- Fix regular GGUF checksum
- *(spec)* Design large-image preprocessing bound and record contract
- Bump project version to 0.4.0
- *(spec)* Settle preprocess modes, true-ceil alignment, and record contract
- *(preprocess)* Single resample to bounded target dims, true-ceil ctx sizing
- Record contract, preprocess modes, D2EMB v2
- *(parity)* Record measured hf-mode residual for tench
- *(parity)* Record rerun code state and CLI 0.4.0

### Miscellaneous
- Add project-scoped mise.toml for clang-format-18 + git-cliff
- Define DINOV2_VERSION from project version
- Add cmake buildPresets so --preset build works
- Merge pull request #10 from espetro/feat/embeddings-dropin
- Merge pull request #11 from espetro/feat/batched-inference
- Merge pull request #12 from espetro/feat/batched-inference
- Merge pull request #13 from espetro/feat/batched-inference
- Merge pull request #14 from espetro/fix/large-image-preprocessing
## [0.3.0] - 2026-09-18

### Fixed
- *(cli)* Guard sys/resource.h and getrusage behind POSIX; use GetProcessMemoryInfo on Windows
- *(ci)* Drop hallucinated git-cliff --tag flag from changelog PR body
- *(ci)* Install libblake3 alongside ccache on macos arm64
- *(ci)* Install blake3 (not libblake3) on macos
- *(ci)* Flip push-to-fork polarity in changelog PR opener
- *(ci)* Drop push-to-fork entirely from changelog PR opener

### Changed
- *(plan)* Post-v0.2.0 work plan
- *(dinov2)* Add pure-function coverage for do_quantize, hparams, pos_embed, preprocess
- Add --bench flag to inference, refactor scripts/bench.sh
- *(plan)* Update post-v0.2.0 plan with v0.3.0 cleanup scope
- *(plan)* Expand v0.3.0 cleanup to full minimal-repo sweep (23 file deletions)
- *(memory)* Commit .agents/memory/dinov2_cpp.md
- *(rename)* Rename inference -> dinov2-cli, drop quantize, minimal-repo sweep
- *(quantize)* Drop quantize binary & dino_model_quantize/do_quantize
- Rename inference -> dinov2-cli in user-facing surface
- Drop quantize loop, f16-only path
- Prune .gitignore, slim pyproject, refresh benchmark docs
- *(changelog)* Add CHANGELOG.md, retire per-release RELEASE_NOTES files
- *(changelog)* Regenerate CHANGELOG.md via git-cliff
- Add git-cliff config + v*-pushed changelog regeneration workflow
- *(contributing)* Changelog-auto editorial rules + breaking-change flag note
- *(changelog)* Add v0.1.0/v0.2.0/v0.3.0 comparison link rows
- *(readme)* Add topic shield + Topics section reflecting repo metadata
- *(memory)* Record changelog-automation + GitHub topics context
- Manual-trigger bench workflow (ubuntu-latest, f16 only)
## [0.2.0] - 2026-09-18

### Added
- *(gguf)* Structured per-variant log file for audit/debug

### Fixed
- *(ci)* Add missing <cmath> include for ggml v0.24 transitive drop
- *(ci)* Install huggingface_hub cli in gguf workflow
- *(ci)* Dedupe upload steps after 3-way merge
- *(ci)* Dedupe upload step after 3-way merge duplication
- *(scripts)* Use --type/--exist-ok flags for new hf CLI
- *(ci)* Emit JSON array from plan job so fromJson(matrix) parses correctly

### Changed
- Rewrite README for clarity and conversion, split details into docs/ and CONTRIBUTING
- Hf gguf conversion + publishing workflow
- Hf publishing guide
- Point HF repo references at dinov2-cpp-core
- Add logo assets in multiple sizes for README and HF avatar
- Restructure README, move weights+build to CONTRIBUTING, add logo + agent quickstart
- Restore GGUF weights table to README, facebook checkpoints to CONTRIBUTING only
- Fix README structure, fork credit, broken features table
- *(refactor)* Port gguf publish to bash script
- Use ubuntu-latest-large runner for GGUF publish job
- *(cache)* Cache .venv-publish across matrix legs
- Document HF_TOKEN in scripts/.env.example
- *(gguf)* Fit conversion within ubuntu-latest (~14 GB) via lean deps + cache cleanup (#5)
- *(gguf)* Hardcode HF_HOME path (runner.home not allowed in env:) (#6)
- *(gguf)* Relocate caches to /mnt scratch space, use hf_xet + HF_XET_HIGH_PERFORMANCE
- *(gguf)* Dynamic matrix to skip ghost runner spawns on single-variant dispatch
- V0.2.0 release notes

### Miscellaneous
- Merge pull request #2 from espetro/hf-gguf-pipeline
- Merge remote-tracking branch 'origin/hf-gguf-pipeline'
- Merge pull request #3 from espetro/gguf-bash-port
- Merge pull request #4 from espetro/gguf-bash-port
- Merge pull request #7 from espetro/gguf-perf
- Merge pull request #8 from espetro/gguf-perf-fix
## [0.1.0] - 2026-09-18

### Fixed
- Fixed citations.bib
- Fixed module structure and package deps
- Fixed linear probe impl
- Fixed linear probe impl
- Fixed warnings in codebase
- Fixed time init issue
- Fixed benchmark code
- Fixed benchmark code
- Fixes to model loading key names
- Fixed model loading
- Fixed dino model loading
- Fixed model loading
- Fixed gguf file saving
- Fixed gguf file saving
- Fixed memory layout issue
- Fixed image byte offset
- Fixed image byte offset
- Fixed interp code
- Fixed memory issue in inference
- Fixed dinov2 giant
- Fixed dinov2 inference
- Fixed benchmark code
- Fixed benchmark code
- Fixed benchmark code
- Fixed silu for metal
- Fixed benchmark.sh
- Fixed benchmark.py
- Fixed timing of mps,cuda
- Fixed registers save script
- Fixed rebase issues
- Fixed regular pipeline attn ordering
- Fixed cmakelists.txt
- Fixed CMakeLists.txt
- Fixed benchmark .sh file
- Fixed cmakelists.txt
- Fix to quantize
- Fixed compilation issues
- Fixed redundant op in forward_features
- Fixed realtime.cpp
- Fixed benchmark code
- *(image)* Use patch-grid dims in pca_project_3d to avoid OOB read
- *(ci)* Missing cstring include and clang-format version reconciliation

### Changed
- *(submodule)* Switch ggml url to https; pin gguf>=0.18.0,<0.20
- *(image)* Drop opencv + vendored stb, port vit.cpp preprocessing + power-iter pca
- 4-job matrix with apt/brew/vcpkg + ccache caching
- Drop opencv install, add hf model links, document backend-less path
- Dev harness - cmake presets, doctest suite, asan/ubsan ci job, clang-format gate
- *(assets)* Regenerate pca_visual.jpg with new pca output
- Pin clang-format-18 in lint job

### Miscellaneous
- Init repo
- Added initial proposal
- Update proposal.tex
- Added citations.bib
- Added pyproject.toml and starter code
- Moved run.py to scripts
- Changed build system to setuptools
- Updated uv.lock
- Added evaluate_track1.py
- Added data module
- Added calculate_accuracy function
- Added TiTok codebase
- Merge branch 'dev'
- Added environment.yml + updated pyproject.toml
- Updated environment.yml
- Updated requirements.txt
- Updated requirements.txt and fixed bugs in convert_imagenet_to_wds.py
- Added SCC batch submission script for dataset
- Increased time of SCC job
- Pretokenization file job for scc
- Update pretokenization.py for webdataset file type
- Update pretokenization.sh, should be placed in a cloned titok repo
- Latest results from attempting to run pretokenization
- Began to add missing pieces from TiTok codebase. Also added test folder.
- Added pretokenization fixes
- Added pretokenization fixes
- Added distributed pretokenization fixes
- Added distributed pretokenization fixes
- Added distributed pretokenization fixes
- Changed pretokenization script
- Changed pretokenization script
- Added save_id to pretokenization.py
- Added save_id to pretokenization.py
- Pretokenization file output from the scc
- Updated pretokenization file output
- Output from pretokenization script
- Changed to OMP=4 in pretokenization.sh
- Added fixes to pretokenization.py
- Added fixes to pretokenization.py
- Added fixes to pretokenization.py
- Added latest SCC log file
- Added linear probe script
- Attempt to load TiTok model for linear probing
- Changed modified forwarding to be simpler
- Added wds version of lin prob
- Added initial fixes to linear probe code
- Added interp pos encoding
- Create midpoint-checkin.tex
- Added midpoint checkin and fixed directories
- Added wandb integration
- Changed linprobe to use patch features
- Changed linprobe wds
- Changed linprobe scc script
- Added vit.cpp base repo
- Added ggml submodule + pyproject.toml
- Updated README.md
- Added hf_transfer package
- Updated .gitignore
- Removed ggml submodule
- Changed ggml submodule
- Updated README.md to work for now
- Added to .gitignore
- Changed project name to dinov2
- Migrated to latest version of ggml
- Possible fix to quantize.cpp
- Changed submodule branch
- Changed submodule branch
- Added torch-backend opt in uv
- Added visual studio folders
- Updated README for windows
- Updated README for windows
- Updated README for windows
- Added Windows build details
- Moved convert-pth-to-ggml script
- Moved convert-pth-to-ggml script README update
- Added dinov2 conversion script
- Updated pyproject.toml
- Updated pyproject.toml
- Modified benchmark code
- Added cpu device
- Updated bash script to work with dino models
- Added original benchmark code
- Added MacOS compatibility with benchmark.sh script
- Enabled experimental mode for uv
- Updated pyproject.toml
- Changed uv to use torch cpu
- Added dinov2 boilerplate
- Added initial ggml conversion code
- Converted DINO weights to fp16
- Altered vit code to load weights of dinov2
- Added main.cpp code for vit
- Edited main.cpp and dinov2.cpp to test loading the model
- Updated model context size to reflect dinov2 architecture
- Updated allocating context for dino
- Added fixes to forward pass
- Added working version of inference
- Updated README and added patch_tokens state
- Added gguf conversion script
- Replaced mean with sum op for mps support
- Added metal instructions in readme
- Added forward pass time
- Updated README.md
- Updated README.md
- Simplified model loadin
- Added classify and feature outputs
- Added optional types
- Updated README.md
- Added reg token support
- Updated dinov2.h and CmakeLists.txt
- Updated dinov2.h and CmakeLists.txt
- Added cast
- Changed to uint types
- Added mods
- Added return type
- Sync code
- Added latest cmake config
- Fix OpenCV include with CMake
- Merge pull request #1 from lavaman131/alicja/opencv-cmake
- Added cmake changes
- Updated CMakeLists.txt
- Converted pipeline to cv::Mat
- Added direct float scaling
- Added baseline PCA code
- Added pca image conversion and saving
- Image output now a parameter
- Added different default val for image name
- Actually fixed byte offset
- Corrected hparam types
- Initial realtime code framework
- Added classify and feature transforms
- Updated realtime.cpp
- Merge branch 'main' into dev
- Merge pull request #2 from lavaman131/dev
- Updated benchmark script
- Removed thread limit in benchmark
- Changed to single qkv
- Added multi-threading support
- Updated to cxx 20
- Changed realtime video size
- Improved attn eff
- Added inplace graph optimizations
- Removed slower inplace ops
- Fixed benchmark.sh. It is now compatible with dinov2.cpp.
- Added gtime in benchmark.sh for Darwin
- Added gtime in benchmark.sh for Darwin
- Added threads opt
- Changed benchmarks for fair comparison
- Removed old source files from vit.cpp
- Added script timing
- Updated readme and documentation
- Updated images in pca output
- Added benchmarking results to readme
- Small change to readme
- Rebase
- Added to flash_attn cli arg
- Cleaned up scripts
- Cleaned up scripts and added camera feature
- Update README.md
- Cleaned up realtime visual
- Removed old vit cpp code
- Initial code for quantization
- Renamed main.cpp to classify.cpp
- Initial quantization code
- Enabled quantize by default
- Added deps for quantize
- Added quantize code
- Hacky quantize fix
- Quantize fix
- Renamed inference file
- Updated CMakeLists.txt
- Inference.cpp and adjustments to CMakeLists.txt
- Additions to readme
- Refining Readme, updating TOC
- Added readme assets folder and tuned README
- Added OpenCV env variable instructions
- Minor tweak to readme
- Minor tweak to readme
- Fixed TOC
- Tweak readme - env variables
- Merge branch 'main' into quantize
- Merge pull request #3 from lavaman131/quantize
- Added initial quantization
- Quant fixes
- Added quantization fixes
- Added compiler optimizations
- Added compiler opt flags
- Updated cmakelists.txt proj desc
- Removed llama.cpp submod
- Update README.md
- Updated .gitignore and benchmark.sh
- Changed compile opts
- Added partial pca changes
- Added quantization benchmarking results
- Fixes to readme.md
- Added new quantization results with registers
- Edits to readme, added updated results
- Fixed second quantization table
- Various tweaks to readme
- Added demo video
- Updated demo video
- Added latest benchmark scripts
- Removed compiler opt
- Added latest results
- Merge pull request #1 from espetro/ci-release
<!-- git-cliff: end of body -->

## Release link table

<!-- Auto-managed; compare links for each shipped version. New rows
must be added by hand when a new tag ships, since git-cliff does
not derive a comparison base URL automatically. -->

[Unreleased]: https://github.com/espetro/dinov2.cpp/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/espetro/dinov2.cpp/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/espetro/dinov2.cpp/compare/v0.3.0...v0.4.0
