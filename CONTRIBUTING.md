# Contributing

Thanks for considering a contribution. Keep PRs focused and small.

## Dev harness

CMake presets (defined in `CMakePresets.json`) configure the standard workflow:

```bash
cmake --preset debug    # or: release, asan, ubsan
cmake --build --preset debug
ctest --preset debug
```

- `debug` / `release`: standard configurations. Both enable `DINOV2_FATAL_WARNINGS` by default, so warnings are errors.
- `asan` / `ubsan`: AddressSanitizer and UndefinedBehaviorSanitizer builds for debugging memory and UB issues.

Build directories are `build-<preset>/`.

## Tests

Unit tests use [doctest](https://github.com/doctest/doctest). Run the suite with:

```bash
cmake --build --preset debug && ctest --preset debug
```

Run the same suite under the `asan` and `ubsan` presets before submitting anything that touches memory handling or the compute path.

## Formatting

CI enforces `clang-format-18`. Before committing:

```bash
clang-format-18 -i <changed files>
```

Format violations fail the lint job.

## Warnings

`DINOV2_FATAL_WARNINGS=ON` (the preset default) promotes warnings to errors. Keep the build clean under this option; do not disable it to mask warnings.

## PR guidelines

- One logical change per PR. Keep refactors separate from features.
- Ensure the full test suite passes under `debug`, `asan` and `ubsan`.
- Update docs (`README.md`, `docs/`) if you change behavior or interfaces.
- Follow conventional commit format (e.g. `feat:`, `fix:`, `docs:`, `ci:`).
