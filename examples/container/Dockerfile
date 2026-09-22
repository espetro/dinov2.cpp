# syntax=docker/dockerfile:1
#
# dinov2.cpp container demo: a FROM scratch image carrying a statically
# linked dinov2-cli plus the pinned dinov2-small GGUF weight. The build
# context only needs this directory; the source tree and the model are
# fetched and verified inside the build stage.

ARG DINOV2_REF=v0.4.0
ARG MODEL_URL=https://huggingface.co/dinov2-cpp-core/dinov2-small-gguf/resolve/main/model.gguf
ARG MODEL_SHA256=b7ca009aa416f6be85ea95363f4ef13d3c139999f96ca3fd47226849afb712e6

FROM ubuntu:24.04 AS build
ARG DINOV2_REF MODEL_URL MODEL_SHA256
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential ca-certificates cmake curl git \
 && rm -rf /var/lib/apt/lists/*

# GGUF weights, pinned by sha256 so a silently changed upstream file fails
# the build instead of shipping different embeddings.
RUN curl -fSL --retry 3 "$MODEL_URL" -o /model.gguf \
 && echo "$MODEL_SHA256  /model.gguf" | sha256sum -c -

# dinov2-cli, fully static. BUILD_SHARED_LIBS=OFF makes libdinov2 and the
# vendored ggml build as .a archives (ggml defaults it ON on Linux), and
# GGML_BACKEND_DL defaults OFF so the CPU backend is compiled in rather than
# dlopen'd; -static then covers libc, libstdc++, libgomp and pthread.
# FindOpenMP records the shared libgomp.so path, which -static cannot use,
# so OpenMP_gomp_LIBRARY is preset to the gcc-shipped static archive.
# GGML_NATIVE=OFF avoids baking -march=native so the image stays portable
# across x86-64 and arm64 hosts.
RUN git clone --depth 1 --branch "$DINOV2_REF" \
      https://github.com/espetro/dinov2.cpp /src \
 && (git -C /src submodule update --init --depth 1 \
      || git -C /src submodule update --init) \
 && cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DDINOV2_BUILD_TESTS=OFF \
      -DGGML_NATIVE=OFF \
      -DOpenMP_gomp_LIBRARY="$(gcc -print-file-name=libgomp.a)" \
      -DCMAKE_EXE_LINKER_FLAGS="-static -static-libgcc -static-libstdc++" \
 && cmake --build /src/build --target dinov2-cli -j"$(nproc)" \
 && strip -o /dinov2-cli /src/build/bin/dinov2-cli \
 && if readelf -d /dinov2-cli | grep -q NEEDED; then \
      echo "error: dinov2-cli is not fully static" >&2; exit 1; \
    fi

FROM scratch
COPY --from=build /dinov2-cli /dinov2-cli
COPY --from=build /model.gguf /model.gguf
ENTRYPOINT ["/dinov2-cli"]
CMD ["--help"]
