# dinov2-server

Minimal HTTP embeddings microservice over the public C API. **Tier-2,
best-effort**: `DINOV2_BUILD_SERVER` is OFF by default, it is not shipped in
release archives, and it is not covered by the tier-1 stability contract
([docs/tiers.md](../../docs/tiers.md)).

## Build

```bash
cmake -B build-server -G Ninja -DCMAKE_BUILD_TYPE=Release -DDINOV2_BUILD_SERVER=ON
cmake --build build-server --target dinov2-server -j
```

The only dependency is a vendored single-header copy of
[cpp-httplib](https://github.com/yhirose/cpp-httplib) v0.57.1 (`httplib.h`,
MIT), the same vendoring pattern as the stb headers in `src/`. No system
packages, no configure-time fetches.

## Run

```bash
./build-server/bin/dinov2-server -m models/model.gguf --port 8080 --threads 4
```

`-m` is required; `--host` defaults to `127.0.0.1`, `--port` to 8080.

## Endpoints

| Method | Path | Description |
|:-------|:-----|:------------|
| GET | `/health` | `{"status":"ok"}` |
| GET | `/v1/models` | model metadata: hidden size, patch size, registers, classifier |
| GET | `/version` | library version |
| POST | `/v1/embeddings` | embed an uploaded image |

`POST /v1/embeddings` accepts either a `multipart/form-data` upload with the
image in a file field named `image` (or `file`), or a JSON body
`{"image": "<base64>"}` (data-URI prefixes are tolerated). Options can be
passed as query parameters, multipart form fields, or JSON keys:

| Option | Default | Effect |
|:-------|:--------|:-------|
| `patches` | off | include flat row-major patch tokens (`n_patches * hidden` floats) |
| `l2_normalize` | off | L2-normalize `cls`/`pooled`/`patches` (cosine = dot) |
| `classify` | off | run the classifier head; adds `topk` |
| `topk` | 5 | number of classes in `topk` when `classify` is set |

Response (feature mode):

```json
{
  "cls": [0.46, ...],
  "pooled": [0.46, ...],
  "n_patches": 1369,
  "grid": {"h": 37, "w": 37},
  "meta": {"model": "models/model.gguf", "version": "0.4.0",
           "hidden_size": 384, "patch_size": 14,
           "n_register_tokens": 0, "has_classifier": true, "n_classes": 1000}
}
```

`cls`/`pooled`/`patches`/`topk`/`grid` carry the same meaning as the
`dinov2-cli --print-embeddings` JSONL fields ([docs/cli.md](../../docs/cli.md)).
Errors are `{"error": "..."}` with a 4xx/5xx status.

### Examples

```bash
curl -s localhost:8080/health

# multipart upload
curl -s -F image=@assets/tench.jpg localhost:8080/v1/embeddings | jq '.cls | length'

# base64 JSON
curl -s -X POST localhost:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d "{\"image\": \"$(base64 -i assets/tench.jpg | tr -d '\n')\"}" | jq '.grid'

# patch tokens + unit-length vectors
curl -s -F image=@assets/tench.jpg 'localhost:8080/v1/embeddings?patches=1&l2_normalize=1' | jq '.n_patches'
```

## Limits

- One inference context serialized behind a mutex: HTTP connections are
  accepted concurrently but encodes run one at a time. `--threads` sets the
  ggml compute threads per encode.
- Uploads capped at 64 MB (`set_payload_max_length`).
- The JSON request parser is intentionally minimal (flat objects, the keys
  above only). Multipart is the robust path; send JSON only for simple
  clients.
- No TLS, auth, batching across requests, or queueing. Put it behind a
  reverse proxy if you need those.
