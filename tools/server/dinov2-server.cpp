// dinov2-server: minimal HTTP embeddings microservice (tier-2).
//
// Built only with -DDINOV2_BUILD_SERVER=ON. Sits on the public C API
// (include/dinov2.h) plus src/image.h for decoding upload bytes; it is a
// consumer of the library, not part of it. Single inference context
// serialized behind a mutex: requests are served concurrently by httplib's
// thread pool but encodes run one at a time. Best-effort surface, no SLA;
// see docs/tiers.md and docs/stability.md.
//
// Endpoints:
//   GET  /health           -> {"status":"ok"}
//   GET  /v1/models        -> model metadata (also GET /version)
//   POST /v1/embeddings    -> multipart file field "image"/"file", or
//                             JSON {"image": "<base64>"}. Optional flags
//                             "patches", "l2_normalize", "classify", "topk"
//                             (JSON keys or ?query=/form params).
//                             -> {"cls":[...], "pooled":[...], "grid":{...},
//                                 "n_patches":N, "meta":{...}, ...}

#include "dinov2.h"
#include "image.h"

#include "httplib.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// tiny JSON field extraction (flat objects only; documented best-effort)
// ---------------------------------------------------------------------------

// Find "key" : <value> at any depth and return the raw value start. nullptr
// when absent. Enough for the flat request bodies this service accepts.
const char *json_find(const std::string &body, const char *key) {
    const std::string needle = std::string("\"") + key + "\"";
    size_t            pos    = 0;
    while ((pos = body.find(needle, pos)) != std::string::npos) {
        size_t i = pos + needle.size();
        while (i < body.size() && (body[i] == ' ' || body[i] == '\t' || body[i] == '\n' || body[i] == '\r')) {
            ++i;
        }
        if (i < body.size() && body[i] == ':') {
            ++i;
            while (i < body.size() && (body[i] == ' ' || body[i] == '\t' || body[i] == '\n' || body[i] == '\r')) {
                ++i;
            }
            return body.c_str() + i;
        }
        pos += needle.size();
    }
    return nullptr;
}

// Extract a JSON string value for `key`, decoding \\ and \" escapes.
bool json_get_string(const std::string &body, const char *key, std::string &out) {
    const char *v = json_find(body, key);
    if (!v || *v != '"') {
        return false;
    }
    ++v;
    out.clear();
    const char *end = body.c_str() + body.size();
    while (v < end && *v != '"') {
        if (*v == '\\' && v + 1 < end) {
            ++v; // keep the escaped char; data URIs and base64 never need more
        }
        out += *v++;
    }
    return v < end;
}

bool json_get_bool(const std::string &body, const char *key, bool fallback) {
    const char *v = json_find(body, key);
    if (!v) {
        return fallback;
    }
    if (*v == '"') {
        ++v; // tolerate quoted booleans ("true", "1")
    }
    return strncmp(v, "true", 4) == 0 || *v == '1';
}

int json_get_int(const std::string &body, const char *key, int fallback) {
    const char *v = json_find(body, key);
    if (!v) {
        return fallback;
    }
    char *end = nullptr;
    long  n   = strtol(v, &end, 10);
    return end != v ? (int)n : fallback;
}

// ---------------------------------------------------------------------------
// base64 decode (request side) and JSON float printing (response side)
// ---------------------------------------------------------------------------

std::vector<uint8_t> base64_decode(const std::string &in) {
    static const signed char T[256] = {
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, -1, -1, 63, 52, 53, 54, 55,
        56, 57, 58, 59, 60, 61, -1, -1, -1, -2, -1, -1, -1, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, -1, -1, 26, 27, 28, 29, 30, 31, 32,
        33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    // tolerate a data URI prefix ("data:image/png;base64,....")
    size_t start = in.find("base64,");
    start        = start == std::string::npos ? 0 : start + 7;
    std::vector<uint8_t> out;
    out.reserve(in.size() / 4 * 3);
    int val = 0, bits = -8;
    for (size_t i = start; i < in.size(); ++i) {
        const signed char c = T[(uint8_t)in[i]];
        if (c == -2) {
            break; // '=' padding
        }
        if (c == -1) {
            continue; // skip whitespace and other noise
        }
        val = (val << 6) + c;
        bits += 6;
        if (bits >= 0) {
            out.push_back((uint8_t)((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return out;
}

void append_f32(std::string &out, const float *v, size_t n) {
    char buf[32];
    out += '[';
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            out += ',';
        }
        snprintf(buf, sizeof(buf), "%.6g", (double)v[i]); // matches dinov2-cli
        out += buf;
    }
    out += ']';
}

std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
        }
        out += c;
    }
    return out;
}

void send_error(httplib::Response &res, int code, const std::string &msg) {
    res.status = code;
    res.set_content("{\"error\":\"" + json_escape(msg) + "\"}", "application/json");
}

// ---------------------------------------------------------------------------
// server state: one model + one ctx, encodes serialized on g_encode_mu
// ---------------------------------------------------------------------------

struct server_state {
    dino_model *model = nullptr;
    dino_ctx   *ctx   = nullptr;
    std::string model_path;
};

server_state g_state;
std::mutex   g_encode_mu;

std::string model_meta_json() {
    const dino_model *m = g_state.model;
    char              buf[512];
    snprintf(buf, sizeof(buf),
             "{\"model\":\"%s\",\"version\":\"%s\",\"hidden_size\":%u,\"patch_size\":%u,"
             "\"n_register_tokens\":%u,\"has_classifier\":%s,\"n_classes\":%u}",
             json_escape(g_state.model_path).c_str(), dino_version(), dino_model_hidden_size(m),
             dino_model_patch_size(m), dino_model_n_register_tokens(m), dino_model_has_classifier(m) ? "true" : "false",
             dino_model_n_classes(m));
    return buf;
}

void handle_embeddings(const httplib::Request &req, httplib::Response &res) {
    // ---- extract image bytes ------------------------------------------------
    std::vector<uint8_t> bytes;
    bool                 want_patches = req.has_param("patches");
    bool                 l2           = req.has_param("l2_normalize");
    bool                 classify     = req.has_param("classify");
    int                  topk         = req.has_param("topk") ? atoi(req.get_param_value("topk").c_str()) : 5;

    if (req.is_multipart_form_data()) {
        const char *field = req.form.has_file("image") ? "image" : "file";
        if (!req.form.has_file(field)) {
            send_error(res, 400, "multipart request needs a file field named 'image' or 'file'");
            return;
        }
        const httplib::FormData file = req.form.get_file(field);
        bytes.assign(file.content.begin(), file.content.end());
        want_patches = want_patches || req.form.has_field("patches");
        l2           = l2 || req.form.has_field("l2_normalize");
        classify     = classify || req.form.has_field("classify");
        if (req.form.has_field("topk")) {
            topk = atoi(req.form.get_field("topk").c_str());
        }
    } else {
        std::string b64;
        if (!json_get_string(req.body, "image", b64)) {
            send_error(res, 400, "expected multipart upload or JSON {\"image\": \"<base64>\"}");
            return;
        }
        bytes        = base64_decode(b64);
        want_patches = json_get_bool(req.body, "patches", want_patches);
        l2           = json_get_bool(req.body, "l2_normalize", l2);
        classify     = json_get_bool(req.body, "classify", classify);
        topk         = json_get_int(req.body, "topk", topk);
    }
    if (bytes.empty()) {
        send_error(res, 400, "empty image payload");
        return;
    }

    Image img = load_image_from_memory(bytes.data(), bytes.size());
    if (img.data.empty()) {
        send_error(res, 400, "could not decode image bytes (stb supports png/jpeg/bmp/tga/...)");
        return;
    }

    // ---- encode (serialized: a dino_ctx is single-threaded) -----------------
    dino_image      din{img.data.data(), img.nx, img.ny, 0};
    dino_run_params run = dino_run_default_params();
    run.classify        = classify;
    run.topk            = topk;
    run.l2_normalize    = l2;

    std::lock_guard<std::mutex> lock(g_encode_mu);
    const dino_status           status = dino_encode(g_state.ctx, &din, 1, run);
    if (status != DINO_STATUS_SUCCESS || dino_output_n_images(g_state.ctx) < 1) {
        send_error(res, status == DINO_STATUS_NO_CLASSIFIER ? 400 : 500,
                   std::string("encode failed with dino_status ") + std::to_string((int)status));
        return;
    }

    const uint32_t  hidden     = dino_model_hidden_size(g_state.model);
    int32_t         n_patches  = 0;
    int32_t         grid_w     = 0;
    int32_t         grid_h     = 0;
    const float    *cls        = dino_output_cls(g_state.ctx, 0);
    const float    *pooled     = dino_output_pooled(g_state.ctx, 0);
    const float    *patch_data = dino_output_patches(g_state.ctx, 0, &n_patches, &grid_w, &grid_h);
    const uint32_t *topk_idx   = nullptr;
    const float    *topk_prob  = nullptr;
    const int32_t   n_topk     = dino_output_topk(g_state.ctx, 0, &topk_idx, &topk_prob);

    std::string out;
    out.reserve((size_t)hidden * 3 * 8 + 256);
    out += '{';
    if (cls) {
        out += "\"cls\":";
        append_f32(out, cls, hidden);
    }
    if (pooled) {
        out += ",\"pooled\":";
        append_f32(out, pooled, 2 * (size_t)hidden);
    }
    char buf[128];
    snprintf(buf, sizeof(buf), ",\"n_patches\":%d,\"grid\":{\"h\":%d,\"w\":%d}", n_patches, grid_h, grid_w);
    out += buf;
    if (want_patches && patch_data) {
        out += ",\"patches\":";
        append_f32(out, patch_data, (size_t)n_patches * hidden);
    }
    if (n_topk > 0) {
        out += ",\"topk\":[";
        for (int32_t i = 0; i < n_topk; ++i) {
            const char *label = dino_model_label(g_state.model, topk_idx[i]);
            snprintf(buf, sizeof(buf), "%s{\"idx\":%u,\"label\":\"%s\",\"prob\":%.6f}", i > 0 ? "," : "", topk_idx[i],
                     json_escape(label ? label : std::to_string(topk_idx[i])).c_str(), (double)topk_prob[i]);
            out += buf;
        }
        out += ']';
    }
    out += ",\"meta\":";
    out += model_meta_json();
    out += '}';
    res.set_content(out, "application/json");
}

void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s -m model.gguf [options]\n"
            "\n"
            "  -m FNAME, --model FNAME  GGUF model path (required)\n"
            "  --host HOST              bind address (default: 127.0.0.1)\n"
            "  --port N                 listen port (default: 8080)\n"
            "  -t N, --threads N        ggml compute threads (default: 4)\n"
            "  -h, --help               show this help and exit\n"
            "  --version                print version and exit\n"
            "\n"
            "tier-2 best-effort embeddings service; see docs/tiers.md\n",
            argv0);
}

} // namespace

int main(int argc, char **argv) {
    std::string model_path;
    std::string host = "127.0.0.1";
    int         port = 8080, n_threads = 4;

    for (int i = 1; i < argc; ++i) {
        const std::string arg  = argv[i];
        auto              next = [&]() -> const char              *{ return ++i < argc ? argv[i] : nullptr; };
        if (arg == "-m" || arg == "--model") {
            const char *v = next();
            if (!v) {
                fprintf(stderr, "error: %s needs a value\n", arg.c_str());
                return 1;
            }
            model_path = v;
        } else if (arg == "--host") {
            const char *v = next();
            if (!v) {
                fprintf(stderr, "error: --host needs a value\n");
                return 1;
            }
            host = v;
        } else if (arg == "--port") {
            const char *v = next();
            if (!v || (port = atoi(v)) <= 0 || port > 65535) {
                fprintf(stderr, "error: --port needs a value in 1..65535\n");
                return 1;
            }
        } else if (arg == "-t" || arg == "--threads") {
            const char *v = next();
            if (!v || (n_threads = atoi(v)) < 1) {
                fprintf(stderr, "error: %s needs a value >= 1\n", arg.c_str());
                return 1;
            }
        } else if (arg == "-h" || arg == "--help") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--version") {
            printf("%s\n", dino_version());
            return 0;
        } else {
            fprintf(stderr, "error: unknown option '%s'\n", arg.c_str());
            usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty()) {
        fprintf(stderr, "error: -m/--model is required\n");
        usage(argv[0]);
        return 1;
    }

    g_state.model_path = model_path;
    g_state.model      = dino_model_load_from_file(model_path.c_str(), dino_model_default_params());
    if (!g_state.model) {
        fprintf(stderr, "error: failed to load model '%s'\n", model_path.c_str());
        return 1;
    }
    dino_ctx_params cparams = dino_ctx_default_params();
    cparams.n_threads       = n_threads;
    g_state.ctx             = dino_init_from_model(g_state.model, cparams);
    if (!g_state.ctx) {
        fprintf(stderr, "error: failed to create inference context\n");
        dino_model_free(g_state.model);
        return 1;
    }
    fprintf(stderr, "dinov2-server %s: loaded %s (hidden=%u, patch=%u, registers=%u, classifier=%s)\n", dino_version(),
            model_path.c_str(), dino_model_hidden_size(g_state.model), dino_model_patch_size(g_state.model),
            dino_model_n_register_tokens(g_state.model), dino_model_has_classifier(g_state.model) ? "yes" : "no");

    httplib::Server svr;
    svr.set_payload_max_length(64 * 1024 * 1024); // 64 MB uploads

    svr.Get("/health", [](const httplib::Request &, httplib::Response &res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    svr.Get("/v1/models", [](const httplib::Request &, httplib::Response &res) {
        res.set_content(model_meta_json(), "application/json");
    });
    svr.Get("/version", [](const httplib::Request &, httplib::Response &res) {
        res.set_content(std::string("{\"version\":\"") + dino_version() + "\"}", "application/json");
    });
    svr.Post("/v1/embeddings", handle_embeddings);

    fprintf(stderr, "listening on http://%s:%d (POST /v1/embeddings, GET /health, GET /v1/models)\n", host.c_str(),
            port);
    if (!svr.listen(host.c_str(), port)) {
        fprintf(stderr, "error: could not bind %s:%d\n", host.c_str(), port);
        dino_free(g_state.ctx);
        dino_model_free(g_state.model);
        return 1;
    }
    // unreachable until svr.stop(); keep teardown explicit for embedders
    dino_free(g_state.ctx);
    dino_model_free(g_state.model);
    return 0;
}
