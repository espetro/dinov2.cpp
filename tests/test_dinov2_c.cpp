// C API tests: exercise include/dinov2.h end to end on a synthetic GGUF
// fixture (TinyModel-equivalent: hidden 16, patch 2, img 8, 2 layers,
// classifier with 7 classes) and compare against the internal C++ path.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "dinov2-impl.h" // internal engine header
#include "dinov2.h"      // public C API under test (include/ on the dinov2 interface path)
#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

static const std::filesystem::path &test_directory() {
    static const std::filesystem::path directory = [] {
        const auto            now  = std::chrono::steady_clock::now().time_since_epoch().count();
        std::filesystem::path base = std::filesystem::temp_directory_path();
        for (unsigned int attempt = 0; attempt < 100; ++attempt) {
            const auto      candidate = base / ("dinov2-c-test-" + std::to_string(now) + "-" + std::to_string(attempt));
            std::error_code ec;
            if (std::filesystem::create_directory(candidate, ec)) {
                return candidate;
            }
        }
        std::fprintf(stderr, "failed to create unique test directory\n");
        std::abort();
    }();
    return directory;
}

struct TempDirectoryCleanup {
    ~TempDirectoryCleanup() {
        std::error_code ec;
        std::filesystem::remove_all(test_directory(), ec);
    }
};

TempDirectoryCleanup cleanup;

// Deterministic pseudo-random fill, same scheme as test_dinov2.cpp so a GGUF
// fixture and an in-memory TinyModel agree on weight magnitudes.
static void fill_tensor(ggml_tensor *t, uint32_t seed, float base, float scale) {
    float        *d = (float *)t->data;
    const int64_t n = ggml_nelements(t);
    for (int64_t i = 0; i < n; ++i) {
        uint32_t x = (uint32_t)i * 2654435761u ^ seed;
        x ^= x >> 13;
        x *= 1274126177u;
        x ^= x >> 16;
        d[i] = base + scale * ((float)(x % 1000) / 1000.0f);
    }
}

// Write a TinyModel-equivalent GGUF: hidden=16, heads=2, patch=2, img=8,
// 2 encoder layers (plain MLP), 7-class classifier when with_classifier.
static bool write_tiny_gguf(const std::string &path, bool with_classifier = true) {
    constexpr int64_t hsz    = 16;
    constexpr int64_t patch  = 2;
    constexpr int64_t imgsz  = 8;
    constexpr int64_t layers = 2;
    constexpr int64_t ncls   = 7;
    constexpr int64_t mlp    = 4 * hsz;
    constexpr int64_t grid   = imgsz / patch; // pos table rows = grid*grid + 1

    gguf_context *gguf = gguf_init_empty();
    gguf_set_val_u32(gguf, "hidden_size", (uint32_t)hsz);
    gguf_set_val_u32(gguf, "num_hidden_layers", (uint32_t)layers);
    gguf_set_val_u32(gguf, "num_attention_heads", 2);
    gguf_set_val_u32(gguf, "patch_size", (uint32_t)patch);
    gguf_set_val_u32(gguf, "img_size", (uint32_t)imgsz);
    gguf_set_val_u32(gguf, "ftype", 1);
    if (with_classifier) {
        gguf_set_val_u32(gguf, "num_classes", (uint32_t)ncls);
        for (int64_t i = 0; i < ncls; ++i) {
            const std::string key = std::to_string(i);
            gguf_set_val_str(gguf, key.c_str(), ("class_" + key).c_str());
        }
    }

    const int     n_tensors = 5 + 14 * (int)layers + 4;
    ggml_context *tctx      = ggml_init({/*mem_size*/ ggml_tensor_overhead() * n_tensors +
                                        sizeof(float) * (hsz * (grid * grid + 1) + patch * patch * 3 * hsz + 64 * 1024),
                                    /*mem_buffer*/ nullptr, /*no_alloc*/ false});
    if (!tctx) {
        gguf_free(gguf);
        return false;
    }

    uint32_t seed = 0;
    auto     add  = [&](const char *name, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
        ggml_tensor *t = ggml_new_tensor_4d(tctx, GGML_TYPE_F32, n0, n1, n2, n3);
        ggml_set_name(t, name);
        const std::string n         = name;
        const bool        is_scale  = n.find("norm") != std::string::npos && n.find("weight") != std::string::npos;
        const bool        is_lambda = n.find("lambda") != std::string::npos;
        if (is_scale) {
            fill_tensor(t, ++seed, 0.8f, 0.4f);
        } else if (is_lambda) {
            fill_tensor(t, ++seed, 0.4f, 0.6f);
        } else if (n.find("bias") != std::string::npos) {
            fill_tensor(t, ++seed, 0.0f, 0.2f);
        } else {
            fill_tensor(t, ++seed, -0.15f, 0.3f);
        }
        gguf_add_tensor(gguf, t);
    };

    add("embeddings.patch_embeddings.projection.weight", patch, patch, 3, hsz);
    add("embeddings.patch_embeddings.projection.bias", 1, 1, hsz, 1);
    add("embeddings.position_embeddings", hsz, grid * grid + 1, 1, 1);
    add("embeddings.cls_token", hsz, 1, 1, 1);

    for (int64_t il = 0; il < layers; ++il) {
        const std::string b = "encoder.layer." + std::to_string(il) + ".";
        add((b + "norm1.weight").c_str(), hsz, 1, 1, 1);
        add((b + "norm1.bias").c_str(), hsz, 1, 1, 1);
        add((b + "attention.attention.qkv.weight").c_str(), hsz, 3 * hsz, 1, 1);
        add((b + "attention.attention.qkv.bias").c_str(), 3 * hsz, 1, 1, 1);
        add((b + "attention.output.dense.weight").c_str(), hsz, hsz, 1, 1);
        add((b + "attention.output.dense.bias").c_str(), hsz, 1, 1, 1);
        add((b + "layer_scale1.lambda1").c_str(), hsz, 1, 1, 1);
        add((b + "norm2.weight").c_str(), hsz, 1, 1, 1);
        add((b + "norm2.bias").c_str(), hsz, 1, 1, 1);
        add((b + "mlp.fc1.weight").c_str(), hsz, mlp, 1, 1);
        add((b + "mlp.fc1.bias").c_str(), mlp, 1, 1, 1);
        add((b + "mlp.fc2.weight").c_str(), mlp, hsz, 1, 1);
        add((b + "mlp.fc2.bias").c_str(), hsz, 1, 1, 1);
        add((b + "layer_scale2.lambda1").c_str(), hsz, 1, 1, 1);
    }

    add("layernorm.weight", hsz, 1, 1, 1);
    add("layernorm.bias", hsz, 1, 1, 1);
    if (with_classifier) {
        add("classifier.weight", 2 * hsz, ncls, 1, 1);
        add("classifier.bias", ncls, 1, 1, 1);
    }

    const bool written = gguf_write_to_file(gguf, path.c_str(), false);
    ggml_free(tctx);
    gguf_free(gguf);
    return written;
}

// Deterministic 8-bit RGB pattern; seed shifts the content per image.
static std::vector<uint8_t> make_rgb8(int w, int h, uint32_t seed) {
    std::vector<uint8_t> px((size_t)w * h * 3);
    for (size_t i = 0; i < px.size(); ++i) {
        px[i] = (uint8_t)((i * 37 + seed * 101) % 256);
    }
    return px;
}

static dino_image as_dino_image(const std::vector<uint8_t> &px, int w, int h, int stride = 0) {
    dino_image img;
    img.pixels = px.data();
    img.width  = w;
    img.height = h;
    img.stride = stride;
    return img;
}

// streaming reader over an in-memory buffer for dino_model_load_from_callback
struct buf_reader {
    const uint8_t *data;
    size_t         size;
};

static size_t buf_read(void *userdata, void *output, uint64_t offset, size_t len) {
    const buf_reader &r = *static_cast<const buf_reader *>(userdata);
    if (offset > r.size) {
        return 0;
    }
    const size_t n = std::min(len, r.size - (size_t)offset);
    std::memcpy(output, r.data + offset, n);
    return n;
}

static std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        return {};
    }
    const std::streamsize size = file.tellg();
    file.seekg(0);
    std::vector<uint8_t> buf((size_t)size);
    if (!file.read(reinterpret_cast<char *>(buf.data()), size)) {
        return {};
    }
    return buf;
}

static bool floats_equal(const float *a, const std::vector<float> &b) {
    return a && std::memcmp(a, b.data(), b.size() * sizeof(float)) == 0;
}

const std::string tiny_path = (test_directory() / "tiny.gguf").string();

} // namespace

TEST_CASE("c api: version and defaults") {
    CHECK(dino_version() != nullptr);
    const dino_model_params mp = dino_model_default_params();
    CHECK(mp.device == nullptr);
    CHECK(mp.require_classifier == false);
    const dino_ctx_params cp = dino_ctx_default_params();
    CHECK(cp.n_threads >= 1);
    CHECK(cp.n_batch == 1);
    CHECK(cp.preprocess == DINO_PREPROCESS_BOUNDED);
    CHECK(cp.max_tokens == -1);
    const dino_run_params rp = dino_run_default_params();
    CHECK(rp.classify == false);
    CHECK(rp.topk == 5);
    CHECK(rp.l2_normalize == false);
    // NULL-tolerant teardown
    dino_model_free(nullptr);
    dino_free(nullptr);
}

TEST_CASE("c api: batch encode matches the C++ path") {
    REQUIRE(write_tiny_gguf(tiny_path));

    dino_model *model = dino_model_load_from_file(tiny_path.c_str(), dino_model_default_params());
    REQUIRE(model != nullptr);
    CHECK(dino_model_hidden_size(model) == 16);
    CHECK(dino_model_patch_size(model) == 2);
    CHECK(dino_model_n_register_tokens(model) == 0);
    CHECK(dino_model_has_classifier(model));
    CHECK(dino_model_n_classes(model) == 7);
    CHECK(std::string(dino_model_label(model, 3)) == "class_3");
    CHECK(dino_model_label(model, 99) == nullptr);

    dino_ctx_params cp = dino_ctx_default_params();
    cp.n_batch         = 2;
    dino_ctx *ctx      = dino_init_from_model(model, cp);
    REQUIRE(ctx != nullptr);

    const std::vector<uint8_t> px0     = make_rgb8(8, 8, 1);
    const std::vector<uint8_t> px1     = make_rgb8(8, 8, 2);
    const dino_image           imgs[2] = {as_dino_image(px0, 8, 8), as_dino_image(px1, 8, 8)};
    REQUIRE(dino_encode(ctx, imgs, 2, dino_run_default_params()) == DINO_STATUS_SUCCESS);
    CHECK(dino_output_n_images(ctx) == 2);

    // feature outputs: cls [16], pooled [32], patches [16 x 16]
    const float *cls0 = dino_output_cls(ctx, 0);
    const float *cls1 = dino_output_cls(ctx, 1);
    REQUIRE(cls0 != nullptr);
    REQUIRE(cls1 != nullptr);
    CHECK(dino_output_pooled(ctx, 0) != nullptr);
    int32_t n_patches = 0, grid_w = 0, grid_h = 0;
    CHECK(dino_output_patches(ctx, 0, &n_patches, &grid_w, &grid_h) != nullptr);
    CHECK(n_patches == 16);
    CHECK(grid_w == 4);
    CHECK(grid_h == 4);

    // same model + same pixels through the internal C++ path must produce
    // bit-identical outputs
    dino_model cpp_model;
    REQUIRE(dino_model_load(tiny_path, cpp_model, dino_model_options{}));
    dino_ctx cpp_ctx;
    REQUIRE(dino_ctx_init(cpp_ctx, cpp_model, dino_ctx_options{.n_batch = 2}));

    auto to_image_f = [&](const std::vector<uint8_t> &px) {
        Image img;
        img.nx = 8;
        img.ny = 8;
        img.c  = 3;
        img.data.assign(px.begin(), px.end());
        return dino_feature_preprocess(img, cpp_model.hparams, cpp_ctx.options);
    };
    const std::vector<dino_output> &cpp_outs =
        dino_predict(cpp_model, cpp_ctx, {to_image_f(px0), to_image_f(px1)}, dino_run_options{});
    REQUIRE(cpp_outs.size() == 2);
    CHECK(floats_equal(cls0, *cpp_outs[0].cls_token));
    CHECK(floats_equal(cls1, *cpp_outs[1].cls_token));
    CHECK(floats_equal(dino_output_pooled(ctx, 0), *cpp_outs[0].pooled));
    CHECK(floats_equal(dino_output_patches(ctx, 1, nullptr, nullptr, nullptr), *cpp_outs[1].patch_tokens));

    dino_ctx_free(cpp_ctx);
    dino_model_unload(cpp_model);
    dino_free(ctx);
    dino_model_free(model);
}

TEST_CASE("c api: buffer and callback loads match the file load") {
    const std::vector<uint8_t> buf = read_file(tiny_path);
    REQUIRE(!buf.empty());

    dino_model *m_buf = dino_model_load_from_buffer(buf.data(), buf.size(), dino_model_default_params());
    REQUIRE(m_buf != nullptr);
    buf_reader  r{buf.data(), buf.size()};
    dino_model *m_cb = dino_model_load_from_callback(buf_read, &r, dino_model_default_params());
    REQUIRE(m_cb != nullptr);
    CHECK(dino_model_hidden_size(m_buf) == dino_model_hidden_size(m_cb));
    CHECK(dino_model_has_classifier(m_cb));

    dino_ctx *c_buf = dino_init_from_model(m_buf, dino_ctx_default_params());
    dino_ctx *c_cb  = dino_init_from_model(m_cb, dino_ctx_default_params());
    REQUIRE(c_buf != nullptr);
    REQUIRE(c_cb != nullptr);

    const std::vector<uint8_t> px  = make_rgb8(8, 8, 3);
    const dino_image           img = as_dino_image(px, 8, 8);
    REQUIRE(dino_encode(c_buf, &img, 1, dino_run_default_params()) == DINO_STATUS_SUCCESS);
    const std::vector<float> cls_buf(dino_output_cls(c_buf, 0), dino_output_cls(c_buf, 0) + 16);
    REQUIRE(dino_encode(c_cb, &img, 1, dino_run_default_params()) == DINO_STATUS_SUCCESS);
    CHECK(std::memcmp(dino_output_cls(c_cb, 0), cls_buf.data(), cls_buf.size() * sizeof(float)) == 0);

    dino_free(c_buf);
    dino_free(c_cb);
    dino_model_free(m_buf);
    dino_model_free(m_cb);
}

TEST_CASE("c api: classify mode returns topk") {
    dino_model_params mp  = dino_model_default_params();
    mp.require_classifier = true; // exercises the strict label preflight
    dino_model *model     = dino_model_load_from_file(tiny_path.c_str(), mp);
    REQUIRE(model != nullptr);

    dino_ctx_params cp = dino_ctx_default_params();
    cp.n_batch         = 2;
    dino_ctx *ctx      = dino_init_from_model(model, cp);
    REQUIRE(ctx != nullptr);

    const std::vector<uint8_t> px0     = make_rgb8(8, 8, 4);
    const std::vector<uint8_t> px1     = make_rgb8(8, 8, 5);
    const dino_image           imgs[2] = {as_dino_image(px0, 8, 8), as_dino_image(px1, 8, 8)};
    dino_run_params            rp      = dino_run_default_params();
    rp.classify                        = true;
    rp.topk                            = 3;
    REQUIRE(dino_encode(ctx, imgs, 2, rp) == DINO_STATUS_SUCCESS);
    CHECK(dino_output_n_images(ctx) == 2);

    const uint32_t *indices = nullptr;
    const float    *probs   = nullptr;
    CHECK(dino_output_topk(ctx, 0, &indices, &probs) == 3);
    REQUIRE(indices != nullptr);
    REQUIRE(probs != nullptr);
    for (int i = 0; i < 3; ++i) {
        CHECK(indices[i] < 7);
        CHECK(probs[i] >= 0.0f);
        CHECK(probs[i] <= 1.0f);
        if (i > 0) {
            CHECK(probs[i - 1] >= probs[i]);
        }
    }
    // feature-only accessors return NULL in classify mode
    CHECK(dino_output_pooled(ctx, 0) == nullptr);
    CHECK(dino_output_patches(ctx, 0, nullptr, nullptr, nullptr) == nullptr);
    CHECK(dino_output_cls(ctx, 0) != nullptr); // cls is stored in both modes

    // compare against the C++ path
    dino_model cpp_model;
    REQUIRE(dino_model_load(tiny_path, cpp_model, dino_model_options{}));
    dino_ctx cpp_ctx;
    REQUIRE(dino_ctx_init(cpp_ctx, cpp_model, dino_ctx_options{.n_batch = 2}));
    auto to_image = [&](const std::vector<uint8_t> &px) {
        Image img;
        img.nx = 8;
        img.ny = 8;
        img.c  = 3;
        img.data.assign(px.begin(), px.end());
        return img;
    };
    const dino_run_options          run{/*.classify*/ true, /*.topk*/ 3, /*.l2_normalize*/ false};
    const std::vector<dino_output> &cpp_outs =
        dino_predict(cpp_model, cpp_ctx,
                     {dino_classify_preprocess(to_image(px0), cpp_model.hparams),
                      dino_classify_preprocess(to_image(px1), cpp_model.hparams)},
                     run);
    REQUIRE(cpp_outs.size() == 2);
    REQUIRE(cpp_outs[0].preds);
    CHECK(std::memcmp(indices, cpp_outs[0].preds->data(), 3 * sizeof(uint32_t)) == 0);
    CHECK(std::memcmp(probs, cpp_outs[0].pred_scores->data(), 3 * sizeof(float)) == 0);

    dino_ctx_free(cpp_ctx);
    dino_model_unload(cpp_model);
    dino_free(ctx);
    dino_model_free(model);
}

TEST_CASE("c api: strided rows and mixed sizes") {
    dino_model *model = dino_model_load_from_file(tiny_path.c_str(), dino_model_default_params());
    REQUIRE(model != nullptr);
    dino_ctx_params cp = dino_ctx_default_params();
    cp.n_batch         = 2;
    dino_ctx *ctx      = dino_init_from_model(model, cp);
    REQUIRE(ctx != nullptr);

    // padded stride: rows are width*3 + 4 bytes apart
    const int                  w = 8, h = 8, stride = w * 3 + 4;
    const std::vector<uint8_t> packed = make_rgb8(w, h, 6);
    std::vector<uint8_t>       strided((size_t)stride * h, 0xAB);
    for (int y = 0; y < h; ++y) {
        std::memcpy(strided.data() + (size_t)y * stride, packed.data() + (size_t)y * w * 3, (size_t)w * 3);
    }
    dino_image img = as_dino_image(strided, w, h, stride);
    REQUIRE(dino_encode(ctx, &img, 1, dino_run_default_params()) == DINO_STATUS_SUCCESS);
    const std::vector<float> cls_strided(dino_output_cls(ctx, 0), dino_output_cls(ctx, 0) + 16);

    const dino_image packed_img = as_dino_image(packed, w, h);
    REQUIRE(dino_encode(ctx, &packed_img, 1, dino_run_default_params()) == DINO_STATUS_SUCCESS);
    CHECK(std::memcmp(dino_output_cls(ctx, 0), cls_strided.data(), cls_strided.size() * sizeof(float)) == 0);

    // mixed sizes in one call group into separate single-size chunks
    const std::vector<uint8_t> big      = make_rgb8(16, 16, 7);
    const dino_image           mixed[2] = {as_dino_image(packed, 8, 8), as_dino_image(big, 16, 16)};
    REQUIRE(dino_encode(ctx, mixed, 2, dino_run_default_params()) == DINO_STATUS_SUCCESS);
    CHECK(dino_output_n_images(ctx) == 2);
    int32_t grid_w = 0, grid_h = 0;
    REQUIRE(dino_output_patches(ctx, 1, nullptr, &grid_w, &grid_h) != nullptr);
    CHECK(grid_w == 8);
    CHECK(grid_h == 8);

    // n_images > n_batch still processes every image (chunked by the wrapper)
    const dino_image three[2] = {as_dino_image(packed, 8, 8), as_dino_image(packed, 8, 8)};
    REQUIRE(dino_encode(ctx, three, 2, dino_run_default_params()) == DINO_STATUS_SUCCESS);

    dino_free(ctx);
    dino_model_free(model);
}

TEST_CASE("c api: error paths return status codes") {
    dino_model *model = dino_model_load_from_file(tiny_path.c_str(), dino_model_default_params());
    REQUIRE(model != nullptr);
    dino_ctx *ctx = dino_init_from_model(model, dino_ctx_default_params());
    REQUIRE(ctx != nullptr);

    const std::vector<uint8_t> px  = make_rgb8(8, 8, 8);
    const dino_image           img = as_dino_image(px, 8, 8);
    const dino_run_params      run = dino_run_default_params();

    CHECK(dino_encode(nullptr, &img, 1, run) == DINO_STATUS_INVALID_ARGUMENT);
    CHECK(dino_encode(ctx, nullptr, 1, run) == DINO_STATUS_INVALID_ARGUMENT);
    CHECK(dino_encode(ctx, &img, 0, run) == DINO_STATUS_INVALID_ARGUMENT);
    CHECK(dino_encode(ctx, &img, DINO_MAX_BATCH + 1, run) == DINO_STATUS_INVALID_ARGUMENT);

    dino_image bad = img;
    bad.pixels     = nullptr;
    CHECK(dino_encode(ctx, &bad, 1, run) == DINO_STATUS_INVALID_ARGUMENT);
    bad       = img;
    bad.width = 0;
    CHECK(dino_encode(ctx, &bad, 1, run) == DINO_STATUS_INVALID_ARGUMENT);
    bad        = img;
    bad.stride = img.width * 3 - 1; // stride shorter than one row
    CHECK(dino_encode(ctx, &bad, 1, run) == DINO_STATUS_INVALID_ARGUMENT);

    dino_run_params bad_run = run;
    bad_run.topk            = 0;
    CHECK(dino_encode(ctx, &img, 1, bad_run) == DINO_STATUS_INVALID_ARGUMENT);
    bad_run          = run;
    bad_run.classify = true;
    bad_run.topk     = 8; // more than the fixture's 7 classes
    CHECK(dino_encode(ctx, &img, 1, bad_run) == DINO_STATUS_INVALID_ARGUMENT);

    // classify on a backbone-only model: NO_CLASSIFIER, not a crash
    const std::string backbone_path = (test_directory() / "backbone.gguf").string();
    REQUIRE(write_tiny_gguf(backbone_path, /*with_classifier*/ false));
    dino_model *backbone = dino_model_load_from_file(backbone_path.c_str(), dino_model_default_params());
    REQUIRE(backbone != nullptr);
    CHECK(!dino_model_has_classifier(backbone));
    CHECK(dino_model_n_classes(backbone) == 0);
    dino_ctx *bctx = dino_init_from_model(backbone, dino_ctx_default_params());
    REQUIRE(bctx != nullptr);
    dino_run_params classify_run = dino_run_default_params();
    classify_run.classify        = true;
    CHECK(dino_encode(bctx, &img, 1, classify_run) == DINO_STATUS_NO_CLASSIFIER);

    // require_classifier fails the load of a backbone-only GGUF
    dino_model_params strict  = dino_model_default_params();
    strict.require_classifier = true;
    CHECK(dino_model_load_from_file(backbone_path.c_str(), strict) == nullptr);

    // invalid ctx params are rejected
    dino_ctx_params bad_cp = dino_ctx_default_params();
    bad_cp.n_batch         = 0;
    CHECK(dino_init_from_model(model, bad_cp) == nullptr);
    bad_cp           = dino_ctx_default_params();
    bad_cp.n_threads = 0;
    CHECK(dino_init_from_model(model, bad_cp) == nullptr);
    bad_cp            = dino_ctx_default_params();
    bad_cp.preprocess = (dino_preprocess)99;
    CHECK(dino_init_from_model(model, bad_cp) == nullptr);
    bad_cp            = dino_ctx_default_params();
    bad_cp.max_tokens = -2;
    CHECK(dino_init_from_model(model, bad_cp) == nullptr);
    CHECK(dino_init_from_model(nullptr, dino_ctx_default_params()) == nullptr);

    // invalid loads
    CHECK(dino_model_load_from_file(nullptr, dino_model_default_params()) == nullptr);
    CHECK(dino_model_load_from_file((test_directory() / "missing.gguf").string().c_str(),
                                    dino_model_default_params()) == nullptr);
    CHECK(dino_model_load_from_buffer(nullptr, 128, dino_model_default_params()) == nullptr);
    CHECK(dino_model_load_from_buffer(px.data(), 0, dino_model_default_params()) == nullptr);
    CHECK(dino_model_load_from_callback(nullptr, nullptr, dino_model_default_params()) == nullptr);
    const uint8_t junk[64] = {};
    CHECK(dino_model_load_from_buffer(junk, sizeof(junk), dino_model_default_params()) == nullptr);

    // accessor bounds
    CHECK(dino_output_n_images(nullptr) == 0);
    CHECK(dino_output_cls(nullptr, 0) == nullptr);
    CHECK(dino_output_cls(ctx, -1) == nullptr);
    CHECK(dino_output_cls(ctx, 99) == nullptr);
    CHECK(dino_output_topk(ctx, 0, nullptr, nullptr) == 0); // feature mode: no preds

    dino_free(bctx);
    dino_model_free(backbone);
    dino_free(ctx);
    dino_model_free(model);
}
