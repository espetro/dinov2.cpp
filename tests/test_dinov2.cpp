// unit tests for dinov2.cpp pure functions: dino_hparams math,
// interpolate_pos_embed, dino_preprocess fallback.
// No GGUF fixtures required - all tests run on synthetic inputs.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "dinov2.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

namespace {

// RAII guard for ggml_context so test failures don't leak memory.
struct CtxGuard {
    ggml_context *ctx = nullptr;
    explicit CtxGuard(size_t mem_size) {
        ggml_init_params p = {/*mem_size*/ mem_size, /*mem_buffer*/ nullptr, /*no_alloc*/ true};
        ctx                = ggml_init(p);
    }
    ~CtxGuard() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

// Deterministic pseudo-random fill so a batch run and a single-image run see
// identical weights. base + scale * hash keeps norm weights near 1 and other
// tensors small enough to avoid degenerate softmax saturation.
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

// A tiny fully in-memory dino_model (no GGUF fixture needed):
// hidden=16, heads=2, patch=2, img_size=8 -> 4x4 = 16 patches per image.
struct TinyModel {
    dino_model     model;
    ggml_gallocr_t allocr = nullptr;

    explicit TinyModel(uint32_t n_registers = 0, uint32_t n_layers = 2) {
        dino_hparams &h       = model.hparams;
        h.hidden_size         = 16;
        h.num_attention_heads = 2;
        h.num_hidden_layers   = n_layers;
        h.num_classes         = 7;
        h.num_register_tokens = n_registers;
        h.patch_size          = 2;
        h.img_size            = 8;

        model.backend = ggml_backend_cpu_init();

        // embeddings (5) + per-layer tensors (14) + final norm + classifier (4)
        const int        n_tensors = 5 + 14 * (int)n_layers + 4;
        ggml_init_params p         = {/*mem_size*/ ggml_tensor_overhead() * n_tensors,
                              /*mem_buffer*/ nullptr, /*no_alloc*/ true};
        model.ctx                  = ggml_init(p);

        uint32_t seed   = 0;
        auto     weight = [&](const char *name, int64_t n0, int64_t n1, int64_t n2, int64_t n3) {
            ggml_tensor *t = ggml_new_tensor_4d(model.ctx, GGML_TYPE_F32, n0, n1, n2, n3);
            ggml_set_name(t, name);
            model.tensors[name] = t;
            return t;
        };
        auto bias = [&](const char *name, int64_t n0) { return weight(name, n0, 1, 1, 1); };

        const uint32_t hsz  = h.hidden_size;
        const uint32_t ncls = h.num_classes;
        const uint32_t mlp  = 4 * hsz;
        const uint32_t grid = h.n_img_embd(); // patches per side at native img_size

        weight("embeddings.patch_embeddings.projection.weight", h.patch_size, h.patch_size, 3, hsz);
        // conv bias broadcasts over (W, H): stored as (1, 1, hidden, 1) in GGUF
        weight("embeddings.patch_embeddings.projection.bias", 1, 1, hsz, 1);
        weight("embeddings.position_embeddings", hsz, grid * grid + 1, 1, 1);
        bias("embeddings.cls_token", hsz);
        if (n_registers > 0) {
            weight("embeddings.register_tokens", hsz, n_registers, 1, 1);
        }

        for (uint32_t il = 0; il < n_layers; ++il) {
            const std::string b = "encoder.layer." + std::to_string(il) + ".";
            bias((b + "norm1.weight").c_str(), hsz);
            bias((b + "norm1.bias").c_str(), hsz);
            weight((b + "attention.attention.qkv.weight").c_str(), hsz, 3 * hsz, 1, 1);
            bias((b + "attention.attention.qkv.bias").c_str(), 3 * hsz);
            weight((b + "attention.output.dense.weight").c_str(), hsz, hsz, 1, 1);
            bias((b + "attention.output.dense.bias").c_str(), hsz);
            bias((b + "layer_scale1.lambda1").c_str(), hsz);
            bias((b + "norm2.weight").c_str(), hsz);
            bias((b + "norm2.bias").c_str(), hsz);
            weight((b + "mlp.fc1.weight").c_str(), hsz, mlp, 1, 1);
            bias((b + "mlp.fc1.bias").c_str(), mlp);
            weight((b + "mlp.fc2.weight").c_str(), mlp, hsz, 1, 1);
            bias((b + "mlp.fc2.bias").c_str(), hsz);
            bias((b + "layer_scale2.lambda1").c_str(), hsz);
        }

        bias("layernorm.weight", hsz);
        bias("layernorm.bias", hsz);
        weight("classifier.weight", 2 * hsz, ncls, 1, 1);
        bias("classifier.bias", ncls);

        model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

        for (auto &[name, t] : model.tensors) {
            ++seed;
            // multiplicative params get a positive base so activations stay sane
            const bool is_scale  = name.find("norm") != std::string::npos && name.find("weight") != std::string::npos;
            const bool is_lambda = name.find("lambda") != std::string::npos;
            if (is_scale) {
                fill_tensor(t, seed, 0.8f, 0.4f);
            } else if (is_lambda) {
                fill_tensor(t, seed, 0.4f, 0.6f);
            } else if (name.find("bias") != std::string::npos) {
                fill_tensor(t, seed, 0.0f, 0.2f);
            } else {
                fill_tensor(t, seed, -0.15f, 0.3f);
            }
        }

        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    }

    ~TinyModel() {
        // same teardown order as the CLI
        if (model.ctx) {
            ggml_free(model.ctx);
        }
        if (allocr) {
            ggml_gallocr_free(allocr);
        }
        if (model.buffer) {
            ggml_backend_buffer_free(model.buffer);
        }
        if (model.backend) {
            ggml_backend_free(model.backend);
        }
    }
};

static ImageF make_test_image(int nx, int ny, uint32_t seed) {
    ImageF img;
    img.nx = nx;
    img.ny = ny;
    img.c  = 3;
    img.data.resize((size_t)nx * ny * 3);
    for (size_t i = 0; i < img.data.size(); ++i) {
        img.data[i] = 0.5f + 0.25f * std::sin((float)(i + 1) * (float)seed * 0.37f);
    }
    return img;
}

} // namespace

TEST_CASE("dino_hparams default math") {
    dino_hparams h;
    // 224 / 14 == 16
    h.img_size   = 224;
    h.patch_size = 14;
    CHECK(h.n_img_embd() == 16);
    CHECK(h.n_img_size() == 224u);
    CHECK(h.n_patch_size() == 14u);

    // hidden_size / num_attention_heads (e.g., ViT-B: 768/12 == 64)
    h.hidden_size         = 768;
    h.num_attention_heads = 12;
    CHECK(h.n_enc_head_dim() == 64u);

    // ViT-L: 1024/16 == 64
    h.hidden_size         = 1024;
    h.num_attention_heads = 16;
    CHECK(h.n_enc_head_dim() == 64u);

    // ViT-S: 384/6 == 64
    h.hidden_size         = 384;
    h.num_attention_heads = 6;
    CHECK(h.n_enc_head_dim() == 64u);

    // ViT-g: 1536/24 == 64
    h.hidden_size         = 1536;
    h.num_attention_heads = 24;
    CHECK(h.n_enc_head_dim() == 64u);
}

TEST_CASE("dino_hparams divisibility invariant across DINOv2 family") {
    // For every canonical config, hidden_size must divide cleanly by
    // num_attention_heads (otherwise n_enc_head_dim truncates and breaks QKV
    // reshape). All four canonical head_dim == 64.
    struct cfg {
        uint32_t h;
        uint32_t n;
    };
    const std::vector<cfg> cfgs = {{384, 6}, {768, 12}, {1024, 16}, {1536, 24}};
    for (auto c : cfgs) {
        dino_hparams hp;
        hp.hidden_size         = c.h;
        hp.num_attention_heads = c.n;
        CHECK(hp.n_enc_head_dim() * c.n == c.h);
        CHECK(hp.n_enc_head_dim() == 64u);
    }
}

TEST_CASE("interpolate_pos_embed: identity when grid matches img_size") {
    // ViT-S/14: hidden=384, img_size=224, patch_size=14 -> grid 16x16 = 256
    // patches + 1 CLS = 257 tokens x 384 hidden.
    dino_hparams h;
    h.hidden_size = 384;
    h.img_size    = 224;
    h.patch_size  = 14;

    const int          num_patches = (int)h.n_img_embd() * (int)h.n_img_embd(); // 256
    const int          total_rows  = num_patches + 1;                           // 257
    std::vector<float> pos_embed((size_t)total_rows * h.hidden_size);

    // Initialize with deterministic values: row r, col c -> r * 1000 + c.
    for (int r = 0; r < total_rows; ++r) {
        for (int c = 0; c < (int)h.hidden_size; ++c) {
            pos_embed[(size_t)r * h.hidden_size + c] = (float)(r * 1000 + c);
        }
    }

    ImgSize    img_size{(int)h.img_size, (int)h.img_size};
    const auto out = interpolate_pos_embed(img_size, pos_embed.data(), h);

    REQUIRE(out.size() == pos_embed.size());
    for (size_t i = 0; i < pos_embed.size(); ++i) {
        CHECK(out[i] == doctest::Approx(pos_embed[i]));
    }
}

TEST_CASE("interpolate_pos_embed: identity when grid matches for non-square img_size") {
    // Different but still grid-aligned img_size (196/14 = 14).
    dino_hparams h;
    h.hidden_size = 384;
    h.img_size    = 196;
    h.patch_size  = 14;

    const int          num_patches = 14 * 14;
    const int          total_rows  = num_patches + 1;
    std::vector<float> pos_embed((size_t)total_rows * h.hidden_size);
    for (int r = 0; r < total_rows; ++r) {
        for (int c = 0; c < (int)h.hidden_size; ++c) {
            pos_embed[(size_t)r * h.hidden_size + c] = (float)r + (float)c * 0.1f;
        }
    }
    ImgSize    img_size{(int)h.img_size, (int)h.img_size};
    const auto out = interpolate_pos_embed(img_size, pos_embed.data(), h);
    REQUIRE(out.size() == pos_embed.size());
    for (size_t i = 0; i < pos_embed.size(); ++i) {
        CHECK(out[i] == doctest::Approx(pos_embed[i]));
    }
}

TEST_CASE("interpolate_pos_embed: CLS row preserved when grid differs") {
    // input grid 16x16 (img_size=224, patch=14). Request 518/14 = 37x37.
    dino_hparams h;
    h.hidden_size = 384;
    h.img_size    = 224;
    h.patch_size  = 14;

    constexpr int M           = 16; // 224/14
    constexpr int h_new       = 37; // 518/14
    constexpr int w_new       = 37;
    const int     hidden_size = (int)h.hidden_size;

    const int          total_in_rows = M * M + 1;
    std::vector<float> pos_embed((size_t)total_in_rows * hidden_size);
    for (int c = 0; c < hidden_size; ++c) {
        pos_embed[(size_t)c] = 1000.0f + (float)c; // CLS row sentinel
    }
    for (int r = 1; r < total_in_rows; ++r) {
        for (int c = 0; c < hidden_size; ++c) {
            pos_embed[(size_t)r * hidden_size + c] = (float)(r * 7 + c) * 0.5f;
        }
    }

    ImgSize    img_size{518, 518};
    const auto out = interpolate_pos_embed(img_size, pos_embed.data(), h);

    const int expected_total = h_new * w_new + 1;
    REQUIRE(out.size() == (size_t)expected_total * (size_t)hidden_size);

    // CLS row preserved exactly.
    for (int c = 0; c < hidden_size; ++c) {
        CHECK(out[(size_t)c] == doctest::Approx(1000.0f + (float)c));
    }

    // All values finite.
    for (float v : out) {
        CHECK(std::isfinite(v));
    }
}

TEST_CASE("interpolate_pos_embed: output size when grid differs") {
    dino_hparams h;
    h.hidden_size = 768; // ViT-B
    h.img_size    = 224;
    h.patch_size  = 14;

    constexpr int      M        = 16;
    constexpr int      h_new    = 37; // 518/14
    constexpr int      w_new    = 37;
    const int          total_in = M * M + 1;
    std::vector<float> pos_embed((size_t)total_in * h.hidden_size, 0.5f);

    ImgSize    img_size{518, 518};
    const auto out = interpolate_pos_embed(img_size, pos_embed.data(), h);

    const size_t expected = (size_t)(h_new * w_new + 1) * h.hidden_size;
    CHECK(out.size() == expected);

    // Constant input -> constant output (bicubic of constant field == field).
    for (float v : out) {
        CHECK(v == doctest::Approx(0.5f));
    }
}

TEST_CASE("dino_preprocess: pads non-aligned input to next patch-multiple") {
    // 100x50 input with patch_size=14:
    //   new_w = (100/14 + 1) * 14 = 8 * 14 = 112
    //   new_h = (50/14 + 1) * 14 = 4 * 14 = 56
    dino_hparams h;
    h.img_size   = 224;
    h.patch_size = 14;

    Image img;
    img.nx = 100;
    img.ny = 50;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 128);

    const auto out = dino_preprocess(img, h);

    CHECK(out.nx == (img.nx / (int)h.patch_size + 1) * (int)h.patch_size);
    CHECK(out.ny == (img.ny / (int)h.patch_size + 1) * (int)h.patch_size);
    CHECK(out.c == 3);

    CHECK(out.nx % (int)h.patch_size == 0);
    CHECK(out.ny % (int)h.patch_size == 0);

    REQUIRE(out.data.size() == (size_t)out.nx * (size_t)out.ny * 3);

    // Spot-check normalization on a constant input of 128.
    const float means[3] = {0.485f, 0.456f, 0.406f};
    const float stds[3]  = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
        const float expect = (128.0f / 255.0f - means[c]) / stds[c];
        CHECK(out.data[(size_t)c] == doctest::Approx(expect).epsilon(1e-4));
    }
}

TEST_CASE("dino_preprocess: image smaller than patch triggers resize") {
    // 7x3 source with patch_size=14: (7/14)+1 = 1, *14 = 14 each dim.
    dino_hparams h;
    h.img_size   = 224;
    h.patch_size = 14;

    Image img;
    img.nx = 7;
    img.ny = 3;
    img.c  = 3;
    img.data.assign((size_t)7 * 3 * 3, 64);

    const auto out = dino_preprocess(img, h);

    CHECK(out.nx == 14);
    CHECK(out.ny == 14);
    CHECK(out.c == 3);
    REQUIRE(out.data.size() == (size_t)14 * 14 * 3);

    // All values finite (no NaN from interpolation at the small source).
    for (float v : out.data) {
        CHECK(std::isfinite(v));
    }
}

TEST_CASE("dino_preprocess: normalization formula across all channels") {
    Image img;
    img.nx = 14;
    img.ny = 14;
    img.c  = 3;
    img.data.assign((size_t)14 * 14 * 3, 0);

    // R plane 255, G 128, B 0.
    for (size_t i = 0; i < img.data.size(); i += 3) {
        img.data[i + 0] = 255;
        img.data[i + 1] = 128;
        img.data[i + 2] = 0;
    }

    dino_hparams h;
    h.img_size   = 224;
    h.patch_size = 14;

    const auto out = dino_preprocess(img, h);
    REQUIRE(out.nx == 28);
    REQUIRE(out.ny == 28);

    const float means[3] = {0.485f, 0.456f, 0.406f};
    const float stds[3]  = {0.229f, 0.224f, 0.225f};

    for (size_t i = 0; i < out.data.size(); i += 3) {
        const float r_expect = (255.0f / 255.0f - means[0]) / stds[0];
        const float g_expect = (128.0f / 255.0f - means[1]) / stds[1];
        const float b_expect = (0.0f / 255.0f - means[2]) / stds[2];
        CHECK(out.data[i + 0] == doctest::Approx(r_expect).epsilon(1e-4));
        CHECK(out.data[i + 1] == doctest::Approx(g_expect).epsilon(1e-4));
        CHECK(out.data[i + 2] == doctest::Approx(b_expect).epsilon(1e-4));
    }
}

TEST_CASE("dino_classify_preprocess: square input yields 224x224 crop") {
    dino_hparams h;

    Image img;
    img.nx = 100;
    img.ny = 100;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 128);

    const auto out = dino_classify_preprocess(img, h);

    CHECK(out.nx == 224);
    CHECK(out.ny == 224);
    CHECK(out.c == 3);
    REQUIRE(out.data.size() == (size_t)224 * 224 * 3);

    // Constant input -> constant output equal to the normalized value.
    const float means[3] = {0.485f, 0.456f, 0.406f};
    const float stds[3]  = {0.229f, 0.224f, 0.225f};
    for (size_t i = 0; i < out.data.size(); i += 3) {
        for (int c = 0; c < 3; ++c) {
            const float expect = (128.0f / 255.0f - means[c]) / stds[c];
            CHECK(out.data[i + c] == doctest::Approx(expect).epsilon(1e-3));
        }
    }
}

TEST_CASE("dino_classify_preprocess: wide input preserves aspect (shortest-edge resize)") {
    // 512x128 input: aspect-preserving resize -> 1024x256, then a 224x224
    // center crop (x range 400..623 of the resized image). A 256x256 squash
    // would instead keep the left-edge stripe inside the crop.
    dino_hparams h;

    Image img;
    img.nx = 512;
    img.ny = 128;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 0);
    // left 10% stripe at 255
    for (int y = 0; y < img.ny; ++y) {
        for (int x = 0; x < img.nx / 10; ++x) {
            for (int c = 0; c < 3; ++c) {
                img.data[((size_t)y * img.nx + x) * 3 + c] = 255;
            }
        }
    }

    const auto out = dino_classify_preprocess(img, h);

    CHECK(out.nx == 224);
    CHECK(out.ny == 224);
    CHECK(out.c == 3);
    REQUIRE(out.data.size() == (size_t)224 * 224 * 3);

    for (float v : out.data) {
        CHECK(std::isfinite(v));
    }

    // The left-edge stripe must be cropped away under aspect-preserving
    // resize, so every output channel equals normalized 0.
    const float means[3] = {0.485f, 0.456f, 0.406f};
    const float stds[3]  = {0.229f, 0.224f, 0.225f};
    for (size_t i = 0; i < out.data.size(); i += 3) {
        for (int c = 0; c < 3; ++c) {
            const float expect = (0.0f / 255.0f - means[c]) / stds[c];
            CHECK(out.data[i + c] == doctest::Approx(expect).epsilon(5e-2));
        }
    }
}

TEST_CASE("binary embeddings preview writes little-endian header and payload") {
    const std::string path = "/tmp/dinov2-binary-preview-test.d2e";
    dino_output       output;
    output.cls_token    = std::vector<float>{1.0f, 2.0f};
    output.pooled       = std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f};
    output.patch_tokens = std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f};

    std::string error;
    REQUIRE(write_embeddings_binary(path, output, 2, 2, false, false, error));
    REQUIRE(error.empty());

    std::ifstream                    file(path, std::ios::binary);
    const std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    std::remove(path.c_str());
    REQUIRE(bytes.size() == 32 + (2 + 4) * sizeof(float));
    CHECK(std::memcmp(bytes.data(), "D2EMB\\0\\0\\0", 8) == 0);
    CHECK(bytes[8] == 1);
    CHECK(bytes[9] == 0);
    CHECK(bytes[10] == 32);
    CHECK(bytes[11] == 0);
    CHECK(bytes[12] == 2);
    CHECK(bytes[16] == 4);
    CHECK(bytes[20] == 0);
    CHECK(bytes[24] == 0);
    CHECK(bytes[28] == 0);

    const auto read_float = [&](size_t offset) {
        uint32_t bits = static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8) |
                        (static_cast<uint32_t>(bytes[offset + 2]) << 16) |
                        (static_cast<uint32_t>(bytes[offset + 3]) << 24);
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    CHECK(read_float(32) == doctest::Approx(1.0f));
    CHECK(read_float(36) == doctest::Approx(2.0f));
    CHECK(read_float(40) == doctest::Approx(1.0f));
    CHECK(read_float(44) == doctest::Approx(2.0f));
    CHECK(read_float(48) == doctest::Approx(3.0f));
    CHECK(read_float(52) == doctest::Approx(4.0f));

    REQUIRE(write_embeddings_binary(path, output, 2, 2, true, true, error));
    file.clear();
    file.open(path, std::ios::binary);
    const std::vector<unsigned char> patch_bytes((std::istreambuf_iterator<char>(file)),
                                                 std::istreambuf_iterator<char>());
    file.close();
    std::remove(path.c_str());
    REQUIRE(patch_bytes.size() == 32 + (2 + 4 + 4) * sizeof(float));
    CHECK(patch_bytes[20] == 2);
    CHECK(patch_bytes[24] == 3);
    CHECK(patch_bytes[56] == 0); // payload remains CLS, pooled, then patches
    CHECK(patch_bytes[57] == 0);
    CHECK(patch_bytes[58] == 160);
    CHECK(patch_bytes[59] == 64); // 5.0f in little-endian
}

TEST_CASE("l2_normalize: produces unit norm and preserves direction") {
    std::vector<float> v   = {3.0f, 4.0f, 0.0f, -1.0f, 2.0f};
    const float        in0 = v[0], in1 = v[1], in3 = v[3], in4 = v[4];
    const float        n = std::sqrt(in0 * in0 + in1 * in1 + in3 * in3 + in4 * in4);

    l2_normalize(v);

    float norm = 0.0f;
    for (float x : v) {
        norm += x * x;
    }
    CHECK(norm == doctest::Approx(1.0f));
    CHECK(v[0] == doctest::Approx(in0 / n));
    CHECK(v[1] == doctest::Approx(in1 / n));
    CHECK(v[2] == 0.0f);
    CHECK(v[3] == doctest::Approx(in3 / n));
    CHECK(v[4] == doctest::Approx(in4 / n));
}

TEST_CASE("l2_normalize: zero vector left unchanged") {
    std::vector<float> v(16, 0.0f);

    l2_normalize(v);

    for (float x : v) {
        CHECK(x == 0.0f);
    }
}

TEST_CASE("l2_normalize: already-unit vector unchanged") {
    std::vector<float> v = {1.0f, 0.0f, 0.0f, 0.0f};

    l2_normalize(v);

    CHECK(v[0] == doctest::Approx(1.0f));
    CHECK(v[1] == doctest::Approx(0.0f));
}

TEST_CASE("dino_batch_size_valid: bounds") {
    CHECK_FALSE(dino_batch_size_valid(-1));
    CHECK_FALSE(dino_batch_size_valid(0));
    CHECK(dino_batch_size_valid(1));
    CHECK(dino_batch_size_valid(DINO_MAX_BATCH));
    CHECK_FALSE(dino_batch_size_valid(DINO_MAX_BATCH + 1));
    CHECK_FALSE(dino_batch_size_valid(1'000'000));
}

TEST_CASE("dino_params: n_batch defaults to 1") {
    dino_params p;
    CHECK(p.n_batch == 1);
}

TEST_CASE("dino_params_parse: accepts valid numeric boundaries") {
    dino_params p;
    char        a0[] = "prog", a1[] = "--seed", a2[] = "-2147483648", a3[] = "--threads", a4[] = "1";
    char        a5[] = "--topk", a6[] = "1", a7[] = "--batch", a8[] = "64";
    char        a9[] = "--bench-runs", a10[] = "1", a11[] = "--bench-warmup", a12[] = "0";
    char       *argv[] = {a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12};

    CHECK(dino_params_parse(13, argv, p));
    CHECK(p.seed == INT32_MIN);
    CHECK(p.n_threads == 1);
    CHECK(p.topk == 1);
    CHECK(p.n_batch == DINO_MAX_BATCH);
    CHECK(p.bench_repeats == 1);
    CHECK(p.bench_warmup == 0);
}

TEST_CASE("dino_params_parse: --batch sets n_batch") {
    dino_params p;
    char        a0[] = "prog", a1[] = "--batch", a2[] = "4";
    char       *argv[] = {a0, a1, a2};
    CHECK(dino_params_parse(3, argv, p));
    CHECK(p.n_batch == 4);
}

TEST_CASE("dino_params: fnames_inp defaults to the bundled sample image") {
    dino_params p;
    REQUIRE(p.fnames_inp.size() == 1);
    CHECK(p.fnames_inp[0] == "../assets/tench.jpg");
}

TEST_CASE("dino_params_parse: repeated -i and comma lists collect images") {
    dino_params p;
    char        a0[] = "prog", a1[] = "-i", a2[] = "a.jpg";
    char        a3[] = "-i", a4[] = "b.jpg, c.jpg ,d.jpg";
    char       *argv[] = {a0, a1, a2, a3, a4};
    CHECK(dino_params_parse(5, argv, p));
    REQUIRE(p.fnames_inp.size() == 4);
    CHECK(p.fnames_inp[0] == "a.jpg");
    CHECK(p.fnames_inp[1] == "b.jpg");
    CHECK(p.fnames_inp[2] == "c.jpg");
    CHECK(p.fnames_inp[3] == "d.jpg");
}

TEST_CASE("dino_params_parse: -i with only empty tokens yields no images") {
    dino_params p;
    char        a0[] = "prog", a1[] = "-i", a2[] = " , ,";
    char       *argv[] = {a0, a1, a2};
    CHECK(dino_params_parse(3, argv, p));
    CHECK(p.fnames_inp.empty());
}

TEST_CASE("dino_predict: batch of 2 equals two single-image runs") {
    ImageF img0 = make_test_image(8, 8, 1);
    ImageF img1 = make_test_image(8, 8, 2);

    dino_params params;
    params.n_batch = 2;

    SUBCASE("no register tokens") {
        TinyModel m(/*n_registers=*/0);

        const auto batch = dino_predict(m.model, std::vector<ImageF>{img0, img1}, params, m.allocr);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, std::vector<ImageF>{img0}, params, m.allocr);
        const auto single1 = dino_predict(m.model, std::vector<ImageF>{img1}, params, m.allocr);
        REQUIRE(single0.size() == 1);
        REQUIRE(single1.size() == 1);

        // every graph op is per-batch-element independent, so equality is bitwise
        CHECK(batch[0].cls_token == single0[0].cls_token);
        CHECK(batch[1].cls_token == single1[0].cls_token);
        CHECK(batch[0].pooled == single0[0].pooled);
        CHECK(batch[1].pooled == single1[0].pooled);
        CHECK(batch[0].patch_tokens == single0[0].patch_tokens);
        CHECK(batch[1].patch_tokens == single1[0].patch_tokens);
    }

    SUBCASE("with register tokens") {
        TinyModel m(/*n_registers=*/2);

        const auto batch = dino_predict(m.model, std::vector<ImageF>{img0, img1}, params, m.allocr);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, std::vector<ImageF>{img0}, params, m.allocr);
        const auto single1 = dino_predict(m.model, std::vector<ImageF>{img1}, params, m.allocr);
        REQUIRE(single0.size() == 1);
        REQUIRE(single1.size() == 1);

        CHECK(batch[0].cls_token == single0[0].cls_token);
        CHECK(batch[1].cls_token == single1[0].cls_token);
        CHECK(batch[0].pooled == single0[0].pooled);
        CHECK(batch[1].pooled == single1[0].pooled);
        CHECK(batch[0].patch_tokens == single0[0].patch_tokens);
        CHECK(batch[1].patch_tokens == single1[0].patch_tokens);
    }
}

TEST_CASE("dino_predict: batch of 2 equals two single-image runs (classify)") {
    TinyModel m(/*n_registers=*/0);
    ImageF    img0 = make_test_image(8, 8, 1);
    ImageF    img1 = make_test_image(8, 8, 2);

    dino_params params;
    params.n_batch  = 2;
    params.classify = true;
    params.topk     = 3;

    const auto batch = dino_predict(m.model, std::vector<ImageF>{img0, img1}, params, m.allocr);
    REQUIRE(batch.size() == 2);
    const auto single0 = dino_predict(m.model, std::vector<ImageF>{img0}, params, m.allocr);
    const auto single1 = dino_predict(m.model, std::vector<ImageF>{img1}, params, m.allocr);
    REQUIRE(single0.size() == 1);
    REQUIRE(single1.size() == 1);

    CHECK(batch[0].cls_token == single0[0].cls_token);
    CHECK(batch[1].cls_token == single1[0].cls_token);
    CHECK(batch[0].preds == single0[0].preds);
    CHECK(batch[1].preds == single1[0].preds);
    CHECK(batch[0].pred_scores == single0[0].pred_scores);
    CHECK(batch[1].pred_scores == single1[0].pred_scores);
}

TEST_CASE("dino_predict: batch of 2 equals two single-image runs (flash attention)") {
    ImageF img0 = make_test_image(8, 8, 1);
    ImageF img1 = make_test_image(8, 8, 2);

    dino_params params;
    params.n_batch           = 2;
    params.enable_flash_attn = true;

    // the flash path pads the sequence to a multiple of 32; the TinyModel
    // seq len (16 patches + 1 cls + registers = 17 or 19) always pads, so
    // these subcases exercise the padded-KV and B>1 unpad-reshape path
    SUBCASE("no register tokens") {
        TinyModel m(/*n_registers=*/0);

        const auto batch = dino_predict(m.model, std::vector<ImageF>{img0, img1}, params, m.allocr);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, std::vector<ImageF>{img0}, params, m.allocr);
        const auto single1 = dino_predict(m.model, std::vector<ImageF>{img1}, params, m.allocr);
        REQUIRE(single0.size() == 1);
        REQUIRE(single1.size() == 1);

        // flash attention computes each (batch, head, q) row independently,
        // so equality with single-image runs is bitwise
        CHECK(batch[0].cls_token == single0[0].cls_token);
        CHECK(batch[1].cls_token == single1[0].cls_token);
        CHECK(batch[0].pooled == single0[0].pooled);
        CHECK(batch[1].pooled == single1[0].pooled);
        CHECK(batch[0].patch_tokens == single0[0].patch_tokens);
        CHECK(batch[1].patch_tokens == single1[0].patch_tokens);
    }

    SUBCASE("with register tokens") {
        TinyModel m(/*n_registers=*/2);

        const auto batch = dino_predict(m.model, std::vector<ImageF>{img0, img1}, params, m.allocr);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, std::vector<ImageF>{img0}, params, m.allocr);
        const auto single1 = dino_predict(m.model, std::vector<ImageF>{img1}, params, m.allocr);
        REQUIRE(single0.size() == 1);
        REQUIRE(single1.size() == 1);

        CHECK(batch[0].cls_token == single0[0].cls_token);
        CHECK(batch[1].cls_token == single1[0].cls_token);
        CHECK(batch[0].pooled == single0[0].pooled);
        CHECK(batch[1].pooled == single1[0].pooled);
        CHECK(batch[0].patch_tokens == single0[0].patch_tokens);
        CHECK(batch[1].patch_tokens == single1[0].patch_tokens);
    }

    SUBCASE("classify") {
        TinyModel m(/*n_registers=*/0);

        params.classify = true;
        params.topk     = 3;

        const auto batch = dino_predict(m.model, std::vector<ImageF>{img0, img1}, params, m.allocr);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, std::vector<ImageF>{img0}, params, m.allocr);
        const auto single1 = dino_predict(m.model, std::vector<ImageF>{img1}, params, m.allocr);
        REQUIRE(single0.size() == 1);
        REQUIRE(single1.size() == 1);

        CHECK(batch[0].cls_token == single0[0].cls_token);
        CHECK(batch[1].cls_token == single1[0].cls_token);
        CHECK(batch[0].preds == single0[0].preds);
        CHECK(batch[1].preds == single1[0].preds);
        CHECK(batch[0].pred_scores == single0[0].pred_scores);
        CHECK(batch[1].pred_scores == single1[0].pred_scores);
    }
}

TEST_CASE("dino_predict: single-image overload matches batch of 1") {
    TinyModel m;
    ImageF    img = make_test_image(8, 8, 3);

    dino_params params;

    const auto                   batch1 = dino_predict(m.model, std::vector<ImageF>{img}, params, m.allocr);
    std::unique_ptr<dino_output> single = dino_predict(m.model, img, params, m.allocr);
    REQUIRE(batch1.size() == 1);
    REQUIRE(single != nullptr);

    CHECK(single->cls_token == batch1[0].cls_token);
    CHECK(single->pooled == batch1[0].pooled);
    CHECK(single->patch_tokens == batch1[0].patch_tokens);
}

TEST_CASE("dino_predict: rejects invalid batch inputs") {
    TinyModel m;
    ImageF    img  = make_test_image(8, 8, 1);
    ImageF    wide = make_test_image(10, 8, 1); // different width

    dino_params params;
    params.n_batch = 2;

    // empty batch
    CHECK(dino_predict(m.model, std::vector<ImageF>{}, params, m.allocr).empty());

    // more images than n_batch allows
    CHECK(dino_predict(m.model, std::vector<ImageF>{img, img, img}, params, m.allocr).empty());

    // mismatched dimensions cannot share one graph
    CHECK(dino_predict(m.model, std::vector<ImageF>{img, wide}, params, m.allocr).empty());
}
