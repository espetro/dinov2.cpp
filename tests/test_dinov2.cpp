// unit tests for dinov2.cpp pure functions: dino_hparams math,
// interpolate_pos_embed, dino_preprocess_padded fallback.
// No GGUF fixtures required - all tests run on synthetic inputs.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "dinov2-impl.h"
#include "ggml.h"
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
    dino_model model;
    dino_ctx   ctx;

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

        dino_ctx_init(ctx, model, dino_ctx_options{});
    }

    ~TinyModel() {
        // same teardown order as the CLI
        if (model.ctx) {
            ggml_free(model.ctx);
        }
        dino_ctx_free(ctx);
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

TEST_CASE("interpolate_pos_embed: equal patch count with different aspect still interpolates") {
    // 16x16 source grid (img_size=224, patch=14). A 448x112 request is a
    // 32x8 grid: same 256 patches but a different aspect, so the table must
    // be resampled, not returned verbatim.
    dino_hparams h;
    h.hidden_size = 8;
    h.img_size    = 224;
    h.patch_size  = 14;

    constexpr int M        = 16;
    constexpr int hidden   = 8;
    const int     in_rows  = M * M + 1;
    const int     w_new    = 32;
    const int     h_new    = 8;
    const int     out_rows = w_new * h_new + 1;

    std::vector<float> pos_embed((size_t)in_rows * hidden);
    for (int r = 0; r < in_rows; ++r) {
        for (int c = 0; c < hidden; ++c) {
            pos_embed[(size_t)r * hidden + c] = (float)r + (float)c * 0.25f;
        }
    }

    const auto out = interpolate_pos_embed({w_new * 14, h_new * 14}, pos_embed.data(), h);

    REQUIRE(out.size() == (size_t)out_rows * hidden);
    // same patch count, so the size matches the input, but the content must
    // not be an identity copy
    REQUIRE(out.size() == pos_embed.size());
    CHECK(out != pos_embed);

    // CLS row is preserved verbatim
    for (int c = 0; c < hidden; ++c) {
        CHECK(out[(size_t)c] == doctest::Approx((float)c * 0.25f));
    }

    // the patch block equals one interleaved resize_planes call on the
    // [patch, hidden] row-major source
    const std::vector<float> expected = resize_planes(pos_embed.data() + hidden, M, M, w_new, h_new, hidden);
    for (size_t i = 0; i < expected.size(); ++i) {
        CHECK(out[(size_t)hidden + i] == doctest::Approx(expected[i]));
    }
}

TEST_CASE("dino_preprocess_padded: pads non-aligned input to next patch-multiple") {
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

    const auto out = dino_preprocess_padded(img, h);

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

TEST_CASE("dino_preprocess_padded: image smaller than patch triggers resize") {
    // 7x3 source with patch_size=14: (7/14)+1 = 1, *14 = 14 each dim.
    dino_hparams h;
    h.img_size   = 224;
    h.patch_size = 14;

    Image img;
    img.nx = 7;
    img.ny = 3;
    img.c  = 3;
    img.data.assign((size_t)7 * 3 * 3, 64);

    const auto out = dino_preprocess_padded(img, h);

    CHECK(out.nx == 14);
    CHECK(out.ny == 14);
    CHECK(out.c == 3);
    REQUIRE(out.data.size() == (size_t)14 * 14 * 3);

    // All values finite (no NaN from interpolation at the small source).
    for (float v : out.data) {
        CHECK(std::isfinite(v));
    }
}

TEST_CASE("dino_preprocess_padded: normalization formula across all channels") {
    Image img;
    img.nx = 15;
    img.ny = 15;
    img.c  = 3;
    img.data.assign((size_t)15 * 15 * 3, 0);

    // R plane 255, G 128, B 0.
    for (size_t i = 0; i < img.data.size(); i += 3) {
        img.data[i + 0] = 255;
        img.data[i + 1] = 128;
        img.data[i + 2] = 0;
    }

    dino_hparams h;
    h.img_size   = 224;
    h.patch_size = 14;

    const auto out = dino_preprocess_padded(img, h);
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

TEST_CASE("dino_preprocess_padded: true ceil keeps patch-aligned dims unchanged") {
    // Under true ceil a dimension already at a multiple of patch_size is
    // unchanged: 518 -> 518 (37 patches), not 532 (38) as strict round-up did.
    dino_hparams h;
    h.img_size   = 224;
    h.patch_size = 14;

    Image img;
    img.nx = 518;
    img.ny = 518;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 128);

    const auto out = dino_preprocess_padded(img, h);
    CHECK(out.nx == 518);
    CHECK(out.ny == 518);

    Image img112;
    img112.nx = 112;
    img112.ny = 56;
    img112.c  = 3;
    img112.data.assign((size_t)112 * 56 * 3, 128);

    const auto out112 = dino_preprocess_padded(img112, h);
    CHECK(out112.nx == 112);
    CHECK(out112.ny == 56);
}

TEST_CASE("dino_feature_preprocess: bounded resizes shortest edge to 518") {
    // 2000x800 -> shortest edge 518, aspect preserved: 800 -> 518,
    // 2000 -> lround(2000 * 518/800) = 1295, then true-ceil to 1298.
    dino_hparams h;
    h.patch_size = 14;
    dino_ctx_options params;

    Image img;
    img.nx = 2000;
    img.ny = 800;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 128);

    const auto out = dino_feature_preprocess(img, h, params);

    const int expected_w = ((int)std::lround(2000 * 518.0 / 800.0) + 13) / 14 * 14;
    CHECK(out.ny == 518);
    CHECK(out.nx == expected_w);
    CHECK(out.nx % 14 == 0);
    CHECK(out.ny % 14 == 0);
    // bounded grid stays under the default cap (4 * 37 * 37 = 5476)
    CHECK((int64_t)(out.ny / 14) * (out.nx / 14) <= (int64_t)dino_default_max_tokens(h.patch_size));
}

TEST_CASE("dino_feature_preprocess: bounded leaves images under the bound untouched") {
    // 500x375: shortest edge <= 518, so only true-ceil alignment applies:
    // 500 -> 504, 375 -> 378.
    dino_hparams h;
    h.patch_size = 14;
    dino_ctx_options params;

    Image img;
    img.nx = 500;
    img.ny = 375;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 128);

    const auto out    = dino_feature_preprocess(img, h, params);
    const auto native = dino_preprocess_padded(img, h);

    CHECK(out.nx == 504);
    CHECK(out.ny == 378);
    CHECK(out.nx == native.nx);
    CHECK(out.ny == native.ny);
}

TEST_CASE("dino_feature_preprocess: no_resize keeps native resolution") {
    dino_hparams h;
    h.patch_size = 14;
    dino_ctx_options params;
    params.no_resize = true;

    Image img;
    img.nx = 2000;
    img.ny = 800;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 128);

    const auto out = dino_feature_preprocess(img, h, params);
    // no bound: 2000 -> 2002, 800 -> 812 (true ceil to patch multiples)
    CHECK(out.nx == ((2000 + 13) / 14) * 14);
    CHECK(out.ny == ((800 + 13) / 14) * 14);
}

TEST_CASE("dino_feature_preprocess: hf yields 224x224") {
    dino_hparams h;
    h.patch_size = 14;
    dino_ctx_options params;
    params.preprocess_mode = dino_preprocess_mode::hf;

    Image img;
    img.nx = 2000;
    img.ny = 800;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 128);

    const auto out = dino_feature_preprocess(img, h, params);
    CHECK(out.nx == 224);
    CHECK(out.ny == 224);
}

TEST_CASE("dino_feature_preprocess: crop518 yields a fixed 518x518 grid") {
    dino_hparams h;
    h.patch_size = 14;
    dino_ctx_options params;
    params.preprocess_mode = dino_preprocess_mode::crop518;

    Image img;
    img.nx = 2000;
    img.ny = 800;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 128);

    const auto out = dino_feature_preprocess(img, h, params);
    CHECK(out.nx == 518);
    CHECK(out.ny == 518);
    // different aspect, same output dims: batch-safe
    img.nx = 800;
    img.ny = 2000;
    img.data.assign((size_t)img.nx * img.ny * 3, 128);
    const auto out2 = dino_feature_preprocess(img, h, params);
    CHECK(out2.nx == 518);
    CHECK(out2.ny == 518);
}

TEST_CASE("dino_feature_output_size matches dino_feature_preprocess dims") {
    dino_hparams h;
    h.patch_size = 14;

    const int sizes[][2] = {{2000, 800}, {518, 518}, {612, 408}, {3440, 5601}, {15, 15}, {1, 1}, {112, 56}};
    for (const auto &s : sizes) {
        for (auto mode : {dino_preprocess_mode::bounded, dino_preprocess_mode::hf, dino_preprocess_mode::crop518}) {
            dino_ctx_options params;
            params.preprocess_mode = mode;
            Image img;
            img.nx = s[0];
            img.ny = s[1];
            img.c  = 3;
            img.data.assign((size_t)img.nx * img.ny * 3, 128);
            const ImgSize target = dino_feature_output_size(img, h, params);
            const ImageF  out    = dino_feature_preprocess(img, h, params);
            CHECK(out.nx == target.width);
            CHECK(out.ny == target.height);
            CHECK(out.nx % 14 == 0);
            CHECK(out.ny % 14 == 0);
        }
    }
    // --no-resize under bounded: dims equal the true-ceil native alignment
    {
        dino_ctx_options params;
        params.no_resize = true;
        Image img;
        img.nx = 2000;
        img.ny = 800;
        img.c  = 3;
        img.data.assign((size_t)img.nx * img.ny * 3, 128);
        const ImgSize target = dino_feature_output_size(img, h, params);
        const ImageF  out    = dino_feature_preprocess(img, h, params);
        CHECK(out.nx == target.width);
        CHECK(out.ny == target.height);
    }
}

TEST_CASE("dino_feature_preprocess: bounded output identical to dino_preprocess_padded under the bound") {
    // shortest edge <= 518: bounded does a single resample to the same
    // patch-aligned dims dino_preprocess_padded computes, so bytes must match.
    dino_hparams h;
    h.patch_size = 14;
    dino_ctx_options params;

    Image img;
    img.nx = 500;
    img.ny = 375;
    img.c  = 3;
    img.data.resize((size_t)img.nx * img.ny * 3);
    for (size_t i = 0; i < img.data.size(); ++i) {
        img.data[i] = (uint8_t)(i % 251);
    }

    const auto a = dino_feature_preprocess(img, h, params);
    const auto b = dino_preprocess_padded(img, h);
    CHECK(a.nx == b.nx);
    CHECK(a.ny == b.ny);
    CHECK(a.data == b.data);
}

TEST_CASE("dino_classify_preprocess: shared helper keeps 224x224 output") {
    // classify and hf feature mode run the same (256, 224) recipe, so their
    // outputs must be identical byte for byte.
    dino_hparams     h;
    dino_ctx_options params;
    params.preprocess_mode = dino_preprocess_mode::hf;

    Image img;
    img.nx = 512;
    img.ny = 128;
    img.c  = 3;
    img.data.assign((size_t)img.nx * img.ny * 3, 200);

    const auto cls = dino_classify_preprocess(img, h);
    const auto hf  = dino_feature_preprocess(img, h, params);
    CHECK(cls.nx == hf.nx);
    CHECK(cls.ny == hf.ny);
    CHECK(cls.data == hf.data);
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
    CHECK(dino_batch_size_valid(dino_max_batch));
    CHECK_FALSE(dino_batch_size_valid(dino_max_batch + 1));
    CHECK_FALSE(dino_batch_size_valid(1'000'000));
}

TEST_CASE("dino_ctx_options: n_batch defaults to 1") {
    dino_ctx_options options;
    CHECK(options.n_batch == 1);
}

TEST_CASE("dino_predict: batch of 2 equals two single-image runs") {
    ImageF img0 = make_test_image(8, 8, 1);
    ImageF img1 = make_test_image(8, 8, 2);

    dino_ctx_options copts;
    copts.n_batch = 2;
    dino_run_options ropts;

    SUBCASE("no register tokens") {
        TinyModel m(/*n_registers=*/0);
        m.ctx.options = copts;

        const auto batch = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0, img1}, ropts);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0}, ropts);
        const auto single1 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img1}, ropts);
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
        m.ctx.options = copts;

        const auto batch = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0, img1}, ropts);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0}, ropts);
        const auto single1 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img1}, ropts);
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

    m.ctx.options.n_batch = 2;
    dino_run_options ropts;
    ropts.classify = true;
    ropts.topk     = 3;

    const auto batch = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0, img1}, ropts);
    REQUIRE(batch.size() == 2);
    const auto single0 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0}, ropts);
    const auto single1 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img1}, ropts);
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

    dino_ctx_options copts;
    copts.n_batch           = 2;
    copts.enable_flash_attn = true;
    dino_run_options ropts;

    // the flash path pads the sequence to a multiple of 32; the TinyModel
    // seq len (16 patches + 1 cls + registers = 17 or 19) always pads, so
    // these subcases exercise the padded-KV and B>1 unpad-reshape path
    SUBCASE("no register tokens") {
        TinyModel m(/*n_registers=*/0);
        m.ctx.options = copts;

        const auto batch = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0, img1}, ropts);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0}, ropts);
        const auto single1 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img1}, ropts);
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
        m.ctx.options = copts;

        const auto batch = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0, img1}, ropts);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0}, ropts);
        const auto single1 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img1}, ropts);
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
        m.ctx.options = copts;

        ropts.classify = true;
        ropts.topk     = 3;

        const auto batch = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0, img1}, ropts);
        REQUIRE(batch.size() == 2);
        const auto single0 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img0}, ropts);
        const auto single1 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img1}, ropts);
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

TEST_CASE("dino_predict: flash attention matches the non-flash path") {
    // Regression net for the K/V zero-padding bug: padded keys used to enter
    // the softmax with logit 0 and their zero V rows diluted the numerator,
    // so -fa outputs drifted from the reference path. The KV seq dim is no
    // longer padded, so both paths must agree within float tolerance.
    ImageF img = make_test_image(8, 8, 1);

    const auto max_abs_diff = [](const std::optional<std::vector<float>> &a,
                                 const std::optional<std::vector<float>> &b) {
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        REQUIRE(a->size() == b->size());
        float worst = 0.0f;
        for (size_t i = 0; i < a->size(); ++i) {
            worst = std::max(worst, std::fabs((*a)[i] - (*b)[i]));
        }
        return worst;
    };

    SUBCASE("no register tokens") {
        TinyModel m(/*n_registers=*/0);

        dino_run_options ropts;

        m.ctx.options.enable_flash_attn = false;
        const auto ref                  = dino_predict(m.model, m.ctx, std::vector<ImageF>{img}, ropts);
        m.ctx.options.enable_flash_attn = true;
        const auto fa                   = dino_predict(m.model, m.ctx, std::vector<ImageF>{img}, ropts);
        REQUIRE(ref.size() == 1);
        REQUIRE(fa.size() == 1);

        CHECK(max_abs_diff(ref[0].cls_token, fa[0].cls_token) < 1e-5f);
        CHECK(max_abs_diff(ref[0].pooled, fa[0].pooled) < 1e-5f);
        CHECK(max_abs_diff(ref[0].patch_tokens, fa[0].patch_tokens) < 1e-5f);
    }

    SUBCASE("with register tokens") {
        TinyModel m(/*n_registers=*/2);

        dino_run_options ropts;

        m.ctx.options.enable_flash_attn = false;
        const auto ref                  = dino_predict(m.model, m.ctx, std::vector<ImageF>{img}, ropts);
        m.ctx.options.enable_flash_attn = true;
        const auto fa                   = dino_predict(m.model, m.ctx, std::vector<ImageF>{img}, ropts);
        REQUIRE(ref.size() == 1);
        REQUIRE(fa.size() == 1);

        CHECK(max_abs_diff(ref[0].cls_token, fa[0].cls_token) < 1e-5f);
        CHECK(max_abs_diff(ref[0].pooled, fa[0].pooled) < 1e-5f);
        CHECK(max_abs_diff(ref[0].patch_tokens, fa[0].patch_tokens) < 1e-5f);
    }

    SUBCASE("classify") {
        TinyModel m(/*n_registers=*/0);

        dino_run_options ropts;
        ropts.classify = true;
        ropts.topk     = 3;

        m.ctx.options.enable_flash_attn = false;
        const auto ref                  = dino_predict(m.model, m.ctx, std::vector<ImageF>{img}, ropts);
        m.ctx.options.enable_flash_attn = true;
        const auto fa                   = dino_predict(m.model, m.ctx, std::vector<ImageF>{img}, ropts);
        REQUIRE(ref.size() == 1);
        REQUIRE(fa.size() == 1);

        CHECK(max_abs_diff(ref[0].cls_token, fa[0].cls_token) < 1e-5f);
        REQUIRE(ref[0].preds.has_value());
        REQUIRE(fa[0].preds.has_value());
        CHECK(ref[0].preds == fa[0].preds);
        CHECK(max_abs_diff(ref[0].pred_scores, fa[0].pred_scores) < 1e-5f);
    }
}

TEST_CASE("dino_predict: single-image overload matches batch of 1") {
    TinyModel m;
    ImageF    img = make_test_image(8, 8, 3);

    dino_run_options ropts;

    const auto        batch1 = dino_predict(m.model, m.ctx, std::vector<ImageF>{img}, ropts);
    const dino_output single = *dino_predict(m.model, m.ctx, img, ropts);
    REQUIRE(batch1.size() == 1);

    CHECK(single.cls_token == batch1[0].cls_token);
    CHECK(single.pooled == batch1[0].pooled);
    CHECK(single.patch_tokens == batch1[0].patch_tokens);
}

TEST_CASE("dino_predict: rejects invalid batch inputs") {
    TinyModel m;
    ImageF    img  = make_test_image(8, 8, 1);
    ImageF    wide = make_test_image(10, 8, 1); // different width

    m.ctx.options.n_batch = 2;
    dino_run_options ropts;

    // empty batch
    CHECK(dino_predict(m.model, m.ctx, std::vector<ImageF>{}, ropts).empty());

    // more images than n_batch allows
    CHECK(dino_predict(m.model, m.ctx, std::vector<ImageF>{img, img, img}, ropts).empty());

    // mismatched dimensions cannot share one graph
    CHECK(dino_predict(m.model, m.ctx, std::vector<ImageF>{img, wide}, ropts).empty());
}
