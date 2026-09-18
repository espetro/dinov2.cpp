// unit tests for dinov2.cpp pure functions: dino_hparams math,
// interpolate_pos_embed, dino_preprocess fallback.
// No GGUF fixtures required - all tests run on synthetic inputs.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "dinov2.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

    const int num_patches = (int)h.n_img_embd() * (int)h.n_img_embd(); // 256
    const int total_rows  = num_patches + 1;                          // 257
    std::vector<float> pos_embed((size_t)total_rows * h.hidden_size);

    // Initialize with deterministic values: row r, col c -> r * 1000 + c.
    for (int r = 0; r < total_rows; ++r) {
        for (int c = 0; c < (int)h.hidden_size; ++c) {
            pos_embed[(size_t)r * h.hidden_size + c] = (float)(r * 1000 + c);
        }
    }

    ImgSize img_size{(int)h.img_size, (int)h.img_size};
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

    const int num_patches = 14 * 14;
    const int total_rows  = num_patches + 1;
    std::vector<float> pos_embed((size_t)total_rows * h.hidden_size);
    for (int r = 0; r < total_rows; ++r) {
        for (int c = 0; c < (int)h.hidden_size; ++c) {
            pos_embed[(size_t)r * h.hidden_size + c] = (float)r + (float)c * 0.1f;
        }
    }
    ImgSize img_size{(int)h.img_size, (int)h.img_size};
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

    const int total_in_rows = M * M + 1;
    std::vector<float> pos_embed((size_t)total_in_rows * hidden_size);
    for (int c = 0; c < hidden_size; ++c) {
        pos_embed[(size_t)c] = 1000.0f + (float)c; // CLS row sentinel
    }
    for (int r = 1; r < total_in_rows; ++r) {
        for (int c = 0; c < hidden_size; ++c) {
            pos_embed[(size_t)r * hidden_size + c] = (float)(r * 7 + c) * 0.5f;
        }
    }

    ImgSize img_size{518, 518};
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

    constexpr int M         = 16;
    constexpr int h_new     = 37; // 518/14
    constexpr int w_new     = 37;
    const int     total_in  = M * M + 1;
    std::vector<float> pos_embed((size_t)total_in * h.hidden_size, 0.5f);

    ImgSize img_size{518, 518};
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
    img.nx   = 100;
    img.ny   = 50;
    img.c    = 3;
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
    img.nx   = 7;
    img.ny   = 3;
    img.c    = 3;
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
    img.nx   = 14;
    img.ny   = 14;
    img.c    = 3;
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
