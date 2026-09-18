// unit tests for src/image.cpp
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "image.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// reference Catmull-Rom kernel (mirrors cubic_kernel in src/image.cpp)
float ref_kernel(float x) {
    x = std::fabs(x);
    if (x < 1.0f) {
        return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    }
    return 0.0f;
}

// 4x4 ramp image, value = y * 4 + x
std::vector<float> ramp4() {
    std::vector<float> img(16);
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            img[(size_t)y * 4 + x] = (float)(y * 4 + x);
        }
    }
    return img;
}

} // namespace

TEST_CASE("cubic_kernel properties") {
    CHECK(cubic_kernel(0.0f) == doctest::Approx(1.0f));
    CHECK(cubic_kernel(1.0f) == doctest::Approx(0.0f));
    CHECK(cubic_kernel(0.5f) == doctest::Approx(0.5625f));
    CHECK(cubic_kernel(-0.5f) == doctest::Approx(0.5625f));

    // even symmetry kernel(t) == kernel(-t); non-negativity on the central
    // lobe [0, 1] (Catmull-Rom lobes dip slightly negative in (1, 2))
    for (int i = 0; i <= 10; ++i) {
        const float t = i / 10.0f;
        CHECK(cubic_kernel(t) == doctest::Approx(ref_kernel(t)));
        CHECK(cubic_kernel(t) == doctest::Approx(cubic_kernel(-t)));
        CHECK(cubic_kernel(t) >= 0.0f);
    }
    // outside support
    CHECK(cubic_kernel(2.5f) == 0.0f);
    CHECK(cubic_kernel(-2.5f) == 0.0f);
}

TEST_CASE("sample_cubic constant image") {
    std::vector<float> img(9 * 9, 7.25f);
    for (float fx : {0.0f, 2.5f, 4.0f, 8.0f}) {
        for (float fy : {0.0f, 1.5f, 4.0f, 8.0f}) {
            CHECK(sample_cubic(img.data(), 9, 9, 0, 1, fx, fy) == doctest::Approx(7.25f));
        }
    }
}

TEST_CASE("sample_cubic identity on 4x4 ramp") {
    const auto img = ramp4();
    // integer sample points map exactly onto pixel centers under the
    // (x + 0.5) * scale - 0.5 mapping with scale = 1
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            const float v = sample_cubic(img.data(), 4, 4, 0, 1, (float)x, (float)y);
            CHECK(v == doctest::Approx((float)(y * 4 + x)).epsilon(1e-5));
        }
    }
    // hand-computed fractional points (edge taps clamp at the left border)
    CHECK(sample_cubic(img.data(), 4, 4, 0, 1, 0.5f, 0.0f) == doctest::Approx(0.4375f).epsilon(1e-5));
    CHECK(sample_cubic(img.data(), 4, 4, 0, 1, 1.0f, 1.5f) == doctest::Approx(7.0f).epsilon(1e-5));
}

TEST_CASE("sample_cubic out-of-bounds clamping at all borders") {
    const auto  img = ramp4();
    const float c00 = sample_cubic(img.data(), 4, 4, 0, 1, -3.0f, -3.0f);
    const float c10 = sample_cubic(img.data(), 4, 4, 0, 1, 7.0f, -3.0f);
    const float c01 = sample_cubic(img.data(), 4, 4, 0, 1, -3.0f, 7.0f);
    const float c11 = sample_cubic(img.data(), 4, 4, 0, 1, 7.0f, 7.0f);
    for (float v : {c00, c10, c01, c11}) {
        CHECK(std::isfinite(v));
    }
    // far outside the image the kernel is evaluated only at clamped pixels;
    // the clamped-neighborhood mean of the corners is bounded by the ramp range
    CHECK(c00 >= 0.0f);
    CHECK(c00 <= 15.0f);
    CHECK(c11 >= 0.0f);
    CHECK(c11 <= 15.0f);
    // extreme extrapolation far off-image evaluates all taps at clamped
    // corners: expected values computed against the same clamped mean
    CHECK(c00 == doctest::Approx(0.0f).epsilon(1e-5));
    CHECK(c11 == doctest::Approx(15.0f).epsilon(1e-5));
}

TEST_CASE("resize_bicubic_f32 2x2 to 4x4 golden values") {
    const float src2[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    const auto  out     = resize_bicubic_f32(src2, 2, 2, 4, 4);
    REQUIRE(out.size() == 16);
    // golden values from the same Catmull-Rom math (OpenCV pixel-center mapping)
    const float expected[16] = {
        7.890625f,  10.625000f, 16.562500f, 19.296875f, //
        13.359375f, 16.093750f, 22.031250f, 24.765625f, //
        25.234375f, 27.968750f, 33.906250f, 36.640625f, //
        30.703125f, 33.437500f, 39.375000f, 42.109375f,
    };
    for (size_t i = 0; i < 16; ++i) {
        CHECK(out[i] == doctest::Approx(expected[i]).epsilon(1e-3));
    }
}

TEST_CASE("resize_bicubic 2x2 to 4x4 matches f32 result") {
    Image src;
    src.nx         = 2;
    src.ny         = 2;
    src.c          = 1;
    src.data       = {10, 20, 30, 40};
    const auto out = resize_bicubic(src, 4, 4);
    REQUIRE(out.nx == 4);
    REQUIRE(out.ny == 4);
    REQUIRE(out.data.size() == 16);
    const float src2[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    const auto  f32     = resize_bicubic_f32(src2, 2, 2, 4, 4);
    for (size_t i = 0; i < 16; ++i) {
        CHECK(std::abs((float)out.data[i] - f32[i]) <= 0.5f); // uint8 rounding
    }
}

TEST_CASE("resize 1x1 up and down does not crash") {
    const float one[1] = {42.0f};
    auto        up     = resize_bicubic_f32(one, 1, 1, 8, 8);
    REQUIRE(up.size() == 64);
    for (float v : up) {
        CHECK(std::isfinite(v));
        CHECK(v == doctest::Approx(42.0f).epsilon(1e-3));
    }
    auto down = resize_bicubic_f32(one, 1, 1, 1, 1);
    REQUIRE(down.size() == 1);
    CHECK(down[0] == doctest::Approx(42.0f).epsilon(1e-3));

    Image img;
    img.nx   = 1;
    img.ny   = 1;
    img.c    = 3;
    img.data = {1, 2, 3};
    auto up8 = resize_bicubic(img, 5, 5);
    REQUIRE(up8.data.size() == 75);
}

TEST_CASE("preprocess_for_dinov2 dims and normalization") {
    Image img;
    img.nx = 100;
    img.ny = 50;
    img.c  = 3;
    img.data.assign((size_t)100 * 50 * 3, 200);
    const auto out = preprocess_for_dinov2(img, 224);
    // short side becomes the next multiple of target_size above each dim
    CHECK(out.nx == 224);
    CHECK(out.ny == 224);
    REQUIRE(out.data.size() == (size_t)224 * 224 * 3);
    // constant channel 200 -> (200/255 - mean) / std
    const float means[3] = {0.485f, 0.456f, 0.406f};
    const float stds[3]  = {0.229f, 0.224f, 0.225f};
    for (int c = 0; c < 3; ++c) {
        const float expect = (200.0f / 255.0f - means[c]) / stds[c];
        CHECK(out.data[(size_t)c] == doctest::Approx(expect).epsilon(1e-4));
    }
    for (size_t i = 0; i < out.data.size(); i += 3) {
        for (int c = 0; c < 3; ++c) {
            CHECK(out.data[i + c] == doctest::Approx((200.0f / 255.0f - means[c]) / stds[c]).epsilon(1e-4));
        }
    }
}

TEST_CASE("preprocess_for_dinov2 empty input") {
    Image      img;
    const auto out = preprocess_for_dinov2(img, 224);
    CHECK(out.data.empty());
}

TEST_CASE("pca_project_3d: rank-3 axis-aligned data") {
    // 6 points on the 3 axes in 3D; dominant directions are the axes (up to sign)
    const std::vector<float> tokens = {
        10,  0,  0, //
        -10, 0,  0, //
        0,   5,  0, //
        0,   -5, 0, //
        0,   0,  1, //
        0,   0,  -1,
    };
    // write to a temp path; verify no crash and file created. the projection
    // itself is min-max normalized, so we validate via the written image path.
    const std::string out_path = "/tmp/dinov2_test_pca_axes.png";
    std::remove(out_path.c_str());
    pca_project_3d(tokens, 6, 3, 3, 2, 8, 8, out_path);
    FILE *f = fopen(out_path.c_str(), "rb");
    REQUIRE(f != nullptr);
    fseek(f, 0, SEEK_END);
    CHECK(ftell(f) > 0);
    fclose(f);
    std::remove(out_path.c_str());
}

TEST_CASE("pca_project_3d: constant data produces no NaN") {
    std::vector<float> tokens(8 * 3, 3.0f);
    const std::string  out_path = "/tmp/dinov2_test_pca_const.png";
    std::remove(out_path.c_str());
    pca_project_3d(tokens, 8, 3, 4, 2, 8, 8, out_path);
    FILE *f = fopen(out_path.c_str(), "rb");
    REQUIRE(f != nullptr);
    fclose(f);
    std::remove(out_path.c_str());
}

TEST_CASE("pca_project_3d: zero-range channel produces no crash") {
    // dim-2 channel is constant zero across all points
    std::vector<float> tokens;
    for (int i = 0; i < 8; ++i) {
        tokens.push_back((float)i);
        tokens.push_back(0.0f);
        tokens.push_back((float)(8 - i));
    }
    const std::string out_path = "/tmp/dinov2_test_pca_zeroch.png";
    std::remove(out_path.c_str());
    pca_project_3d(tokens, 8, 3, 2, 4, 8, 8, out_path);
    FILE *f = fopen(out_path.c_str(), "rb");
    REQUIRE(f != nullptr);
    fclose(f);
    std::remove(out_path.c_str());
}

TEST_CASE("pca_project_3d: invalid inputs do not crash") {
    std::vector<float> tokens(4, 0.0f);
    pca_project_3d(tokens, -1, 3, 2, 2, 8, 8, "/tmp/dinov2_test_pca_invalid.png");
    pca_project_3d(tokens, 7, 3, 0, 0, 8, 8, "/tmp/dinov2_test_pca_invalid.png"); // un-factorable grid
}
