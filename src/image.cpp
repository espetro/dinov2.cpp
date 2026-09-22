#include "image.h"

constexpr float IMAGENET_DEFAULT_MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float IMAGENET_DEFAULT_STD[3]  = {0.229f, 0.224f, 0.225f};

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// load / write
// ---------------------------------------------------------------------------

Image load_image(const std::string &path) {
    Image    img;
    int      w, h, n;
    uint8_t *pixels = stbi_load(path.c_str(), &w, &h, &n, 3);
    if (!pixels) {
        fprintf(stderr, "%s: failed to load image from '%s': %s\n", __func__, path.c_str(), stbi_failure_reason());
        return img;
    }
    img.nx = w;
    img.ny = h;
    img.c  = 3;
    img.data.assign(pixels, pixels + (size_t)w * h * 3);
    stbi_image_free(pixels);
    return img;
}

Image load_image_from_memory(const uint8_t *data, size_t size) {
    Image img;
    if (!data || size == 0 || size > (size_t)INT_MAX) {
        fprintf(stderr, "%s: invalid buffer (size %zu)\n", __func__, size);
        return img;
    }
    int      w, h, n;
    uint8_t *pixels = stbi_load_from_memory(data, (int)size, &w, &h, &n, 3);
    if (!pixels) {
        fprintf(stderr, "%s: failed to decode image: %s\n", __func__, stbi_failure_reason());
        return img;
    }
    img.nx = w;
    img.ny = h;
    img.c  = 3;
    img.data.assign(pixels, pixels + (size_t)w * h * 3);
    stbi_image_free(pixels);
    return img;
}

void write_png(const std::string &path, const Image &img) {
    if (img.data.empty() || img.nx <= 0 || img.ny <= 0) {
        fprintf(stderr, "%s: nothing to write to '%s'\n", __func__, path.c_str());
        return;
    }
    if (!stbi_write_png(path.c_str(), img.nx, img.ny, img.c, img.data.data(), img.nx * img.c)) {
        fprintf(stderr, "%s: failed to save image to '%s'\n", __func__, path.c_str());
    }
}

// ---------------------------------------------------------------------------
// bicubic resize (Catmull-Rom kernel, matches OpenCV INTER_CUBIC)
// ---------------------------------------------------------------------------

float cubic_kernel(float x) {
    // Catmull-Rom, support = 2
    x = std::fabs(x);
    if (x < 1.0f) {
        return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    }
    if (x < 2.0f) {
        return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    }
    return 0.0f;
}

float sample_cubic(const float *src, int sw, int sh, int c, int channels, float fx, float fy) {
    const int cx  = (int)std::floor(fx);
    const int cy  = (int)std::floor(fy);
    float     acc = 0.0f, wsum = 0.0f;
    for (int m = -1; m <= 2; ++m) {
        const float wy = cubic_kernel(fy - (cy + m));
        if (wy == 0.0f) {
            continue;
        }
        int sy = cy + m;
        sy     = std::max(0, std::min(sh - 1, sy));
        for (int n = -1; n <= 2; ++n) {
            const float wx = cubic_kernel(fx - (cx + n));
            if (wx == 0.0f) {
                continue;
            }
            int sx        = cx + n;
            sx            = std::max(0, std::min(sw - 1, sx));
            const float w = wx * wy;
            acc += w * src[((size_t)sy * sw + sx) * channels + c];
            wsum += w;
        }
    }
    return wsum != 0.0f ? acc / wsum : 0.0f;
}

// bicubic resize of float image
std::vector<float> resize_planes(const float *src, int sw, int sh, int dw, int dh, int channels) {
    const float scale_x = (float)sw / dw;
    const float scale_y = (float)sh / dh;
    // OpenCV pixel-center mapping
    std::vector<float> dst((size_t)dw * dh * channels);
    for (int y = 0; y < dh; ++y) {
        const float fy = (y + 0.5f) * scale_y - 0.5f;
        for (int x = 0; x < dw; ++x) {
            const float fx  = (x + 0.5f) * scale_x - 0.5f;
            float      *out = &dst[((size_t)y * dw + x) * channels];
            for (int c = 0; c < channels; ++c) {
                out[c] = sample_cubic(src, sw, sh, c, channels, fx, fy);
            }
        }
    }
    return dst;
}

std::vector<float> resize_bicubic_f32(const float *src, int sw, int sh, int dw, int dh) {
    if (!src || sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) {
        return {};
    }
    return resize_planes(src, sw, sh, dw, dh, 1);
}

Image resize_bicubic(const Image &src, int w, int h) {
    Image dst;
    if (src.data.empty() || w <= 0 || h <= 0) {
        return dst;
    }
    // convert to float
    std::vector<float> fsrc(src.data.begin(), src.data.end());
    std::vector<float> fdst = resize_planes(fsrc.data(), src.nx, src.ny, w, h, src.c);
    dst.nx                  = w;
    dst.ny                  = h;
    dst.c                   = src.c;
    dst.data.resize((size_t)w * h * src.c);
    for (size_t i = 0; i < dst.data.size(); ++i) {
        float v     = std::round(fdst[i]);
        dst.data[i] = (uint8_t)std::max(0.0f, std::min(255.0f, v));
    }
    return dst;
}

Image resize_shortest_edge(const Image &src, int short_edge) {
    const float scale = (float)short_edge / (float)std::min(src.nx, src.ny);
    const int   new_w = std::max((int)std::lround(src.nx * scale), 1);
    const int   new_h = std::max((int)std::lround(src.ny * scale), 1);
    return resize_bicubic(src, new_w, new_h);
}

ImageF preprocess_resize_crop(const Image &src, int short_edge, int crop) {
    ImageF out;
    if (src.data.empty() || crop > short_edge) {
        return out;
    }
    // 1) shortest-edge resize preserving aspect ratio (HF BitImageProcessor
    //    parity)
    Image image = resize_shortest_edge(src, short_edge);

    // clamp >= 0 for safety (min dimension is short_edge >= crop for the
    // recipes used here)
    const int offset_w = std::max((image.nx - crop) / 2, 0);
    const int offset_h = std::max((image.ny - crop) / 2, 0);

    // 2) center crop
    Image cropped;
    cropped.nx = crop;
    cropped.ny = crop;
    cropped.c  = 3;
    cropped.data.resize((size_t)crop * crop * 3);
    for (int y = 0; y < crop; ++y) {
        const uint8_t *src_row = &image.data[((size_t)(offset_h + y) * image.nx + offset_w) * 3];
        std::memcpy(&cropped.data[(size_t)y * crop * 3], src_row, (size_t)crop * 3);
    }

    // 3) convert to float, scale to [0,1] and channel-wise standardization (RGB)
    out.nx = crop;
    out.ny = crop;
    out.c  = 3;
    out.data.resize((size_t)crop * crop * 3);
    for (size_t i = 0; i < cropped.data.size(); i += 3) {
        out.data[i + 0] = (cropped.data[i + 0] / 255.0f - IMAGENET_DEFAULT_MEAN[0]) / IMAGENET_DEFAULT_STD[0];
        out.data[i + 1] = (cropped.data[i + 1] / 255.0f - IMAGENET_DEFAULT_MEAN[1]) / IMAGENET_DEFAULT_STD[1];
        out.data[i + 2] = (cropped.data[i + 2] / 255.0f - IMAGENET_DEFAULT_MEAN[2]) / IMAGENET_DEFAULT_STD[2];
    }
    return out;
}

// ---------------------------------------------------------------------------
// DINOv2 preprocessing
// ---------------------------------------------------------------------------

ImageF preprocess_resize_normalized(const Image &src, int w, int h) {
    ImageF out;
    if (src.data.empty() || w <= 0 || h <= 0) {
        return out;
    }
    // convert to float in [0, 1]
    std::vector<float> fsrc(src.data.size());
    for (size_t i = 0; i < src.data.size(); ++i) {
        fsrc[i] = src.data[i] / 255.0f;
    }

    std::vector<float> resized = resize_planes(fsrc.data(), src.nx, src.ny, w, h, 3);

    out.nx = w;
    out.ny = h;
    out.c  = 3;
    out.data.resize((size_t)w * h * 3);
    for (size_t i = 0; i < out.data.size(); i += 3) {
        // RGB channel order (stb gives RGB), ImageNet normalization
        out.data[i + 0] = (resized[i + 0] - IMAGENET_DEFAULT_MEAN[0]) / IMAGENET_DEFAULT_STD[0];
        out.data[i + 1] = (resized[i + 1] - IMAGENET_DEFAULT_MEAN[1]) / IMAGENET_DEFAULT_STD[1];
        out.data[i + 2] = (resized[i + 2] - IMAGENET_DEFAULT_MEAN[2]) / IMAGENET_DEFAULT_STD[2];
    }
    return out;
}

ImageF preprocess_for_dinov2(const Image &src, int target_size) {
    // resize each dimension up to the next multiple of target_size (true
    // ceil: a dimension already at a multiple is unchanged)
    auto mult = [](int v, int t) { return ((v + t - 1) / t) * t; };
    return preprocess_resize_normalized(src, mult(src.nx, target_size), mult(src.ny, target_size));
}

// ---------------------------------------------------------------------------
// PCA via power iteration with deflation
// ---------------------------------------------------------------------------

void pca_project_3d(const std::vector<float> &patch_tokens, int n_patches, int dim, int grid_w, int grid_h, int out_w,
                    int out_h, const std::string &out_path) {
    if (n_patches <= 0 || dim <= 0 || (int)patch_tokens.size() < n_patches * dim) {
        fprintf(stderr, "%s: invalid input\n", __func__);
        return;
    }

    // patch-grid dims must cover exactly n_patches
    int gw = grid_w;
    int gh = grid_h;
    if (gw <= 0 || gh <= 0 || gw * gh != n_patches) {
        // fall back: try divisors of n_patches closest to a square grid
        gw = (int)std::lround(std::sqrt((double)n_patches));
        while (gw > 1 && n_patches % gw != 0) {
            --gw;
        }
        gh = n_patches / gw;
        if (gw * gh != n_patches) {
            fprintf(stderr, "%s: could not derive patch-grid dims for %d patches, skipping\n", __func__, n_patches);
            return;
        }
        fprintf(stderr, "%s: invalid grid dims %d x %d, using %d x %d instead\n", __func__, grid_w, grid_h, gw, gh);
    }

    // mean-center tokens
    std::vector<float> X((size_t)n_patches * dim);
    std::vector<float> mean(dim, 0.0f);
    for (int i = 0; i < n_patches; ++i) {
        for (int d = 0; d < dim; ++d) {
            mean[d] += patch_tokens[(size_t)i * dim + d];
        }
    }
    for (int d = 0; d < dim; ++d) {
        mean[d] /= n_patches;
    }
    for (size_t i = 0; i < X.size(); ++i) {
        X[i] = patch_tokens[i] - mean[i % dim];
    }

    // power iteration for top-3 eigenvectors of the covariance with deflation
    std::vector<float> comps[3];
    std::vector<float> residual = X; // deflated data
    for (int k = 0; k < 3; ++k) {
        std::vector<float> v(dim, 0.0f);
        // deterministic init
        for (int d = 0; d < dim; ++d) {
            v[d] = 1.0f / (1.0f + d + k);
        }
        for (int iter = 0; iter < 100; ++iter) {
            // w = residual * v  (n_patches)
            std::vector<float> w(n_patches, 0.0f);
            for (int i = 0; i < n_patches; ++i) {
                float        s   = 0.0f;
                const float *row = &residual[(size_t)i * dim];
                for (int d = 0; d < dim; ++d) {
                    s += row[d] * v[d];
                }
                w[i] = s;
            }
            // v' = residual^T * w, normalize
            std::vector<float> vn(dim, 0.0f);
            for (int i = 0; i < n_patches; ++i) {
                const float wi = w[i];
                if (wi == 0.0f) {
                    continue;
                }
                const float *row = &residual[(size_t)i * dim];
                for (int d = 0; d < dim; ++d) {
                    vn[d] += wi * row[d];
                }
            }
            float norm = 0.0f;
            for (int d = 0; d < dim; ++d) {
                norm += vn[d] * vn[d];
            }
            norm = std::sqrt(norm);
            if (norm < 1e-12f) {
                break;
            }
            for (int d = 0; d < dim; ++d) {
                v[d] = vn[d] / norm;
            }
        }
        comps[k] = v;
        // deflation: residual -= (residual * v) v^T
        for (int i = 0; i < n_patches; ++i) {
            float  s   = 0.0f;
            float *row = &residual[(size_t)i * dim];
            for (int d = 0; d < dim; ++d) {
                s += row[d] * v[d];
            }
            for (int d = 0; d < dim; ++d) {
                row[d] -= s * v[d];
            }
        }
    }

    // project tokens to 3D
    std::vector<float> proj((size_t)n_patches * 3);
    for (int i = 0; i < n_patches; ++i) {
        const float *row = &X[(size_t)i * dim];
        for (int k = 0; k < 3; ++k) {
            float s = 0.0f;
            for (int d = 0; d < dim; ++d) {
                s += row[d] * comps[k][d];
            }
            proj[(size_t)i * 3 + k] = s;
        }
    }

    // min-max normalize per channel to 0-255
    uint8_t *px = new uint8_t[(size_t)n_patches * 3];
    for (int k = 0; k < 3; ++k) {
        float mn = proj[k], mx = proj[k];
        for (int i = 0; i < n_patches; ++i) {
            const float v = proj[(size_t)i * 3 + k];
            mn            = std::min(mn, v);
            mx            = std::max(mx, v);
        }
        const float range = (mx - mn) != 0.0f ? (mx - mn) : 1.0f;
        for (int i = 0; i < n_patches; ++i) {
            px[(size_t)i * 3 + k] = (uint8_t)std::lround((proj[(size_t)i * 3 + k] - mn) / range * 255.0f);
        }
    }

    // reshape to patch grid and upscale to out_w x out_h
    Image patch_img;
    patch_img.nx = gw;
    patch_img.ny = gh;
    patch_img.c  = 3;
    patch_img.data.assign(px, px + (size_t)n_patches * 3);
    delete[] px;

    Image out_img = resize_bicubic(patch_img, out_w, out_h);
    write_png(out_path, out_img);
    fprintf(stderr, "%s: saved PCA visualization to '%s' (%d x %d)\n", __func__, out_path.c_str(), out_w, out_h);
}
