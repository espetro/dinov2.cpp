#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct Image {
    int                  nx = 0, ny = 0, c = 3;
    std::vector<uint8_t> data; // interleaved, c channels per pixel
};

struct ImageF {
    int                nx = 0, ny = 0, c = 3;
    std::vector<float> data; // interleaved, c channels per pixel
};

// Load an image from disk as 3-channel RGB (8 bit per channel).
Image load_image(const std::string &path);

// Write an 8-bit RGB image to disk as PNG.
void write_png(const std::string &path, const Image &img);

// Bicubic (Catmull-Rom, matching OpenCV INTER_CUBIC) resize of a single-channel
// float plane.
std::vector<float> resize_bicubic_f32(const float *src, int sw, int sh, int dw, int dh);

// Bicubic (Catmull-Rom, matching OpenCV INTER_CUBIC) resize of an 8-bit RGB image.
Image resize_bicubic(const Image &src, int w, int h);

// Catmull-Rom bicubic kernel (support = 2). Exposed for unit tests.
float cubic_kernel(float x);

// Bicubic sampling of a single channel at fractional coordinates, with
// clamped edge handling. Exposed for unit tests.
float sample_cubic(const float *src, int sw, int sh, int c, int channels, float fx, float fy);

// DINOv2 preprocessing: resize to (target_size/patch+1)*patch-sized short side with
// bicubic, scale to [0,1], ImageNet mean/std normalization.
// c = 3, interleaved RGB float.
ImageF preprocess_for_dinov2(const Image &src, int target_size);

// Project patch tokens (n_patches x dim, row-major) to their top-3 principal
// components via power iteration with deflation, normalize per channel to
// 0-255, and write an upscaled RGB PNG to out_path.
void pca_project_3d(const std::vector<float> &patch_tokens, int n_patches, int dim, int grid_w, int grid_h, int out_w,
                    int out_h, const std::string &out_path);
