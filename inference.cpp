#define CRT_SECURE_NO_DEPRECATE // disables "unsafe" warnings on Windows

#include "dinov2.h"
#include "ggml.h"
#include "src/image.h"
#include "ggml-alloc.h"
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>

#include "ggml-backend.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267) // possible loss of data
#endif

// main function
int main(int argc, char **argv) {
    ggml_time_init();
    dino_params params;
    dino_model  model;

    if (dino_params_parse(argc, argv, params) == false) {
        return 1;
    }

    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);

    // load the image
    Image img = load_image(params.fname_inp);
    if (img.data.empty()) {
        fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, params.fname_inp.c_str());
        return 1;
    }
    fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, params.fname_inp.c_str(), img.nx, img.ny);

    // load the model
    if (!dino_model_load({img.nx, img.ny}, params.model, model, params)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
        return 1;
    }

    ImageF img_f;
    if (params.classify) {
        img_f = dino_classify_preprocess(img, model.hparams);
    } else {
        img_f = dino_preprocess(img, model.hparams);
    }

    fprintf(stderr, "%s: preprocessed image (%d x %d)\n", __func__, img_f.nx, img_f.ny);

    // prepare for graph computation, memory allocation and results processing
    {
        ggml_backend_synchronize(model.backend);
        ggml_gallocr_t               allocr     = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
        int64_t                      start_time = ggml_time_ms();
        std::unique_ptr<dino_output> output     = dino_predict(model, img_f, params, allocr);
        ggml_backend_synchronize(model.backend);
        int64_t end_time = ggml_time_ms();
        fprintf(stderr, "%s: graph computation took %lld ms\n", __func__, end_time - start_time);

        ggml_free(model.ctx);
        ggml_gallocr_free(allocr);
        ggml_backend_buffer_free(model.buffer);
        ggml_backend_free(model.backend);

        if (!params.classify && output->patch_tokens) {
            const int patch_size = model.hparams.patch_size;
            const int out_w      = img_f.nx;
            const int out_h      = img_f.ny;
            const int n_patches  = (img_f.ny / patch_size) * (img_f.nx / patch_size);
            const int grid_w     = img_f.nx / patch_size;
            const int grid_h     = img_f.ny / patch_size;

            pca_project_3d(*output->patch_tokens, n_patches, model.hparams.hidden_size, grid_w, grid_h, out_w, out_h,
                           params.image_out);
            fprintf(stderr, "%s: Saved image to: %s\n", __func__, params.image_out.c_str());
        }
    }

    return 0;
}
