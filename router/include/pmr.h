#pragma once

#include "avx2_bhv.h"
#include <onnxruntime_cxx_api.h>
#include <vector>

class ProbabilisticMaskRouter {
public:
    ProbabilisticMaskRouter(const std::string& model_path);

    // Generate a 128-bit mask with 4 bits set to 1
    std::vector<uint8_t> get_mask(const BHV& bhv);

private:
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
};
