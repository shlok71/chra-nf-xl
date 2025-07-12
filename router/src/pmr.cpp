#include "pmr.h"
#include <algorithm>
#include <vector>
#include <numeric>

ProbabilisticMaskRouter::ProbabilisticMaskRouter(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "PMR"),
      session(env, model_path.c_str(), Ort::SessionOptions{nullptr}) {

    // Get input and output shapes
    input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    output_shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

std::vector<uint8_t> ProbabilisticMaskRouter::get_mask(const BHV& bhv) {
    // 1. Prepare input tensor
    // The model expects a float tensor, so we convert the BHV data.
    std::vector<float> input_tensor_values(BHV_BITS);
    for (int i = 0; i < BHV_WORDS; ++i) {
        for (int j = 0; j < 64; ++j) {
            input_tensor_values[i * 64 + j] = (bhv.data[i] >> j) & 1;
        }
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    // 2. Run inference
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names, &input_tensor, 1,
                                      output_names, 1);

    // 3. Process output to get top-k mask
    float* float_array = output_tensors[0].GetTensorMutableData<float>();

    // Get indices of the top 4 scores
    std::vector<int> indices(output_shape[1]);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + 4, indices.end(),
                      [&](int A, int B) {
                          return float_array[A] > float_array[B];
                      });

    // Create the 128-bit mask (16 bytes)
    std::vector<uint8_t> mask(16, 0);
    for (int i = 0; i < 4; ++i) {
        int idx = indices[i];
        mask[idx / 8] |= (1 << (idx % 8));
    }

    return mask;
}
