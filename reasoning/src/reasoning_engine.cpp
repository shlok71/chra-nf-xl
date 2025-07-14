#include "reasoning_engine.h"
#include "onnxruntime_cxx_api.h"

static Ort::Env env;
static Ort::Session session{nullptr};

ReasoningEngine::ReasoningEngine() {
    // Load the ONNX model
    session = Ort::Session(env, "reasoning.onnx", Ort::SessionOptions{nullptr});
}

std::string ReasoningEngine::reason(const std::string& input) {
    // Prepare the input tensor
    BHV input_bhv = BHV::encode(input);
    std::vector<float> input_tensor_values(BHV_BITS);
    for (int i = 0; i < BHV_WORDS; ++i) {
        for (int j = 0; j < 64; ++j) {
            input_tensor_values[i * 64 + j] = (input_bhv.data[i] >> j) & 1;
        }
    }
    std::vector<int64_t> input_shape = {1, BHV_BITS};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Run inference
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Get the output
    float* output = output_tensors[0].GetTensorMutableData<float>();
    // This is a placeholder for decoding the output.
    // A real implementation would convert the output tensor to a string.
    return "The answer is " + std::to_string(output[0]);
}
