#include "sgr.h"
#include <limits>

#include <fstream>
#include <dlfcn.h>

SGR::SGR(const std::string& registry_path) {
    std::ifstream registry_file(registry_path);
    registry_file >> module_registry;

    for (const auto& module : module_registry["modules"]) {
        std::string task_name = module["name"].asString();
        TaskType task_type;
        if (task_name == "text") {
            task_type = TaskType::TEXT;
        } else if (task_name == "ocr") {
            task_type = TaskType::OCR;
        } else if (task_name == "canvas") {
            task_type = TaskType::CANVAS;
        } else if (task_name == "retrieval") {
            task_type = TaskType::RETRIEVAL;
        }
        task_vectors[task_type] = BHV::encode(task_name);
    }
}

TaskType SGR::route(const BHV& input) {
    int min_dist = std::numeric_limits<int>::max();
    TaskType best_task = TaskType::TEXT;

    for (auto const& [task_type, task_vector] : task_vectors) {
        int dist = BHV::hamming_distance(input, task_vector);
        if (dist < min_dist) {
            min_dist = dist;
            best_task = task_type;
        }
    }

    load_module(best_task);

    return best_task;
}

#include "onnxruntime_cxx_api.h"

static Ort::Env env;
static Ort::Session session{nullptr};

SGR::SGR(const std::string& registry_path) {
    std::ifstream registry_file(registry_path);
    registry_file >> module_registry;

    for (const auto& module : module_registry["modules"]) {
        std::string task_name = module["name"].asString();
        TaskType task_type;
        if (task_name == "text") {
            task_type = TaskType::TEXT;
        } else if (task_name == "ocr") {
            task_type = TaskType::OCR;
        } else if (task_name == "canvas") {
            task_type = TaskType::CANVAS;
        } else if (task_name == "retrieval") {
            task_type = TaskType::RETRIEVAL;
        }
        task_vectors[task_type] = BHV::encode(task_name);
    }

    // Load the ONNX model
    session = Ort::Session(env, "router.onnx", Ort::SessionOptions{nullptr});
}

TaskType SGR::route(const BHV& input) {
    // Prepare the input tensor
    std::vector<float> input_tensor_values(BHV_BITS);
    for (int i = 0; i < BHV_WORDS; ++i) {
        for (int j = 0; j < 64; ++j) {
            input_tensor_values[i * 64 + j] = (input.data[i] >> j) & 1;
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
    int best_task_index = 0;
    for (int i = 1; i < 4; ++i) {
        if (output[i] > output[best_task_index]) {
            best_task_index = i;
        }
    }

    return static_cast<TaskType>(best_task_index);
}
