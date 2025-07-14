#include "nca_canvas.h"
#include <cmath>

NCACanvas::NCACanvas(int width, int height) : width(width), height(height), grid(width* height, 0) {}

void NCACanvas::edit_pixel(int x, int y, uint8_t value) {
    if (x >= 0 && x < width && y >= 0 && y < height) {
        grid[y * width + x] = value;
    }
}

void NCACanvas::draw_circle(int cx, int cy, int r, uint8_t value) {
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (std::sqrt(std::pow(x - cx, 2) + std::pow(y - cy, 2)) < r) {
                grid[y * width + x] = value;
            }
        }
    }
}

#include "onnxruntime_cxx_api.h"

static Ort::Env env;
static Ort::Session session{nullptr};

NCACanvas::NCACanvas(int width, int height) : width(width), height(height), grid(width* height, 0) {
    // Load the ONNX model
    session = Ort::Session(env, "nca.onnx", Ort::SessionOptions{nullptr});
}

void NCACanvas::step() {
    // Prepare the input tensor
    std::vector<float> input_tensor_values(width * height * 9);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = (x + i + width) % width;
                    int ny = (y + j + height) % height;
                    input_tensor_values[(y * width + x) * 9 + (j + 1) * 3 + (i + 1)] = grid[ny * width + nx];
                }
            }
        }
    }
    std::vector<int64_t> input_shape = {1, width, height, 9};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());

    // Run inference
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Get the output
    float* output = output_tensors[0].GetTensorMutableData<float>();
    for (int i = 0; i < width * height; ++i) {
        grid[i] = static_cast<uint8_t>(output[i] * 255);
    }
}

BHV NCACanvas::to_bhv() {
    // This is a placeholder for converting the canvas to a BHV.
    // A real implementation would use a deterministic hashing or
    // permutation logic to convert the grid to a BHV.
    return BHV::encode("canvas_bhv");
}

std::vector<uint8_t> NCACanvas::generate_image(const BHV& seed) {
    // This is a placeholder for image generation.
    // A real implementation would use the seed to initialize the grid
    // and then evolve it over a number of steps.
    grid.assign(grid.size(), seed.data[0] % 256);
    for (int i = 0; i < 10; ++i) {
        step();
    }
    return grid;
}

std::vector<std::vector<uint8_t>> NCACanvas::generate_video(const BHV& seed, int num_frames) {
    // This is a placeholder for video generation.
    // A real implementation would generate a sequence of frames
    // by evolving the grid.
    std::vector<std::vector<uint8_t>> frames;
    grid.assign(grid.size(), seed.data[0] % 256);
    for (int i = 0; i < num_frames; ++i) {
        step();
        frames.push_back(grid);
    }
    return frames;
}
