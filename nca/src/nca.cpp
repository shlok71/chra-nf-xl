#include "../include/nca.h"
#include <iostream>

NCA::NCA(int width, int height) : width(width), height(height), history_index(-1) {
    grid.resize(width * height * 4); // 4 channels: RGBA

    // Define the MLP architecture
    mlp.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(1, 4, 32, 3, 3, 1, 1));
    mlp.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(1, 32, 64, 3, 3, 1, 1));
    mlp.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(1, 64, 128, 3, 3, 1, 1));
    mlp.add_layer(new MiniDNN::Convolutional<MiniDNN::ReLU>(1, 128, 64, 3, 3, 1, 1));
    mlp.add_layer(new MiniDNN::Convolutional<MiniDNN::Identity>(1, 64, 4, 3, 3, 1, 1)); // 4 output channels

    save_history();
}

void NCA::save_history() {
    if (history_index < (int)history.size() - 1) {
        history.erase(history.begin() + history_index + 1, history.end());
    }
    history.push_back(grid);
    history_index++;
}

void NCA::undo() {
    if (history_index > 0) {
        history_index--;
        grid = history[history_index];
    }
}

void NCA::redo() {
    if (history_index < (int)history.size() - 1) {
        history_index++;
        grid = history[history_index];
    }
}

void NCA::step() {
    save_history();
    // Convert grid to MiniDNN::Matrix
    MiniDNN::Matrix input(height, width, 4);
    for (int i = 0; i < grid.size(); ++i) {
        input.data()[i] = grid[i] / 255.0;
    }

    // Predict the next state
    MiniDNN::Matrix output = mlp.predict(input);

    // Convert output back to grid
    for (int i = 0; i < grid.size(); ++i) {
        grid[i] = std::min(255, std::max(0, (int)(output.data()[i] * 255.0)));
    }
}

const std::vector<uint8_t>& NCA::getGrid() const {
    return grid;
}

void NCA::load_weights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    mlp.load_parameters(file);
}

void NCA::set_pixel(int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    save_history();
    if (x < 0 || x >= width || y < 0 || y >= height) {
        return;
    }
    int index = (y * width + x) * 4;
    grid[index] = r;
    grid[index + 1] = g;
    grid[index + 2] = b;
    grid[index + 3] = a;
}

void NCA::draw_circle(int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    save_history();
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            if (x * x + y * y <= radius * radius) {
                if (cx + x < 0 || cx + x >= width || cy + y < 0 || cy + y >= height) {
                    continue;
                }
                int index = ((cy + y) * width + (cx + x)) * 4;
                grid[index] = r;
                grid[index + 1] = g;
                grid[index + 2] = b;
                grid[index + 3] = a;
            }
        }
    }
}

void NCA::settle() {
    for (int i = 0; i < 100; ++i) {
        step();
    }
}

BHV NCA::transcribe_to_bhv() {
    // This is a placeholder
    return BHV();
}
