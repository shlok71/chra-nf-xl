#include "nca.h"
#include <iostream>

NCA::NCA(int width, int height) : width(width), height(height), grid(width * height, 0) {
    // Initialize grid with some pattern
    for (int i = 0; i < width * height; ++i) {
        grid[i] = i % 256;
    }
}

void NCA::step() {
    std::vector<uint8_t> nextGrid = grid;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // This is a placeholder for the 5-layer MLP
            // It should take the 3x3 neighborhood as input
            // and produce a new state for the cell (x, y)
            uint8_t neighborhood[9];
            for (int j = -1; j <= 1; ++j) {
                for (int i = -1; i <= 1; ++i) {
                    int nx = (x + i + width) % width;
                    int ny = (y + j + height) % height;
                    neighborhood[(j + 1) * 3 + (i + 1)] = grid[ny * width + nx];
                }
            }
            // Simple averaging for now
            uint32_t sum = 0;
            for (int i = 0; i < 9; ++i) {
                sum += neighborhood[i];
            }
            nextGrid[y * width + x] = sum / 9;
        }
    }
    grid = nextGrid;
}

const std::vector<uint8_t>& NCA::getGrid() const {
    return grid;
}

void NCA::applyMLP(int x, int y) {
    // This is a placeholder for the 5-layer MLP
    // It should take the 3x3 neighborhood as input
    // and produce a new state for the cell (x, y)
    uint8_t neighborhood[9];
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int nx = (x + i + width) % width;
            int ny = (y + j + height) % height;
            neighborhood[(j + 1) * 3 + (i + 1)] = grid[ny * width + nx];
        }
    }
    // Simple averaging for now
    uint32_t sum = 0;
    for (int i = 0; i < 9; ++i) {
        sum += neighborhood[i];
    }
    grid[y * width + x] = sum / 9;
}
