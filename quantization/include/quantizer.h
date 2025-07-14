#pragma once

#include <vector>
#include <cstdint>

class Quantizer {
public:
    Quantizer();
    std::vector<uint8_t> quantize(const std::vector<float>& weights);
    std::vector<float> dequantize(const std::vector<uint8_t>& quantized_weights);
};
