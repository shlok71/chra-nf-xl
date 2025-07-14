#pragma once

#include <vector>
#include <cstdint>

class Compressor {
public:
    Compressor();
    std::vector<uint8_t> compress(const std::vector<uint8_t>& data);
    std::vector<uint8_t> decompress(const std::vector<uint8_t>& compressed_data);
};
