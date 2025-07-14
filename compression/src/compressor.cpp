#include "compressor.h"
#include <iostream>

#include <lz4.h>

Compressor::Compressor() {}

std::vector<uint8_t> Compressor::compress(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> compressed_data(data.size());
    int compressed_size = LZ4_compress_default(
        reinterpret_cast<const char*>(data.data()),
        reinterpret_cast<char*>(compressed_data.data()),
        data.size(),
        compressed_data.size()
    );
    compressed_data.resize(compressed_size);
    return compressed_data;
}

std::vector<uint8_t> Compressor::decompress(const std::vector<uint8_t>& compressed_data) {
    // This is a placeholder for the decompressed size.
    // In a real implementation, the compressed data would include
    // the original size.
    std::vector<uint8_t> decompressed_data(compressed_data.size() * 2);
    int decompressed_size = LZ4_decompress_safe(
        reinterpret_cast<const char*>(compressed_data.data()),
        reinterpret_cast<char*>(decompressed_data.data()),
        compressed_data.size(),
        decompressed_data.size()
    );
    decompressed_data.resize(decompressed_size);
    return decompressed_data;
}
