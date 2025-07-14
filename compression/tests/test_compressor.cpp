#include <gtest/gtest.h>
#include "compressor.h"

TEST(CompressorTest, CompressDecompress) {
    Compressor compressor;
    std::vector<uint8_t> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<uint8_t> compressed_data = compressor.compress(data);
    std::vector<uint8_t> decompressed_data = compressor.decompress(compressed_data);
    ASSERT_EQ(data, decompressed_data);
}
