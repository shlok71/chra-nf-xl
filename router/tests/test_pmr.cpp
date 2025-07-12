#include "gtest/gtest.h"
#include "pmr.h"
#include <numeric>

TEST(PMRTest, MaskCorrectness) {
    // The ONNX model should be copied to the build directory by CMake
    ProbabilisticMaskRouter router("router.onnx");
    BHV bhv = BHV::encode_text({"some", "test", "tokens"});

    std::vector<uint8_t> mask = router.get_mask(bhv);

    // Verify mask size
    ASSERT_EQ(mask.size(), 16); // 128 bits = 16 bytes

    // Verify that exactly 4 bits are set
    int set_bits = 0;
    for (uint8_t byte : mask) {
        set_bits += _mm_popcnt_u32(byte); // Count set bits in each byte
    }
    ASSERT_EQ(set_bits, 4);
}
