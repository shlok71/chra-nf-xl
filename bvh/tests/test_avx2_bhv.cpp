#include "gtest/gtest.h"
#include "avx2_bhv.h"

// Test fixture for BHV tests
class BHVTest : public ::testing::Test {
protected:
    BHV v1, v2, v_zero;
    std::vector<std::string> tokens1 = {"hello", "world"};
    std::vector<std::string> tokens2 = {"hello", "gtest"};
};

// Test that a default-initialized BHV is all zeros.
TEST_F(BHVTest, Initialization) {
    for (int i = 0; i < BHV_WORDS; ++i) {
        ASSERT_EQ(v_zero.data[i], 0);
    }
}

// Test Hamming distance
TEST_F(BHVTest, HammingDistance) {
    v1 = BHV::encode_text(tokens1);
    v2 = BHV::encode_text(tokens2);

    // Distance to self should be 0
    ASSERT_EQ(BHV::hamming_distance(v1, v1), 0);
    // Distance to zero vector
    int dist_to_zero = BHV::hamming_distance(v1, v_zero);
    ASSERT_GT(dist_to_zero, 0);
    // Distance between two different vectors
    int dist_v1_v2 = BHV::hamming_distance(v1, v2);
    ASSERT_GT(dist_v1_v2, 0);
    ASSERT_NE(dist_v1_v2, dist_to_zero);
}

// Test bind operation (XOR)
TEST_F(BHVTest, Bind) {
    v1 = BHV::encode_text(tokens1);
    v2 = BHV::encode_text(tokens2);

    BHV v3 = BHV::bind(v1, v2);
    // bind(v1, v2) should have a non-zero Hamming distance to both v1 and v2
    ASSERT_GT(BHV::hamming_distance(v3, v1), 0);
    ASSERT_GT(BHV::hamming_distance(v3, v2), 0);

    // Test associativity: bind(v1, v2) == bind(v2, v1)
    BHV v4 = BHV::bind(v2, v1);
    ASSERT_EQ(BHV::hamming_distance(v3, v4), 0);

    // Test identity: bind(v1, v1) == zero vector
    BHV v5 = BHV::bind(v1, v1);
    ASSERT_EQ(BHV::hamming_distance(v5, v_zero), 0);
}

// Test round-trip encoding and comparison
TEST_F(BHVTest, RoundTrip) {
    v1 = BHV::encode_text(tokens1);
    BHV v1_again = BHV::encode_text(tokens1);
    ASSERT_EQ(BHV::hamming_distance(v1, v1_again), 0);
}

// Test a small vocabulary
TEST_F(BHVTest, SmallVocabulary) {
    std::vector<std::string> vocab = {"apple", "banana", "cherry"};
    BHV b_apple = BHV::encode_text({"apple"});
    BHV b_banana = BHV::encode_text({"banana"});
    BHV b_cherry = BHV::encode_text({"cherry"});

    ASSERT_GT(BHV::hamming_distance(b_apple, b_banana), 0);
    ASSERT_GT(BHV::hamming_distance(b_apple, b_cherry), 0);
    ASSERT_GT(BHV::hamming_distance(b_banana, b_cherry), 0);

    // Test binding two words
    BHV b_apple_banana = BHV::bind(b_apple, b_banana);
    BHV b_banana_apple = BHV::bind(b_banana, b_apple);
    ASSERT_EQ(BHV::hamming_distance(b_apple_banana, b_banana_apple), 0);
}
