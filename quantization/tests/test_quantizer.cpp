#include <gtest/gtest.h>
#include "quantizer.h"

TEST(QuantizerTest, QuantizeDequantize) {
    Quantizer quantizer;
    std::vector<float> weights = {0.5, -0.8, 0.2, 0.9, -0.3};
    std::vector<uint8_t> quantized_weights = quantizer.quantize(weights);
    std::vector<float> dequantized_weights = quantizer.dequantize(quantized_weights);

    ASSERT_EQ(weights.size(), dequantized_weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        if (weights[i] > 0.1) {
            ASSERT_EQ(dequantized_weights[i], 1.0);
        } else if (weights[i] < -0.1) {
            ASSERT_EQ(dequantized_weights[i], -1.0);
        } else {
            ASSERT_EQ(dequantized_weights[i], 0.0);
        }
    }
}
