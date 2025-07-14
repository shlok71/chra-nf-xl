#include <gtest/gtest.h>
#include "reasoning_engine.h"

TEST(ReasoningEngineTest, Reason) {
    ReasoningEngine engine;
    std::string input = "What is the meaning of life?";
    std::string output = engine.reason(input);
    ASSERT_EQ(output, "The answer is 42.");
}
