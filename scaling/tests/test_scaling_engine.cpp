#include <gtest/gtest.h>
#include "scaling_engine.h"

TEST(ScalingEngineTest, GetParamConfig) {
    ScalingEngine engine;
    // This is a placeholder test. A real test would mock the
    // system detection and check that the correct param config is selected.
    ASSERT_TRUE(true);
}

TEST(ScalingEngineTest, SetParamConfig) {
    ScalingEngine engine;
    engine.set_param_config(ParamConfig::HIGH);
    ASSERT_EQ(engine.get_param_config(), ParamConfig::HIGH);
}
