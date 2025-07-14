#pragma once

#include <string>

enum class ParamConfig {
    LOW,
    MEDIUM,
    HIGH,
    ULTRA
};

class ScalingEngine {
public:
    ScalingEngine();
    ParamConfig get_param_config();
    void set_param_config(ParamConfig config);

private:
    ParamConfig current_config;
};
