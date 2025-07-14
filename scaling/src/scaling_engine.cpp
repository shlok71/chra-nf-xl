#include "scaling_engine.h"
#include "system_detection.h"
#include <iostream>

#include <cpuid.h>

bool has_avx512() {
    unsigned int eax, ebx, ecx, edx;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & (1 << 16));
}

ScalingEngine::ScalingEngine() {
    SystemDetection system_detection;
    SystemSpecs specs = system_detection.get_system_specs();

    if (specs.ram_gb < 6) {
        current_config = ParamConfig::LOW;
    } else if (specs.ram_gb >= 6 && specs.ram_gb <= 12) {
        current_config = ParamConfig::MEDIUM;
    } else if (specs.ram_gb > 32 && has_avx512()) {
        current_config = ParamConfig::HIGH;
    } else if (specs.ram_gb > 128) {
        current_config = ParamConfig::ULTRA;
    } else {
        current_config = ParamConfig::MEDIUM;
    }
}

ParamConfig ScalingEngine::get_param_config() {
    return current_config;
}

void ScalingEngine::set_param_config(ParamConfig config) {
    current_config = config;
}
