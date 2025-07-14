#pragma once

#include "bhv.h"
#include <vector>
#include <string>

#include "quantizer.h"

class ReasoningEngine {
public:
    ReasoningEngine();
    std::string reason(const std::string& input);

private:
    Quantizer quantizer;
    std::vector<uint8_t> quantized_weights;
};
