#pragma once

#include <string>

class WhisperWrapper {
public:
    WhisperWrapper();
    std::string transcribe(const std::string& audio_file);
};
