#pragma once

#include <string>

class TTSWrapper {
public:
    TTSWrapper();
    void speak(const std::string& text);
};
