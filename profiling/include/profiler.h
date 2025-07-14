#pragma once

#include <string>
#include <vector>

class Profiler {
public:
    Profiler();
    void start_profiling();
    void stop_profiling();

private:
    void profile();
    void optimize();
};
