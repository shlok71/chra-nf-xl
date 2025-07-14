#pragma once

#include <string>

struct SystemSpecs {
    int ram_gb;
    int cpu_cores;
};

class SystemDetection {
public:
    SystemDetection();
    SystemSpecs get_system_specs();
    void scale_architecture(int ram_allocation_gb, int cpu_allocation_cores);
};
