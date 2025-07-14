#include "system_detection.h"
#include <iostream>

SystemDetection::SystemDetection() {}

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#include <unistd.h>
#endif

SystemSpecs SystemDetection::get_system_specs() {
    SystemSpecs specs;
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    GlobalMemoryStatusEx(&statex);
    specs.ram_gb = statex.ullTotalPhys / (1024 * 1024 * 1024);
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    specs.cpu_cores = sysInfo.dwNumberOfProcessors;
#else
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    specs.ram_gb = memInfo.totalram / (1024 * 1024 * 1024);
    specs.cpu_cores = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    return specs;
}

void SystemDetection::scale_architecture(int ram_allocation_gb, int cpu_allocation_cores) {
    // This is a placeholder for scaling the AI's architecture.
    // A real implementation would adjust the model size, context length,
    // and other parameters based on the allocated resources.
    if (ram_allocation_gb >= 128 && cpu_allocation_cores >= 16) {
        std::cout << "Scaling up to 1T parameters and 256k context." << std::endl;
    } else {
        std::cout << "Using 300M parameter model and short context." << std::endl;
    }
}
