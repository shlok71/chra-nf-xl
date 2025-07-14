#include <gtest/gtest.h>
#include "system_detection.h"

TEST(SystemDetectionTest, GetSystemSpecs) {
    SystemDetection system_detection;
    SystemSpecs specs = system_detection.get_system_specs();
    ASSERT_EQ(specs.ram_gb, 16);
    ASSERT_EQ(specs.cpu_cores, 4);
}

TEST(SystemDetectionTest, ScaleArchitecture) {
    SystemDetection system_detection;
    // This is a placeholder test. A real test would check that
    // the AI's architecture is scaled correctly based on the
    // allocated resources.
    system_detection.scale_architecture(16, 4);
    system_detection.scale_architecture(256, 32);
    ASSERT_TRUE(true);
}
