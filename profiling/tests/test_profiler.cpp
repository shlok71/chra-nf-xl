#include <gtest/gtest.h>
#include "profiler.h"

TEST(ProfilerTest, StartStop) {
    Profiler profiler;
    // This is a placeholder test. A real test would check that
    // the profiler starts and stops correctly.
    profiler.start_profiling();
    profiler.stop_profiling();
    ASSERT_TRUE(true);
}
