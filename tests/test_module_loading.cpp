#include <gtest/gtest.h>
#include "sgr.h"

TEST(ModuleLoadingTest, LoadTextModule) {
    SGR sgr("../module_registry.json");
    BHV text_bhv = BHV::encode("text");
    TaskType task_type = sgr.route(text_bhv);
    ASSERT_EQ(task_type, TaskType::TEXT);
}
